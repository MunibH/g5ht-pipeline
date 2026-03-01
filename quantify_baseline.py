"""
quantify_baseline.py — Baseline fluorescence analysis across strains.

For recordings that have a defined baseline_window in metadata.json, loads the
registered TIF stacks over that window and computes per-ROI baseline statistics
for both the GFP and RFP channels individually, as well as the ratiometric
signal R = GFP/RFP.

Produces the following plots:
    1. Ratiometric baseline: mean ± std of R per ROI, grouped by strain (bar plot).
    2. GFP vs RFP scatter: mean baseline GFP vs mean baseline RFP per ROI,
       colored by strain.
    3. Per-channel bar plots: mean ± std of raw GFP and RFP per ROI, grouped
       by strain.
    4. Coefficient of variation (CV = std/mean) of R per ROI, grouped by strain
       — captures how stable the baseline is.
    5. Strip/swarm plot of per-frame ratiometric values during baseline, grouped
       by strain — shows the full distribution, not just mean/std.

Usage:
    python quantify_baseline.py <data_root>

    From a notebook:
        import quantify_baseline
        sys.argv = ["", data_root]
        quantify_baseline.main()

Arguments:
    sys.argv[1]  data_root : str — Root data directory (e.g. D:\\DATA\\g5ht-free).
                 The script walks all date/worm subdirectories looking for
                 metadata.json, roi.tif, and registered TIFs.
    sys.argv[2]  reg_dir   : str — Name of registered TIF subdirectory
                 (default: 'registered_elastix').
    sys.argv[3]  drop_dnc  : int — 0 or 1 (default: 1). Drop dorsal_nerve_cord.
"""

import json
import sys
import os
import glob

import numpy as np
import pandas as pd
import tifffile
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import pretty_plot, default_plt_params
from quantify_roi import (
    load_metadata, load_roi, LABEL_COLORS, get_label_color, _FALLBACK_COLORS,
)

default_plt_params()

# ── Strain color palette ──────────────────────────────────────────────
# Use a qualitative colormap so each strain gets a clearly distinct color.
_STRAIN_CMAP = plt.cm.get_cmap('tab10')


def _strain_color(strain, strain_list):
    """Return a unique color for each strain."""
    idx = strain_list.index(strain) if strain in strain_list else 0
    return _STRAIN_CMAP(idx % 10)


# ── Data collection ───────────────────────────────────────────────────

def collect_baseline_stats(data_root, reg_dir='registered_elastix', drop_dnc=True):
    """Walk data_root, find recordings with baseline windows, compute stats.

    Returns
    -------
    records : list[dict]
        Each dict has keys: worm_pth, strain, condition, roi_labels,
        and per-ROI dicts with gfp_mean, gfp_std, rfp_mean, rfp_std,
        ratio_mean, ratio_std, ratio_cv, ratio_frames (array of per-frame R).
    """
    worm_tuple = tuple(f'worm{i:03d}' for i in range(1, 20))

    date_pths = sorted([
        os.path.join(data_root, d)
        for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])

    records = []

    for date_pth in date_pths:
        worm_pths = sorted([
            os.path.join(date_pth, d)
            for d in os.listdir(date_pth)
            if os.path.isdir(os.path.join(date_pth, d)) and d.endswith(worm_tuple)
        ])
        for worm_pth in worm_pths:
            meta_path = os.path.join(worm_pth, 'metadata.json')
            roi_path = os.path.join(worm_pth, 'roi.tif')
            if not os.path.exists(meta_path) or not os.path.exists(roi_path):
                continue

            meta = load_metadata(worm_pth, require=False)
            if meta is None or meta.get('baseline_window') is None:
                continue

            bw = meta['baseline_window']
            bad_frames = set(meta['bad_frames'].tolist())

            # Parse strain / condition from folder name
            basename = os.path.basename(worm_pth)
            parts = {
                p.split('-', 1)[0]: p.split('-', 1)[1]
                for p in basename.split('_') if '-' in p
            }
            strain = parts.get('strain', 'unknown')
            condition = parts.get('condition', 'unknown')

            # Load ROIs
            roi, roi_labels = load_roi(worm_pth)
            if drop_dnc:
                roi_labels = [l for l in roi_labels if l not in ('dorsal_nerve_cord', 'DNC', 'ventral_nerve_cord', 'VNC', 
                                                                 'procorpus', 'metacorpus', 'isthmus', 'terminal_bulb', 'mtea')]

            # Discover registered TIFs
            registered_dir = os.path.join(worm_pth, reg_dir)
            if not os.path.isdir(registered_dir):
                continue
            tif_paths = sorted(
                glob.glob(os.path.join(registered_dir, '*.tif')),
                key=lambda x: int(os.path.basename(x).split('.')[0]),
            )
            if not tif_paths:
                continue

            nlabels = len(roi_labels)
            T = len(tif_paths)

            # Restrict to baseline window frames, skip bad frames
            bl_frames = [f for f in range(bw[0], min(bw[1], T)) if f not in bad_frames]
            if len(bl_frames) == 0:
                continue

            # Accumulate per-frame GFP and RFP means within each ROI
            gfp_per_frame = np.full((len(bl_frames), nlabels), np.nan)
            rfp_per_frame = np.full((len(bl_frames), nlabels), np.nan)

            # Pre-compute ROI masks (original label index in roi.tif)
            with tifffile.TiffFile(os.path.join(worm_pth, 'roi.tif')) as tif:
                all_labels = tif.imagej_metadata['Labels']
            roi_masks = {}
            for j, label in enumerate(roi_labels):
                roi_idx = all_labels.index(label) + 1
                roi_masks[j] = roi == roi_idx

            for fi, frame_idx in enumerate(bl_frames):
                stack = tifffile.imread(tif_paths[frame_idx])  # (Z, 2, H, W)
                for j, label in enumerate(roi_labels):
                    mask = roi_masks[j]
                    gfp_vals = stack[:, 0][mask].astype(np.float64)
                    rfp_vals = stack[:, 1][mask].astype(np.float64)
                    if len(gfp_vals) > 0:
                        gfp_per_frame[fi, j] = np.mean(gfp_vals)
                    if len(rfp_vals) > 0:
                        rfp_per_frame[fi, j] = np.mean(rfp_vals)

            # Compute per-ROI statistics
            roi_stats = {}
            for j, label in enumerate(roi_labels):
                gfp = gfp_per_frame[:, j]
                rfp = rfp_per_frame[:, j]
                # Per-frame ratio
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where(rfp > 0, gfp / rfp, np.nan)

                gfp_finite = gfp[np.isfinite(gfp)]
                rfp_finite = rfp[np.isfinite(rfp)]
                ratio_finite = ratio[np.isfinite(ratio)]

                roi_stats[label] = {
                    'gfp_mean': np.mean(gfp_finite) if len(gfp_finite) else np.nan,
                    'gfp_std': np.std(gfp_finite) if len(gfp_finite) else np.nan,
                    'rfp_mean': np.mean(rfp_finite) if len(rfp_finite) else np.nan,
                    'rfp_std': np.std(rfp_finite) if len(rfp_finite) else np.nan,
                    'ratio_mean': np.mean(ratio_finite) if len(ratio_finite) else np.nan,
                    'ratio_std': np.std(ratio_finite) if len(ratio_finite) else np.nan,
                    'ratio_cv': (np.std(ratio_finite) / np.mean(ratio_finite)
                                 if len(ratio_finite) and np.mean(ratio_finite) != 0
                                 else np.nan),
                    'ratio_frames': ratio_finite,
                }

            worm_id = basename.split('_')[-1]
            records.append({
                'worm_pth': worm_pth,
                'worm_id': worm_id,
                'strain': strain,
                'condition': condition,
                'roi_labels': roi_labels,
                'roi_stats': roi_stats,
                'baseline_window': bw,
            })
            print(f"  {basename}: {len(bl_frames)} baseline frames, "
                  f"{nlabels} ROIs, strain={strain}")

    return records


# ── Plotting ──────────────────────────────────────────────────────────

def _build_summary_df(records):
    """Flatten records into a tidy DataFrame for plotting."""
    rows = []
    for r in records:
        for label, stats in r['roi_stats'].items():
            rows.append({
                'strain': r['strain'],
                'condition': r['condition'],
                'worm_id': r['worm_id'],
                'worm_pth': r['worm_pth'],
                'roi': label,
                'gfp_mean': stats['gfp_mean'],
                'gfp_std': stats['gfp_std'],
                'rfp_mean': stats['rfp_mean'],
                'rfp_std': stats['rfp_std'],
                'ratio_mean': stats['ratio_mean'],
                'ratio_std': stats['ratio_std'],
                'ratio_cv': stats['ratio_cv'],
            })
    return pd.DataFrame(rows)


def plot_ratio_bar(summary_df, strains, roi_labels):
    """Plot 1: Ratiometric mean ± std per ROI, grouped by strain."""
    n_rois = len(roi_labels)
    n_strains = len(strains)
    bar_width = 0.8 / max(n_strains, 1)

    fig, ax = pretty_plot(figsize=(max(10, 2 * n_rois * n_strains), 5))
    x_base = np.arange(n_rois)

    for si, strain in enumerate(strains):
        sub = summary_df[summary_df['strain'] == strain]
        means = []
        stds = []
        for roi in roi_labels:
            roi_sub = sub[sub['roi'] == roi]
            means.append(roi_sub['ratio_mean'].mean())
            stds.append(roi_sub['ratio_mean'].std())  # across-recording variability
        x = x_base + si * bar_width
        ax.bar(x, means, bar_width, yerr=stds, label=strain,
               color=_strain_color(strain, strains), alpha=0.85,
               capsize=4, edgecolor='k', linewidth=0.5)

    ax.set_xticks(x_base + bar_width * (n_strains - 1) / 2)
    ax.set_xticklabels(roi_labels, rotation=30, ha='right')
    ax.set_ylabel('Baseline R (GFP/RFP)')
    ax.set_title('Baseline ratiometric level per ROI (mean ± std across recordings)')
    ax.legend(title='Strain')
    plt.tight_layout()
    plt.show()


def plot_gfp_vs_rfp_scatter(summary_df, strains, roi_labels):
    """Plot 2: Scatter of mean baseline GFP vs mean baseline RFP per ROI."""
    n_rois = len(roi_labels)
    ncols = min(4, n_rois)
    nrows = int(np.ceil(n_rois / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for ri, roi in enumerate(roi_labels):
        row, col = divmod(ri, ncols)
        ax = axes[row, col]
        for strain in strains:
            sub = summary_df[(summary_df['strain'] == strain) & (summary_df['roi'] == roi)]
            ax.scatter(sub['rfp_mean'], sub['gfp_mean'],
                       color=_strain_color(strain, strains), label=strain,
                       s=60, edgecolor='k', linewidth=0.4, zorder=3)
        ax.set_xlabel('Mean baseline RFP')
        ax.set_ylabel('Mean baseline GFP')
        ax.set_title(roi, fontweight='bold', color=get_label_color(roi))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legend on first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0, 0].legend(by_label.values(), by_label.keys(), fontsize=9, title='Strain')

    # Hide unused
    for ri in range(n_rois, nrows * ncols):
        row, col = divmod(ri, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle('Mean baseline GFP vs RFP per ROI', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_channel_bars(summary_df, strains, roi_labels):
    """Plot 3: Per-channel (GFP, RFP) bar plots per ROI, grouped by strain."""
    n_rois = len(roi_labels)
    n_strains = len(strains)
    bar_width = 0.8 / max(n_strains, 1)

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 2 * n_rois * n_strains), 5))

    for ch_idx, (ch_name, mean_col, std_col) in enumerate([
        ('GFP', 'gfp_mean', 'gfp_std'),
        ('RFP', 'rfp_mean', 'rfp_std'),
    ]):
        ax = axes[ch_idx]
        x_base = np.arange(n_rois)
        for si, strain in enumerate(strains):
            sub = summary_df[summary_df['strain'] == strain]
            means = []
            stds = []
            for roi in roi_labels:
                roi_sub = sub[sub['roi'] == roi]
                means.append(roi_sub[mean_col].mean())
                stds.append(roi_sub[mean_col].std())
            x = x_base + si * bar_width
            ax.bar(x, means, bar_width, yerr=stds, label=strain,
                   color=_strain_color(strain, strains), alpha=0.85,
                   capsize=4, edgecolor='k', linewidth=0.5)
        ax.set_xticks(x_base + bar_width * (n_strains - 1) / 2)
        ax.set_xticklabels(roi_labels, rotation=30, ha='right')
        ax.set_ylabel(f'Baseline {ch_name} (mean intensity)')
        ax.set_title(f'{ch_name} baseline level per ROI')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(title='Strain', fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_ratio_cv(summary_df, strains, roi_labels):
    """Plot 4: Coefficient of variation of R per ROI, grouped by strain."""
    n_rois = len(roi_labels)
    n_strains = len(strains)
    bar_width = 0.8 / max(n_strains, 1)

    fig, ax = pretty_plot(figsize=(max(10, 2 * n_rois * n_strains), 5))
    x_base = np.arange(n_rois)

    for si, strain in enumerate(strains):
        sub = summary_df[summary_df['strain'] == strain]
        cvs = []
        for roi in roi_labels:
            roi_sub = sub[sub['roi'] == roi]
            cvs.append(roi_sub['ratio_cv'].mean())
        x = x_base + si * bar_width
        ax.bar(x, cvs, bar_width, label=strain,
               color=_strain_color(strain, strains), alpha=0.85,
               edgecolor='k', linewidth=0.5)

    ax.set_xticks(x_base + bar_width * (n_strains - 1) / 2)
    ax.set_xticklabels(roi_labels, rotation=30, ha='right')
    ax.set_ylabel('CV (std / mean)')
    ax.set_title('Baseline ratiometric CV per ROI — lower = more stable baseline')
    ax.legend(title='Strain')
    plt.tight_layout()
    plt.show()


def plot_ratio_strip(records, strains, roi_labels):
    """Plot 5: Strip plot of per-frame R values during baseline, by strain."""
    n_rois = len(roi_labels)
    ncols = min(4, n_rois)
    nrows = int(np.ceil(n_rois / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for ri, roi in enumerate(roi_labels):
        row, col = divmod(ri, ncols)
        ax = axes[row, col]
        jitter_width = 0.3

        for si, strain in enumerate(strains):
            strain_records = [r for r in records if r['strain'] == strain]
            all_vals = []
            for r in strain_records:
                if roi in r['roi_stats']:
                    all_vals.extend(r['roi_stats'][roi]['ratio_frames'].tolist())
            if not all_vals:
                continue
            all_vals = np.array(all_vals)
            x_jitter = si + np.random.uniform(-jitter_width / 2, jitter_width / 2,
                                               size=len(all_vals))
            ax.scatter(x_jitter, all_vals, s=6, alpha=0.4,
                       color=_strain_color(strain, strains), zorder=2)
            # Overlay mean ± std
            ax.errorbar(si, np.mean(all_vals), yerr=np.std(all_vals),
                        fmt='o', color='k', markersize=6, capsize=5, zorder=3)

        ax.set_xticks(range(len(strains)))
        ax.set_xticklabels(strains, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('R (GFP/RFP)')
        ax.set_title(roi, fontweight='bold', color=get_label_color(roi))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for ri in range(n_rois, nrows * ncols):
        row, col = divmod(ri, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle('Per-frame baseline R distribution by strain', fontsize=16)
    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    """
    Baseline fluorescence analysis across strains.

    Usage
    -----
    From the command line:
        python quantify_baseline.py <data_root> [reg_dir] [drop_dnc]

    From a notebook:
        import quantify_baseline
        sys.argv = ["", r"D:\\DATA\\g5ht-free"]
        quantify_baseline.main()

    Arguments
    ---------
    sys.argv[1]  data_root : str — Root data directory.
    sys.argv[2]  reg_dir   : str — Registered TIF subdirectory (default: 'registered_elastix').
    sys.argv[3]  drop_dnc  : int — 0 or 1 (default: 1). Drop dorsal_nerve_cord.
    """
    if len(sys.argv) < 2:
        print("Usage: python quantify_baseline.py <data_root> [reg_dir] [drop_dnc]")
        sys.exit(1)

    data_root = sys.argv[1]
    reg_dir = sys.argv[2] if len(sys.argv) > 2 else 'registered_elastix'
    drop_dnc = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True

    print(f"Scanning {data_root} for recordings with baselines …")
    records = collect_baseline_stats(data_root, reg_dir=reg_dir, drop_dnc=drop_dnc)
    print(f"\nCollected {len(records)} recordings with defined baselines")

    if len(records) == 0:
        print("No recordings with baseline windows found. Exiting.")
        return

    strains = sorted(set(r['strain'] for r in records))
    print(f"Strains: {strains}")

    # Union of ROI labels across recordings
    all_roi_labels = set()
    for r in records:
        all_roi_labels.update(r['roi_labels'])
    roi_labels = sorted(all_roi_labels)
    print(f"ROI labels: {roi_labels}")

    # Build tidy summary DataFrame
    summary_df = _build_summary_df(records)

    # ── Plot 1: Ratiometric baseline bar ───────────────────────────────
    print("\nPlot 1: Ratiometric baseline mean ± std")
    plot_ratio_bar(summary_df, strains, roi_labels)

    # ── Plot 2: GFP vs RFP scatter ────────────────────────────────────
    print("Plot 2: GFP vs RFP scatter")
    plot_gfp_vs_rfp_scatter(summary_df, strains, roi_labels)

    # ── Plot 3: Per-channel bar plots ─────────────────────────────────
    print("Plot 3: Per-channel (GFP, RFP) baseline levels")
    plot_channel_bars(summary_df, strains, roi_labels)

    # ── Plot 4: CV of baseline R ──────────────────────────────────────
    print("Plot 4: Coefficient of variation of baseline R")
    plot_ratio_cv(summary_df, strains, roi_labels)

    # ── Plot 5: Strip plot of per-frame R ─────────────────────────────
    print("Plot 5: Per-frame baseline R distribution (strip plot)")
    plot_ratio_strip(records, strains, roi_labels)


if __name__ == '__main__':
    main()
