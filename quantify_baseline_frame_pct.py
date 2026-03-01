"""
quantify_baseline_frame_pct.py — Whole-frame baseline intensity analysis.

Two analyses are performed for each recording with a baseline window:

  A) **Percentile mode**: For every baseline frame, compute the user-specified
     percentile of GFP and RFP intensity across all pixels in the frame.
     Plots:
       1. Per-frame percentile values during baseline (strip plot, by strain)
          — three panels: GFP, RFP, and R = GFP/RFP.
       2. Per-recording mean of the per-frame percentiles (bar, by strain)
          — three panels: GFP, RFP, and R.
       3. GFP-percentile vs RFP-percentile scatter, one dot per recording.

  B) **Whole-frame mean mode**: For every baseline frame, compute the average
     intensity of GFP and RFP across all pixels in the frame (i.e. percentile
     is not used — just the arithmetic mean).
     Plots:
       4. Per-frame mean values during baseline (strip plot, by strain)
          — three panels: GFP, RFP, and R = GFP/RFP.
       5. Per-recording mean of the per-frame means (bar, by strain)
          — three panels: GFP, RFP, and R.
       6. Mean-GFP vs Mean-RFP scatter, one dot per recording.

Usage:
    python quantify_baseline_frame_pct.py <data_root> [percentile] [reg_dir]

    From a notebook:
        import quantify_baseline_frame_pct as qbf
        sys.argv = ["", data_root]
        qbf.main()

Arguments:
    sys.argv[1]  data_root  : str  — Root data directory (e.g. D:\\DATA\\g5ht-free).
    sys.argv[2]  percentile : float — Percentile to compute (default: 95).
    sys.argv[3]  reg_dir    : str  — Registered TIF subdirectory
                               (default: 'registered_elastix').
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
from quantify_roi import load_metadata

default_plt_params()

# ── Strain colour palette ─────────────────────────────────────────────
_STRAIN_CMAP = plt.cm.get_cmap('tab10')


def _strain_color(strain, strain_list):
    idx = strain_list.index(strain) if strain in strain_list else 0
    return _STRAIN_CMAP(idx % 10)


# ── Data collection ───────────────────────────────────────────────────

def collect_frame_baseline(data_root, percentile=95.0,
                           reg_dir='registered_elastix'):
    """Walk *data_root*, find recordings with baseline windows.

    For each baseline frame (skipping bad frames) compute:
        * ``np.percentile(channel, percentile)``   (percentile mode)
        * ``np.mean(channel)``                     (mean mode)

    Returns
    -------
    records : list[dict]
        One entry per valid recording.  Keys include:
        ``strain``, ``condition``, ``worm_id``, ``worm_pth``,
        ``pct_gfp``  (1-D array, per-frame percentile GFP),
        ``pct_rfp``  (1-D array),
        ``mean_gfp`` (1-D array, per-frame mean GFP),
        ``mean_rfp`` (1-D array).
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
            if not os.path.exists(meta_path):
                continue

            meta = load_metadata(worm_pth, require=False)
            if meta is None or meta.get('baseline_window') is None:
                continue

            bw = meta['baseline_window']
            bad_frames = set(meta['bad_frames'].tolist())

            # Parse strain / condition
            basename = os.path.basename(worm_pth)
            parts = {
                p.split('-', 1)[0]: p.split('-', 1)[1]
                for p in basename.split('_') if '-' in p
            }
            strain = parts.get('strain', 'unknown')
            condition = parts.get('condition', 'unknown')

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

            T = len(tif_paths)

            # Baseline frames, excluding bad
            bl_frames = [f for f in range(bw[0], min(bw[1], T))
                         if f not in bad_frames]
            if len(bl_frames) == 0:
                continue

            pct_gfp = np.full(len(bl_frames), np.nan)
            pct_rfp = np.full(len(bl_frames), np.nan)
            mean_gfp = np.full(len(bl_frames), np.nan)
            mean_rfp = np.full(len(bl_frames), np.nan)

            for fi, frame_idx in enumerate(tqdm(
                    bl_frames,
                    desc=f"  {basename}",
                    leave=False)):
                stack = tifffile.imread(tif_paths[frame_idx])  # (Z, 2, H, W)
                gfp = stack[:, 0].astype(np.float64).ravel()
                rfp = stack[:, 1].astype(np.float64).ravel()

                pct_gfp[fi] = np.percentile(gfp, percentile)
                pct_rfp[fi] = np.percentile(rfp, percentile)
                mean_gfp[fi] = np.mean(gfp)
                mean_rfp[fi] = np.mean(rfp)

            # Per-frame ratios
            with np.errstate(divide='ignore', invalid='ignore'):
                pct_ratio = np.where(pct_rfp > 0, pct_gfp / pct_rfp, np.nan)
                mean_ratio = np.where(mean_rfp > 0, mean_gfp / mean_rfp, np.nan)

            worm_id = basename.split('_')[-1]
            records.append({
                'worm_pth': worm_pth,
                'worm_id': worm_id,
                'strain': strain,
                'condition': condition,
                'pct_gfp': pct_gfp,
                'pct_rfp': pct_rfp,
                'pct_ratio': pct_ratio,
                'mean_gfp': mean_gfp,
                'mean_rfp': mean_rfp,
                'mean_ratio': mean_ratio,
                'n_baseline_frames': len(bl_frames),
            })
            print(f"  {basename}: {len(bl_frames)} baseline frames, strain={strain}")

    return records


# ── Helper: tidy summary DataFrame ───────────────────────────────────

def _build_summary(records):
    """One row per recording with aggregated statistics."""
    rows = []
    for r in records:
        rows.append({
            'strain': r['strain'],
            'condition': r['condition'],
            'worm_id': r['worm_id'],
            'worm_pth': r['worm_pth'],
            # percentile aggregates
            'pct_gfp_mean': np.nanmean(r['pct_gfp']),
            'pct_gfp_std':  np.nanstd(r['pct_gfp']),
            'pct_rfp_mean': np.nanmean(r['pct_rfp']),
            'pct_rfp_std':  np.nanstd(r['pct_rfp']),
            'pct_ratio_mean': np.nanmean(r['pct_ratio']),
            'pct_ratio_std':  np.nanstd(r['pct_ratio']),
            # whole-frame mean aggregates
            'mean_gfp_mean': np.nanmean(r['mean_gfp']),
            'mean_gfp_std':  np.nanstd(r['mean_gfp']),
            'mean_rfp_mean': np.nanmean(r['mean_rfp']),
            'mean_rfp_std':  np.nanstd(r['mean_rfp']),
            'mean_ratio_mean': np.nanmean(r['mean_ratio']),
            'mean_ratio_std':  np.nanstd(r['mean_ratio']),
        })
    return pd.DataFrame(rows)


# ── Plotting helpers ──────────────────────────────────────────────────

def _strip_plot(records, strains, gfp_key, rfp_key, ratio_key,
                title_suffix, ylabel_gfp, ylabel_rfp, ylabel_ratio):
    """Per-frame strip/swarm plot for a given per-frame array key.

    Creates a 1×3 figure: GFP, RFP, and R = GFP/RFP.
    Within each panel, x-positions correspond to strains.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    jitter_w = 0.3

    for ci, (ch_key, ylabel, ch_label) in enumerate([
        (gfp_key, ylabel_gfp, 'GFP'),
        (rfp_key, ylabel_rfp, 'RFP'),
        (ratio_key, ylabel_ratio, 'R (GFP/RFP)'),
    ]):
        ax = axes[ci]
        for si, strain in enumerate(strains):
            strain_recs = [r for r in records if r['strain'] == strain]
            all_vals = np.concatenate([r[ch_key] for r in strain_recs])
            all_vals = all_vals[np.isfinite(all_vals)]
            if len(all_vals) == 0:
                continue
            x_jitter = si + np.random.uniform(-jitter_w / 2, jitter_w / 2,
                                               size=len(all_vals))
            ax.scatter(x_jitter, all_vals, s=6, alpha=0.35,
                       color=_strain_color(strain, strains), zorder=2)
            # mean ± std overlay
            ax.errorbar(si, np.mean(all_vals), yerr=np.std(all_vals),
                        fmt='o', color='k', markersize=6, capsize=5,
                        zorder=3)

        ax.set_xticks(range(len(strains)))
        ax.set_xticklabels(strains, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ch_label} — {title_suffix}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


def _bar_plot(summary_df, strains, mean_col_gfp, mean_col_rfp,
              mean_col_ratio, ylabel_gfp, ylabel_rfp, ylabel_ratio,
              title_suffix):
    """Bar plot: per-recording mean (± across-recording std) for each strain.

    Creates a 1×3 figure: GFP, RFP, and R = GFP/RFP.
    """
    n_strains = len(strains)
    bar_width = 0.6

    fig, axes = plt.subplots(1, 3, figsize=(max(12, 2.5 * n_strains * 1.5), 5))

    for ci, (col, ylabel, ch_label) in enumerate([
        (mean_col_gfp, ylabel_gfp, 'GFP'),
        (mean_col_rfp, ylabel_rfp, 'RFP'),
        (mean_col_ratio, ylabel_ratio, 'R (GFP/RFP)'),
    ]):
        ax = axes[ci]
        means = []
        stds = []
        for strain in strains:
            sub = summary_df[summary_df['strain'] == strain]
            means.append(sub[col].mean())
            stds.append(sub[col].std())

        x = np.arange(n_strains)
        ax.bar(x, means, bar_width, yerr=stds,
               color=[_strain_color(s, strains) for s in strains],
               alpha=0.85, capsize=5, edgecolor='k', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(strains, rotation=30, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ch_label} — {title_suffix}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


def _scatter_plot(summary_df, strains, rfp_col, gfp_col,
                  xlabel, ylabel, title):
    """Scatter plot: one point per recording, RFP on x, GFP on y."""
    fig, ax = pretty_plot(figsize=(6, 5))

    for strain in strains:
        sub = summary_df[summary_df['strain'] == strain]
        ax.scatter(sub[rfp_col], sub[gfp_col],
                   color=_strain_color(strain, strains), label=strain,
                   s=60, edgecolor='k', linewidth=0.4, zorder=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title='Strain', fontsize=9,
              bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.show()


# ── Percentile-mode plots ─────────────────────────────────────────────

def plot_pct_strip(records, strains, percentile):
    """Plot 1: Per-frame percentile GFP/RFP/R strip plot by strain."""
    _strip_plot(
        records, strains,
        gfp_key='pct_gfp',
        rfp_key='pct_rfp',
        ratio_key='pct_ratio',
        title_suffix=f'per-frame P{percentile:.0f} during baseline',
        ylabel_gfp=f'GFP P{percentile:.0f}',
        ylabel_rfp=f'RFP P{percentile:.0f}',
        ylabel_ratio=f'R (GFP/RFP) P{percentile:.0f}',
    )


def plot_pct_bar(summary_df, strains, percentile):
    """Plot 2: Mean P(percentile) across sessions, per strain."""
    _bar_plot(
        summary_df, strains,
        mean_col_gfp='pct_gfp_mean',
        mean_col_rfp='pct_rfp_mean',
        mean_col_ratio='pct_ratio_mean',
        ylabel_gfp=f'Mean P{percentile:.0f} GFP',
        ylabel_rfp=f'Mean P{percentile:.0f} RFP',
        ylabel_ratio=f'Mean P{percentile:.0f} R',
        title_suffix=f'session-mean P{percentile:.0f} (± std across recordings)',
    )


def plot_pct_scatter(summary_df, strains, percentile):
    """Plot 3: P(percentile) GFP vs RFP scatter, one dot per recording."""
    _scatter_plot(
        summary_df, strains,
        rfp_col='pct_rfp_mean',
        gfp_col='pct_gfp_mean',
        xlabel=f'Mean P{percentile:.0f} RFP',
        ylabel=f'Mean P{percentile:.0f} GFP',
        title=f'Baseline P{percentile:.0f}: GFP vs RFP per recording',
    )


# ── Whole-frame mean plots ────────────────────────────────────────────

def plot_mean_strip(records, strains):
    """Plot 4: Per-frame whole-frame mean GFP/RFP/R strip plot by strain."""
    _strip_plot(
        records, strains,
        gfp_key='mean_gfp',
        rfp_key='mean_rfp',
        ratio_key='mean_ratio',
        title_suffix='per-frame mean during baseline',
        ylabel_gfp='Mean GFP intensity',
        ylabel_rfp='Mean RFP intensity',
        ylabel_ratio='R (GFP/RFP)',
    )


def plot_mean_bar(summary_df, strains):
    """Plot 5: Mean whole-frame intensity across sessions, per strain."""
    _bar_plot(
        summary_df, strains,
        mean_col_gfp='mean_gfp_mean',
        mean_col_rfp='mean_rfp_mean',
        mean_col_ratio='mean_ratio_mean',
        ylabel_gfp='Mean GFP intensity',
        ylabel_rfp='Mean RFP intensity',
        ylabel_ratio='Mean R (GFP/RFP)',
        title_suffix='session-mean intensity (± std across recordings)',
    )


def plot_mean_scatter(summary_df, strains):
    """Plot 6: Mean GFP vs Mean RFP scatter, one dot per recording."""
    _scatter_plot(
        summary_df, strains,
        rfp_col='mean_rfp_mean',
        gfp_col='mean_gfp_mean',
        xlabel='Mean RFP intensity',
        ylabel='Mean GFP intensity',
        title='Baseline mean intensity: GFP vs RFP per recording',
    )


# ── Main ──────────────────────────────────────────────────────────────

def main():
    """
    Whole-frame baseline intensity analysis.

    Usage
    -----
    From the command line:
        python quantify_baseline_frame_pct.py <data_root> [percentile] [reg_dir]

    From a notebook:
        import quantify_baseline_frame_pct as qbf
        sys.argv = ["", r"D:\\DATA\\g5ht-free"]
        qbf.main()

    Arguments
    ---------
    sys.argv[1]  data_root  : str   — Root data directory.
    sys.argv[2]  percentile : float — Percentile to compute (default: 95).
    sys.argv[3]  reg_dir    : str   — Registered TIF subdirectory
                               (default: 'registered_elastix').
    """
    if len(sys.argv) < 2:
        print("Usage: python quantify_baseline_frame_pct.py "
              "<data_root> [percentile] [reg_dir]")
        sys.exit(1)

    data_root = sys.argv[1]
    percentile = float(sys.argv[2]) if len(sys.argv) > 2 else 95.0
    reg_dir = sys.argv[3] if len(sys.argv) > 3 else 'registered_elastix'

    print(f"Scanning {data_root} for recordings with baselines …")
    print(f"Percentile = {percentile}")

    records = collect_frame_baseline(data_root, percentile=percentile,
                                     reg_dir=reg_dir)
    print(f"\nCollected {len(records)} recordings with defined baselines")

    if len(records) == 0:
        print("No recordings with baseline windows found. Exiting.")
        return

    strains = sorted(set(r['strain'] for r in records))
    print(f"Strains: {strains}")

    summary_df = _build_summary(records)

    # ── Percentile-mode plots ──────────────────────────────────────────
    print(f"\n=== Percentile mode (P{percentile:.0f}) ===")

    print("Plot 1: Per-frame percentile distribution (strip)")
    plot_pct_strip(records, strains, percentile)

    print("Plot 2: Session-mean percentile (bar)")
    plot_pct_bar(summary_df, strains, percentile)

    print("Plot 3: Percentile GFP vs RFP scatter")
    plot_pct_scatter(summary_df, strains, percentile)

    # ── Whole-frame mean plots ─────────────────────────────────────────
    print("\n=== Whole-frame mean mode ===")

    print("Plot 4: Per-frame mean distribution (strip)")
    plot_mean_strip(records, strains)

    print("Plot 5: Session-mean intensity (bar)")
    plot_mean_bar(summary_df, strains)

    print("Plot 6: Mean GFP vs RFP scatter")
    plot_mean_scatter(summary_df, strains)


if __name__ == '__main__':
    main()
