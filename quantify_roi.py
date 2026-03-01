"""
quantify_roi.py — ROI-based ratiometric quantification for dual-channel 5-HT imaging.

Reads registered TIF stacks (GFP + RFP channels), computes the ratiometric
signal R = GFP / RFP per ROI, NaN-fills bad frames, and saves traces as CSV.
Loads metadata.json for fps, bad_frames, baseline, encounter information.

Usage:
    python quantify_roi.py <input_dir> <reg_dir> [plot_only] [time_type]
                           [normalize_to_max] [ylim_pct_lo] [ylim_pct_hi] [drop_dnc]

    From a notebook:
        import quantify_roi
        sys.argv = ["" , input_dir, reg_dir]
        quantify_roi.main()

Arguments:
    input_dir           Path to the worm data directory (must contain metadata.json,
                        roi.tif, and fixed_*.tif).
    reg_dir             Subdirectory containing registered TIF stacks.
    plot_only           0 or 1 (default: 0). If 1, skip quantification and re-plot.
    time_type           'min', 'sec', or 'frame' (default: 'frame').
    normalize_to_max    0 or 1 (default: 0). Normalize each trace to its max.
    ylim_pct_lo         Lower y-axis percentile limit (default: 0).
    ylim_pct_hi         Upper y-axis percentile limit (default: 99.75).
    drop_dnc            0 or 1 (default: 1). Drop dorsal_nerve_cord from plots.

Output:
    <input_dir>/quantified_roi.csv       — ROI traces with frame and time_sec columns.
    <input_dir>/quantified_roi.png/svg   — Trace plot.
    <input_dir>/roi_overlay.png/svg      — ROI contour overlay on fixed image.
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
from skimage import measure
from tqdm import tqdm

from utils import pretty_plot, default_plt_params

default_plt_params()

# ---------------------------------------------------------------------------
# Predefined hex colors for each ROI label (consistent across datasets)
# ---------------------------------------------------------------------------
LABEL_COLORS = {
    # long-form labels
    'procorpus':          '#F94144',
    'metacorpus':         '#FF9129',
    'isthmus':            '#F3BD3E',
    'terminal_bulb':      '#FFD981',
    'nerve_ring':         '#90BE6D',
    'ventral_nerve_cord': '#43AA8B',
    'dorsal_nerve_cord':  '#000000',
    # short-form labels
    'PC':                 '#F94144',
    'MC':                 '#FF9129',
    'IM':                 '#F3BD3E',
    'TB':                 '#FFD981',
    'NR':                 '#90BE6D',
    'VNC':                '#43AA8B',
    'DNC':                '#000000',
}

_FALLBACK_COLORS = ['#17becf', '#bcbd22', '#7f7f7f', '#aec7e8', '#ffbb78', '#98df8a']


def get_label_color(label, fallback_idx=0):
    """Return the hex color for a given label, with fallback for unknown labels."""
    if label in LABEL_COLORS:
        return LABEL_COLORS[label]
    return _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_metadata(input_dir, require=True):
    """Load metadata.json and return a dict with numpy arrays where appropriate.

    Returns None if the file is missing and require is False.
    """
    metadata_path = os.path.join(input_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        if require:
            raise FileNotFoundError(f"metadata.json not found in {input_dir}")
        return None

    with open(metadata_path, 'r') as fj:
        meta = json.load(fj)

    meta['bad_frames'] = np.array(meta['bad_frames'], dtype=int)
    meta['frame_index'] = np.array(meta['frame_index'], dtype=int)
    meta['fps'] = float(meta['fps'])
    meta['nframes'] = int(meta['nframes'])
    meta['encounter_frame'] = meta.get('encounter_frame')  # int or None

    bsf = meta.get('baseline_start_frame')
    bef = meta.get('baseline_end_frame')
    if bsf is not None and bef is not None:
        meta['baseline_window'] = (int(bsf), int(bef))
    else:
        meta['baseline_window'] = None

    return meta


def load_roi(input_dir):
    """Load roi.tif and return (roi_array, roi_labels)."""
    roi_path = os.path.join(input_dir, 'roi.tif')
    roi = tifffile.imread(roi_path)
    with tifffile.TiffFile(roi_path) as tif:
        labels = tif.imagej_metadata['Labels']
    return roi, labels


def load_fixed(input_dir):
    """Load the fixed_XXXX*.tif reference image."""
    fixed_fn = glob.glob(os.path.join(input_dir, 'fixed_[0-9][0-9][0-9][0-9]*.tif'))
    if not fixed_fn:
        raise FileNotFoundError(f"No fixed_XXXX*.tif found in {input_dir}")
    return tifffile.imread(fixed_fn[0])


# ---------------------------------------------------------------------------
# Quantification
# ---------------------------------------------------------------------------

def quantify_rois(input_dir, reg_dir, meta, roi, labels):
    """Compute per-ROI ratiometric GFP/RFP traces.

    Parameters
    ----------
    input_dir : str
    reg_dir : str
    meta : dict
        Loaded metadata (fps, bad_frames, frame_index, etc.).
    roi : ndarray (Z, H, W)
        ROI label mask (1-indexed).
    labels : list[str]
        ROI label names.

    Returns
    -------
    df : DataFrame
        Columns: 'frame', 'time_sec', plus ROI label columns.
    """
    registered_dir = os.path.join(input_dir, reg_dir)
    tif_paths = sorted(
        glob.glob(os.path.join(registered_dir, '*.tif')),
        key=lambda x: int(os.path.basename(x).split('.')[0]),
    )
    if not tif_paths:
        raise FileNotFoundError(f"No .tif files found in {registered_dir}")

    nlabels = len(labels)
    T = len(tif_paths)
    out = np.full((T, nlabels), np.nan, dtype=np.float64)

    print(f"Quantifying {T} frames across {nlabels} ROIs …")
    for i in tqdm(range(T), desc="Quantifying ROIs"):
        stack = tifffile.imread(tif_paths[i])
        for j in range(nlabels):
            mask = roi == (j + 1)
            denominator = np.sum(stack[:, 1][mask])
            if denominator > 0:
                out[i, j] = np.sum(stack[:, 0][mask]) / denominator

    # Build time vector from metadata
    fps = meta['fps']
    frame_index = meta['frame_index']
    t_sec = frame_index[:T] / fps  # time in seconds

    # Interpolate NaN from zero-denominator frames first
    df = pd.DataFrame(out, columns=labels)
    df = df.interpolate(method='linear', limit_direction='both')

    # NaN-fill bad frames AFTER interpolation so they stay NaN
    bad_frames = meta['bad_frames']
    if len(bad_frames) > 0:
        valid_bad = bad_frames[bad_frames < T]
        if len(valid_bad) > 0:
            df.iloc[valid_bad] = np.nan
            print(f"NaN-filled {len(valid_bad)} bad frames")

    # Insert frame index and time columns at the front
    df.insert(0, 'frame', frame_index[:T])
    df.insert(1, 'time_sec', t_sec)

    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_traces(
    df, meta,
    input_dir=None,
    time_type='frame',
    normalize_to_max=False,
    ylim_pct=(0, 99.75),
    drop_dnc=True,
):
    """Plot R/R_baseline traces for each ROI.

    Parameters
    ----------
    df : DataFrame
        ROI traces with 'frame' and 'time_sec' columns plus ROI label columns.
    meta : dict
        Metadata dict (must have baseline_window, encounter_frame, fps, frame_index).
    input_dir : str or None
        If given, save figures here.
    time_type : str
        'min', 'sec', or 'frame'.
    normalize_to_max : bool
        Normalize each R/R_baseline trace to its max.
    ylim_pct : tuple (lo, hi)
        Percentiles for y-axis limits (applied to the plotted data).
    drop_dnc : bool
        If True, drop dorsal_nerve_cord / DNC from the plot.
    """
    lw = 2.5

    # Extract frame/time columns and ROI-only data
    t_sec = df['time_sec'].values if 'time_sec' in df.columns else np.arange(len(df)) / meta['fps']
    df_roi = df.drop(columns=['frame', 'time_sec'], errors='ignore').copy()

    if drop_dnc:
        for dnc_label in ('dorsal_nerve_cord', 'DNC'):
            if dnc_label in df_roi.columns:
                df_roi = df_roi.drop(columns=[dnc_label])

    labels = df_roi.columns.tolist()
    out = df_roi.values
    nlabels = out.shape[1]

    # Sort labels alphabetically (and reorder data/colors accordingly)
    sorted_indices = np.argsort(labels)
    labels = [labels[i] for i in sorted_indices]
    out = out[:, sorted_indices]

    # Determine baseline window
    baseline_window = meta.get('baseline_window')
    if baseline_window is None:
        baseline_window = (0, min(60, out.shape[0]))

    # Build time axis based on time_type
    fps = meta['fps']
    if time_type == 'frame':
        t_plot = np.arange(len(out)).astype(float)
        xlabel = 'Frame'
    elif time_type == 'sec':
        t_plot = t_sec.copy()
        xlabel = 'Time (sec)'
    elif time_type == 'min':
        t_plot = t_sec / 60.0
        xlabel = 'Time (min)'
    else:
        raise ValueError('time_type must be "min", "sec", or "frame"')

    # Encounter alignment
    encounter_frame = meta.get('encounter_frame')
    if encounter_frame is not None:
        if time_type == 'frame':
            encounter_val = float(encounter_frame)
        elif time_type == 'sec':
            encounter_val = meta['frame_index'][encounter_frame] / fps
        elif time_type == 'min':
            encounter_val = meta['frame_index'][encounter_frame] / fps / 60.0
        t_plot = t_plot - encounter_val
    else:
        encounter_val = None

    # Compute R / R_baseline for each ROI
    plot_data = np.zeros_like(out)
    for i in range(nlabels):
        r_baseline = np.nanmean(out[baseline_window[0]:baseline_window[1], i])
        if r_baseline == 0 or np.isnan(r_baseline):
            r_baseline = 1.0  # avoid division by zero
        r_over_baseline = out[:, i] / r_baseline
        if normalize_to_max:
            mx = np.nanmax(r_over_baseline)
            if mx > 0:
                r_over_baseline = r_over_baseline / mx
        plot_data[:, i] = r_over_baseline

    # Determine y-limits from percentiles (across all ROIs, ignoring NaN)
    finite_vals = plot_data[np.isfinite(plot_data)]
    if len(finite_vals) > 0:
        ylo = np.percentile(finite_vals, ylim_pct[0])
        yhi = np.percentile(finite_vals, ylim_pct[1])
        # Add a small margin
        margin = (yhi - ylo) * 0.05
        ylo -= margin
        yhi += margin
    else:
        ylo, yhi = None, None

    fig, ax = pretty_plot(figsize=(12, 3.5))
    fallback_idx = 0
    for i in range(nlabels):
        c = get_label_color(labels[i], fallback_idx)
        if labels[i] not in LABEL_COLORS:
            fallback_idx += 1
        ax.plot(t_plot, plot_data[:, i], label=labels[i], color=c, lw=lw)

    if encounter_val is not None:
        # Shade ±3 frames around encounter, converted to current time units
        shade_frame = 3  # frames
        if time_type == 'min':
            shade_half = shade_frame * 0.533 / 60
        elif time_type == 'sec':
            shade_half = shade_frame * 0.533
        elif time_type == 'frame':
            shade_half = shade_frame
        ax.axvspan(-shade_half, shade_half, color='gray', alpha=0.3, label='Encounter')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if encounter_val is not None:
        ax.set_xlabel(f'{xlabel} from encounter')
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$R/R_{\mathrm{baseline}}$')

    # Determine x-limits: skip leading/trailing all-NaN regions
    any_finite = np.any(np.isfinite(plot_data), axis=1)  # (T,) True where at least one ROI is finite
    finite_indices = np.where(any_finite)[0]
    if len(finite_indices) > 0:
        xlo = t_plot[finite_indices[0]]
        xhi = t_plot[finite_indices[-1]]
    elif encounter_val is not None:
        xlo, xhi = -encounter_val, t_plot[-1]
    else:
        xlo, xhi = t_plot[0], t_plot[-1]
    ax.set_xlim(xlo, xhi)

    # if ylo is not None and yhi is not None:
    #     ax.set_ylim(ylo, yhi)

    ax.axhline(1, ls='--', c='k', zorder=0)
    plt.tight_layout()

    if input_dir is not None:
        fig.savefig(os.path.join(input_dir, 'quantified_roi.png'), dpi=300)
        fig.savefig(os.path.join(input_dir, 'quantified_roi.svg'), dpi=300)
    plt.show()


def plot_rois(fixed, roi, labels=None, input_dir=None, drop_dnc=True):
    """Visualize ROI contours overlaid on the max-projection of the fixed image."""
    if drop_dnc and labels is not None:
        labels = [l for l in labels if l not in ('dorsal_nerve_cord', 'DNC')]

    img = np.zeros((fixed.shape[-2], fixed.shape[-1], 3), np.float32)
    # Red-channel max projection (channel index 1 = RFP structural channel)
    img[..., 0] = np.max(fixed[:, 1], axis=0)
    img[..., 0] = np.clip(img[..., 0] / 400, 0, 1)
    img = (img * 255).astype(np.ubyte)
    # Convert to grayscale for clearer overlay
    img = np.stack([img[..., 0]] * 3, axis=-1)

    nlabels = np.max(roi)

    plt.figure(figsize=(10, 4))
    fallback_idx = 0
    for i in range(nlabels):
        if labels is not None and i < len(labels):
            c = get_label_color(labels[i], fallback_idx)
            if labels[i] not in LABEL_COLORS:
                fallback_idx += 1
        else:
            c = _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]
        contours = measure.find_contours(np.max(roi == i + 1, axis=0), level=0.5)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=3, color=c)
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()

    if input_dir is not None:
        plt.savefig(os.path.join(input_dir, 'roi_overlay.png'), dpi=300)
        plt.savefig(os.path.join(input_dir, 'roi_overlay.svg'), dpi=300)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    ROI-based ratiometric quantification for dual-channel 5-HT imaging.

    Usage
    -----
    From the command line:
        python quantify_roi.py <input_dir> <reg_dir> [plot_only] [time_type]
               [normalize_to_max] [ylim_pct_lo] [ylim_pct_hi] [drop_dnc]

    From a notebook:
        import quantify_roi
        sys.argv = ["", input_dir, reg_dir]
        quantify_roi.main()

        # With all options:
        sys.argv = ["", input_dir, reg_dir, 0, 'frame', 0, 0, 99.75, 1]
        quantify_roi.main()

    Arguments
    ---------
    sys.argv[1]  input_dir        : str   — Path to worm data directory (must contain
                                            metadata.json, roi.tif, fixed_*.tif).
    sys.argv[2]  reg_dir          : str   — Subdirectory with registered TIF stacks
                                            (e.g. 'registered_elastix').
    sys.argv[3]  plot_only        : int   — 0 or 1 (default: 0). If 1, skip
                                            quantification and load existing CSV.
    sys.argv[4]  time_type        : str   — 'min', 'sec', or 'frame' (default: 'frame').
    sys.argv[5]  normalize_to_max : int   — 0 or 1 (default: 0). Normalize each
                                            R/R_baseline trace to its max.
    sys.argv[6]  ylim_pct_lo      : float — Lower y-axis percentile limit (default: 0).
    sys.argv[7]  ylim_pct_hi      : float — Upper y-axis percentile limit (default: 100).
    sys.argv[8]  drop_dnc         : int   — 0 or 1 (default: 1). Drop dorsal_nerve_cord
                                            from plots.
    """
    if len(sys.argv) < 3:
        print("Usage: python quantify_roi.py <input_dir> <reg_dir> [plot_only] "
              "[time_type] [normalize_to_max] [ylim_pct_lo] [ylim_pct_hi] [drop_dnc]")
        sys.exit(1)

    input_dir = sys.argv[1]
    reg_dir = sys.argv[2]
    plot_only = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
    time_type = sys.argv[4] if len(sys.argv) > 4 else 'frame'
    normalize_to_max = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False
    ylim_pct_lo = float(sys.argv[6]) if len(sys.argv) > 6 else 0
    ylim_pct_hi = float(sys.argv[7]) if len(sys.argv) > 7 else 99.75
    drop_dnc = bool(int(sys.argv[8])) if len(sys.argv) > 8 else True
    ylim_pct = (ylim_pct_lo, ylim_pct_hi)

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------
    meta = load_metadata(input_dir, require=True)
    print(f"Loaded metadata: fps={meta['fps']:.4f}, nframes={meta['nframes']}, "
          f"baseline_window={meta['baseline_window']}, "
          f"encounter_frame={meta['encounter_frame']}, "
          f"bad_frames={len(meta['bad_frames'])}")

    # ------------------------------------------------------------------
    # Load ROI and fixed image
    # ------------------------------------------------------------------
    roi, roi_labels = load_roi(input_dir)
    fixed = load_fixed(input_dir)
    print(f"ROI labels: {roi_labels}")

    # ------------------------------------------------------------------
    # Quantify or load
    # ------------------------------------------------------------------
    csv_path = os.path.join(input_dir, 'quantified_roi.csv')

    if not plot_only:
        df = quantify_rois(input_dir, reg_dir, meta, roi, roi_labels)
        df.to_csv(csv_path, index=False)
        print(f"Saved traces to {csv_path}")
    else:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"{csv_path} not found — run without plot_only first."
            )
        df = pd.read_csv(csv_path)
        print(f"Loaded existing traces from {csv_path}")

    # ------------------------------------------------------------------
    # Plot traces
    # ------------------------------------------------------------------
    plot_traces(
        df, meta,
        input_dir=input_dir,
        time_type=time_type,
        normalize_to_max=normalize_to_max,
        ylim_pct=ylim_pct,
        drop_dnc=drop_dnc,
    )

    # ------------------------------------------------------------------
    # Plot ROIs
    # ------------------------------------------------------------------
    plot_rois(fixed, roi, labels=roi_labels, input_dir=input_dir, drop_dnc=drop_dnc)


if __name__ == '__main__':
    main()
