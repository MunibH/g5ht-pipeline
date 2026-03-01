"""
latency_analysis.py — Latency analysis for dual-channel ratiometric 5-HT imaging.

Loads consolidated processed_data.h5 files (produced by quantify_voxels.py),
normalizes the ratiometric signal (R/R0 or R/F20), applies spatial masking,
detects response onset latency per voxel, and provides visualization functions.

All frame indices in this module are absolute (i.e. they index directly into the
ratio array from the h5 file). Bad frames are already NaN-filled in the h5.

Key data flow:
    load_from_h5  →  normalize_ratio  →  preprocess_ratio  →  flatten_and_mask
    →  select_top_n_voxels  →  apply_preprocessing  →  compute_response_statistics

Dependencies:
    numpy, h5py, matplotlib, scipy, tifffile
"""

import os
import glob
import json
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List, Any

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import convolve1d

from utils import pretty_plot


# ============================================================================
# CUSTOM NORMS
# ============================================================================

class CenteredPowerNorm(mcolors.Normalize):
    """Normalize that centres *vcenter* at 0.5 in the colour-map and applies
    a power-law (gamma) stretch independently on each side.

    Useful for diverging colour-maps (e.g. RdBu) where the neutral colour
    should sit at a specific data value (e.g. encounter time) and the data
    distribution is skewed.

    Parameters
    ----------
    gamma : float
        Exponent for the power-law.  ``gamma < 1`` expands small
        deviations from *vcenter*; ``gamma > 1`` compresses them.
    vcenter : float
        Data value mapped to the middle of the colour-map (0.5).
    vmin, vmax : float
        Data limits.
    """

    def __init__(self, gamma: float = 1.0, vcenter: float = 0.0,
                 vmin: float = None, vmax: float = None, clip: bool = False):
        self.gamma = gamma
        self._vcenter = vcenter
        super().__init__(vmin, vmax, clip)

    @property
    def vcenter(self):
        return self._vcenter

    @vcenter.setter
    def vcenter(self, value):
        self._vcenter = value

    def __call__(self, value, clip=None):
        result = np.ma.asarray(value, dtype=float)
        self.autoscale_None(result)
        vmin, vcenter, vmax = self.vmin, self._vcenter, self.vmax

        # Clamp vcenter inside [vmin, vmax]
        vcenter = np.clip(vcenter, vmin, vmax)

        out = np.ma.empty_like(result)

        # --- below vcenter: map [vmin, vcenter] → [0, 0.5] ---
        below = result <= vcenter
        if vcenter != vmin:
            x = np.clip((vcenter - result[below]) / (vcenter - vmin), 0, 1)
            out[below] = 0.5 - 0.5 * np.power(x, self.gamma)
        else:
            out[below] = 0.5

        # --- above vcenter: map [vcenter, vmax] → [0.5, 1.0] ---
        above = ~below
        if vmax != vcenter:
            x = np.clip((result[above] - vcenter) / (vmax - vcenter), 0, 1)
            out[above] = 0.5 + 0.5 * np.power(x, self.gamma)
        else:
            out[above] = 0.5

        return np.ma.clip(out, 0, 1)

    def inverse(self, value):
        value = np.ma.asarray(value, dtype=float)
        vmin, vcenter, vmax = self.vmin, self._vcenter, self.vmax
        vcenter = np.clip(vcenter, vmin, vmax)

        out = np.ma.empty_like(value)
        below = value <= 0.5
        above = ~below

        if vcenter != vmin:
            x = np.clip((0.5 - value[below]) / 0.5, 0, 1)
            out[below] = vcenter - np.power(x, 1.0 / self.gamma) * (vcenter - vmin)
        else:
            out[below] = vcenter

        if vmax != vcenter:
            x = np.clip((value[above] - 0.5) / 0.5, 0, 1)
            out[above] = vcenter + np.power(x, 1.0 / self.gamma) * (vmax - vcenter)
        else:
            out[above] = vcenter

        return out


# ============================================================================
# DATA LOADING
# ============================================================================

def load_from_h5(
    input_dir: str,
    h5_glob: str = '*_processed_data.h5',
) -> Dict[str, Any]:
    """Load all datasets from a processed_data.h5 file.

    Parameters
    ----------
    input_dir : str
        Path to the worm data directory.
    h5_glob : str
        Glob pattern to find the h5 file inside *input_dir*.

    Returns
    -------
    dict with keys:
        ratio            (T, Z, H, W) float32 — raw ratiometric signal R = GFP/<RFP>_t.
                         Bad frames are already NaN-filled.
        rfp_mean         (Z, H, W) float32
        gfp_mean         (Z, H, W) float32
        f20              (Z, H, W) float32 — 20th-percentile of R across time.
        baseline         (Z, H, W) float32 or None — mean of R over baseline_window.
        fixed_mask       (H, W) uint8
        time_vec         (T,) float64
        frame_index      (T,) int
        fps              float
        binning_factor   int
        baseline_window  (int, int) or None
        encounter_frame  int or None
        bad_frames       (N,) int
        nframes          int
    """
    matches = sorted(glob.glob(os.path.join(input_dir, h5_glob)))
    if not matches:
        raise FileNotFoundError(
            f"No h5 file matching '{h5_glob}' in {input_dir}"
        )
    h5_path = matches[0]
    print(f"Loading {h5_path}")

    data = {}
    with h5py.File(h5_path, 'r') as f:
        data['ratio'] = f['ratio'][:]
        data['rfp_mean'] = f['rfp_mean'][:]
        data['gfp_mean'] = f['gfp_mean'][:]
        data['f20'] = f['f20'][:]
        data['baseline'] = f['baseline'][:] if 'baseline' in f else None
        data['fixed_mask'] = f['fixed_mask'][:]
        data['time_vec'] = f['time_vec'][:]
        data['frame_index'] = f['frame_index'][:]
        data['fps'] = float(f['fps'][()])
        data['binning_factor'] = int(f['binning_factor'][()])
        data['nframes'] = int(f['nframes'][()])

        bw = f['baseline_window'][:]
        data['baseline_window'] = tuple(bw) if bw[0] != -1 else None

        ef = int(f['encounter_frame'][()])
        data['encounter_frame'] = ef if ef != -1 else None

        data['bad_frames'] = f['bad_frames'][:]

    T, Z, H, W = data['ratio'].shape
    bw_str = data['baseline_window'] if data['baseline_window'] else 'None'
    print(f"  ratio: ({T}, {Z}, {H}, {W}), fps={data['fps']:.4f}")
    print(f"  baseline_window={bw_str}, encounter_frame={data['encounter_frame']}")
    print(f"  bad_frames: {len(data['bad_frames'])} frames")
    return data


def load_metadata(input_dir: str) -> Dict[str, Any]:
    """Load metadata.json and return a dict (with numpy arrays for lists)."""
    meta_path = os.path.join(input_dir, 'metadata.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    meta['bad_frames'] = np.array(meta['bad_frames'], dtype=int)
    meta['frame_index'] = np.array(meta['frame_index'], dtype=int)
    return meta


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_ratio(
    ratio: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    f20: Optional[np.ndarray] = None,
    baseline_window: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Normalize the raw ratiometric signal.

    Strategy:
      - If *baseline_window* is set (not None) **and** a *baseline* array is
        provided → ``normalized = ratio / baseline``  (i.e. R / R₀).
      - Otherwise → ``normalized = ratio / f20``  (i.e. R / R₂₀).

    Division-by-zero voxels (denominator == 0) are set to 0.

    Parameters
    ----------
    ratio : (T, Z, H, W) float32
        Raw ratiometric signal from h5.
    baseline : (Z, H, W) float32 or None
        Mean of R over the baseline window (from h5).
    f20 : (Z, H, W) float32 or None
        20th percentile of R across time (always present in h5).
    baseline_window : tuple of int or None
        If set, triggers R/R₀ normalization.

    Returns
    -------
    normalized : (T, Z, H, W) float32
    """
    if baseline_window is not None and baseline is not None:
        denom = baseline
        method = 'R/R0 (baseline mean)'
    elif f20 is not None:
        denom = f20
        method = 'R/F20 (20th percentile)'
    else:
        raise ValueError(
            "Either baseline+baseline_window or f20 must be provided."
        )

    normalized = np.divide(
        ratio,
        denom[np.newaxis, :, :, :],
        out=np.zeros_like(ratio),
        where=denom[np.newaxis, :, :, :] != 0,
    )
    print(f"  Normalization: {method}")
    print(f"  Normalized range: [{np.nanmin(normalized):.3f}, {np.nanmax(normalized):.3f}]")
    return normalized


# ============================================================================
# SPATIAL BINNING
# ============================================================================

def bin_spatial(data: np.ndarray, bin_factor: int) -> np.ndarray:
    """Bin the spatial (H, W) dimensions by *bin_factor* via reshape-mean.

    Supports 2-D (H, W), 3-D (Z, H, W), and 4-D (T, Z, H, W) inputs.
    """
    if bin_factor <= 1:
        return data

    if data.ndim == 4:
        t, z, h, w = data.shape
        hb, wb = h // bin_factor, w // bin_factor
        return (
            data[:, :, :hb * bin_factor, :wb * bin_factor]
            .reshape(t, z, hb, bin_factor, wb, bin_factor)
            .mean(axis=(3, 5))
        )
    elif data.ndim == 3:
        z, h, w = data.shape
        hb, wb = h // bin_factor, w // bin_factor
        return (
            data[:, :hb * bin_factor, :wb * bin_factor]
            .reshape(z, hb, bin_factor, wb, bin_factor)
            .mean(axis=(2, 4))
        )
    elif data.ndim == 2:
        h, w = data.shape
        hb, wb = h // bin_factor, w // bin_factor
        return (
            data[:hb * bin_factor, :wb * bin_factor]
            .reshape(hb, bin_factor, wb, bin_factor)
            .mean(axis=(1, 3))
        )
    else:
        raise ValueError(f"Unsupported ndim={data.ndim}")


# ============================================================================
# PREPROCESSING / MASKING
# ============================================================================

def preprocess_ratio(
    normalized: np.ndarray,
    rfp_mean: np.ndarray,
    fixed_mask: np.ndarray,
    bin_factor: int = 1,
    rfp_thresh: float = 20.0,
    zero_prob_thresh: float = 5.0,
    keep_width: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Apply spatial binning, masking, and quality filtering.

    Parameters
    ----------
    normalized : (T, Z, H, W)
        Already-normalised ratio array.
    rfp_mean : (Z, H, W)
        Time-averaged RFP channel.
    fixed_mask : (H, W)
        2-D binary worm mask.
    bin_factor : int
        Additional spatial binning on top of whatever was used during h5
        creation.  1 = no extra binning.
    rfp_thresh : float
        Minimum RFP intensity for a voxel to be included.
    zero_prob_thresh : float
        Maximum percentage of time a voxel can be exactly zero before it is
        excluded (accounts for registration edge artifacts).
    keep_width : (int, int) or None
        Optional (x_start, x_end) crop in the width dimension.

    Returns
    -------
    dict with keys:
        data          (T, Z, Hb, Wb) — masked, binned ratio
        mask          (Z, Hb, Wb) — boolean mask
        spatial_shape (Z, Hb, Wb) tuple
    """
    data = bin_spatial(normalized, bin_factor)
    rfp = bin_spatial(rfp_mean, bin_factor)
    mask_2d = bin_spatial(fixed_mask.astype(np.float32), bin_factor)

    # Expand 2-D mask to 3-D (Z, H, W)
    Z = data.shape[1]
    mask = np.repeat(mask_2d[np.newaxis, :, :], Z, axis=0) > 0.5

    # --- RFP threshold ---
    mask_rfp = mask.copy()
    mask_rfp[rfp < rfp_thresh] = False

    # --- Zero-probability threshold ---
    data_masked = data * mask[np.newaxis, :, :, :]
    zero_frac = np.mean(data_masked == 0, axis=0)  # (Z, H, W)
    zero_frac[~mask] = np.nan
    good_voxels = zero_frac < (zero_prob_thresh / 100.0)

    mask_final = mask_rfp & good_voxels

    n_removed = int(np.sum(~good_voxels & mask))
    print(f"  Removed {n_removed} voxels with >{zero_prob_thresh}% zeros")

    # Apply mask
    data = data * mask_final[np.newaxis, :, :, :]

    # Optional width trim
    if keep_width is not None:
        data = data[:, :, :, keep_width[0]:keep_width[1]]
        mask_final = mask_final[:, :, keep_width[0]:keep_width[1]]

    spatial_shape = data.shape[1:]  # (Z, H, W)
    n_valid = int(np.sum(mask_final))
    print(f"  Preprocessed: {data.shape}, mask voxels: {n_valid}")
    return {'data': data, 'mask': mask_final, 'spatial_shape': spatial_shape}


# ============================================================================
# FLATTEN / SELECT / PREPROCESS
# ============================================================================

def flatten_and_mask(
    data_4d: np.ndarray,
    mask_3d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten (T, Z, H, W) → (T, n_valid) keeping only masked voxels.

    Returns
    -------
    data_flat : (T, n_valid)
    mask_flat : (Z*H*W,) bool — positions of valid voxels in the flat volume.
    """
    flat = data_4d.reshape(data_4d.shape[0], -1)
    mask_flat = mask_3d.reshape(-1) > 0
    return flat[:, mask_flat], mask_flat


def select_top_n_voxels(
    data_flat: np.ndarray,
    n: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select the top *n* most variable voxels (by variance along time).

    Parameters
    ----------
    data_flat : (T, V)
    n : int or None
        None keeps all voxels.

    Returns
    -------
    data_selected : (T, n)
    selected_indices : (n,) — indices into *data_flat* columns.
    """
    voxel_vars = np.nanvar(data_flat, axis=0)
    sort_idx = np.argsort(voxel_vars)[::-1]

    if n is not None and n < len(sort_idx):
        sort_idx = sort_idx[:n]
        print(f"  Selected top {n} most variable voxels")
    else:
        print(f"  Using all {len(sort_idx)} voxels")

    return data_flat[:, sort_idx], sort_idx


def apply_preprocessing(
    data: np.ndarray,
    method: str = 'raw',
) -> np.ndarray:
    """Apply optional preprocessing to the (T, V) array.

    Methods
    -------
    'raw'    : no change (already R/R₀ or R/F₂₀).
    'zscore' : per-voxel z-score.
    'center' : per-voxel mean subtraction.
    """
    if method == 'raw':
        print("  Preprocessing: raw")
        return data
    elif method == 'zscore':
        print("  Preprocessing: z-score")
        mean = np.nanmean(data, axis=0, keepdims=True)
        std = np.nanstd(data, axis=0, keepdims=True)
        std[std == 0] = 1.0
        return (data - mean) / std
    elif method == 'center':
        print("  Preprocessing: mean-centered")
        mean = np.nanmean(data, axis=0, keepdims=True)
        return data - mean
    else:
        raise ValueError(f"Unknown method: {method!r}")


# ============================================================================
# CAUSAL SMOOTHING
# ============================================================================

def causal_smooth(
    data: np.ndarray,
    sigma: float,
    truncate: float = 4.0,
) -> np.ndarray:
    """One-sided (causal) Gaussian smoothing along axis 0 (time).

    Parameters
    ----------
    data : (T, ...) array
    sigma : float — std of Gaussian kernel in frames.
    truncate : float — number of sigmas for kernel width.

    Returns
    -------
    smoothed : same shape as *data*.
    """
    kernel_size = int(truncate * sigma) + 1
    x = np.arange(kernel_size)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return convolve1d(data, kernel, axis=0, mode='nearest')


# ============================================================================
# RESPONSE STATISTICS & LATENCY DETECTION
# ============================================================================

@dataclass
class ResponseStats:
    """Container for per-voxel response statistics."""

    mean_baseline: np.ndarray      # (V,)
    mean_pre: np.ndarray           # (V,) or NaN
    mean_post: np.ndarray          # (V,) or NaN
    var_baseline: np.ndarray       # (V,)
    var_pre: np.ndarray            # (V,) or NaN
    var_post: np.ndarray           # (V,) or NaN
    significant_mask: np.ndarray   # (T, V) bool
    voxel_response_prob: np.ndarray  # (V,)
    latency: np.ndarray            # (V,) frame index or NaN
    is_responsive: np.ndarray      # (V,) bool


def compute_response_statistics(
    data: np.ndarray,
    baseline_window: Tuple[int, int],
    pre_window: Optional[Tuple[int, int]] = None,
    post_window: Optional[Tuple[int, int]] = None,
    n_std: float = 3.0,
    min_consecutive: int = 3,
    smooth_sigma: Optional[float] = None,
    detection_start: Optional[int] = None,
    min_response_prob: float = 0.05,
) -> ResponseStats:
    """Compute response onset latency and significance for each voxel.

    Pipeline
    --------
    1. Baseline mean/std per voxel over *baseline_window*.
    2. Optionally smooth data with causal Gaussian (*smooth_sigma*).
    3. Threshold: voxel is significant when value > baseline_mean + n_std * baseline_std.
    4. Latency = first frame of ≥ *min_consecutive* consecutive significant frames.
    5. Responsive if latency exists **and** response probability ≥ *min_response_prob*.

    Parameters
    ----------
    data : (T, V) — preprocessed voxel array.
    baseline_window : (start, end) — frame-index slice for baseline.
    pre_window, post_window : optional epoch windows for summary stats.
    n_std : significance threshold in baseline std units.
    min_consecutive : sustained-activation requirement (frames).
    smooth_sigma : Gaussian sigma for pre-detection smoothing (None = skip).
    detection_start : frame to begin latency search (default = baseline_window[1]).
    min_response_prob : minimum fraction of time significant to count as responsive.

    Returns
    -------
    ResponseStats dataclass.
    """
    n_time, n_voxels = data.shape

    # Baseline statistics
    bl = data[baseline_window[0]:baseline_window[1]]
    mean_bl = np.nanmean(bl, axis=0)
    var_bl = np.nanvar(bl, axis=0, ddof=1)
    std_bl = np.sqrt(var_bl)
    std_bl[std_bl == 0] = np.inf  # prevent false positives

    # Epoch summaries
    def _epoch_stats(win):
        if win is None:
            return np.full(n_voxels, np.nan), np.full(n_voxels, np.nan)
        d = data[win[0]:win[1]]
        return np.nanmean(d, axis=0), np.nanvar(d, axis=0, ddof=1)

    mean_pre, var_pre = _epoch_stats(pre_window)
    mean_post, var_post = _epoch_stats(post_window)

    # Optional smoothing
    det = causal_smooth(data, sigma=smooth_sigma) if smooth_sigma else data

    # Threshold & significance
    threshold = mean_bl + n_std * std_bl
    sig_mask = det > threshold[np.newaxis, :]
    response_prob = np.mean(sig_mask, axis=0)

    # Latency detection (vectorized)
    start = detection_start if detection_start is not None else baseline_window[1]
    latency = _detect_sustained_onset(sig_mask, min_consecutive, start)

    is_resp = ~np.isnan(latency) & (response_prob >= min_response_prob)

    return ResponseStats(
        mean_baseline=mean_bl,
        mean_pre=mean_pre,
        mean_post=mean_post,
        var_baseline=var_bl,
        var_pre=var_pre,
        var_post=var_post,
        significant_mask=sig_mask,
        voxel_response_prob=response_prob,
        latency=latency,
        is_responsive=is_resp,
    )


def _detect_sustained_onset(
    significant_mask: np.ndarray,
    min_consecutive: int,
    start_frame: int = 0,
) -> np.ndarray:
    """Vectorized detection of first sustained threshold crossing.

    Uses ``numpy.lib.stride_tricks.sliding_window_view`` to find the first
    run of ≥ *min_consecutive* consecutive True values per voxel, starting
    from *start_frame*.  No per-voxel Python loop.

    Parameters
    ----------
    significant_mask : (T, V) bool
    min_consecutive : int
    start_frame : int

    Returns
    -------
    latency : (V,) float — absolute frame index of onset, or NaN.
    """
    mask_sub = significant_mask[start_frame:].astype(np.float32)
    n_frames, n_voxels = mask_sub.shape
    latency = np.full(n_voxels, np.nan)

    if n_frames < min_consecutive:
        return latency

    if min_consecutive <= 1:
        # Simple: first True per column
        has_any = mask_sub.any(axis=0)
        first = np.argmax(mask_sub, axis=0)
        latency[has_any] = first[has_any] + start_frame
        return latency

    # Sliding-window sum along time axis
    windows = np.lib.stride_tricks.sliding_window_view(
        mask_sub, min_consecutive, axis=0,
    )  # shape (n_frames - k + 1, n_voxels, k)
    window_sums = windows.sum(axis=-1)  # (n_frames - k + 1, n_voxels)

    # First window where all frames are True
    hits = window_sums >= min_consecutive  # bool
    has_hit = hits.any(axis=0)  # (V,)
    first_hit = np.argmax(hits, axis=0)  # (V,) — index of first True row
    latency[has_hit] = first_hit[has_hit] + start_frame

    return latency


# ============================================================================
# SPATIAL COORDINATE MAPPING
# ============================================================================

def preprocessed_idx_to_spatial(
    preprocessed_indices: np.ndarray,
    selected_indices: np.ndarray,
    mask_flat: np.ndarray,
    spatial_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Map column indices of the preprocessed (T, V) array back to (z, h, w).

    Parameters
    ----------
    preprocessed_indices : indices into the preprocessed array's axis-1.
    selected_indices : from ``select_top_n_voxels`` (maps preprocessed → masked).
    mask_flat : from ``flatten_and_mask`` (maps masked → full flat volume).
    spatial_shape : (Z, H, W).

    Returns
    -------
    coords : (N, 3) int array of (z, h, w) coordinates.
    """
    masked_idx = selected_indices[preprocessed_indices]
    flat_positions = np.where(mask_flat)[0][masked_idx]
    coords = np.column_stack(np.unravel_index(flat_positions, spatial_shape))
    return coords


# ============================================================================
# PLOTTING — BASIC VISUALIZATIONS
# ============================================================================

def plot_representative_slices(
    data_4d: np.ndarray,
    mask_3d: np.ndarray,
    frame: int,
    z_indices: Optional[np.ndarray] = None,
    title: str = 'Activity',
    cmap: str = 'viridis',
) -> plt.Figure:
    """Show a few representative z-slices at a single time frame.

    Parameters
    ----------
    data_4d : (T, Z, H, W)
    mask_3d : (Z, H, W) bool
    frame : time index to display.
    z_indices : which z-slices to show (default: first, middle, last).
    """
    n_z = data_4d.shape[1]
    if z_indices is None:
        z_indices = [0, n_z // 2, n_z - 1] if n_z >= 3 else list(range(n_z))

    n = len(z_indices)
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 4), constrained_layout=True)
    if n == 1:
        axs = [axs]

    for i, zi in enumerate(z_indices):
        ax = axs[i]
        sl = data_4d[frame, zi]
        im = ax.imshow(sl, cmap=cmap)
        ax.contour(
            mask_3d[zi].astype(float), colors='white',
            linewidths=0.5, levels=[0.5],
        )
        ax.set_title(f'z{zi}', fontsize=12)
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal',
                            fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_ticks([np.nanmin(sl), np.nanmax(sl)])
        ax.axis('off')

    fig.suptitle(f'{title} (frame {frame})', y=0.95)
    return fig


def plot_voxel_heatmap(
    data_2d: np.ndarray,
    fps: float,
    time_window: Optional[Tuple[int, int]] = None,
    max_voxels: int = 50000,
    vmin_pct: float = 1.0,
    vmax_pct: float = 99.0,
) -> plt.Figure:
    """Heatmap of voxel traces, sorted by variance."""
    if time_window is not None:
        sub = data_2d[time_window[0]:time_window[1]]
        t_off = time_window[0]
    else:
        sub = data_2d
        t_off = 0

    var = np.nanvar(sub, axis=0)
    order = np.argsort(var)[::-1]
    n_show = min(max_voxels, sub.shape[1])
    sorted_data = sub[:, order[:n_show]]

    vmin = np.nanpercentile(sorted_data, vmin_pct)
    vmax = np.nanpercentile(sorted_data, vmax_pct)
    time = (np.arange(sub.shape[0]) + t_off) / fps

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        sorted_data.T, aspect='auto', cmap='viridis',
        extent=[time[0], time[-1], 0, n_show],
        vmin=vmin, vmax=vmax,
    )
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voxel (sorted by variance)')
    ax.set_title(f'Voxel Activity Heatmap (top {n_show})')
    fig.colorbar(im, ax=ax, label='Activity')
    fig.tight_layout()
    return fig


def plot_voxel_spatial_map(
    voxel_data: np.ndarray,
    voxel_indices: np.ndarray,
    selected_indices: np.ndarray,
    mask_flat: np.ndarray,
    mask_3d: np.ndarray,
    spatial_shape: Tuple[int, int, int],
    cmap: str = 'viridis',
    vmin: float = 0,
    vmax: float = 6,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Tuple[float, float] = (12, 10),
    mask_color: str = 'black',
    norm_cmap: Optional[mcolors.Normalize] = None,
    encounter_time: Optional[float] = None,
    num_cbar_ticks: int = 6,
    mask_bg: Optional[str] = None,
) -> plt.Figure:
    """Multi-panel spatial map of voxel values across z-planes.

    Parameters
    ----------
    voxel_data : (N,) — one value per voxel.
    voxel_indices : (N,) — column indices into the preprocessed array.
    selected_indices, mask_flat : from flatten/select pipeline.
    mask_3d : (Z, H, W) bool.
    spatial_shape : (Z, H, W).
    nrows, ncols : grid layout (auto-computed from z-planes if None).
    norm_cmap : optional matplotlib Normalize (e.g. PowerNorm).
    encounter_time : if set, draw arrow on colorbar.
    mask_bg : optional colour string (e.g. ``'black'``).  When set, pixels
        inside the mask that have no data (NaN) are filled with this colour
        instead of the default transparent background.
    """
    coords = preprocessed_idx_to_spatial(
        voxel_indices, selected_indices, mask_flat, spatial_shape,
    )

    if voxel_data.ndim > 1:
        voxel_data = voxel_data.squeeze()
    if len(voxel_data) != len(voxel_indices):
        raise ValueError("voxel_data and voxel_indices length mismatch")

    z_planes = np.unique(coords[:, 0])
    n_planes = len(z_planes)

    if nrows is None or ncols is None:
        ncols = min(n_planes, 5)
        nrows = int(np.ceil(n_planes / ncols))

    # Prepare colour-map copy with transparent bad values so the
    # optional mask_bg layer shows through.
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(alpha=0)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows, ncols, wspace=0.005, hspace=0.005)
    axs = [fig.add_subplot(gs[i // ncols, i % ncols])
           for i in range(nrows * ncols)]

    z, h, w = spatial_shape
    im = None

    for i, z_plane in enumerate(z_planes):
        ax = axs[i]

        # --- optional solid background inside the mask ---
        if mask_bg is not None:
            bg = np.full((h, w, 4), 0.0)  # fully transparent
            bg[mask_3d[z_plane]] = mcolors.to_rgba(mask_bg)
            ax.imshow(bg, interpolation='nearest')

        spatial_map = np.full((h, w), np.nan)
        zmask = coords[:, 0] == z_plane
        zc = coords[zmask]
        spatial_map[zc[:, 1], zc[:, 2]] = voxel_data[zmask]

        if norm_cmap is not None:
            im = ax.imshow(spatial_map, cmap=cmap_obj, norm=norm_cmap)
        else:
            im = ax.imshow(spatial_map, cmap=cmap_obj, vmin=vmin, vmax=vmax)

        ax.contour(
            mask_3d[z_plane].astype(float), colors=mask_color,
            linewidths=0.5, levels=[0.5],
        )
        ax.axis('off')

    for j in range(len(z_planes), len(axs)):
        axs[j].axis('off')

    fig.subplots_adjust(
        left=0.01, right=0.99, top=0.99, bottom=0.06,
        wspace=0.005, hspace=0.005,
    )

    if im is not None:
        cbar_ax = fig.add_axes([0.3, 0.025, 0.4, 0.025])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=12)
        # When using a custom norm, read vmin/vmax from it
        cb_vmin = norm_cmap.vmin if norm_cmap is not None else vmin
        cb_vmax = norm_cmap.vmax if norm_cmap is not None else vmax
        ticks = np.round(np.linspace(cb_vmin, cb_vmax, num_cbar_ticks), 2)
        cbar.set_ticks(ticks)
        if encounter_time is not None:
            cbar.ax.annotate(
                '', xy=(encounter_time, 0), xytext=(encounter_time, 1),
                arrowprops=dict(facecolor='magenta', shrink=0.5),
            )

    return fig


# ============================================================================
# PLOTTING — LATENCY VISUALIZATIONS
# ============================================================================

def plot_latency_histogram(
    latency: np.ndarray,
    fps: float,
    encounter_frame: Optional[int] = None,
    xlim: Optional[Tuple[float, float]] = None,
    bins: int = 100,
) -> plt.Figure:
    """Histogram of response latency in seconds.

    Parameters
    ----------
    latency : (V,) — absolute frame indices (NaN for non-responsive).
    fps : frames per second.
    encounter_frame : absolute frame index of food encounter (optional).
    xlim : x-axis limits in seconds.
    """
    fig, ax = pretty_plot(figsize=(8, 5))
    lat_sec = latency / fps
    ax.hist(lat_sec[~np.isnan(lat_sec)], bins=bins, edgecolor='none')
    ax.set_xlabel('Response latency (s)')
    ax.set_ylabel('# voxels')
    if encounter_frame is not None:
        ax.axvline(encounter_frame / fps, color='k', ls='--', label='encounter')
        ax.legend(fontsize=12)
    if xlim is not None:
        ax.set_xlim(xlim)
    fig.tight_layout()
    return fig


def plot_latency_sorted_heatmap(
    data_2d: np.ndarray,
    latency: np.ndarray,
    significant_mask: np.ndarray,
    num2plot: int = 35000,
) -> Tuple[plt.Figure, plt.Figure]:
    """Two figures: activity heatmap and significance mask, sorted by latency.

    Returns (fig_activity, fig_significance).
    """
    order = np.argsort(latency)
    n = min(num2plot, data_2d.shape[1])
    sorted_data = data_2d[:, order[:n]]
    sorted_sig = significant_mask[:, order[:n]]
    sorted_lat = latency[order[:n]]

    # Activity heatmap
    fig1, ax1 = plt.subplots(figsize=(4.5, 6))
    im = ax1.imshow(
        sorted_data.T, aspect='auto', cmap='plasma',
        vmax=np.nanpercentile(sorted_data, 99),
    )
    fig1.colorbar(im, ax=ax1, label='activity')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Voxel')
    fig1.tight_layout()

    # Significance mask + latency scatter
    fig2, ax2 = plt.subplots(figsize=(4.5, 6))
    ax2.imshow(sorted_sig.T, aspect='auto', cmap='gray')
    ax2.scatter(sorted_lat, np.arange(n), color='red', s=1)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Voxel')
    fig2.tight_layout()

    return fig1, fig2


def plot_latency_spatial_map(
    latency: np.ndarray,
    fps: float,
    selected_indices: np.ndarray,
    mask_flat: np.ndarray,
    mask_3d: np.ndarray,
    spatial_shape: Tuple[int, int, int],
    encounter_frame: Optional[int] = None,
    gamma: float = 1.0,
    cmap: str = 'RdBu_r',
    max_latency_offset: float = 20.0,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Tuple[float, float] = (12, 10),
    num_cbar_ticks: int = 6,
    norm_cmap: Optional[mcolors.Normalize] = None,
    mask_bg: Optional[str] = None,
) -> plt.Figure:
    """Spatial map coloured by response latency.

    When *encounter_frame* is provided and no explicit *norm_cmap* is given,
    a ``CenteredPowerNorm`` is used so that the encounter time sits at the
    centre of the colour-map (e.g. white for ``RdBu_r``).  This combines a
    ``TwoSlopeNorm``-style centre with a power-law (*gamma*) stretch that
    compensates for skewed latency distributions.

    Parameters
    ----------
    norm_cmap : optional Normalize override.  When supplied the default
        auto-construction is skipped and this norm is used directly.
    mask_bg : optional colour for NaN pixels inside the mask (e.g. ``'black'``).
    """
    lat_sec = latency.copy() / fps

    if encounter_frame is not None:
        enc_sec = encounter_frame / fps
        lat_sec[lat_sec > enc_sec + max_latency_offset] = np.nan
        encounter_time = enc_sec
    else:
        encounter_time = None
        enc_sec = None

    vmin = float(np.nanmin(lat_sec)) if np.any(~np.isnan(lat_sec)) else 0
    vmax = float(np.nanmax(lat_sec)) if np.any(~np.isnan(lat_sec)) else 1

    if norm_cmap is None:
        if enc_sec is not None:
            norm = CenteredPowerNorm(
                gamma=gamma, vcenter=enc_sec, vmin=vmin, vmax=vmax,
            )
        else:
            norm = mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    else:
        norm = norm_cmap

    fig = plot_voxel_spatial_map(
        lat_sec, np.arange(len(lat_sec)),
        selected_indices, mask_flat, mask_3d, spatial_shape,
        cmap=cmap, vmin=vmin, vmax=vmax,
        nrows=nrows, ncols=ncols, figsize=figsize,
        norm_cmap=norm, encounter_time=encounter_time,
        num_cbar_ticks=num_cbar_ticks,
        mask_bg=mask_bg,
    )
    return fig


def plot_latency_min_projection(
    latency: np.ndarray,
    fps: float,
    selected_indices: np.ndarray,
    mask_flat: np.ndarray,
    mask_3d: np.ndarray,
    spatial_shape: Tuple[int, int, int],
    encounter_frame: Optional[int] = None,
    gamma: float = 1.0,
    cmap: str = 'RdBu_r',
    max_latency_offset: float = 20.0,
    figsize: Tuple[float, float] = (8, 6),
    num_cbar_ticks: int = 6,
    norm_cmap: Optional[mcolors.Normalize] = None,
    mask_bg: Optional[str] = None,
) -> plt.Figure:
    """Min-latency projection across all z-slices.

    For each (h, w) pixel, the shortest response latency across z-planes
    is shown.  Uses the same encounter-centred norm logic as
    ``plot_latency_spatial_map``.

    Parameters
    ----------
    norm_cmap : optional Normalize override.
    mask_bg : optional background colour for masked NaN pixels.
    """
    lat_sec = latency.copy() / fps

    if encounter_frame is not None:
        enc_sec = encounter_frame / fps
        lat_sec[lat_sec > enc_sec + max_latency_offset] = np.nan
        encounter_time = enc_sec
    else:
        enc_sec = None
        encounter_time = None

    # --- reconstruct 3-D latency volume and project ---
    coords = preprocessed_idx_to_spatial(
        np.arange(len(latency)), selected_indices, mask_flat, spatial_shape,
    )
    z, h, w = spatial_shape
    lat_vol = np.full((z, h, w), np.nan)
    lat_vol[coords[:, 0], coords[:, 1], coords[:, 2]] = lat_sec

    with np.errstate(all='ignore'):
        min_proj = np.nanmin(lat_vol, axis=0)  # (H, W)

    valid = min_proj[~np.isnan(min_proj)]
    if len(valid) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No valid latencies', ha='center', va='center',
                transform=ax.transAxes)
        return fig

    vmin = float(np.nanmin(valid))
    vmax = float(np.nanmax(valid))

    # --- build norm ---
    if norm_cmap is None:
        if enc_sec is not None:
            norm = CenteredPowerNorm(
                gamma=gamma, vcenter=enc_sec, vmin=vmin, vmax=vmax,
            )
        else:
            norm = mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    else:
        norm = norm_cmap

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # mask background: union of mask across z
    mask_2d = np.any(mask_3d, axis=0)
    if mask_bg is not None:
        bg = np.full((h, w, 4), 0.0)
        bg[mask_2d] = mcolors.to_rgba(mask_bg)
        ax.imshow(bg, interpolation='nearest')

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(alpha=0)

    im = ax.imshow(min_proj, cmap=cmap_obj, norm=norm)
    ax.contour(mask_2d.astype(float), colors='gray', linewidths=0.5,
               levels=[0.5])
    ax.axis('off')
    ax.set_title('Min latency projection (across z)', fontsize=13)

    cbar = fig.colorbar(im, ax=ax, orientation='horizontal',
                        fraction=0.06, pad=0.08)
    ticks = np.round(np.linspace(vmin, vmax, num_cbar_ticks), 2)
    cbar.set_ticks(ticks)
    cbar.set_label('Latency (s)')
    if encounter_time is not None:
        cbar.ax.annotate(
            '', xy=(encounter_time, 0), xytext=(encounter_time, 1),
            arrowprops=dict(facecolor='magenta', shrink=0.5),
        )

    fig.tight_layout()
    return fig


def plot_latency_window_maps(
    latency: np.ndarray,
    fps: float,
    data_2d: np.ndarray,
    encounter_frame: int,
    selected_indices: np.ndarray,
    mask_flat: np.ndarray,
    mask_3d: np.ndarray,
    spatial_shape: Tuple[int, int, int],
    windows: Optional[List[Tuple[float, float]]] = None,
    cmap: str = 'viridis',
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Tuple[float, float] = (12, 10),
) -> List[Tuple[plt.Figure, str]]:
    """Spatial maps for voxels whose latency falls in specific time windows.

    Each window is specified in seconds relative to the encounter frame.
    Default windows: [(-30,-21), (-20,-11), (-10,-1), (0,2), (3,6), (7,15)].

    Returns list of (figure, window_label) tuples.
    """
    if windows is None:
        windows = [(-30, -21), (-20, -11), (-10, -1), (0, 2), (3, 6), (7, 15)]

    enc_sec = encounter_frame / fps
    lat_sec = latency / fps
    lat_aligned = lat_sec - enc_sec

    vmin_global = np.nanpercentile(data_2d, 1)
    vmax_global = np.nanpercentile(data_2d, 99)

    figs = []
    for w_start, w_end in windows:
        in_window = (lat_aligned >= w_start) & (lat_aligned <= w_end)
        voxel_idx = np.where(in_window)[0]
        label = f'[{w_start:+.0f}, {w_end:+.0f}] s (n={len(voxel_idx)})'

        if len(voxel_idx) == 0:
            print(f"  Window {label}: no voxels — skipping.")
            continue

        # Mean activity of these voxels over a window around encounter
        t_start = max(0, int((enc_sec + w_start) * fps))
        t_end = min(data_2d.shape[0], int((enc_sec + w_end) * fps))
        if t_end <= t_start:
            t_start, t_end = 0, data_2d.shape[0]
        mean_act = np.nanmean(data_2d[t_start:t_end, voxel_idx], axis=0)

        fig = plot_voxel_spatial_map(
            mean_act, voxel_idx,
            selected_indices, mask_flat, mask_3d, spatial_shape,
            cmap=cmap, vmin=vmin_global, vmax=vmax_global,
            nrows=nrows, ncols=ncols, figsize=figsize,
        )
        fig.suptitle(label, y=1.0, fontsize=14)
        figs.append((fig, label))

    return figs


# ============================================================================
# PLOTTING — NEW POPULATION-LEVEL ANALYSES
# ============================================================================

def plot_latency_cdf(
    latency: np.ndarray,
    fps: float,
    encounter_frame: Optional[int] = None,
) -> plt.Figure:
    """Cumulative fraction of responsive voxels activated by each time point.

    Shows the CDF (empirical distribution function) of latency.
    """
    valid = latency[~np.isnan(latency)]
    if len(valid) == 0:
        print("  No responsive voxels for CDF plot.")
        fig, ax = pretty_plot()
        return fig

    lat_sec = np.sort(valid) / fps
    cdf = np.arange(1, len(lat_sec) + 1) / len(lat_sec)

    fig, ax = pretty_plot(figsize=(8, 5))
    ax.step(lat_sec, cdf, where='post', color='tab:blue', lw=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Fraction of responsive voxels activated')
    ax.set_ylim(0, 1.05)

    if encounter_frame is not None:
        ax.axvline(encounter_frame / fps, color='k', ls='--', label='encounter')
        ax.legend(fontsize=12)

    fig.tight_layout()
    return fig


def plot_mean_response_onset(
    data_2d: np.ndarray,
    latency: np.ndarray,
    fps: float,
    encounter_frame: Optional[int] = None,
    window_frames: int = 60,
    smooth_sigma: float = 2.0,
) -> plt.Figure:
    """Average response waveform aligned to each voxel's onset.

    For every responsive voxel, a snippet of *window_frames* after onset is
    extracted and all snippets are averaged to show the mean response shape.
    Also shows ± SEM.
    """
    valid_mask = ~np.isnan(latency)
    valid_lat = latency[valid_mask].astype(int)
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        print("  No responsive voxels for onset plot.")
        fig, ax = pretty_plot()
        return fig

    T = data_2d.shape[0]
    snippets = []
    for vi, onset in zip(np.where(valid_mask)[0], valid_lat):
        end = min(onset + window_frames, T)
        snippet = data_2d[onset:end, vi]
        if len(snippet) < window_frames:
            snippet = np.pad(snippet, (0, window_frames - len(snippet)),
                             constant_values=np.nan)
        snippets.append(snippet)

    snippets = np.array(snippets)  # (n_valid, window_frames)
    mean_resp = np.nanmean(snippets, axis=0)
    sem_resp = np.nanstd(snippets, axis=0) / np.sqrt(n_valid)

    if smooth_sigma > 0:
        mean_resp = causal_smooth(mean_resp[:, None], sigma=smooth_sigma).ravel()

    time_ax = np.arange(window_frames) / fps

    fig, ax = pretty_plot(figsize=(8, 5))
    ax.plot(time_ax, mean_resp, color='tab:blue', lw=2, label='mean')
    ax.fill_between(
        time_ax, mean_resp - sem_resp, mean_resp + sem_resp,
        alpha=0.25, color='tab:blue',
    )
    ax.set_xlabel('Time from onset (s)')
    ax.set_ylabel('Response')
    ax.legend(fontsize=12)
    fig.tight_layout()
    return fig


# ============================================================================
# ROI-LEVEL LATENCY ANALYSIS
# ============================================================================

def compute_roi_latency_stats(
    latency: np.ndarray,
    selected_indices: np.ndarray,
    mask_flat: np.ndarray,
    roi: np.ndarray,
    roi_labels: List[str],
    spatial_shape: Tuple[int, int, int],
    bin_factor: int = 1,
) -> Dict[str, np.ndarray]:
    """Group voxel latencies by ROI label.

    Parameters
    ----------
    latency : (V,) — per-voxel latency.
    selected_indices, mask_flat : from the flatten/select pipeline.
    roi : (Z, H, W) uint8 — label mask (1-indexed regions).
    roi_labels : list of str — region names (index 0 → label 1 in roi).
    spatial_shape : (Z, H, W) of the preprocessed volume.
    bin_factor : extra binning applied during preprocessing (to downsample roi
                 to match spatial_shape).

    Returns
    -------
    dict mapping label name → latency array (only responsive voxels, NaN-free).
    """
    roi_binned = bin_spatial(roi.astype(np.float32), bin_factor)

    all_idx = np.arange(len(latency))
    coords = preprocessed_idx_to_spatial(
        all_idx, selected_indices, mask_flat, spatial_shape,
    )  # (V, 3)

    roi_vals = roi_binned[coords[:, 0], coords[:, 1], coords[:, 2]]

    result = {}
    for i, label in enumerate(roi_labels):
        roi_id = i + 1  # 1-indexed
        in_roi = np.round(roi_vals) == roi_id
        lat_roi = latency[in_roi]
        lat_valid = lat_roi[~np.isnan(lat_roi)]
        result[label] = lat_valid

    return result


def plot_roi_latency_comparison(
    roi_latency_dict: Dict[str, np.ndarray],
    fps: float,
    encounter_frame: Optional[int] = None,
) -> plt.Figure:
    """Box/violin plot comparing latency distributions across ROIs."""
    labels = list(roi_latency_dict.keys())
    data_sec = [v / fps for v in roi_latency_dict.values()]

    fig, ax = pretty_plot(figsize=(10, 6))
    parts = ax.violinplot(
        data_sec, showmedians=True, showextrema=False,
    )
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax.set_ylabel('Response latency (s)')

    if encounter_frame is not None:
        ax.axhline(encounter_frame / fps, color='k', ls='--', label='encounter')
        ax.legend(fontsize=12)

    fig.tight_layout()
    return fig


# ============================================================================
# RESULT PERSISTENCE
# ============================================================================

def save_results(
    output_dir: str,
    stats: ResponseStats,
    selected_indices: np.ndarray,
    mask_flat: np.ndarray,
    spatial_shape: Tuple[int, int, int],
    analysis_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Save latency results to ``latency_results.npz``.

    Returns the path to the saved file.
    """
    path = os.path.join(output_dir, 'latency_results.npz')
    save_dict = dict(
        latency=stats.latency,
        is_responsive=stats.is_responsive,
        voxel_response_prob=stats.voxel_response_prob,
        mean_baseline=stats.mean_baseline,
        mean_pre=stats.mean_pre,
        mean_post=stats.mean_post,
        var_baseline=stats.var_baseline,
        selected_indices=selected_indices,
        mask_flat=mask_flat,
        spatial_shape=np.array(spatial_shape),
    )
    if analysis_params is not None:
        # Store params as a string repr (npz can't store dicts natively)
        save_dict['analysis_params_repr'] = np.array(repr(analysis_params))

    np.savez_compressed(path, **save_dict)
    print(f"Saved latency results → {path}")
    return path


def load_results(output_dir: str) -> Dict[str, np.ndarray]:
    """Load ``latency_results.npz`` and return as a dict."""
    path = os.path.join(output_dir, 'latency_results.npz')
    data = dict(np.load(path, allow_pickle=True))
    data['spatial_shape'] = tuple(data['spatial_shape'])
    return data
