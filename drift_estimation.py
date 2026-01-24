import glob
import sys
import os
import pandas as pd
import numpy as np
import tifffile
from nd2reader import ND2Reader
import imageio
import matplotlib.pyplot as plt
import itk
import warnings; warnings.filterwarnings('ignore', category=UserWarning, module='itk')
from tqdm import tqdm
from scipy.ndimage import laplace


def compute_laplacian_variance(image):
    """
    Compute Laplacian variance as a sharpness/focus metric.
    Higher values indicate sharper (more in-focus) images.
    """
    lap = laplace(image.astype(np.float64))
    return np.var(lap)


def causal_ewma(x, alpha=0.15):
    """
    Causal exponential weighted moving average.
    Only uses current and past values.
    alpha: smoothing factor in (0,1]. Smaller = smoother.
    """
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = alpha * x[t] + (1 - alpha) * y[t - 1]
    return y


def select_consistent_z_slices(
    out_dir,
    tif_dir,
    stack_range,
    n_slices=8,
    smooth_alpha=0.15,
    sharpness_threshold_percentile=30,
    save_diagnostics=True,
):
    """
    Select a consistent subset of z-slices across all frames using sharpness-based
    focus detection with drift compensation.
    
    Parameters
    ----------
    out_dir : str
        Output directory path.
    tif_dir : str
        Subdirectory containing the TIF stacks.
    stack_range : range
        Range of frame indices to process.
    n_slices : int
        Number of z-slices to select per frame (constant across all frames).
    smooth_alpha : float
        Smoothing parameter for causal EWMA (0.1-0.3 recommended, smaller = smoother).
    sharpness_threshold_percentile : float
        Percentile threshold for determining if a z-slice is sharp enough.
    save_diagnostics : bool
        If True, save diagnostic plots and CSV files.
    
    Returns
    -------
    z_indices_per_frame : list of np.ndarray
        List where each element is an array of z-slice indices for that frame.
        If padding is needed, indices will include -1 to indicate padding positions.
    needs_padding : np.ndarray of bool
        Boolean array indicating which frames require zero-padding.
    focus_tracked : np.ndarray
        Tracked focus position (center z-slice) per frame.
    sharpness : np.ndarray
        Sharpness metric array of shape (n_frames, n_z_slices).
    diagnostics : dict
        Dictionary containing diagnostic information.
    """
    print("Computing sharpness-based focus metrics...")
    
    tif_files = check_files(os.path.join(out_dir, tif_dir), stack_range, 'tif')
    n_frames = len(tif_files)
    
    # Read first file to get dimensions
    first_stack = tifffile.imread(tif_files[0])
    if len(first_stack.shape) == 3:
        first_stack = first_stack[np.newaxis, :, :, :]
    n_z = first_stack.shape[0]
    
    # Compute sharpness (Laplacian variance) for each z-slice of each frame
    sharpness = np.zeros((n_frames, n_z))
    
    for i in tqdm(range(n_frames), desc="Computing sharpness"):
        stack = tifffile.imread(tif_files[i])
        if len(stack.shape) == 3:
            stack = stack[np.newaxis, :, :, :]
        
        # Use RFP channel (index 1) for sharpness computation
        rfp = stack[:, 1]
        
        for z in range(n_z):
            sharpness[i, z] = compute_laplacian_variance(rfp[z])
    
    # Step 1: Find peak sharpness z-slice per frame (raw)
    z_peak_raw = np.argmax(sharpness, axis=1).astype(float)
    
    # Step 2: Sub-slice refinement using parabolic interpolation
    z_peak_refined = np.zeros(n_frames)
    for t in range(n_frames):
        k = int(z_peak_raw[t])
        if 0 < k < n_z - 1:
            y0, y1, y2 = sharpness[t, k - 1], sharpness[t, k], sharpness[t, k + 1]
            denom = y0 - 2.0 * y1 + y2
            if denom != 0:
                delta = 0.5 * (y0 - y2) / denom
                delta = np.clip(delta, -1.0, 1.0)
            else:
                delta = 0.0
            z_peak_refined[t] = k + delta
        else:
            z_peak_refined[t] = k
    
    # Step 3: Causal smoothing to track drift
    focus_tracked = causal_ewma(z_peak_refined, alpha=smooth_alpha)
    
    # Step 4: Determine sharpness threshold per frame
    # A z-slice is considered "usable" if its sharpness is above this threshold
    sharpness_thresholds = np.percentile(sharpness, sharpness_threshold_percentile, axis=1)
    
    # Step 5: Select z-slices centered around tracked focus
    half_window = n_slices // 2
    z_indices_per_frame = []
    needs_padding = np.zeros(n_frames, dtype=bool)
    n_usable_per_frame = np.zeros(n_frames, dtype=int)
    
    for t in range(n_frames):
        center = focus_tracked[t]
        threshold = sharpness_thresholds[t]
        
        # Determine candidate z-slices around the center
        z_lo = int(np.floor(center - half_window))
        z_hi = z_lo + n_slices  # exclusive upper bound
        
        # Clamp to valid range
        if z_lo < 0:
            z_lo = 0
            z_hi = min(n_slices, n_z)
        elif z_hi > n_z:
            z_hi = n_z
            z_lo = max(0, n_z - n_slices)
        
        # Get candidate indices
        candidates = np.arange(z_lo, z_hi)
        
        # Filter by sharpness threshold
        usable_mask = sharpness[t, candidates] >= threshold
        usable_indices = candidates[usable_mask]
        n_usable = len(usable_indices)
        n_usable_per_frame[t] = n_usable
        
        # Build final index array
        if n_usable >= n_slices:
            # Enough usable slices - take the n_slices centered around the peak
            # within the usable set
            final_indices = usable_indices[:n_slices]
        elif n_usable > 0:
            # Some usable slices but not enough - use what we have and pad
            final_indices = np.full(n_slices, -1, dtype=int)  # -1 indicates padding
            final_indices[:n_usable] = usable_indices
            needs_padding[t] = True
        else:
            # No usable slices - fall back to the candidate range and pad everything
            final_indices = np.full(n_slices, -1, dtype=int)
            # Still try to use the candidates even if below threshold
            n_to_use = min(len(candidates), n_slices)
            final_indices[:n_to_use] = candidates[:n_to_use]
            needs_padding[t] = True
        
        z_indices_per_frame.append(final_indices)
    
    # Compute drift
    drift = focus_tracked - focus_tracked[0]
    
    # Diagnostics
    diagnostics = {
        'z_peak_raw': z_peak_raw,
        'z_peak_refined': z_peak_refined,
        'focus_tracked': focus_tracked,
        'drift': drift,
        'sharpness_thresholds': sharpness_thresholds,
        'n_usable_per_frame': n_usable_per_frame,
        'n_slices_requested': n_slices,
        'n_frames_needing_padding': needs_padding.sum(),
    }
    
    if save_diagnostics:
        _save_z_selection_diagnostics(
            out_dir, sharpness, z_peak_raw, focus_tracked, drift,
            z_indices_per_frame, needs_padding, n_usable_per_frame, n_slices
        )
    
    print(f"\nZ-slice selection complete:")
    print(f"  Total frames: {n_frames}")
    print(f"  Z-slices per frame: {n_slices}")
    print(f"  Frames needing padding: {needs_padding.sum()} ({100 * needs_padding.mean():.1f}%)")
    print(f"  Total drift: {drift[-1]:.2f} z-slices")
    print(f"  Max drift from start: {np.abs(drift).max():.2f} z-slices")
    
    return z_indices_per_frame, needs_padding, focus_tracked, sharpness, diagnostics


def _save_z_selection_diagnostics(
    out_dir, sharpness, z_peak_raw, focus_tracked, drift,
    z_indices_per_frame, needs_padding, n_usable_per_frame, n_slices
):
    """Save diagnostic plots and CSV for z-slice selection."""
    n_frames, n_z = sharpness.shape
    
    # Save sharpness matrix
    df = pd.DataFrame(sharpness, columns=[f'Z{i}' for i in range(n_z)])
    df.to_csv(os.path.join(out_dir, 'sharpness.csv'), index_label='frame')
    
    # Save z-indices per frame
    z_indices_array = np.array(z_indices_per_frame)
    df_indices = pd.DataFrame(z_indices_array, columns=[f'slot_{i}' for i in range(n_slices)])
    df_indices['needs_padding'] = needs_padding
    df_indices['n_usable'] = n_usable_per_frame
    df_indices['focus_tracked'] = focus_tracked
    df_indices['drift'] = drift
    df_indices.to_csv(os.path.join(out_dir, 'z_selection.csv'), index_label='frame')
    
    # Create diagnostic plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1) Sharpness heatmap with tracking overlay
    ax1 = axes[0, 0]
    im = ax1.imshow(sharpness.T, aspect='auto', cmap='viridis', origin='lower')
    ax1.plot(np.arange(n_frames), z_peak_raw, 'w.', markersize=1, alpha=0.3, label='Raw peaks')
    ax1.plot(np.arange(n_frames), focus_tracked, 'r-', linewidth=1.5, label='Tracked')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Z-slice')
    ax1.set_title('Sharpness (Laplacian Variance) with Focus Tracking')
    ax1.legend(loc='upper right', fontsize=8)
    plt.colorbar(im, ax=ax1, label='Sharpness', shrink=0.8)
    
    # 2) Drift over time
    ax2 = axes[0, 1]
    ax2.plot(drift, 'b-', linewidth=1)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.fill_between(np.arange(len(drift)), drift, alpha=0.3)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Drift (z-slices)')
    ax2.set_title(f'Focus Drift (total: {drift[-1]:.2f} slices)')
    
    # 3) Usable slices per frame
    ax3 = axes[1, 0]
    colors = ['orange' if p else 'green' for p in needs_padding]
    ax3.bar(np.arange(n_frames), n_usable_per_frame, color=colors, width=1.0)
    ax3.axhline(n_slices, color='r', linestyle='--', label=f'Target: {n_slices}')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Usable z-slices')
    ax3.set_title(f'Usable Slices per Frame (orange = needs padding)')
    ax3.legend(fontsize=8)
    
    # 4) Selected z-range visualization
    ax4 = axes[1, 1]
    z_min_selected = np.array([idx[idx >= 0].min() if (idx >= 0).any() else np.nan 
                                for idx in z_indices_per_frame])
    z_max_selected = np.array([idx[idx >= 0].max() if (idx >= 0).any() else np.nan 
                                for idx in z_indices_per_frame])
    ax4.fill_between(np.arange(n_frames), z_min_selected, z_max_selected, 
                     alpha=0.5, color='blue', label='Selected range')
    ax4.plot(focus_tracked, 'r-', linewidth=1, label='Focus center')
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Z-slice')
    ax4.set_title('Selected Z-slice Range per Frame')
    ax4.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'z_selection_diagnostics.png'), dpi=150)
    plt.close()


def extract_z_subset(stack, z_indices):
    """
    Extract a subset of z-slices from a stack, padding with zeros where needed.
    
    Parameters
    ----------
    stack : np.ndarray
        Input stack of shape (Z, C, H, W) or (Z, H, W).
    z_indices : np.ndarray
        Array of z-indices to extract. -1 indicates padding with zeros.
    
    Returns
    -------
    subset : np.ndarray
        Extracted subset with same shape structure but reduced Z dimension.
    """
    n_slices = len(z_indices)
    
    if len(stack.shape) == 4:
        _, n_channels, h, w = stack.shape
        subset = np.zeros((n_slices, n_channels, h, w), dtype=stack.dtype)
    else:
        _, h, w = stack.shape
        subset = np.zeros((n_slices, h, w), dtype=stack.dtype)
    
    for i, z_idx in enumerate(z_indices):
        if z_idx >= 0:
            subset[i] = stack[z_idx]
        # else: remains zero (padding)
    
    return subset

def extract_frame_from_path(path):
    # get the basename (filename with extension): '13845.tif'
    filename_with_ext = os.path.basename(path)
    filename_without_ext = os.path.splitext(filename_with_ext)[0]
    # convert the resulting string to an integer for numerical sorting
    # this step is needed for sorting '0837' correctly before '1675', etc.
    return int(filename_without_ext)

def get_range(input_nd2, stack_length=41):
    """Returns a range object containing valid stack indices from the ND2 file."""
    with ND2Reader(input_nd2) as f:
        stack_range = range(f.metadata['num_frames'] // stack_length)
    return stack_range

def check_files(out_dir, stack_range, extension):
    """Checks if all the expected files in the stack range are in the directory"""
    existing_files = set(glob.glob(os.path.join(out_dir, f'*.{extension}')))
    expected_files = {os.path.join(out_dir, f'{i:04d}.{extension}') for i in stack_range}
    if missing_files := expected_files - existing_files:
        stacks = sorted([int(os.path.splitext(os.path.basename(f))[0]) for f in missing_files])
        raise FileNotFoundError(f"Missing .{extension} files: {','.join([str(i) for i in stacks])}")

    return sorted(expected_files, key=extract_frame_from_path)

def get_transform_params(file, parameter_object):
    """Extracts transform parameters from a text file."""
    parameter_object.ReadParameterFile(file)
    return np.array(parameter_object.GetParameterMap(0)['TransformParameters'], float)






def make_rgb(frame, rmax, gmax, shape=(512, 512, 3)):
    """Creates an RGB image from a frame."""
    gfp, rfp = frame
    rgb = np.zeros(shape, np.ubyte)
    adjust = lambda frame, lo, hi: np.clip((frame.astype(np.float32) - lo) / (hi - lo), 0, 1)
    rgb[..., 0] = adjust(rfp, 0, rmax) * 255
    rgb[..., 1] = adjust(gfp, 0, gmax) * 255
    return rgb


def main():
    """Main pipeline: create parameter DataFrame, save results."""
    input_nd2 = sys.argv[1]
    tif_dir = sys.argv[2]
    stack_length = sys.argv[3]
    num_frames = sys.argv[4] 

    stack_range = range(num_frames) #get_range(input_nd2, stack_length)
    out_dir = os.path.splitext(input_nd2)[0]
    
    z_indices, needs_padding, focus, sharpness, diag = select_consistent_z_slices(
    out_dir, tif_dir, stack_range,
    n_slices=24,           # number of z-slices to select per frame (at 1.08 um per slice, this is roughly the largest we should go)
    smooth_alpha=0.15,    # causal smoothing (smaller = smoother)
    sharpness_threshold_percentile=30,
)


if __name__ == '__main__':
    main()
