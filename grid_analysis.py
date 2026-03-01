import tifffile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import sys
import glob
import os
from tqdm import tqdm

import matplotlib
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 15}
matplotlib.rc('font', **font)


def load_data(input_dir):
    """Load normalized voxel data and mask."""
    # Load normalized voxels
    voxels_path = os.path.join(input_dir, 'normalized_voxels.npy')
    if not os.path.exists(voxels_path):
        raise FileNotFoundError(f"Normalized voxels not found at {voxels_path}")
    
    normalized_data = np.load(voxels_path)
    print(f"Loaded normalized data with shape: {normalized_data.shape}")
    
    # Load mask - find the fixed_mask_*.tif file
    fixed_mask_files = glob.glob(os.path.join(input_dir, 'fixed_mask_*.tif'))
    if fixed_mask_files is None or len(fixed_mask_files) == 0:
        fixed_mask_files = glob.glob(os.path.join(input_dir, 'fixed_mask*.tif'))

    if not fixed_mask_files:
        raise FileNotFoundError(f"No fixed_mask_*.tif found in {input_dir}")
    
    mask = tifffile.imread(fixed_mask_files[0])
    print(f"Loaded mask with shape: {mask.shape}")
    
    return normalized_data, mask


def create_grid(height, width, grid_spacing_x, grid_spacing_y):
    """
    Create grid indices based on spacing.
    
    Returns:
        grid_info: list of tuples (grid_id, y_start, y_end, x_start, x_end)
        n_grids_y: number of grids in y direction
        n_grids_x: number of grids in x direction
    """
    n_grids_y = height // grid_spacing_y
    n_grids_x = width // grid_spacing_x
    
    grid_info = []
    grid_id = 0
    
    for gy in range(n_grids_y):
        for gx in range(n_grids_x):
            y_start = gy * grid_spacing_y
            y_end = (gy + 1) * grid_spacing_y
            x_start = gx * grid_spacing_x
            x_end = (gx + 1) * grid_spacing_x
            grid_info.append((grid_id, y_start, y_end, x_start, x_end))
            grid_id += 1
    
    return grid_info, n_grids_y, n_grids_x

def calculate_grid_intensity_single_frame(frame, mask, grid_info):
    """
    Calculate mean intensity in each grid square for a single frame.
    
    Args:
        frame: (H, W) array for a single time point and z-slice
        mask: (H, W) array
        grid_info: list of tuples (grid_id, y_start, y_end, x_start, x_end)
    Returns:
        grid_intensity: (n_grids,) array of mean intensity for each grid
    """    
    grid_intensity = np.zeros(len(grid_info))
    
    for grid_id, y_start, y_end, x_start, x_end in grid_info:
        grid_mask = mask[y_start:y_end, x_start:x_end]
        grid_data = frame[y_start:y_end, x_start:x_end]
        
        # Calculate mean intensity for this grid (no masking applied to data)
        grid_intensity[grid_id] = grid_data.mean()
    
    return grid_intensity

def calculate_grid_intensities(normalized_data, mask, grid_spacing_x, grid_spacing_y, bin_factor=1):
    """
    Calculate mean intensity in each grid square for each time point and z-slice.
    No normalization is applied - data is used as-is.
    No masking is applied - all grid squares are computed, with mask coverage saved for reference.
    
    Args:
        normalized_data: (T, Z, H, W) array - already normalized data
        mask: (H, W) array
        grid_spacing_x: grid size in x direction
        grid_spacing_y: grid size in y direction
        bin_factor: binning factor if mask needs to be resized
        
    Returns:
        grid_timeseries: (T, Z, n_grids_y, n_grids_x) array
        grid_info: list of grid boundaries
        n_grids_y: number of grids in y direction
        n_grids_x: number of grids in x direction
        mask_binned: binned mask
        grid_mask_coverage: (Z, n_grids_y, n_grids_x) array - mask coverage for each grid (0-1)
        grid_in_mask: (n_grids_y, n_grids_x) boolean array - True if grid is within mask
    """
    
    
    
    n_time, n_z, height, width = normalized_data.shape
    
    # Bin mask if needed
    h_mask, w_mask = mask.shape
    if h_mask != height or w_mask != width:
        h_binned = height
        w_binned = width
        bin_factor_h = h_mask // h_binned
        bin_factor_w = w_mask // w_binned
        mask_binned = mask[:h_binned * bin_factor_h, :w_binned * bin_factor_w].reshape(
            h_binned, bin_factor_h, w_binned, bin_factor_w
        ).max(axis=(1, 3))
    else:
        mask_binned = mask
    
    # Create grid
    grid_info, n_grids_y, n_grids_x = create_grid(height, width, grid_spacing_x, grid_spacing_y)
    
    print(f"Grid: {n_grids_y} rows x {n_grids_x} cols = {len(grid_info)} grids per z-slice")
    print(f"Total grids across all z-slices: {len(grid_info) * n_z}")
    
    # Initialize output arrays
    grid_timeseries = np.zeros((n_time, n_z, n_grids_y, n_grids_x))
    grid_mask_coverage = np.zeros((n_z, n_grids_y, n_grids_x))
    grid_in_mask = np.zeros((n_grids_y, n_grids_x), dtype=bool)
    
    # Calculate intensity for each grid (no masking applied to data)
    for grid_id, y_start, y_end, x_start, x_end in tqdm(grid_info, desc="Processing grids"):
        gy = grid_id // n_grids_x
        gx = grid_id % n_grids_x
        
        # Get mask coverage for this grid region (for reference, not for masking)
        grid_mask = mask_binned[y_start:y_end, x_start:x_end]
        mask_coverage = grid_mask.mean()
        
        # Determine if this grid is within the mask (using threshold of 0.5)
        grid_in_mask[gy, gx] = mask_coverage >= 0.5
        
        for z in range(n_z):
            # Get data for this grid region - NO MASKING applied
            grid_data = normalized_data[:, z, y_start:y_end, x_start:x_end]
            
            # Calculate mean over all voxels in grid (no masking)
            grid_timeseries[:, z, gy, gx] = grid_data.mean(axis=(1, 2))
            
            # Store mask coverage for reference
            grid_mask_coverage[z, gy, gx] = mask_coverage
    
    return grid_timeseries, grid_info, n_grids_y, n_grids_x, mask_binned, grid_mask_coverage, grid_in_mask


def causal_smooth(data, sigma=2.0, truncate=4.0):
    """
    Apply causal Gaussian smoothing to a (T, features) array.
    Only uses past and current time points, never future data.
    
    Args:
        data: (T, features) numpy array
        sigma: standard deviation of the Gaussian kernel (in samples/frames)
               Higher sigma = more smoothing. sigma=1 is mild, sigma=5 is strong.
        truncate: truncate the filter at this many standard deviations (default: 4.0)
        
    Returns:
        smoothed_data: (T, features) array with causal Gaussian smoothing applied
    """
    T, n_features = data.shape
    smoothed = np.zeros_like(data)
    
    # Create one-sided (causal) Gaussian kernel
    # Kernel extends from 0 to truncate*sigma samples into the past
    kernel_radius = int(truncate * sigma)
    
    # Generate Gaussian weights for lags 0, 1, 2, ..., kernel_radius
    # lag 0 = current time point, lag k = k frames in the past
    lags = np.arange(kernel_radius + 1)
    kernel = np.exp(-0.5 * (lags / sigma) ** 2)
    kernel = kernel / kernel.sum()  # Normalize to sum to 1
    
    # Apply causal convolution
    for t in range(T):
        # Determine how many past samples are available
        available_lags = min(t + 1, len(kernel))
        
        # Get the portion of the kernel we can use (from lag 0 to available_lags-1)
        k = kernel[:available_lags]
        
        # Re-normalize kernel for edge cases (when we have fewer samples than kernel size)
        k_normalized = k / k.sum()
        
        # Get past data points (current + past)
        past_data = data[t - available_lags + 1:t + 1, :]  # shape: (available_lags, n_features)
        
        # Apply weights: kernel[0] is for current, kernel[1] for t-1, etc.
        # So we need to flip the data or the kernel
        # past_data is [t-available_lags+1, ..., t-1, t], kernel is [0, 1, 2, ...]
        # We want: kernel[0]*data[t] + kernel[1]*data[t-1] + ...
        # So flip past_data: [t, t-1, ..., t-available_lags+1]
        smoothed[t] = (k_normalized[:, np.newaxis] * past_data[::-1, :]).sum(axis=0)
    
    return smoothed


def flatten_grid_timeseries(grid_timeseries, grid_mask_coverage, grid_baseline, grid_in_mask, min_coverage=0.0):
    """
    Flatten grid timeseries to (T, n_total_grids) with z-slice labels.
    All grids are included (regardless of mask), with in_mask indicator saved.
    
    Args:
        grid_timeseries: (T, Z, n_grids_y, n_grids_x) array
        grid_mask_coverage: (Z, n_grids_y, n_grids_x) array
        grid_baseline: (Z, n_grids_y, n_grids_x) array - pre-computed baseline per grid
        grid_in_mask: (n_grids_y, n_grids_x) boolean array - True if grid is within mask
        min_coverage: minimum mask coverage to include a grid (default: 0.0, include all)
        
    Returns:
        flat_timeseries: (T, n_valid_grids) array
        z_labels: z-slice label for each grid
        grid_labels: (gy, gx) label for each grid
        flat_baseline: (n_valid_grids,) array - baseline for each valid grid
        flat_in_mask: (n_valid_grids,) boolean array - whether each grid is in mask
        unflatten_info: dict containing info needed to unflatten back to original shape
    """
    n_time, n_z, n_grids_y, n_grids_x = grid_timeseries.shape
    
    flat_list = []
    z_labels = []
    grid_labels = []
    baseline_list = []
    in_mask_list = []
    flat_to_grid_idx = []  # Maps flat index to (z, gy, gx)
    
    flat_idx = 0
    for z in range(n_z):
        for gy in range(n_grids_y):
            for gx in range(n_grids_x):
                if grid_mask_coverage[z, gy, gx] >= min_coverage:
                    flat_list.append(grid_timeseries[:, z, gy, gx])
                    z_labels.append(z)
                    grid_labels.append((gy, gx))
                    baseline_list.append(grid_baseline[z, gy, gx])
                    in_mask_list.append(grid_in_mask[gy, gx])
                    flat_to_grid_idx.append((z, gy, gx))
                    flat_idx += 1
    
    flat_timeseries = np.array(flat_list).T  # (T, n_valid_grids)
    z_labels = np.array(z_labels)
    flat_baseline = np.array(baseline_list)  # (n_valid_grids,)
    flat_in_mask = np.array(in_mask_list)  # (n_valid_grids,)
    
    # Create unflatten info for reconstructing original shape
    unflatten_info = {
        'n_time': n_time,
        'n_z': n_z,
        'n_grids_y': n_grids_y,
        'n_grids_x': n_grids_x,
        'flat_to_grid_idx': np.array(flat_to_grid_idx),  # (n_valid_grids, 3) - (z, gy, gx) for each flat idx
        'grid_mask_coverage': grid_mask_coverage,
        'min_coverage': min_coverage
    }
    
    return flat_timeseries, z_labels, grid_labels, flat_baseline, flat_in_mask, unflatten_info


def unflatten_grid_timeseries(flat_timeseries, unflatten_info, fill_value=np.nan):
    """
    Unflatten a (T, n_valid_grids) array back to (T, Z, n_grids_y, n_grids_x) shape.
    
    Args:
        flat_timeseries: (T, n_valid_grids) array
        unflatten_info: dict containing:
            - n_time: int
            - n_z: int
            - n_grids_y: int
            - n_grids_x: int
            - flat_to_grid_idx: (n_valid_grids, 3) array of (z, gy, gx) indices
        fill_value: value to fill for grids that were excluded (default: np.nan)
        
    Returns:
        grid_timeseries: (T, Z, n_grids_y, n_grids_x) array
    """
    n_time = flat_timeseries.shape[0]
    n_z = unflatten_info['n_z']
    n_grids_y = unflatten_info['n_grids_y']
    n_grids_x = unflatten_info['n_grids_x']
    flat_to_grid_idx = unflatten_info['flat_to_grid_idx']
    
    # Initialize with fill value
    grid_timeseries = np.full((n_time, n_z, n_grids_y, n_grids_x), fill_value)
    
    # Fill in the valid grids
    for flat_idx, (z, gy, gx) in enumerate(flat_to_grid_idx):
        grid_timeseries[:, z, gy, gx] = flat_timeseries[:, flat_idx]
    
    return grid_timeseries


def load_flat_timeseries(filepath):
    """
    Load a saved flat timeseries .npz file and return data with unflatten info.
    
    Args:
        filepath: path to grid_timeseries_flat.npz or grid_timeseries_flat_smoothed.npz
        
    Returns:
        flat_timeseries: (T, n_grids) array
        z_labels: z-slice label for each grid
        grid_labels: (gy, gx) label for each grid
        in_mask: boolean array indicating if each grid is within mask
        unflatten_info: dict that can be passed to unflatten_grid_timeseries()
        metadata: dict with additional info (baseline, z_range, etc.)
    """
    data = np.load(filepath, allow_pickle=True)
    
    flat_timeseries = data['timeseries']
    z_labels = data['z_labels']
    grid_labels = data['grid_labels']
    in_mask = data['in_mask']
    
    unflatten_info = {
        'n_z': int(data['n_z']),
        'n_grids_y': int(data['n_grids_y']),
        'n_grids_x': int(data['n_grids_x']),
        'flat_to_grid_idx': data['flat_to_grid_idx'],
        'z_offset': int(data['z_offset'])
    }
    
    metadata = {
        'baseline': data['baseline'],
        'z_range': tuple(data['z_range']),
        'baseline_window_sec': tuple(data['baseline_window_sec'])
    }
    
    return flat_timeseries, z_labels, grid_labels, in_mask, unflatten_info, metadata


def plot_timeseries(flat_timeseries, z_labels, fps=1.0, output_path=None, 
                    data_start_time=0.0, time_offset=0.0, baseline=None):
    """Plot time series for all grids, colored by z-slice.
    
    Args:
        flat_timeseries: (T, n_grids) array
        z_labels: z-slice label for each grid
        fps: frames per second
        output_path: path to save figure
        data_start_time: the actual time (in sec) of the first frame in the data
        time_offset: reference time to subtract (e.g., food delivery time) so t=0 is at this event
        baseline: (n_grids,) array - pre-computed baseline for each grid. If None, uses first 60 frames.
    """
    n_time, n_grids = flat_timeseries.shape
    # Time vector: starts at data_start_time, relative to time_offset
    t = (np.arange(n_time) / fps + data_start_time) - time_offset
    
    unique_z = np.unique(z_labels)
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(unique_z)))
    
    # Use pre-computed baseline if provided, otherwise fall back to first 60 frames
    if baseline is None:
        baseline = flat_timeseries[:60, :].mean(axis=0)
    
    zscore_timeseries = flat_timeseries #/ (baseline + 1e-6)
    
    plt.figure(figsize=(14, 6))
    
    for i in range(n_grids):
        z = z_labels[i]
        color_idx = np.where(unique_z == z)[0][0]
        alpha = 0.3 if n_grids > 50 else 0.6
        plt.plot(t, zscore_timeseries[:, i], color=colors[color_idx], alpha=alpha, lw=0.5)
    
    # Add colorbar for z-slices
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=unique_z.min(), vmax=unique_z.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), label='Z-slice')
    
    plt.xlabel('Time (sec)')
    plt.ylabel('$F/F_{\\mathrm{baseline}}$')
    plt.title('Grid Time Series (colored by z-slice)')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved time series plot to {output_path}")
    
    plt.show()


def plot_heatmap(flat_timeseries, z_labels, fps=1.0, output_path=None,
                 data_start_time=0.0, time_offset=0.0, baseline=None):
    """Plot heatmap of time series, grouped by z-slice with annotations.
    
    Args:
        flat_timeseries: (T, n_grids) array
        z_labels: z-slice label for each grid
        fps: frames per second
        output_path: path to save figure
        data_start_time: the actual time (in sec) of the first frame in the data
        time_offset: reference time to subtract (e.g., food delivery time) so t=0 is at this event
        baseline: (n_grids,) array - pre-computed baseline for each grid. If None, uses first 60 frames.
    """
    n_time, n_grids = flat_timeseries.shape
    # Time vector: starts at data_start_time, relative to time_offset
    t = (np.arange(n_time) / fps + data_start_time) - time_offset
    
    # Sort by z-slice to group together
    sort_idx = np.argsort(z_labels)
    sorted_timeseries = flat_timeseries[:, sort_idx]
    sorted_z_labels = z_labels[sort_idx]
    
    # Use pre-computed baseline if provided, otherwise fall back to first 60 frames
    if baseline is None:
        sorted_baseline = sorted_timeseries[:60, :].mean(axis=0)
    else:
        sorted_baseline = baseline[sort_idx]
    
    # F/F0 normalization
    zscore_timeseries = sorted_timeseries #/ (sorted_baseline + 1e-6)
    
    # determine vmin and vmax, considering outliers
    vmin = np.percentile(zscore_timeseries, 1)
    vmax = np.percentile(zscore_timeseries, 99.8)
    
    # set anything in time series greather than vmax to vmax, and anything less than vmin to vmin, to avoid outliers dominating the color scale
    zscore_timeseries = np.clip(zscore_timeseries, vmin, vmax)
    # set anything greather than vmax to vmin
    # zscore_timeseries[zscore_timeseries > vmax] = vmin
    # zscore_timeseries[zscore_timeseries < vmin] = vmin
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot heatmap
    im = ax.pcolormesh(t, np.arange(n_grids), zscore_timeseries.T, cmap='plasma', 
                       vmin=vmin, vmax=vmax, shading='auto')
    # manual vmin and vmax to avoid outliers dominating the color scale
    # im = ax.pcolormesh(t, np.arange(n_grids), zscore_timeseries.T, cmap='plasma', 
    #                    vmin=0, vmax=2, shading='auto')
    
    # Add z-slice boundaries and labels
    unique_z = np.unique(sorted_z_labels)
    boundary_positions = []
    label_positions = []
    
    current_pos = 0
    for z in unique_z:
        count = np.sum(sorted_z_labels == z)
        boundary_positions.append(current_pos)
        label_positions.append(current_pos + count / 2)
        current_pos += count
    boundary_positions.append(current_pos)
    
    # Draw horizontal lines at z-slice boundaries
    for pos in boundary_positions[1:-1]:
        ax.axhline(pos, color='black', linewidth=0.5, alpha=0.7)
    
    # Add z-slice labels on right side
    ax2 = ax.secondary_yaxis('right')
    ax2.set_yticks(label_positions)
    ax2.set_yticklabels([f'z={z}' for z in unique_z], fontsize=8)
    ax2.set_ylabel('Z-slice')
    
    plt.colorbar(im, ax=ax, label='$F/F_{\\mathrm{baseline}}$', pad=0.12)
    
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Grid Index')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to {output_path}")
    
    plt.show()


def visualize_grid(normalized_data, mask_binned, grid_spacing_x, grid_spacing_y, 
                   time_idx=0, z_idx=20, output_path=None):
    """Visualize the grid overlay on data for a specific time point and z-slice."""
    height, width = normalized_data.shape[2], normalized_data.shape[3]
    
    grid_info, n_grids_y, n_grids_x = create_grid(height, width, grid_spacing_x, grid_spacing_y)
    
    # Get data for specific time and z
    data_slice = normalized_data[time_idx, z_idx, :, :]
    # baseline normalize
    baseline = normalized_data[:60, z_idx, :, :].mean(axis=0)
    data_slice = data_slice #/ (baseline + 1e-6)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot 1: Raw data with grid overlay
    ax1 = axes[0]
    im1 = ax1.imshow(data_slice, cmap='plasma', origin='upper')
    plt.colorbar(im1, ax=ax1, label='$F/F_{\\mathrm{baseline}}$')
    
    # Draw grid lines
    for gy in range(n_grids_y + 1):
        ax1.axhline(gy * grid_spacing_y, color='white', linewidth=0.5, alpha=0.7)
    for gx in range(n_grids_x + 1):
        ax1.axvline(gx * grid_spacing_x, color='white', linewidth=0.5, alpha=0.7)
    
    ax1.set_title(f'Data with Grid (t={time_idx}, z={z_idx})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # Plot 2: Masked data with grid
    ax2 = axes[1]
    masked_slice = data_slice * mask_binned
    im2 = ax2.imshow(masked_slice, cmap='plasma', origin='upper')
    plt.colorbar(im2, ax=ax2, label='$F/F_{\\mathrm{baseline}}$')
    
    for gy in range(n_grids_y + 1):
        ax2.axhline(gy * grid_spacing_y, color='white', linewidth=0.5, alpha=0.7)
    for gx in range(n_grids_x + 1):
        ax2.axvline(gx * grid_spacing_x, color='white', linewidth=0.5, alpha=0.7)
    
    ax2.set_title(f'Masked Data with Grid (t={time_idx}, z={z_idx})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    # Plot 3: Grid mean intensities
    ax3 = axes[2]
    grid_means = np.zeros((n_grids_y, n_grids_x))
    
    for grid_id, y_start, y_end, x_start, x_end in grid_info:
        gy = grid_id // n_grids_x
        gx = grid_id % n_grids_x
        
        grid_mask = mask_binned[y_start:y_end, x_start:x_end]
        grid_data = data_slice[y_start:y_end, x_start:x_end]
        
        if grid_mask.sum() > 0:
            grid_means[gy, gx] = (grid_data * grid_mask).sum() / grid_mask.sum()
        else:
            grid_means[gy, gx] = np.nan
    
    im3 = ax3.imshow(grid_means, cmap='plasma', origin='upper', aspect='equal')
    plt.colorbar(im3, ax=ax3, label='$F/F_{\\mathrm{baseline}}$')
    
    # Add grid labels
    for gy in range(n_grids_y):
        for gx in range(n_grids_x):
            if not np.isnan(grid_means[gy, gx]):
                ax3.text(gx, gy, f'{gy},{gx}', ha='center', va='center', 
                        fontsize=8, color='white', fontweight='bold')
    
    ax3.set_title(f'Grid Mean Intensities (t={time_idx}, z={z_idx})')
    ax3.set_xlabel('Grid X')
    ax3.set_ylabel('Grid Y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved grid visualization to {output_path}")
    
    plt.show()


def main():
    """Main function for grid-based intensity analysis."""
    
    if len(sys.argv) < 2:
        print("Usage: python grid_analysis.py <input_dir> [grid_spacing_x] [grid_spacing_y] [fps] [bin_factor] [z_start] [z_end]")
        print("  input_dir: directory containing normalized_voxels.npy and fixed_mask_*.tif")
        print("  grid_spacing_x: grid size in x direction (default: 20)")
        print("  grid_spacing_y: grid size in y direction (default: 20)")
        print("  fps: frames per second for time axis (default: 1.877)")
        print("  bin_factor: binning factor for mask (default: 1)")
        print("  z_start: starting z-slice index, inclusive (default: 0)")
        print("  z_end: ending z-slice index, inclusive (default: last slice)")
        print("  start_time_sec: start time in seconds, data before this is discarded (default: 0.0)")
        print("  food_time: time of food delivery in seconds, used as t=0 reference (default: None, uses start_time_sec)")
        sys.exit(1)
    
    # Parse arguments
    input_dir = sys.argv[1]
    grid_spacing_x = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    grid_spacing_y = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    fps = float(sys.argv[4]) if len(sys.argv) > 4 else 1.877  # default ~1/0.533
    bin_factor = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    z_start = int(sys.argv[6]) if len(sys.argv) > 6 else None
    z_end = int(sys.argv[7]) if len(sys.argv) > 7 else None
    baseline_start_sec = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0
    baseline_end_sec = float(sys.argv[9]) if len(sys.argv) > 9 else 30.0
    start_time_sec = float(sys.argv[10]) if len(sys.argv) > 10 else 0.0
    time_food = float(sys.argv[11]) if len(sys.argv) > 11 else None
    
    print(f"Input directory: {input_dir}")
    print(f"Grid spacing: ({grid_spacing_x}, {grid_spacing_y})")
    print(f"FPS: {fps}")
    print(f"Binning factor: {bin_factor}")
    print(f"Z slice range: {z_start} to {z_end}")
    print(f"Baseline window: {baseline_start_sec} to {baseline_end_sec} sec")
    print(f"Start time: {start_time_sec} sec")
    print(f"Food time: {time_food} sec")
    
    # Load data
    normalized_data, mask = load_data(input_dir)
    
    # Apply z-slice filtering if specified
    n_z_total = normalized_data.shape[1]
    if z_start is None:
        z_start = 0
    if z_end is None:
        z_end = n_z_total - 1
    
    # Validate z range
    z_start = max(0, z_start)
    z_end = min(n_z_total - 1, z_end)
    
    if z_start > z_end:
        raise ValueError(f"z_start ({z_start}) must be <= z_end ({z_end})")
    
    # Slice data to selected z range (inclusive)
    normalized_data = normalized_data[:, z_start:z_end + 1, :, :]
    print(f"Using z-slices {z_start} to {z_end} (inclusive), {normalized_data.shape[1]} slices")
    
    # Store z_offset for labeling
    z_offset = z_start
    
    # Calculate grid intensities on FULL time series first (before time trimming)
    # This allows us to compute baseline from the correct time window
    # Note: No masking is applied to the data, all grids are computed
    grid_timeseries_full, grid_info, n_grids_y, n_grids_x, mask_binned, grid_mask_coverage, grid_in_mask = \
        calculate_grid_intensities(normalized_data, mask, grid_spacing_x, grid_spacing_y, bin_factor)
    
    print(f"Grids in mask: {grid_in_mask.sum()} / {grid_in_mask.size}")
    
    # Calculate baseline from specified time window (relative to recording start)
    n_time_total = grid_timeseries_full.shape[0]
    baseline_start_frame = int(baseline_start_sec * fps)
    baseline_end_frame = int(baseline_end_sec * fps)
    baseline_start_frame = max(0, min(baseline_start_frame, n_time_total - 1))
    baseline_end_frame = max(0, min(baseline_end_frame, n_time_total - 1))
    
    if baseline_start_frame >= baseline_end_frame:
        raise ValueError(f"baseline_start_frame ({baseline_start_frame}) must be < baseline_end_frame ({baseline_end_frame})")
    
    print(f"Computing baseline from frames {baseline_start_frame} to {baseline_end_frame} ({baseline_end_frame - baseline_start_frame} frames)")
    
    # Compute baseline: mean intensity in baseline window for each grid
    # Shape: (Z, n_grids_y, n_grids_x)
    grid_baseline = grid_timeseries_full[baseline_start_frame:baseline_end_frame, :, :, :].mean(axis=0)
    
    # Now apply time filtering based on start_time_sec
    start_frame = int(start_time_sec * fps)
    start_frame = max(0, min(start_frame, n_time_total - 1))
    
    grid_timeseries = grid_timeseries_full[start_frame:, :, :, :]
    print(f"Using frames from {start_frame} onwards ({grid_timeseries.shape[0]} frames remaining)")
    
    # Calculate time offset for plotting (relative to food_time)
    # If food_time is None, use start_time_sec as the reference (t=0)
    if time_food is None:
        time_offset = start_time_sec
    else:
        time_offset = time_food
    
    # The actual start time of the data after filtering
    data_start_time = start_time_sec
    
    # Flatten with z-labels (include all grids, min_coverage=0.0)
    flat_timeseries, z_labels, grid_labels, flat_baseline, flat_in_mask, unflatten_info = flatten_grid_timeseries(
        grid_timeseries, grid_mask_coverage, grid_baseline, grid_in_mask, min_coverage=0.0
    )
    
    # Adjust z_labels to reflect original z-slice indices
    z_labels = z_labels + z_offset
    
    # Update unflatten_info with z_offset for proper reconstruction
    unflatten_info['z_offset'] = z_offset
    
    print(f"\nFinal time series shape: {flat_timeseries.shape}")
    print(f"Number of grids: {len(z_labels)}")
    print(f"Grids in mask: {flat_in_mask.sum()} / {len(flat_in_mask)}")
    
    # Save grid timeseries
    output_file = os.path.join(input_dir, 'grid_timeseries.npy')
    np.save(output_file, grid_timeseries)
    print(f"Saved grid timeseries to {output_file}")
    
    # Save flat timeseries with z-labels, baseline info, mask info, and unflatten info
    np.savez(os.path.join(input_dir, 'grid_timeseries_flat.npz'),
             timeseries=flat_timeseries,
             z_labels=z_labels,
             grid_labels=np.array(grid_labels),
             z_range=(z_start, z_end),
             baseline=flat_baseline,
             baseline_window_sec=(baseline_start_sec, baseline_end_sec),
             in_mask=flat_in_mask,
             flat_to_grid_idx=unflatten_info['flat_to_grid_idx'],
             n_z=unflatten_info['n_z'],
             n_grids_y=unflatten_info['n_grids_y'],
             n_grids_x=unflatten_info['n_grids_x'],
             z_offset=z_offset)
    print(f"Saved flat grid timeseries to {os.path.join(input_dir, 'grid_timeseries_flat.npz')}")
    
    # Apply causal smoothing for visualization
    print("\nApplying causal Gaussian smoothing for visualization...")
    flat_timeseries_smoothed = causal_smooth(flat_timeseries, sigma=2.0)
    print(f"Smoothed time series shape: {flat_timeseries_smoothed.shape}")
    
    # Save smoothed version as well
    np.savez(os.path.join(input_dir, 'grid_timeseries_flat_smoothed.npz'),
             timeseries=flat_timeseries_smoothed,
             z_labels=z_labels,
             grid_labels=np.array(grid_labels),
             z_range=(z_start, z_end),
             baseline=flat_baseline,
             baseline_window_sec=(baseline_start_sec, baseline_end_sec),
             in_mask=flat_in_mask,
             flat_to_grid_idx=unflatten_info['flat_to_grid_idx'],
             n_z=unflatten_info['n_z'],
             n_grids_y=unflatten_info['n_grids_y'],
             n_grids_x=unflatten_info['n_grids_x'],
             z_offset=z_offset)
    print(f"Saved smoothed flat grid timeseries to {os.path.join(input_dir, 'grid_timeseries_flat_smoothed.npz')}")
    
    # Visualize grid for example time point and z-slice
    example_time = 830  
    example_z = 5  # relative to sliced data
    example_z_original = example_z + z_offset  # original z-index for display
    visualize_grid(normalized_data, mask_binned, grid_spacing_x, grid_spacing_y,
                   time_idx=example_time, z_idx=example_z,
                   output_path=os.path.join(input_dir, 'grid_visualization.png'))
    
    # Plot time series (using smoothed data)
    plot_timeseries(flat_timeseries_smoothed, z_labels, fps=fps,
                    output_path=os.path.join(input_dir, 'grid_timeseries_plot.png'),
                    data_start_time=data_start_time, time_offset=time_offset,
                    baseline=flat_baseline)
    
    # Plot heatmap (using smoothed data)
    plot_heatmap(flat_timeseries_smoothed, z_labels, fps=fps,
                 output_path=os.path.join(input_dir, 'grid_heatmap.png'),
                 data_start_time=data_start_time, time_offset=time_offset,
                 baseline=flat_baseline)
    


if __name__ == "__main__":
    main()
