"""
Dynamic Mode Decomposition (DMD) analysis for fluorescent intensity volume time series.

This script extracts spatiotemporal dynamics from normalized voxel data using DMD,
with options for noise filtering and causal smoothing.

Usage:
    python dmd_analysis.py <input_dir> [options]

Arguments:
    input_dir: Directory containing normalized_voxels.npy and fixed_mask_*.tif
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, sosfiltfilt
import tifffile
import glob
import os
import sys
from tqdm import tqdm

import matplotlib
font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 15}
matplotlib.rc('font', **font)


def load_data(input_dir):
    """Load normalized voxel data and mask."""
    voxels_path = os.path.join(input_dir, 'normalized_voxels.npy')
    if not os.path.exists(voxels_path):
        raise FileNotFoundError(f"Normalized voxels not found at {voxels_path}")
    
    normalized_data = np.load(voxels_path)
    print(f"Loaded normalized data with shape: {normalized_data.shape}")
    
    # Load mask
    fixed_mask_files = glob.glob(os.path.join(input_dir, 'fixed_mask_*.tif'))
    if not fixed_mask_files:
        raise FileNotFoundError(f"No fixed_mask_*.tif found in {input_dir}")
    
    mask = tifffile.imread(fixed_mask_files[0])
    print(f"Loaded mask with shape: {mask.shape}")
    
    return normalized_data, mask


def lowpass_filter(data, cutoff_freq, fs, order=4):
    """
    Apply a lowpass Butterworth filter to remove high-frequency noise.
    
    Args:
        data: (T, n_features) array - time series data
        cutoff_freq: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Filtered data with same shape as input
    """
    nyquist = fs / 2
    normalized_cutoff = cutoff_freq / nyquist
    
    # Clamp to valid range
    normalized_cutoff = min(normalized_cutoff, 0.99)
    
    sos = butter(order, normalized_cutoff, btype='low', output='sos')
    
    # Apply filter along time axis
    filtered_data = sosfiltfilt(sos, data, axis=0)
    
    return filtered_data


def bandpass_filter(data, low_freq, high_freq, fs, order=4):
    """
    Apply a bandpass Butterworth filter.
    
    Args:
        data: (T, n_features) array
        low_freq: Low cutoff frequency in Hz
        high_freq: High cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Filtered data
    """
    nyquist = fs / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Clamp to valid range
    low = max(low, 0.01)
    high = min(high, 0.99)
    
    sos = butter(order, [low, high], btype='band', output='sos')
    filtered_data = sosfiltfilt(sos, data, axis=0)
    
    return filtered_data


def causal_smooth(data, sigma=2.0, truncate=4.0):
    """
    Apply causal Gaussian smoothing to a (T, features) array.
    Only uses past and current time points, never future data.
    
    Args:
        data: (T, features) numpy array
        sigma: standard deviation of the Gaussian kernel (in samples/frames)
        truncate: truncate the filter at this many standard deviations
        
    Returns:
        smoothed_data: (T, features) array with causal Gaussian smoothing applied
    """
    T, n_features = data.shape
    smoothed = np.zeros_like(data)
    
    kernel_radius = int(truncate * sigma)
    lags = np.arange(kernel_radius + 1)
    kernel = np.exp(-0.5 * (lags / sigma) ** 2)
    kernel = kernel / kernel.sum()
    
    for t in range(T):
        available_lags = min(t + 1, len(kernel))
        k = kernel[:available_lags]
        k_normalized = k / k.sum()
        past_data = data[t - available_lags + 1:t + 1, :]
        smoothed[t] = (k_normalized[:, np.newaxis] * past_data[::-1, :]).sum(axis=0)
    
    return smoothed


def prepare_data_for_dmd(normalized_data, mask, bin_factor=2, z_start=None, z_end=None,
                         min_mask_coverage=0.1):
    """
    Prepare volumetric data for DMD by flattening spatial dimensions.
    
    Args:
        normalized_data: (T, Z, H, W) array
        mask: (H, W) array
        bin_factor: Spatial binning factor to reduce dimensionality
        z_start, z_end: Z-slice range to use (inclusive)
        min_mask_coverage: Minimum mask coverage to include a voxel
        
    Returns:
        data_matrix: (T, n_voxels) array - voxels within mask
        voxel_coords: (n_voxels, 3) array - (z, y, x) coordinates
        mask_binned: Binned mask
        spatial_shape: (n_z, h_binned, w_binned)
    """
    n_time, n_z, height, width = normalized_data.shape
    
    # Apply z-slice filtering
    if z_start is None:
        z_start = 0
    if z_end is None:
        z_end = n_z - 1
    
    z_start = max(0, z_start)
    z_end = min(n_z - 1, z_end)
    
    normalized_data = normalized_data[:, z_start:z_end + 1, :, :]
    n_z = normalized_data.shape[1]
    
    print(f"Using z-slices {z_start} to {z_end}, {n_z} slices total")
    
    # Bin spatially if needed
    h_binned = height // bin_factor
    w_binned = width // bin_factor
    
    # Bin mask
    h_mask, w_mask = mask.shape
    if h_mask != height or w_mask != width:
        # Mask has different size, need to resize
        bin_factor_h = h_mask // h_binned
        bin_factor_w = w_mask // w_binned
        mask_binned = mask[:h_binned * bin_factor_h, :w_binned * bin_factor_w].reshape(
            h_binned, bin_factor_h, w_binned, bin_factor_w
        ).max(axis=(1, 3))
    else:
        if bin_factor > 1:
            mask_binned = mask[:h_binned * bin_factor, :w_binned * bin_factor].reshape(
                h_binned, bin_factor, w_binned, bin_factor
            ).max(axis=(1, 3))
        else:
            mask_binned = mask
    
    # Bin data
    if bin_factor > 1:
        data_binned = normalized_data[:, :, :h_binned * bin_factor, :w_binned * bin_factor].reshape(
            n_time, n_z, h_binned, bin_factor, w_binned, bin_factor
        ).mean(axis=(3, 5))
    else:
        data_binned = normalized_data
    
    print(f"Binned data shape: {data_binned.shape}")
    print(f"Binned mask shape: {mask_binned.shape}")
    
    # Create voxel coordinates and data matrix for voxels within mask
    voxel_list = []
    coord_list = []
    
    for z in range(n_z):
        for y in range(h_binned):
            for x in range(w_binned):
                if mask_binned[y, x] >= min_mask_coverage:
                    voxel_list.append(data_binned[:, z, y, x])
                    coord_list.append((z + z_start, y, x))  # Store original z-index
    
    data_matrix = np.array(voxel_list).T  # (T, n_voxels)
    voxel_coords = np.array(coord_list)  # (n_voxels, 3)
    
    print(f"Data matrix shape: {data_matrix.shape}")
    print(f"Number of voxels within mask: {len(voxel_coords)}")
    
    spatial_shape = (n_z, h_binned, w_binned)
    
    return data_matrix, voxel_coords, mask_binned, spatial_shape, z_start


def compute_dmd(X, rank=None, exact=True):
    """
    Compute Dynamic Mode Decomposition.
    
    Args:
        X: (T, n_features) data matrix where each row is a time snapshot
        rank: Truncation rank for SVD (None for automatic selection)
        exact: Use exact DMD formulation
        
    Returns:
        modes: (n_features, rank) - DMD modes (spatial patterns)
        dynamics: (T, rank) - Time evolution of each mode
        eigenvalues: (rank,) - DMD eigenvalues
        freqs: (rank,) - Frequencies (in cycles per frame)
        growth_rates: (rank,) - Growth rates
    """
    # DMD works with X = [x0, x1, ..., x_{T-2}] and X' = [x1, x2, ..., x_{T-1}]
    X1 = X[:-1, :].T  # (n_features, T-1)
    X2 = X[1:, :].T   # (n_features, T-1)
    
    # SVD of X1
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    
    # Determine rank
    if rank is None:
        # Use optimal hard threshold for singular value truncation
        beta = min(X1.shape) / max(X1.shape)
        omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.43 + 1.82 * beta
        tau = omega * np.median(S)
        rank = np.sum(S > tau)
        rank = max(rank, 1)  # At least 1 mode
        print(f"Automatically selected rank: {rank}")
    
    rank = min(rank, len(S), X1.shape[1])
    
    # Truncate
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    
    # Build low-rank A matrix: A_tilde = U_r^H @ X2 @ V_r @ S_r^{-1}
    A_tilde = U_r.conj().T @ X2 @ Vh_r.conj().T @ np.diag(1.0 / S_r)
    
    # Eigendecomposition of A_tilde
    eigenvalues, W = np.linalg.eig(A_tilde)
    
    # Compute DMD modes
    if exact:
        # Exact DMD modes
        modes = X2 @ Vh_r.conj().T @ np.diag(1.0 / S_r) @ W
    else:
        # Projected DMD modes
        modes = U_r @ W
    
    # Normalize modes
    mode_norms = np.linalg.norm(modes, axis=0)
    modes = modes / (mode_norms + 1e-10)
    
    # Compute time dynamics
    T = X.shape[0]
    dynamics = np.zeros((T, rank), dtype=complex)
    
    # Initial condition
    b = np.linalg.lstsq(modes, X[0, :], rcond=None)[0]
    
    for t in range(T):
        dynamics[t, :] = b * (eigenvalues ** t)
    
    # Compute frequencies and growth rates
    dt = 1.0  # 1 frame
    log_eigs = np.log(eigenvalues + 1e-10)
    freqs = np.imag(log_eigs) / (2 * np.pi * dt)  # cycles per frame
    growth_rates = np.real(log_eigs) / dt
    
    return modes, dynamics, eigenvalues, freqs, growth_rates


def sort_modes_by_energy(modes, dynamics):
    """
    Sort DMD modes by their contribution to the overall signal energy.
    
    Args:
        modes: (n_features, n_modes)
        dynamics: (T, n_modes)
        
    Returns:
        sort_idx: Indices to sort modes by decreasing energy
        energy: Energy contribution of each mode
    """
    # Energy is proportional to |mode|^2 * |dynamics|^2 integrated over time
    mode_energy = np.sum(np.abs(modes)**2, axis=0)
    dynamics_energy = np.sum(np.abs(dynamics)**2, axis=0)
    total_energy = mode_energy * dynamics_energy
    
    sort_idx = np.argsort(total_energy)[::-1]
    
    return sort_idx, total_energy


def reconstruct_spatial_mode(mode, voxel_coords, spatial_shape, mask_binned):
    """
    Reconstruct a DMD mode in spatial coordinates.
    
    Args:
        mode: (n_voxels,) - mode values for each voxel
        voxel_coords: (n_voxels, 3) - (z, y, x) coordinates
        spatial_shape: (n_z, h, w)
        mask_binned: (h, w) mask
        
    Returns:
        spatial_mode: (n_z, h, w) - mode in spatial format
    """
    n_z, h, w = spatial_shape
    # Use complex dtype to preserve phase information
    spatial_mode = np.zeros((n_z, h, w), dtype=complex)
    
    z_offset = voxel_coords[:, 0].min()
    
    for i, (z, y, x) in enumerate(voxel_coords):
        z_idx = z - z_offset
        if 0 <= z_idx < n_z:
            spatial_mode[z_idx, y, x] = mode[i]
    
    return spatial_mode


def plot_top_modes(modes, dynamics, eigenvalues, freqs, growth_rates,
                   voxel_coords, spatial_shape, mask_binned,
                   fps=1.0, n_modes=6, output_dir=None, 
                   data_start_time=0.0, time_offset=0.0):
    """
    Visualize top DMD modes.
    
    Args:
        modes: (n_features, n_modes) DMD modes
        dynamics: (T, n_modes) time evolution
        eigenvalues: DMD eigenvalues
        freqs: Mode frequencies
        growth_rates: Mode growth rates
        voxel_coords: Spatial coordinates of voxels
        spatial_shape: (n_z, h, w)
        mask_binned: 2D mask
        fps: Frames per second
        n_modes: Number of top modes to visualize
        output_dir: Directory to save figures
        data_start_time: Actual time of first frame
        time_offset: Reference time for t=0
    """
    # Sort modes by energy
    sort_idx, energy = sort_modes_by_energy(modes, dynamics)
    
    T = dynamics.shape[0]
    t = (np.arange(T) / fps + data_start_time) - time_offset
    
    n_modes = min(n_modes, modes.shape[1])
    
    # Figure 1: Mode overview - eigenvalues, frequencies, growth rates
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Eigenvalue plot (unit circle)
    ax1 = axes[0]
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Unit circle')
    scatter = ax1.scatter(np.real(eigenvalues), np.imag(eigenvalues),
                          c=np.arange(len(eigenvalues)), cmap='viridis', s=50, alpha=0.7)
    ax1.set_xlabel('Real')
    ax1.set_ylabel('Imaginary')
    ax1.set_title('DMD Eigenvalues')
    ax1.axis('equal')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Mode index')
    
    # Frequency spectrum
    ax2 = axes[1]
    freqs_hz = freqs[sort_idx[:n_modes*2]] * fps  # Convert to Hz
    energy_sorted = energy[sort_idx[:n_modes*2]]
    ax2.bar(np.arange(len(freqs_hz)), np.abs(freqs_hz), color='steelblue', alpha=0.7)
    ax2.set_xlabel('Mode Index (sorted by energy)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('Mode Frequencies')
    
    # Growth rates
    ax3 = axes[2]
    growth_sorted = growth_rates[sort_idx[:n_modes*2]]
    colors = ['green' if g > 0 else 'red' for g in growth_sorted]
    ax3.bar(np.arange(len(growth_sorted)), growth_sorted, color=colors, alpha=0.7)
    ax3.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Mode Index (sorted by energy)')
    ax3.set_ylabel('Growth Rate')
    ax3.set_title('Mode Growth Rates')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'dmd_eigenvalues.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Figure 2: Top modes - spatial patterns
    n_z = spatial_shape[0]
    z_slices_to_show = [n_z // 4, n_z // 2, 3 * n_z // 4]  # Show 3 z-slices
    
    fig, axes = plt.subplots(n_modes, len(z_slices_to_show) + 1, 
                              figsize=(4 * (len(z_slices_to_show) + 1), 3 * n_modes))
    
    if n_modes == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_modes):
        mode_idx = sort_idx[i]
        mode = modes[:, mode_idx]
        
        # Reconstruct spatial mode
        spatial_mode = reconstruct_spatial_mode(mode, voxel_coords, spatial_shape, mask_binned)
        
        # Plot z-slices
        for j, z in enumerate(z_slices_to_show):
            ax = axes[i, j]
            mode_slice = np.real(spatial_mode[z, :, :])
            vmax = np.percentile(np.abs(mode_slice), 95)
            im = ax.imshow(mode_slice, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='upper')
            ax.set_title(f'Mode {i+1}, z={z}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Plot time dynamics
        ax = axes[i, -1]
        dyn = dynamics[:, mode_idx]
        ax.plot(t, np.real(dyn), 'b-', label='Real', alpha=0.8)
        ax.plot(t, np.abs(dyn), 'k--', label='Magnitude', alpha=0.6)
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Amplitude')
        freq_hz = freqs[mode_idx] * fps
        ax.set_title(f'Mode {i+1}: f={freq_hz:.4f} Hz')
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'dmd_top_modes_spatial.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Figure 3: Time dynamics comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i in range(min(n_modes, 5)):
        mode_idx = sort_idx[i]
        dyn = np.abs(dynamics[:, mode_idx])
        dyn_normalized = dyn / (dyn.max() + 1e-10)
        ax.plot(t, dyn_normalized + i * 1.2, label=f'Mode {i+1}', lw=1.5)
    
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Normalized Amplitude (offset)')
    ax.set_title('Top DMD Mode Dynamics')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'dmd_mode_dynamics.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_mode_projections(data_matrix, modes, dynamics, sort_idx, fps=1.0,
                          output_dir=None, data_start_time=0.0, time_offset=0.0):
    """
    Plot projection of original data onto top modes and reconstruction error.
    """
    T = data_matrix.shape[0]
    t = (np.arange(T) / fps + data_start_time) - time_offset
    
    # Reconstruction with different numbers of modes
    n_modes_list = [1, 3, 5, 10, 20]
    n_modes_list = [n for n in n_modes_list if n <= modes.shape[1]]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Original mean signal
    mean_signal = data_matrix.mean(axis=1)
    axes[0].plot(t, mean_signal, 'k-', label='Original', lw=2, alpha=0.8)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(n_modes_list)))
    
    errors = []
    for n_modes, color in zip(n_modes_list, colors):
        idx = sort_idx[:n_modes]
        reconstruction = np.real(dynamics[:, idx] @ modes[:, idx].T)
        recon_mean = reconstruction.mean(axis=1)
        axes[0].plot(t, recon_mean, color=color, label=f'{n_modes} modes', alpha=0.7, lw=1.5)
        
        # Compute reconstruction error
        error = np.sqrt(np.mean((data_matrix - reconstruction) ** 2))
        errors.append(error)
    
    axes[0].set_xlabel('Time (sec)')
    axes[0].set_ylabel('Mean Intensity')
    axes[0].set_title('Signal Reconstruction with Different Numbers of Modes')
    axes[0].legend()
    
    # Reconstruction error vs number of modes
    axes[1].plot(n_modes_list, errors, 'o-', lw=2, markersize=8)
    axes[1].set_xlabel('Number of Modes')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('Reconstruction Error vs Number of Modes')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'dmd_reconstruction.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_spatiotemporal_heatmap(dynamics, sort_idx, fps=1.0, n_modes=20,
                                 output_dir=None, data_start_time=0.0, time_offset=0.0):
    """
    Plot heatmap of mode dynamics over time.
    """
    T = dynamics.shape[0]
    t = (np.arange(T) / fps + data_start_time) - time_offset
    
    n_modes = min(n_modes, dynamics.shape[1])
    
    # Get top modes by energy
    dyn_sorted = np.abs(dynamics[:, sort_idx[:n_modes]])
    
    # Normalize each mode
    dyn_normalized = dyn_sorted / (dyn_sorted.max(axis=0, keepdims=True) + 1e-10)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    im = ax.pcolormesh(t, np.arange(n_modes), dyn_normalized.T, 
                       cmap='plasma', shading='auto')
    plt.colorbar(im, ax=ax, label='Normalized Amplitude')
    
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Mode Index (sorted by energy)')
    ax.set_title('DMD Mode Dynamics Heatmap')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'dmd_dynamics_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main function for DMD analysis."""
    
    if len(sys.argv) < 2:
        print("Usage: python dmd_analysis.py <input_dir> [options]")
        print("  input_dir: directory containing normalized_voxels.npy and fixed_mask_*.tif")
        print("  fps: frames per second (default: 1.877)")
        print("  bin_factor: spatial binning factor (default: 2)")
        print("  z_start: starting z-slice (default: 0)")
        print("  z_end: ending z-slice (default: last)")
        print("  lowpass_freq: lowpass filter cutoff in Hz (default: 0.1)")
        print("  dmd_rank: number of DMD modes to compute (default: auto)")
        print("  n_modes_plot: number of top modes to visualize (default: 6)")
        print("  baseline_start_sec: start of baseline window (default: 0)")
        print("  baseline_end_sec: end of baseline window (default: 30)")
        print("  start_time_sec: data start time in sec (default: 0)")
        print("  time_offset_sec: reference time for t=0 (default: 0)")
        sys.exit(1)
    
    # Parse arguments
    input_dir = sys.argv[1]
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 1.877
    bin_factor = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    z_start = int(sys.argv[4]) if len(sys.argv) > 4 else None
    z_end = int(sys.argv[5]) if len(sys.argv) > 5 else None
    lowpass_freq = float(sys.argv[6]) if len(sys.argv) > 6 else 0.1
    dmd_rank = int(sys.argv[7]) if len(sys.argv) > 7 else None
    n_modes_plot = int(sys.argv[8]) if len(sys.argv) > 8 else 6
    baseline_start_sec = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0
    baseline_end_sec = float(sys.argv[10]) if len(sys.argv) > 10 else 30.0
    start_time_sec = float(sys.argv[11]) if len(sys.argv) > 11 else 0.0
    time_offset_sec = float(sys.argv[12]) if len(sys.argv) > 12 else 0.0
    
    print(f"\n{'='*60}")
    print("DMD Analysis Parameters")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"FPS: {fps}")
    print(f"Spatial binning: {bin_factor}")
    print(f"Z-slice range: {z_start} to {z_end}")
    print(f"Lowpass filter cutoff: {lowpass_freq} Hz")
    print(f"DMD rank: {'auto' if dmd_rank is None else dmd_rank}")
    print(f"Modes to visualize: {n_modes_plot}")
    print(f"Baseline window: {baseline_start_sec} to {baseline_end_sec} sec")
    print(f"Start time: {start_time_sec} sec")
    print(f"Time offset: {time_offset_sec} sec")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    normalized_data, mask = load_data(input_dir)
    
    # Prepare data for DMD
    print("\nPreparing data for DMD...")
    data_matrix, voxel_coords, mask_binned, spatial_shape, z_offset = prepare_data_for_dmd(
        normalized_data, mask, bin_factor=bin_factor, z_start=z_start, z_end=z_end
    )
    
    # Compute baseline
    baseline_start_frame = int(baseline_start_sec * fps)
    baseline_end_frame = int(baseline_end_sec * fps)
    baseline_start_frame = max(0, min(baseline_start_frame, data_matrix.shape[0] - 1))
    baseline_end_frame = max(baseline_start_frame + 1, min(baseline_end_frame, data_matrix.shape[0]))
    
    print(f"\nComputing baseline from frames {baseline_start_frame} to {baseline_end_frame}")
    baseline = data_matrix[baseline_start_frame:baseline_end_frame, :].mean(axis=0)
    
    # F/F0 normalization
    data_normalized = data_matrix #/ (baseline + 1e-6)
    
    # Apply lowpass filter to remove high-frequency noise
    if lowpass_freq is not None and lowpass_freq > 0:
        print(f"\nApplying lowpass filter (cutoff: {lowpass_freq} Hz)...")
        data_filtered = lowpass_filter(data_normalized, lowpass_freq, fps, order=4)
    else:
        data_filtered = data_normalized
    
    # Apply causal smoothing for visualization
    print("\nApplying causal smoothing for visualization...")
    data_smoothed = causal_smooth(data_filtered, sigma=2.0)
    
    # Trim data based on start_time
    start_frame = int(start_time_sec * fps)
    start_frame = max(0, min(start_frame, data_filtered.shape[0] - 1))
    data_for_dmd = data_filtered[start_frame:, :]
    data_smoothed = data_smoothed[start_frame:, :]
    
    print(f"\nUsing frames from {start_frame} onwards ({data_for_dmd.shape[0]} frames)")
    
    # Mean-center data for DMD
    data_mean = data_for_dmd.mean(axis=0)
    data_centered = data_for_dmd - data_mean
    
    # Compute DMD
    print("\nComputing DMD...")
    modes, dynamics, eigenvalues, freqs, growth_rates = compute_dmd(
        data_centered, rank=dmd_rank, exact=True
    )
    
    print(f"\nDMD Results:")
    print(f"  Number of modes: {modes.shape[1]}")
    print(f"  Mode shape: {modes.shape}")
    print(f"  Dynamics shape: {dynamics.shape}")
    
    # Sort modes by energy
    sort_idx, energy = sort_modes_by_energy(modes, dynamics)
    
    # Save results
    print("\nSaving results...")
    np.savez(os.path.join(input_dir, 'dmd_results.npz'),
             modes=modes,
             dynamics=dynamics,
             eigenvalues=eigenvalues,
             frequencies=freqs,
             growth_rates=growth_rates,
             sort_idx=sort_idx,
             energy=energy,
             voxel_coords=voxel_coords,
             spatial_shape=spatial_shape,
             mask_binned=mask_binned,
             data_mean=data_mean,
             fps=fps,
             z_offset=z_offset)
    print(f"Saved DMD results to {os.path.join(input_dir, 'dmd_results.npz')}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Actual start time of the trimmed data
    data_start_time = start_time_sec
    
    # Plot top modes
    plot_top_modes(modes, dynamics, eigenvalues, freqs, growth_rates,
                   voxel_coords, spatial_shape, mask_binned,
                   fps=fps, n_modes=n_modes_plot, output_dir=input_dir,
                   data_start_time=data_start_time, time_offset=time_offset_sec)
    
    # Plot reconstruction
    plot_mode_projections(data_centered, modes, dynamics, sort_idx,
                          fps=fps, output_dir=input_dir,
                          data_start_time=data_start_time, time_offset=time_offset_sec)
    
    # Plot dynamics heatmap
    plot_spatiotemporal_heatmap(dynamics, sort_idx, fps=fps, n_modes=20,
                                 output_dir=input_dir,
                                 data_start_time=data_start_time, time_offset=time_offset_sec)
    
    print("\nDMD analysis complete!")
    print(f"Results saved to {input_dir}")


if __name__ == "__main__":
    main()
