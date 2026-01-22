"""
Principal Component Analysis (PCA) for fluorescent intensity volume time series.

This script extracts spatiotemporal dynamics from normalized voxel data using PCA,
with options for noise filtering and causal smoothing.

Usage:
    python pca_analysis.py <input_dir> [options]

Arguments:
    input_dir: Directory containing normalized_voxels.npy and fixed_mask_*.tif
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt
from sklearn.decomposition import PCA
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
    if fixed_mask_files is None or len(fixed_mask_files) == 0:
        fixed_mask_files = glob.glob(os.path.join(input_dir, 'fixed_mask*.tif'))
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


def prepare_data_for_pca(normalized_data, mask, bin_factor=2, z_start=None, z_end=None,
                         min_mask_coverage=0.1):
    """
    Prepare volumetric data for PCA by flattening spatial dimensions.
    
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


def compute_pca(X, n_components=None):
    """
    Compute Principal Component Analysis.
    
    Args:
        X: (T, n_features) data matrix where each row is a time snapshot
        n_components: Number of components to keep (None for all)
        
    Returns:
        components: (n_components, n_features) - PC loadings/weights (spatial patterns)
        scores: (T, n_components) - PC scores (time evolution)
        explained_variance: (n_components,) - Variance explained by each PC
        explained_variance_ratio: (n_components,) - Fraction of variance explained
        mean: (n_features,) - Mean of data (for reconstruction)
    """
    # Determine number of components
    if n_components is None:
        n_components = min(X.shape[0], X.shape[1])
    
    n_components = min(n_components, X.shape[0], X.shape[1])
    
    print(f"Computing PCA with {n_components} components...")
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)  # (T, n_components)
    
    components = pca.components_  # (n_components, n_features)
    explained_variance = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    mean = pca.mean_
    
    print(f"Variance explained by first 5 PCs: {explained_variance_ratio[:5]}")
    print(f"Cumulative variance (first 10): {np.cumsum(explained_variance_ratio[:10])}")
    
    return components, scores, explained_variance, explained_variance_ratio, mean


def reconstruct_spatial_weights(weights, voxel_coords, spatial_shape):
    """
    Reconstruct PC weights in spatial coordinates.
    
    Args:
        weights: (n_voxels,) - weight values for each voxel
        voxel_coords: (n_voxels, 3) - (z, y, x) coordinates
        spatial_shape: (n_z, h, w)
        
    Returns:
        spatial_weights: (n_z, h, w) - weights in spatial format
    """
    n_z, h, w = spatial_shape
    spatial_weights = np.zeros((n_z, h, w))
    
    z_offset = voxel_coords[:, 0].min()
    
    for i, (z, y, x) in enumerate(voxel_coords):
        z_idx = z - z_offset
        if 0 <= z_idx < n_z:
            spatial_weights[z_idx, y, x] = weights[i]
    
    return spatial_weights


def plot_variance_explained(explained_variance_ratio, output_dir=None):
    """Plot variance explained by each PC."""
    n_pcs = min(30, len(explained_variance_ratio))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual variance
    ax1 = axes[0]
    ax1.bar(np.arange(1, n_pcs + 1), explained_variance_ratio[:n_pcs] * 100, 
            color='steelblue', alpha=0.7)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained (%)')
    ax1.set_title('Variance Explained by Each PC')
    ax1.set_xticks(np.arange(1, n_pcs + 1, 2))
    
    # Cumulative variance
    ax2 = axes[1]
    cumvar = np.cumsum(explained_variance_ratio[:n_pcs]) * 100
    ax2.plot(np.arange(1, n_pcs + 1), cumvar, 'o-', color='darkgreen', lw=2, markersize=6)
    ax2.axhline(90, color='red', linestyle='--', alpha=0.7, label='90% threshold')
    ax2.axhline(95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
    ax2.set_xlabel('Number of Principal Components')
    ax2.set_ylabel('Cumulative Variance Explained (%)')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()
    ax2.set_xticks(np.arange(1, n_pcs + 1, 2))
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pca_variance_explained.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print how many PCs needed for 90% and 95% variance
    n_90 = np.argmax(np.cumsum(explained_variance_ratio) >= 0.90) + 1
    n_95 = np.argmax(np.cumsum(explained_variance_ratio) >= 0.95) + 1
    print(f"\nPCs needed for 90% variance: {n_90}")
    print(f"PCs needed for 95% variance: {n_95}")


def plot_top_pcs_spatial(components, voxel_coords, spatial_shape, mask_binned,
                          n_pcs=6, output_dir=None):
    """
    Visualize spatial weights of top PCs across z-slices.
    
    Args:
        components: (n_components, n_features) PC loadings
        voxel_coords: Spatial coordinates of voxels
        spatial_shape: (n_z, h, w)
        mask_binned: 2D mask
        n_pcs: Number of top PCs to visualize
        output_dir: Directory to save figures
    """
    n_z = spatial_shape[0]
    z_slices_to_show = [n_z // 4, n_z // 2, 3 * n_z // 4]
    
    n_pcs = min(n_pcs, components.shape[0])
    
    fig, axes = plt.subplots(n_pcs, len(z_slices_to_show), 
                              figsize=(4 * len(z_slices_to_show), 3 * n_pcs))
    
    if n_pcs == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_pcs):
        weights = components[i, :]
        
        # Reconstruct spatial weights
        spatial_weights = reconstruct_spatial_weights(weights, voxel_coords, spatial_shape)
        
        # Plot z-slices
        for j, z in enumerate(z_slices_to_show):
            ax = axes[i, j]
            weight_slice = spatial_weights[z, :, :]
            vmax = np.percentile(np.abs(weight_slice[weight_slice != 0]), 95) if np.any(weight_slice != 0) else 1
            im = ax.imshow(weight_slice, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='upper')
            ax.set_title(f'PC{i+1}, z={z}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('PC Spatial Weights (Loadings)', fontsize=16, y=1.02)
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pca_spatial_weights.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_pc_scores(scores, explained_variance_ratio, fps=1.0, n_pcs=6, 
                   output_dir=None, data_start_time=0.0, time_offset=0.0):
    """
    Plot PC scores (temporal dynamics) over time.
    
    Args:
        scores: (T, n_components) PC scores
        explained_variance_ratio: Variance explained by each PC
        fps: Frames per second
        n_pcs: Number of top PCs to plot
        output_dir: Directory to save figures
        data_start_time: Actual time of first frame
        time_offset: Reference time for t=0
    """
    T = scores.shape[0]
    t = (np.arange(T) / fps + data_start_time) - time_offset
    
    n_pcs = min(n_pcs, scores.shape[1])
    
    # Individual PC time courses
    fig, axes = plt.subplots(n_pcs, 1, figsize=(14, 2.5 * n_pcs), sharex=True)
    
    if n_pcs == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.arange(n_pcs))
    
    for i in range(n_pcs):
        ax = axes[i]
        ax.plot(t, scores[:, i], color=colors[i], lw=1.5)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        var_pct = explained_variance_ratio[i] * 100
        ax.set_ylabel(f'PC{i+1}\n({var_pct:.1f}%)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    axes[-1].set_xlabel('Time (sec)')
    plt.suptitle('PC Scores (Temporal Dynamics)', fontsize=16)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pca_scores_individual.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Overlay plot (normalized)
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i in range(min(n_pcs, 5)):
        score_norm = scores[:, i] / (np.abs(scores[:, i]).max() + 1e-10)
        ax.plot(t, score_norm + i * 2.5, label=f'PC{i+1} ({explained_variance_ratio[i]*100:.1f}%)', 
                lw=1.5, color=colors[i])
    
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Normalized Score (offset)')
    ax.set_title('Top PC Scores (Normalized, Offset)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pca_scores_overlay.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_scores_heatmap(scores, explained_variance_ratio, fps=1.0, n_pcs=20,
                         output_dir=None, data_start_time=0.0, time_offset=0.0):
    """
    Plot heatmap of PC scores over time.
    """
    T = scores.shape[0]
    t = (np.arange(T) / fps + data_start_time) - time_offset
    
    n_pcs = min(n_pcs, scores.shape[1])
    
    # Normalize each PC for visualization
    scores_subset = scores[:, :n_pcs]
    scores_normalized = scores_subset / (np.abs(scores_subset).max(axis=0, keepdims=True) + 1e-10)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    im = ax.pcolormesh(t, np.arange(n_pcs), scores_normalized.T, 
                       cmap='RdBu_r', shading='auto', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Normalized Score')
    
    # Add variance explained labels on y-axis
    yticks = np.arange(n_pcs)
    yticklabels = [f'PC{i+1} ({explained_variance_ratio[i]*100:.1f}%)' for i in range(n_pcs)]
    ax.set_yticks(yticks[::2])
    ax.set_yticklabels(yticklabels[::2], fontsize=9)
    
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Principal Component')
    ax.set_title('PC Scores Heatmap')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pca_scores_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_weight_distributions(components, n_pcs=6, output_dir=None):
    """
    Plot distributions of PC weights to understand sparsity and magnitude.
    """
    n_pcs = min(n_pcs, components.shape[0])
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for i in range(n_pcs):
        ax = axes[i]
        weights = components[i, :]
        
        ax.hist(weights, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax.axvline(weights.mean(), color='green', linestyle='-', alpha=0.7, label=f'Mean: {weights.mean():.4f}')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Count')
        ax.set_title(f'PC{i+1} Weight Distribution')
        ax.legend(fontsize=8)
    
    plt.suptitle('PC Weight Distributions', fontsize=16)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pca_weight_distributions.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_reconstruction(data_matrix, components, scores, mean, fps=1.0,
                        output_dir=None, data_start_time=0.0, time_offset=0.0):
    """
    Plot reconstruction with different numbers of PCs and reconstruction error.
    """
    T = data_matrix.shape[0]
    t = (np.arange(T) / fps + data_start_time) - time_offset
    
    # Reconstruction with different numbers of PCs
    n_pcs_list = [1, 3, 5, 10, 20, 50]
    n_pcs_list = [n for n in n_pcs_list if n <= components.shape[0]]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Original mean signal
    mean_signal = data_matrix.mean(axis=1)
    axes[0].plot(t, mean_signal, 'k-', label='Original', lw=2, alpha=0.8)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(n_pcs_list)))
    
    errors = []
    for n_pcs, color in zip(n_pcs_list, colors):
        # Reconstruction: X_approx = scores[:, :n_pcs] @ components[:n_pcs, :] + mean
        reconstruction = scores[:, :n_pcs] @ components[:n_pcs, :] + mean
        recon_mean = reconstruction.mean(axis=1)
        axes[0].plot(t, recon_mean, color=color, label=f'{n_pcs} PCs', alpha=0.7, lw=1.5)
        
        # Compute reconstruction error
        error = np.sqrt(np.mean((data_matrix - reconstruction) ** 2))
        errors.append(error)
    
    axes[0].set_xlabel('Time (sec)')
    axes[0].set_ylabel('Mean Intensity')
    axes[0].set_title('Signal Reconstruction with Different Numbers of PCs')
    axes[0].legend()
    
    # Reconstruction error vs number of PCs
    axes[1].plot(n_pcs_list, errors, 'o-', lw=2, markersize=8, color='darkgreen')
    axes[1].set_xlabel('Number of Principal Components')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('Reconstruction Error vs Number of PCs')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pca_reconstruction.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_pc_correlations(scores, n_pcs=10, output_dir=None):
    """
    Plot correlation matrix between PC scores.
    """
    n_pcs = min(n_pcs, scores.shape[1])
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(scores[:, :n_pcs].T)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Correlation')
    
    # Add labels
    ax.set_xticks(np.arange(n_pcs))
    ax.set_yticks(np.arange(n_pcs))
    ax.set_xticklabels([f'PC{i+1}' for i in range(n_pcs)])
    ax.set_yticklabels([f'PC{i+1}' for i in range(n_pcs)])
    
    # Add correlation values as text
    for i in range(n_pcs):
        for j in range(n_pcs):
            if i != j:
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha='center', va='center', fontsize=8,
                              color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
    
    ax.set_title('PC Score Correlations')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pca_correlations.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_combined_spatial_temporal(components, scores, explained_variance_ratio, 
                                    voxel_coords, spatial_shape, mask_binned,
                                    fps=1.0, n_pcs=4, output_dir=None,
                                    data_start_time=0.0, time_offset=0.0):
    """
    Combined plot showing spatial weights and temporal dynamics side by side.
    """
    T = scores.shape[0]
    t = (np.arange(T) / fps + data_start_time) - time_offset
    
    n_z = spatial_shape[0]
    z_mid = n_z // 2
    
    n_pcs = min(n_pcs, components.shape[0])
    
    fig, axes = plt.subplots(n_pcs, 2, figsize=(14, 3 * n_pcs))
    
    if n_pcs == 1:
        axes = axes.reshape(1, -1)
    
    colors = plt.cm.tab10(np.arange(n_pcs))
    
    for i in range(n_pcs):
        # Spatial weights (middle z-slice)
        ax_spatial = axes[i, 0]
        weights = components[i, :]
        spatial_weights = reconstruct_spatial_weights(weights, voxel_coords, spatial_shape)
        weight_slice = spatial_weights[z_mid, :, :]
        vmax = np.percentile(np.abs(weight_slice[weight_slice != 0]), 95) if np.any(weight_slice != 0) else 1
        im = ax_spatial.imshow(weight_slice, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='upper')
        var_pct = explained_variance_ratio[i] * 100
        ax_spatial.set_title(f'PC{i+1} Weights (z={z_mid}) - {var_pct:.1f}% var')
        ax_spatial.axis('off')
        plt.colorbar(im, ax=ax_spatial, fraction=0.046, pad=0.04)
        
        # Temporal dynamics
        ax_temporal = axes[i, 1]
        ax_temporal.plot(t, scores[:, i], color=colors[i], lw=1.5)
        ax_temporal.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax_temporal.set_ylabel('Score')
        ax_temporal.set_title(f'PC{i+1} Score Over Time')
        ax_temporal.spines['top'].set_visible(False)
        ax_temporal.spines['right'].set_visible(False)
        if i == n_pcs - 1:
            ax_temporal.set_xlabel('Time (sec)')
    
    plt.suptitle('PCA: Spatial Weights and Temporal Dynamics', fontsize=16, y=1.01)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'pca_combined.png'), dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main function for PCA analysis."""
    
    if len(sys.argv) < 2:
        print("Usage: python pca_analysis.py <input_dir> [options]")
        print("  input_dir: directory containing normalized_voxels.npy and fixed_mask_*.tif")
        print("  fps: frames per second (default: 1.877)")
        print("  bin_factor: spatial binning factor (default: 2)")
        print("  z_start: starting z-slice (default: 0)")
        print("  z_end: ending z-slice (default: last)")
        print("  lowpass_freq: lowpass filter cutoff in Hz (default: 0.1)")
        print("  n_components: number of PCA components (default: 50)")
        print("  n_pcs_plot: number of top PCs to visualize (default: 6)")
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
    n_components = int(sys.argv[7]) if len(sys.argv) > 7 else 50
    n_pcs_plot = int(sys.argv[8]) if len(sys.argv) > 8 else 6
    baseline_start_sec = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0
    baseline_end_sec = float(sys.argv[10]) if len(sys.argv) > 10 else 30.0
    start_time_sec = float(sys.argv[11]) if len(sys.argv) > 11 else 0.0
    time_offset_sec = float(sys.argv[12]) if len(sys.argv) > 12 else 0.0
    
    print(f"\n{'='*60}")
    print("PCA Analysis Parameters")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"FPS: {fps}")
    print(f"Spatial binning: {bin_factor}")
    print(f"Z-slice range: {z_start} to {z_end}")
    print(f"Lowpass filter cutoff: {lowpass_freq} Hz")
    print(f"Number of components: {n_components}")
    print(f"PCs to visualize: {n_pcs_plot}")
    print(f"Baseline window: {baseline_start_sec} to {baseline_end_sec} sec")
    print(f"Start time: {start_time_sec} sec")
    print(f"Time offset: {time_offset_sec} sec")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    normalized_data, mask = load_data(input_dir)
    
    # Prepare data for PCA
    print("\nPreparing data for PCA...")
    data_matrix, voxel_coords, mask_binned, spatial_shape, z_offset = prepare_data_for_pca(
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
    data_normalized = data_matrix# / (baseline + 1e-6)
    
    # Apply lowpass filter to remove high-frequency noise
    if lowpass_freq is not None and lowpass_freq > 0:
        print(f"\nApplying lowpass filter (cutoff: {lowpass_freq} Hz)...")
        data_filtered = lowpass_filter(data_normalized, lowpass_freq, fps, order=4)
    else:
        data_filtered = data_normalized
    
    # Apply causal smoothing for visualization
    print("\nApplying causal smoothing...")
    # data_smoothed = causal_smooth(data_filtered, sigma=0.1)
    data_smoothed = data_filtered
    
    # Trim data based on start_time
    start_frame = int(start_time_sec * fps)
    start_frame = max(0, min(start_frame, data_filtered.shape[0] - 1))
    data_for_pca = data_filtered[start_frame:, :]
    data_smoothed = data_smoothed[start_frame:, :]
    
    print(f"\nUsing frames from {start_frame} onwards ({data_for_pca.shape[0]} frames)")
    
    # Compute PCA
    print("\nComputing PCA...")
    components, scores, explained_variance, explained_variance_ratio, data_mean = compute_pca(
        data_for_pca, n_components=n_components
    )
    
    print(f"\nPCA Results:")
    print(f"  Number of components: {components.shape[0]}")
    print(f"  Components shape: {components.shape}")
    print(f"  Scores shape: {scores.shape}")
    
    # Save results
    print("\nSaving results...")
    np.savez(os.path.join(input_dir, 'pca_results.npz'),
             components=components,
             scores=scores,
             explained_variance=explained_variance,
             explained_variance_ratio=explained_variance_ratio,
             mean=data_mean,
             voxel_coords=voxel_coords,
             spatial_shape=spatial_shape,
             mask_binned=mask_binned,
             fps=fps,
             z_offset=z_offset)
    print(f"Saved PCA results to {os.path.join(input_dir, 'pca_results.npz')}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Actual start time of the trimmed data
    data_start_time = start_time_sec
    
    # Plot variance explained
    plot_variance_explained(explained_variance_ratio, output_dir=input_dir)
    
    # Plot spatial weights
    plot_top_pcs_spatial(components, voxel_coords, spatial_shape, mask_binned,
                          n_pcs=n_pcs_plot, output_dir=input_dir)
    
    # Plot weight distributions
    plot_weight_distributions(components, n_pcs=n_pcs_plot, output_dir=input_dir)
    
    # Plot PC scores
    plot_pc_scores(scores, explained_variance_ratio, fps=fps, n_pcs=n_pcs_plot,
                   output_dir=input_dir, data_start_time=data_start_time, 
                   time_offset=time_offset_sec)
    
    # Plot scores heatmap
    plot_scores_heatmap(scores, explained_variance_ratio, fps=fps, n_pcs=20,
                         output_dir=input_dir, data_start_time=data_start_time,
                         time_offset=time_offset_sec)
    
    # Plot reconstruction
    plot_reconstruction(data_for_pca, components, scores, data_mean,
                        fps=fps, output_dir=input_dir,
                        data_start_time=data_start_time, time_offset=time_offset_sec)
    
    # Plot PC correlations
    plot_pc_correlations(scores, n_pcs=10, output_dir=input_dir)
    
    # Plot combined spatial-temporal
    plot_combined_spatial_temporal(components, scores, explained_variance_ratio,
                                    voxel_coords, spatial_shape, mask_binned,
                                    fps=fps, n_pcs=4, output_dir=input_dir,
                                    data_start_time=data_start_time, time_offset=time_offset_sec)
    
    print("\nPCA analysis complete!")
    print(f"Results saved to {input_dir}")


if __name__ == "__main__":
    main()
