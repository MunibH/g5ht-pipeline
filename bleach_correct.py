import tifffile
import numpy as np
import os
import math
from scipy import signal
from scipy.optimize import curve_fit
import sys
from glob import glob
import matplotlib.pyplot as plt
import gc
from utils import default_plt_params
default_plt_params()

# most of the 'block' method code is adapted from  https://github.com/xiyuyi-at-LLNL/pysofi/blob/master/pysofi

# Channel names and colors for plotting
CHANNEL_INFO = {
    0: {'name': 'GFP', 'color': 'green', 'color_light': 'lightgreen'},
    1: {'name': 'RFP', 'color': 'red', 'color_light': 'lightcoral'}
}


def med_smooth(ori_signal, kernel_size=251):
    """
    Perform a one-dimensional median filter with 'reflect' padding.
    For more information, please check scipy.signal.medfilt.
    """
    signal_pad = np.append(np.append(ori_signal[0:kernel_size][::-1],
                                     ori_signal),
                           ori_signal[-kernel_size:][::-1])
    filtered_signal = signal.medfilt(signal_pad, kernel_size)
    return filtered_signal[kernel_size:-kernel_size]


def exponential_decay(t, a, b, c):
    """Exponential decay model: y = a * exp(-b * t) + c"""
    return a * np.exp(-b * t) + c


def fit_exponential(signal_data):
    """
    Fit an exponential decay model to the signal data.
    
    Parameters
    ----------
    signal_data : 1darray
        Signal intensity over time.
        
    Returns
    -------
    fitted_signal : 1darray
        Fitted exponential curve.
    params : tuple
        Fitted parameters (a, b, c).
    """
    t = np.arange(len(signal_data))
    
    # Initial parameter estimates
    a0 = signal_data[0] - signal_data[-1]
    c0 = signal_data[-1]
    b0 = 1.0 / len(signal_data)
    
    try:
        params, _ = curve_fit(exponential_decay, t, signal_data, 
                              p0=[a0, b0, c0], maxfev=10000,
                              bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
        fitted_signal = exponential_decay(t, *params)
    except RuntimeError:
        # If fitting fails, return original signal
        print("  Warning: Exponential fit failed, using median smoothing instead")
        fitted_signal = med_smooth(signal_data, kernel_size=min(251, len(signal_data) // 2 * 2 + 1))
        params = (0, 0, 0)
    
    return fitted_signal, params


def cut_frames(signal_level, fbc=0.04):
    """
    Find the list of frame number to cut the whole signal plot into separate
    blocks based on the change of total signal intensity. 

    Parameters
    ----------
    signal_level : 1darray
        Signal change over time (can be derived from 'calc_total_signal').
    fbc : float
        The fraction of signal decrease within each block compared to the
        total signal decrease.

    Returns
    -------
    bounds : list of int
        Signal intensities on the boundary of each block.
    frame_lst : list of int
        Frame number where to cut the whole signal plot into blocks.
    """
    max_intensity, min_intensity = np.max(signal_level), np.min(signal_level)
    frame_num = np.argmax(signal_level)
    total_diff = max_intensity - min_intensity
    block_num = math.ceil(1/fbc)
    frame_lst = []
    # lower bound of intensity for each block
    bounds = [int(max_intensity - total_diff * i * fbc)
              for i in range(1, block_num + 1)]
    i = 0
    while frame_num < len(signal_level) and i < block_num:
        if signal_level[frame_num] < bounds[i]:
            frame_lst.append(frame_num)
            frame_num += 1
            i += 1
        else:
            frame_num += 1
    frame_lst = [0] + frame_lst + [len(signal_level)]
    bounds = [int(max_intensity)] + bounds

    return bounds, frame_lst


def get_sorted_tiff_files(input_dir):
    """
    Get sorted list of tiff files in directory with XXXX.tiff naming convention.
    
    Parameters
    ----------
    input_dir : str
        Path to the directory containing tiff files.
        
    Returns
    -------
    files : list of str
        Sorted list of full file paths.
    """
    # Match both .tiff and .tif extensions
    files = glob(os.path.join(input_dir, '[0-9][0-9][0-9][0-9].tiff'))
    if not files:
        files = glob(os.path.join(input_dir, '[0-9][0-9][0-9][0-9].tif'))
    files = sorted(files)
    return files


def calc_signal(input_dir, channel, mode='total'):
    """
    Calculate the signal intensity of a specific channel 
    for each volume across all tiff files.

    Parameters
    ----------
    input_dir : str
        Path to directory containing tiff files.
    channel : int
        Channel index (0 for GFP, 1 for RFP).
    mode : str
        'total' for sum of all pixel values, 'median' for median pixel value.

    Returns
    -------
    signal : 1darray
        Signal intensity (total or median) of the channel for each volume.
    """
    files = get_sorted_tiff_files(input_dir)
    num_volumes = len(files)
    signal_values = np.zeros(num_volumes)
    channel_name = CHANNEL_INFO[channel]['name']
    mode_label = 'total' if mode == 'total' else 'median'

    for idx, filepath in enumerate(files):
        # Load volume with shape (Z, C, H, W)
        volume = tifffile.imread(filepath)
        # Extract channel
        channel_data = volume[:, channel, :, :]  # Shape: (Z, H, W)
        if mode == 'total':
            signal_values[idx] = np.sum(channel_data)
        else:  # median
            signal_values[idx] = np.median(channel_data)
        sys.stdout.write('\r')
        sys.stdout.write(f"Calculating {channel_name} {mode_label} signal: volume {idx + 1}/{num_volumes}")
        sys.stdout.flush()
    
    print()  # New line after progress
    return signal_values


def average_channel_volume(input_dir, channel, volume_indices):
    """
    Get the average channel volume for specified volume indices.

    Parameters
    ----------
    input_dir : str
        Path to directory containing tiff files.
    channel : int
        Channel index (0 for GFP, 1 for RFP).
    volume_indices : list of int
        Start and end volume indices [start, end) for averaging.

    Returns
    -------
    mean_volume : ndarray
        The average volume with shape (Z, H, W), dtype float32.
    """
    files = get_sorted_tiff_files(input_dir)
    start_idx, end_idx = volume_indices
    
    # Read first volume to get dimensions
    first_volume = tifffile.imread(files[start_idx])
    z_dim, _, h_dim, w_dim = first_volume.shape
    mean_volume = np.zeros((z_dim, h_dim, w_dim), dtype=np.float32)
    del first_volume
    
    for idx in range(start_idx, end_idx):
        volume = tifffile.imread(files[idx])
        channel_data = volume[:, channel, :, :].astype(np.float32)
        mean_volume += channel_data
        del volume, channel_data
    
    mean_volume /= (end_idx - start_idx)
    return mean_volume


def min_channel_volume(corrected_list):
    """
    Get the minimum value across all corrected volumes (element-wise).

    Parameters
    ----------
    corrected_list : list of ndarray
        List of corrected volumes, each with shape (Z, H, W).

    Returns
    -------
    min_volume : ndarray
        The element-wise minimum across all volumes.
    """
    min_volume = corrected_list[0].copy()
    for vol in corrected_list[1:]:
        min_volume = np.minimum(min_volume, vol)
    return min_volume


def plot_bleaching_diagnostics(output_dir, raw_signal, fitted_signal, corrected_signal,
                                channel, method, intensity_mode='total',
                                frame_lst=None, bounds=None, exp_params=None):
    """
    Generate and save diagnostic plots for bleaching correction.

    Parameters
    ----------
    output_dir : str
        Directory to save the plots.
    raw_signal : 1darray
        Original signal per volume (total or median).
    fitted_signal : 1darray
        Fitted bleaching curve (median filtered or exponential).
    corrected_signal : 1darray
        Signal after bleaching correction.
    channel : int
        Channel index (0 for GFP, 1 for RFP).
    method : str
        Correction method ('block' or 'exponential').
    intensity_mode : str
        'total' or 'median' - which intensity metric was used.
    frame_lst : list of int, optional
        Frame indices where blocks are divided (for block method).
    bounds : list of int, optional
        Signal intensity bounds for each block (for block method).
    exp_params : tuple, optional
        Exponential fit parameters (a, b, c).

    Returns
    -------
    None
        Saves plots to output_dir.
    """
    num_volumes = len(raw_signal)
    volume_indices = np.arange(num_volumes)
    
    channel_name = CHANNEL_INFO[channel]['name']
    channel_color = CHANNEL_INFO[channel]['color']
    channel_color_light = CHANNEL_INFO[channel]['color_light']
    method_label = 'block' if method == 'block' else 'exp'
    intensity_label = 'Total' if intensity_mode == 'total' else 'Median'
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Raw signal with fitted curve
    ax1 = axes[0, 0]
    ax1.plot(volume_indices, raw_signal, color=channel_color, alpha=0.5, linewidth=0.8, label=f'Raw {channel_name}')
    fit_label = 'Exponential fit' if method == 'exponential' else 'Median filtered'
    ax1.plot(volume_indices, fitted_signal, 'k-', linewidth=2, label=fit_label)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel(f'{intensity_label} Intensity')
    ax1.legend()
    
    # Plot 2: Raw signal with block boundaries or exponential parameters
    ax2 = axes[0, 1]
    ax2.plot(volume_indices, raw_signal, color=channel_color, alpha=0.7, linewidth=0.8, label=f'Raw {channel_name}')
    if method == 'block' and frame_lst is not None:
        for i, frame_idx in enumerate(frame_lst[1:-1]):
            ax2.axvline(x=frame_idx, color='black', linestyle='--', alpha=0.7, linewidth=1)
        if bounds is not None:
            for i, bound in enumerate(bounds[:-1]):
                ax2.axhline(y=bound, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax2.set_title(f'{channel_name} Block Boundaries')
    else:
        ax2.plot(volume_indices, fitted_signal, 'k-', linewidth=2, label='Exp fit')
        if exp_params is not None and exp_params[0] != 0:
            ax2.text(0.05, 0.95, f'a={exp_params[0]:.2e}\nb={exp_params[1]:.4f}\nc={exp_params[2]:.2e}',
                    transform=ax2.transAxes, verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.set_title(f'{channel_name} Exponential Fit')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel(f'{intensity_label} Intensity')
    ax2.legend()
    
    # Plot 3: Before vs After comparison
    ax3 = axes[1, 0]
    ax3.plot(volume_indices, raw_signal / raw_signal[0], color=channel_color, alpha=0.7, linewidth=1, label='Before (norm)')
    corrected_norm = corrected_signal / corrected_signal[0] if corrected_signal[0] > 0 else corrected_signal
    ax3.plot(volume_indices, corrected_norm, color=channel_color_light, alpha=0.9, linewidth=1.5, label='After (norm)')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Normalized Signal')
    ax3.legend()
    
    # Plot 4: Corrected signal
    ax4 = axes[1, 1]
    ax4.plot(volume_indices, corrected_signal, color=channel_color, linewidth=1, label=f'Corrected {channel_name}')
    z = np.polyfit(volume_indices, corrected_signal, 1)
    p = np.poly1d(z)
    # # trend should be close to zero
    # ax4.plot(volume_indices, p(volume_indices), 'k--', linewidth=2, 
    #          label=f'Trend: {z[0]:.2e}')
    ax4.set_xlabel('Frame')
    ax4.set_ylabel(f'{intensity_label} Intensity')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save figure with descriptive filename (both PNG and SVG)
    plot_basename = f'bleach_diagnostics_{channel_name}_{method_label}'
    plot_path_png = os.path.join(output_dir, f'{plot_basename}.png')
    plot_path_svg = os.path.join(output_dir, f'{plot_basename}.svg')
    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path_svg, dpi=300, bbox_inches='tight', format='svg')
    plt.close()
    
    print(f"  Saved: {plot_path_png}")
    print(f"  Saved: {plot_path_svg}")
    
    # Create a second figure showing the bleaching profile in detail
    fig2, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(volume_indices, raw_signal, color=channel_color, alpha=0.4, linewidth=0.8, label=f'Raw {channel_name}')
    ax.plot(volume_indices, fitted_signal, 'k-', linewidth=2, label='Bleach curve')
    
    if method == 'block' and frame_lst is not None:
        # Shade blocks with alternating colors
        for i in range(len(frame_lst) - 1):
            start = frame_lst[i]
            end = frame_lst[i + 1]
            color = channel_color_light if i % 2 == 0 else 'lightyellow'
            ax.axvspan(start, end, alpha=0.3, color=color)
    
    ax.set_xlabel('Frame')
    ax.set_ylabel(f'{intensity_label} {channel_name} Intensity')
    ax.legend()
    
    plt.tight_layout()
    
    # Save both PNG and SVG
    plot_basename2 = f'bleach_profile_{channel_name}_{method_label}'
    plot_path2_png = os.path.join(output_dir, f'{plot_basename2}.png')
    plot_path2_svg = os.path.join(output_dir, f'{plot_basename2}.svg')
    plt.savefig(plot_path2_png, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path2_svg, dpi=300, bbox_inches='tight', format='svg')
    plt.close()
    
    print(f"  Saved: {plot_path2_png}")
    print(f"  Saved: {plot_path2_svg}")


def correct_channel_block(input_dir, files, channel, fbc, smooth_kernel, intensity_mode='total'):
    """
    Perform block-based bleaching correction on a single channel using
    multiplicative correction.
    
    This method is appropriate for freely moving samples where the same
    pixel location does not correspond to the same biological structure
    across time points.
    
    Steps:
    1. Calculate signal per volume (total or median)
    2. Median filter to get smooth bleaching curve
    3. Divide into blocks based on intensity drops
    4. For each block, compute mean signal (scalar)
    5. Calculate multiplicative correction factor per block
    6. Return correction factors for each volume
    
    The correction factor for each block is: initial_block_signal / current_block_signal
    Each volume is then multiplied by its block's correction factor.
    """
    num_volumes = len(files)
    channel_name = CHANNEL_INFO[channel]['name']
    mode_label = 'total' if intensity_mode == 'total' else 'median'
    
    # Step 1: Calculate signal per volume
    print(f"  Calculating {mode_label} {channel_name} signal...")
    sig_b = calc_signal(input_dir, channel, mode=intensity_mode)
    
    # Step 2: Filter the total signal to get a smoother bleaching profile
    actual_kernel = min(smooth_kernel, num_volumes)
    if actual_kernel % 2 == 0:
        actual_kernel -= 1
    if actual_kernel < 3:
        actual_kernel = 3
    filtered_sig_b = med_smooth(sig_b, kernel_size=actual_kernel)
    
    # Step 3: Calculate block boundaries
    bounds, frame_lst = cut_frames(filtered_sig_b, fbc=fbc)
    block_num = len(frame_lst) - 1
    print(f"  Divided into {block_num} blocks")
    
    # Step 4: Compute mean total signal for each block
    print(f"  Computing block mean signals...")
    block_mean_signals = []
    for block_idx in range(block_num):
        start_vol = frame_lst[block_idx]
        end_vol = frame_lst[block_idx + 1]
        # Use the filtered signal for more stable estimates
        block_mean = np.mean(filtered_sig_b[start_vol:end_vol])
        block_mean_signals.append(block_mean)
        print(f"    Block {block_idx + 1}/{block_num}: volumes {start_vol}-{end_vol-1}, mean signal = {block_mean:.2e}")
    
    # Step 5: Calculate correction factors
    # Reference is the first block's mean signal
    reference_signal = block_mean_signals[0]
    block_correction_factors = [reference_signal / mean_sig for mean_sig in block_mean_signals]
    
    print(f"  Block correction factors:")
    for block_idx, factor in enumerate(block_correction_factors):
        print(f"    Block {block_idx + 1}: {factor:.4f}")
    
    # Create per-volume correction factors array
    correction_factors = np.zeros(num_volumes, dtype=np.float64)
    for block_idx in range(block_num):
        start_vol = frame_lst[block_idx]
        end_vol = frame_lst[block_idx + 1]
        correction_factors[start_vol:end_vol] = block_correction_factors[block_idx]
    
    print(f"  Block correction complete for {channel_name}")
    
    return sig_b, filtered_sig_b, frame_lst, bounds, correction_factors


def correct_channel_exponential(input_dir, files, channel, intensity_mode='total'):
    """
    Perform exponential-fit based bleaching correction on a single channel.
    
    Returns diagnostic data and correction factors for memory-efficient processing.
    Does NOT store all corrected volumes in memory.
    """
    num_volumes = len(files)
    channel_name = CHANNEL_INFO[channel]['name']
    mode_label = 'total' if intensity_mode == 'total' else 'median'
    
    # Calculate signal per volume
    print(f"  Calculating {mode_label} {channel_name} signal...")
    sig_b = calc_signal(input_dir, channel, mode=intensity_mode)
    
    # Fit exponential decay
    print(f"  Fitting exponential decay model...")
    fitted_sig_b, exp_params = fit_exponential(sig_b)
    
    # Calculate correction factors for each volume
    initial_signal = fitted_sig_b[0]
    correction_factors = initial_signal / fitted_sig_b
    
    return sig_b, fitted_sig_b, exp_params, correction_factors


def correct_bleaching(input_dir, output_dir=None, channels=None, method='block',
                      fbc=0.04, smooth_kernel=251, intensity_mode='total'):
    """
    Performs bleaching correction on specified channels of ZCHW tiff volumes.
    
    Memory-efficient implementation: processes one volume at a time.
    
    Input data format:
    - Directory containing tiff files named XXXX.tiff (4-digit numbers: 0000, 0001, ...)
    - Each tiff file is a volume with shape (Z, C, H, W) where C=2
    - Channel 0: GFP
    - Channel 1: RFP

    Parameters
    ----------
    input_dir : str
        Path to directory containing input tiff files.
    output_dir : str, optional
        Path to output directory. If None, creates subdirectory inside input_dir.
    channels : int or list of int, optional
        Channel(s) to correct. Can be 0, 1, or [0, 1]. Default is [1] (RFP only).
    method : str
        Correction method: 'block' (default) or 'exponential'.
    fbc : float
        The fraction of signal decrease within each block (for block method).
    smooth_kernel : int
        The size of the median filter window (for block method).
    intensity_mode : str
        'total' (default) or 'median' - which intensity metric to use for
        estimating the bleaching curve.

    Returns
    -------
    None
        Saves corrected volumes to output directory.
    """
    # Handle channel input
    if channels is None:
        channels = [1]  # Default to RFP only
    elif isinstance(channels, int):
        channels = [channels]
    
    # Validate channels
    for ch in channels:
        if ch not in [0, 1]:
            raise ValueError(f"Invalid channel {ch}. Must be 0 (GFP) or 1 (RFP).")
    
    # Validate method
    if method not in ['block', 'exponential']:
        raise ValueError(f"Invalid method '{method}'. Must be 'block' or 'exponential'.")
    
    # Validate intensity_mode
    if intensity_mode not in ['total', 'median']:
        raise ValueError(f"Invalid intensity_mode '{intensity_mode}'. Must be 'total' or 'median'.")
    
    # Create descriptive output directory name
    channel_str = '_'.join([CHANNEL_INFO[ch]['name'] for ch in sorted(channels)])
    method_str = 'block' if method == 'block' else 'exp'
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), f'bleach_corrected_{channel_str}_{method_str}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of input files
    files = get_sorted_tiff_files(input_dir)
    num_volumes = len(files)
    
    if num_volumes == 0:
        raise ValueError(f"No tiff files found in {input_dir} with XXXX.tiff naming convention")
    
    print(f"Found {num_volumes} volumes in {input_dir}")
    print(f"Channels to correct: {[CHANNEL_INFO[ch]['name'] for ch in channels]}")
    print(f"Method: {method}")
    print(f"Intensity mode: {intensity_mode}")
    
    # Get original dtype from first volume
    first_volume = tifffile.imread(files[0])
    original_dtype = first_volume.dtype
    del first_volume
    
    # Store diagnostic data for each channel
    channel_diagnostics = {}
    channel_correction_factors = {}  # Store correction factors for both methods
    
    # First pass: compute corrections for each channel
    for channel in channels:
        channel_name = CHANNEL_INFO[channel]['name']
        print(f"\n{'='*50}")
        print(f"Processing {channel_name} channel (index {channel})")
        print(f"{'='*50}")
        
        if method == 'block':
            raw_sig, fitted_sig, frame_lst, bounds, correction_factors = correct_channel_block(
                input_dir, files, channel, fbc, smooth_kernel, intensity_mode
            )
            channel_diagnostics[channel] = {
                'raw_signal': raw_sig,
                'fitted_signal': fitted_sig,
                'frame_lst': frame_lst,
                'bounds': bounds,
                'exp_params': None
            }
            channel_correction_factors[channel] = correction_factors
        else:  # exponential
            raw_sig, fitted_sig, exp_params, correction_factors = correct_channel_exponential(
                input_dir, files, channel, intensity_mode
            )
            channel_diagnostics[channel] = {
                'raw_signal': raw_sig,
                'fitted_signal': fitted_sig,
                'frame_lst': None,
                'bounds': None,
                'exp_params': exp_params
            }
            channel_correction_factors[channel] = correction_factors
    
    # Second pass: apply multiplicative corrections and save volumes
    print(f"\n{'='*50}")
    print("Applying corrections and saving volumes...")
    print(f"{'='*50}")
    
    corrected_signals = {ch: np.zeros(num_volumes) for ch in channels}
    
    for vol_idx in range(num_volumes):
        # Load original volume
        volume = tifffile.imread(files[vol_idx])
        output_volume = volume.copy()
        
        # Apply multiplicative corrections for each channel
        for channel in channels:
            channel_data = volume[:, channel, :, :].astype(np.float64)
            correction_factor = channel_correction_factors[channel][vol_idx]
            corrected = channel_data * correction_factor
            
            # Clip and convert to original dtype
            if np.issubdtype(original_dtype, np.integer):
                max_val = np.iinfo(original_dtype).max
            else:
                max_val = np.finfo(original_dtype).max
            corrected = np.clip(corrected, 0, max_val)
            corrected_final = corrected.astype(original_dtype)
            
            # Track total corrected signal for plotting
            corrected_signals[channel][vol_idx] = np.sum(corrected_final)
            
            # Update channel in output volume
            output_volume[:, channel, :, :] = corrected_final
        
        # Save with same filename
        output_filename = os.path.basename(files[vol_idx])
        output_path = os.path.join(output_dir, output_filename)
        tifffile.imwrite(output_path, output_volume, imagej=True)
        
        if (vol_idx + 1) % 50 == 0 or vol_idx == num_volumes - 1:
            sys.stdout.write('\r')
            sys.stdout.write(f"  Saved volume {vol_idx + 1}/{num_volumes}: {output_filename}")
            sys.stdout.flush()
    print()
    
    # diagnostic plots should save to one level above input dir
    plot_dir = os.path.dirname(input_dir)
    # Generate diagnostic plots for each channel
    print(f"\n\nGenerating diagnostic plots...")
    for channel in channels:
        diag = channel_diagnostics[channel]
        plot_bleaching_diagnostics(
            output_dir=plot_dir,
            raw_signal=diag['raw_signal'],
            fitted_signal=diag['fitted_signal'],
            corrected_signal=corrected_signals[channel],
            channel=channel,
            method=method,
            intensity_mode=intensity_mode,
            frame_lst=diag['frame_lst'],
            bounds=diag['bounds'],
            exp_params=diag['exp_params']
        )
    
    print(f"\nBleach correction complete!")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Volumes processed: {num_volumes}")
    print(f"  Channels corrected: {[CHANNEL_INFO[ch]['name'] for ch in channels]}")
    print(f"  Method: {method}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bleach correction for ZCHW tiff volumes")
    parser.add_argument("input_dir", type=str, help="Directory containing input tiff files (XXXX.tiff)")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Output directory (default: auto-generated based on settings)")
    parser.add_argument("--channels", type=int, nargs='+', default=[1],
                        help="Channel(s) to correct: 0 (GFP), 1 (RFP), or both (default: 1)")
    parser.add_argument("--method", type=str, default='block', choices=['block', 'exponential'],
                        help="Correction method: 'block' or 'exponential' (default: block)")
    parser.add_argument("--fbc", type=float, default=0.04,
                        help="Bleaching correction factor for block method (default: 0.04)")
    parser.add_argument("--smooth_kernel", type=int, default=251,
                        help="Median filter kernel size for block method (default: 251)")
    parser.add_argument("--intensity_mode", type=str, default='total', choices=['total', 'median'],
                        help="Intensity metric for bleaching estimation: 'total' or 'median' (default: total)")
    
    args = parser.parse_args()
    
    correct_bleaching(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        channels=args.channels,
        method=args.method,
        fbc=args.fbc,
        smooth_kernel=args.smooth_kernel,
        intensity_mode=args.intensity_mode
    )