"""
Sync NIR and Confocal Data - Simplified Approach with Validation

This script synchronizes NIR camera data with confocal microscopy data using a
peak-detection approach for confocal timing and filtered triggers for NIR timing.
It includes comprehensive validation plots and exports aligned data.

Usage:
    python sync_and_export.py --h5_path <path_to_h5_file> --n_stack <number_of_stacks> [options]
    
Or import and use functions directly:
    from sync_and_export import sync_nir_confocal, export_aligned_data
"""

import h5py
import numpy as np
from scipy import signal
import tifffile as tif
import matplotlib.pyplot as plt
import pandas as pd
import os
import os.path as osp
import argparse
from typing import Tuple, Dict, Optional
import warnings


def load_h5_data(h5_path: str) -> Dict[str, np.ndarray]:
    """
    Load all relevant data from HDF5 file.
    
    Parameters
    ----------
    h5_path : str
        Path to HDF5 file
        
    Returns
    -------
    dict
        Dictionary containing all loaded data arrays
    """
    print(f"Loading data from: {h5_path}")
    
    if not osp.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
    
    data = {}
    
    with h5py.File(h5_path, 'r') as f:
        # Load DAQ data
        print("  Loading daqmx_ai (analog inputs)...")
        data['daqmx_ai'] = f['daqmx_ai'][:]
        
        print("  Loading daqmx_di (digital inputs)...")
        data['daqmx_di'] = f['daqmx_di'][:]
        
        # Load image metadata
        print("  Loading img_metadata...")
        data['img_id'] = np.array(f['img_metadata']['img_id'])
        data['q_iter_save'] = np.array(f['img_metadata']['q_iter_save'])
        data['img_timestamp'] = np.array(f['img_metadata']['img_timestamp'])
        
        # Load stage position if available
        if 'pos_stage' in f:
            print("  Loading pos_stage...")
            data['pos_stage'] = np.array(f['pos_stage'])
        else:
            print("  Warning: pos_stage not found in file")
            data['pos_stage'] = None
            
        # Get NIR image shape
        print("  Getting img_nir shape...")
        data['img_nir_shape'] = f['img_nir'].shape
        data['n_img_nir'] = f['img_nir'].shape[0]
        
    print(f"  Data loaded successfully!")
    print(f"    - NIR images: {data['img_nir_shape']} (n_frames, height, width)")
    print(f"    - DAQ samples: {len(data['daqmx_ai'])}")
    print(f"    - Metadata entries: {len(data['img_id'])}")
    
    return data


def detect_confocal_timing_peaks(piezo_signal: np.ndarray, 
                                 expected_n_volumes: Optional[int] = None,
                                 min_distance: int = 100,
                                 min_prominence: float = 1.0) -> np.ndarray:
    """
    Detect confocal volume timing using peak detection on piezo signal.
    
    This approach identifies the end of each 3D volume (z-stack) by finding peaks
    in the piezo signal (when the objective returns to starting position).
    
    Parameters
    ----------
    piezo_signal : np.ndarray
        Piezo analog signal from daqmx_ai
    expected_n_volumes : int, optional
        Expected number of 3D volumes over time (e.g., 1200 timepoints)
        NOT the number of z-slices per volume
    min_distance : int, optional
        Minimum distance between peaks (default: 100)
    min_prominence : float, optional
        Minimum prominence of peaks (default: 1.0)
        
    Returns
    -------
    np.ndarray
        Indices of confocal volume end times (one per timepoint)
    """
    print("\nDetecting confocal timing from piezo signal...")
    
    # Find peaks in piezo signal
    peaks, properties = signal.find_peaks(
        piezo_signal, 
        distance=min_distance,
        prominence=min_prominence
    )
    
    print(f"  Found {len(peaks)} peaks in piezo signal (confocal volumes/timepoints)")

    # plot peak properties for debugging
    plt.figure(figsize=(10, 4))
    plt.plot(piezo_signal, label='Piezo Signal')
    plt.plot(peaks, piezo_signal[peaks], "x", label='Detected Peaks')
    plt.title('Piezo Signal with Detected Peaks (Each peak = 1 volume/timepoint)')
    plt.xlabel('Sample Index')
    plt.ylabel('Signal Amplitude')
    plt.xlim(0,100000)
    plt.legend()
    plt.show()
    
    # Validate peak count
    if expected_n_volumes is not None:
        print(f"  Validating peak count against expected: {expected_n_volumes} volumes")
        if len(peaks) != expected_n_volumes:
            warnings.warn(
                f"Expected {expected_n_volumes} volumes but found {len(peaks)} peaks. "
                f"Difference: {abs(len(peaks) - expected_n_volumes)}"
            )
            if abs(len(peaks) - expected_n_volumes) > 3:
                raise ValueError(
                    f"Peak count mismatch too large! Expected {expected_n_volumes} volumes, "
                    f"found {len(peaks)} peaks. Check min_distance and min_prominence parameters."
                )
    
    # Validate peak spacing
    if len(peaks) > 1:
        peak_diffs = np.diff(peaks)
        print(f"  Peak spacing: mean={np.mean(peak_diffs):.1f}, "
              f"std={np.std(peak_diffs):.1f}, "
              f"range=[{np.min(peak_diffs)}, {np.max(peak_diffs)}]")
        
        # Check for suspicious spacing
        if np.std(peak_diffs) / np.mean(peak_diffs) > 0.5:
            warnings.warn("Large variability in peak spacing detected. Verify results.")
    
    return peaks


def detect_nir_timing_filtered(di_nir: np.ndarray, 
                               img_id: np.ndarray,
                               q_iter_save: np.ndarray) -> np.ndarray:
    """
    Detect NIR frame timing with filtering based on image IDs and save flags.
    
    Parameters
    ----------
    di_nir : np.ndarray
        NIR digital input signal
    img_id : np.ndarray
        Image ID array from metadata
    q_iter_save : np.ndarray
        Boolean array indicating which frames were saved
        
    Returns
    -------
    np.ndarray
        Indices of NIR frame start times (filtered)
    """
    print("\nDetecting NIR timing from digital input...")
    
    # Binarize NIR digital input
    nir_ons = (di_nir > 0.5).astype(int)
    
    # Find rising edges (frame starts)
    nir_starts = np.where(np.diff(nir_ons) == 1)[0]
    print(f"  Found {len(nir_starts)} total NIR triggers")
    
    # Adjust image IDs to start from 0
    ids = img_id - img_id[0]
    print(f"  Image IDs range: {ids[0]} to {ids[-1]}")
    
    # Filter by save flags
    q = q_iter_save > 0.5
    print(f"  Frames marked for save: {np.sum(q)} / {len(q)}")
    
    # Get the IDs for saved frames
    saved_ids = ids[q]
    print(f"  Saved frame IDs range: {saved_ids.min()} to {saved_ids.max()}")
    
    # Validate that IDs are within bounds
    if saved_ids.max() >= len(nir_starts):
        n_out_of_bounds = np.sum(saved_ids >= len(nir_starts))
        warnings.warn(
            f"{n_out_of_bounds} image IDs are out of bounds (>= {len(nir_starts)}). "
            f"Max ID: {saved_ids.max()}. This may indicate missing triggers or metadata mismatch. "
            f"Clipping IDs to valid range."
        )
        # Clip IDs to valid range
        saved_ids = np.clip(saved_ids, 0, len(nir_starts) - 1)
    
    # Apply filters
    try:
        filtered_nir_starts = nir_starts[saved_ids]
        print(f"  Filtered NIR starts: {len(filtered_nir_starts)}")
    except IndexError as e:
        raise IndexError(
            f"Error indexing NIR starts. "
            f"nir_starts shape: {nir_starts.shape}, "
            f"saved_ids shape: {saved_ids.shape}, "
            f"max saved_id: {np.max(saved_ids) if len(saved_ids) > 0 else 'N/A'}"
        ) from e
    
    return filtered_nir_starts


def create_nir_confocal_mapping(confocal_peaks: np.ndarray,
                                nir_starts: np.ndarray) -> np.ndarray:
    """
    Create mapping from confocal volumes to NIR frames.
    
    For each confocal volume end time, find the NIR frame that was captured
    just before that time.
    
    Parameters
    ----------
    confocal_peaks : np.ndarray
        Confocal volume end times (indices), one per timepoint
    nir_starts : np.ndarray
        NIR frame start times (indices)
        
    Returns
    -------
    np.ndarray
        Array mapping each confocal volume/timepoint to its NIR frame index
    """
    print("\nCreating NIR-to-confocal mapping...")
    
    # Use searchsorted to find NIR frame for each confocal volume
    mapping = np.searchsorted(nir_starts, confocal_peaks, side='right') - 1
    
    print(f"  Created mapping for {len(mapping)} confocal volumes/timepoints")
    print(f"  NIR frame range: {np.min(mapping)} to {np.max(mapping)}")
    
    # Validate mapping
    if np.any(mapping < 0):
        n_invalid = np.sum(mapping < 0)
        warnings.warn(f"{n_invalid} confocal volumes have no corresponding NIR frame")
        
    if np.any(mapping >= len(nir_starts)):
        n_invalid = np.sum(mapping >= len(nir_starts))
        warnings.warn(f"{n_invalid} confocal volumes mapped beyond NIR range")
    
    # Check for duplicate mappings
    unique_mappings = len(np.unique(mapping))
    if unique_mappings < len(mapping):
        warnings.warn(
            f"Multiple confocal volumes map to same NIR frames: "
            f"{len(mapping)} volumes -> {unique_mappings} unique NIR frames"
        )
    
    return mapping


def sync_nir_confocal(h5_path: str, 
                      expected_n_volumes: Optional[int] = None,
                      min_peak_distance: int = 100,
                      min_peak_prominence: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Synchronize NIR and confocal data from HDF5 file.
    
    Parameters
    ----------
    h5_path : str
        Path to HDF5 file
    expected_n_volumes : int, optional
        Expected number of 3D confocal volumes over time (e.g., 1200 timepoints)
        NOT the number of z-slices per volume (which might be ~41)
    min_peak_distance : int, optional
        Minimum distance between peaks for confocal detection (default: 100)
    min_peak_prominence : float, optional
        Minimum prominence for peak detection (default: 1.0)
        
    Returns
    -------
    confocal_peaks : np.ndarray
        Confocal volume end times (one per timepoint)
    nir_starts : np.ndarray
        NIR frame start times (filtered)
    mapping : np.ndarray
        NIR frame index for each confocal volume/timepoint
    data : dict
        All loaded data for further processing
    """
    # Load data
    data = load_h5_data(h5_path)
    
    # Detect confocal timing from piezo signal (column 1 in daqmx_ai)
    confocal_peaks = detect_confocal_timing_peaks(
        piezo_signal=data['daqmx_ai'][1,:],
        expected_n_volumes=expected_n_volumes,
        min_distance=min_peak_distance,
        min_prominence=min_peak_prominence
    )
    
    # Detect NIR timing from digital input (column 1 in daqmx_di)
    nir_starts = detect_nir_timing_filtered(
        di_nir=data['daqmx_di'][1,:],
        img_id=data['img_id'],
        q_iter_save=data['q_iter_save']
    )
    
    # Create mapping
    mapping = create_nir_confocal_mapping(confocal_peaks, nir_starts)
    
    return confocal_peaks, nir_starts, mapping, data


def plot_timing_validation(confocal_peaks: np.ndarray,
                           nir_starts: np.ndarray,
                           mapping: np.ndarray,
                           data: Dict,
                           output_path: str,
                           plot_range: Optional[Tuple[int, int]] = None):
    """
    Create comprehensive validation plots for timing synchronization.
    
    Parameters
    ----------
    confocal_peaks : np.ndarray
        Confocal volume end times (one per timepoint)
    nir_starts : np.ndarray
        NIR frame start times
    mapping : np.ndarray
        NIR-to-confocal mapping (NIR frame for each confocal volume)
    data : dict
        Loaded data dictionary
    output_path : str
        Path to save plots
    plot_range : tuple, optional
        (start, end) indices to plot (for zoomed view)
    """
    print("\nGenerating validation plots...")
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    # Determine plot range
    if plot_range is None:
        plot_start, plot_end = 0, len(data['daqmx_ai'])
    else:
        plot_start, plot_end = plot_range
    
    x_range = np.arange(plot_start, plot_end)
    
    # Plot 1: Piezo signal with detected peaks
    ax = axes[0]
    piezo_signal = data['daqmx_ai'][1,:]
    ax.plot(x_range, piezo_signal[plot_start:plot_end], 'b-', linewidth=0.5, label='Piezo Signal')
    
    # Mark peaks within range
    peaks_in_range = confocal_peaks[(confocal_peaks >= plot_start) & (confocal_peaks < plot_end)]
    ax.plot(peaks_in_range, piezo_signal[peaks_in_range], 'r*', markersize=10, label='Detected Peaks')
    
    ax.set_ylabel('Piezo Signal (V)')
    ax.set_title('Confocal Volume Detection (Piezo Signal - each peak = 1 volume/timepoint)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: NIR digital input with detected starts
    ax = axes[1]
    nir_di = data['daqmx_di'][1,:]
    ax.plot(x_range, nir_di[plot_start:plot_end], 'g-', linewidth=0.5, label='NIR DI')
    
    # Mark NIR starts within range
    nir_in_range = nir_starts[(nir_starts >= plot_start) & (nir_starts < plot_end)]
    if len(nir_in_range) > 0:
        ax.plot(nir_in_range, np.ones(len(nir_in_range)) * 0.9, 'r|', markersize=8, 
                markeredgewidth=1.5, label='NIR Frame Starts')
    
    ax.set_ylabel('Digital Input')
    ax.set_title('NIR Frame Detection (Digital Input)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Laser signal (confocal recording indicator)
    ax = axes[2]
    laser_signal = data['daqmx_ai'][0,:]
    ax.plot(x_range, laser_signal[plot_start:plot_end], 'orange', linewidth=0.5, label='Laser Signal')
    ax.set_ylabel('Laser Signal (V)')
    ax.set_title('Confocal Laser Signal', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Stimulus signal
    ax = axes[3]
    stim_signal = data['daqmx_ai'][2,:]
    ax.plot(x_range, stim_signal[plot_start:plot_end], 'm-', linewidth=0.5, label='Stimulus Signal')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Stimulus (V)')
    ax.set_title('Stimulus Signal', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved validation plot: {output_path}")
    plt.close()
    
    # Create mapping visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot mapping indices
    ax = axes[0]
    ax.plot(mapping, 'o-', markersize=4, linewidth=1)
    ax.set_xlabel('Confocal Volume/Timepoint Index')
    ax.set_ylabel('NIR Frame Index')
    ax.set_title('NIR-to-Confocal Frame Mapping', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot mapping differences
    ax = axes[1]
    if len(mapping) > 1:
        mapping_diff = np.diff(mapping)
        ax.plot(mapping_diff, 'o-', markersize=4, linewidth=1)
        ax.axhline(y=1, color='r', linestyle='--', label='Expected (1:1 mapping)')
        ax.set_xlabel('Confocal Volume/Timepoint Index')
        ax.set_ylabel('NIR Frame Increment')
        ax.set_title('Frame Mapping Increments (Should be mostly 1)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Print statistics
        print(f"\n  Mapping statistics:")
        print(f"    Mean increment: {np.mean(mapping_diff):.2f}")
        print(f"    Std increment: {np.std(mapping_diff):.2f}")
        print(f"    Non-unity mappings: {np.sum(mapping_diff != 1)} / {len(mapping_diff)}")
    
    plt.tight_layout()
    mapping_plot_path = output_path.replace('.png', '_mapping.png')
    plt.savefig(mapping_plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved mapping plot: {mapping_plot_path}")
    plt.close()


def export_aligned_data(h5_path: str,
                       mapping: np.ndarray,
                       output_dir: Optional[str] = None,
                       save_nir: bool = True,
                       save_stage: bool = True,
                       save_metadata: bool = True) -> Dict[str, str]:
    """
    Export aligned NIR images and stage positions.
    
    Parameters
    ----------
    h5_path : str
        Path to HDF5 file
    mapping : np.ndarray
        NIR frame index for each confocal volume/timepoint
    output_dir : str, optional
        Output directory (default: same as h5_path)
    save_nir : bool, optional
        Save NIR images as TIFF (default: True)
    save_stage : bool, optional
        Save stage positions as CSV (default: True)
    save_metadata : bool, optional
        Save synchronization metadata (default: True)
        
    Returns
    -------
    dict
        Dictionary of output file paths
    """
    print("\nExporting aligned data...")
    
    # Setup output paths
    if output_dir is None:
        output_dir = osp.dirname(h5_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    name = osp.basename(h5_path).split('.')[0]
    output_files = {}
    
    # Export NIR images
    if save_nir:
        print(f"  Loading and saving {len(mapping)} NIR frames...")
        
        nir_path = osp.join(output_dir, f'{name}_nir_aligned.tif')
        
        with h5py.File(h5_path, 'r') as f:
            img_shape = f['img_nir'].shape
            nir = np.zeros((len(mapping), img_shape[1], img_shape[2]), dtype=np.uint8)
            
            for i, j in enumerate(mapping):
                try:
                    nir[i] = f['img_nir'][j]
                except (IndexError, KeyError) as e:
                    warnings.warn(f"Could not load NIR frame {j} for confocal volume {i}: {e}")
                    break
        
        tif.imwrite(nir_path, nir, imagej=True)
        output_files['nir'] = nir_path
        print(f"  Saved NIR images: {nir_path}")
        print(f"    Shape: {nir.shape}")
    
    # Export stage positions
    if save_stage:
        print(f"  Loading and saving stage positions...")
        
        stage_path = osp.join(output_dir, f'{name}_stage_aligned.csv')
        
        with h5py.File(h5_path, 'r') as f:
            if 'pos_stage' in f:
                stage_data = np.array(f['pos_stage'])
                
                # Interpolate stage positions
                df = pd.DataFrame(stage_data, columns=['y', 'x'])
                df_interp = df.interpolate(method='cubic')
                
                # Extract aligned positions
                aligned_stage = df_interp.to_numpy()[mapping]
                
                # Save to CSV
                df_aligned = pd.DataFrame(aligned_stage, columns=['y', 'x'])
                df_aligned.index.name = 'confocal_volume'
                df_aligned.to_csv(stage_path)
                
                output_files['stage'] = stage_path
                print(f"  Saved stage positions: {stage_path}")
                print(f"    Shape: {aligned_stage.shape}")
            else:
                print(f"  Warning: pos_stage not found in HDF5 file")
    
    # Export metadata
    if save_metadata:
        print(f"  Saving synchronization metadata...")
        
        metadata_path = osp.join(output_dir, f'{name}_sync_metadata.npz')
        
        with h5py.File(h5_path, 'r') as f:
            timestamps = np.array(f['img_metadata']['img_timestamp'])
            q_save = np.array(f['img_metadata']['q_iter_save'])
        
        # Get timestamps for aligned NIR frames
        nir_timestamps = timestamps[q_save] / 1e9  # Convert to seconds
        aligned_timestamps = nir_timestamps[mapping]
        
        np.savez(
            metadata_path,
            mapping=mapping,
            confocal_volume_indices=np.arange(len(mapping)),
            nir_frame_indices=mapping,
            timestamps=aligned_timestamps,
            description="NIR-to-confocal synchronization mapping (one NIR frame per confocal volume/timepoint)"
        )
        
        output_files['metadata'] = metadata_path
        print(f"  Saved metadata: {metadata_path}")
    
    print(f"\n  All exports complete!")
    return output_files


def run_full_pipeline(h5_path: str,
                     expected_n_volumes: Optional[int] = None,
                     output_dir: Optional[str] = None,
                     min_peak_distance: int = 100,
                     min_peak_prominence: float = 1.0) -> Dict[str, str]:
    """
    Run the complete synchronization and export pipeline.
    
    Parameters
    ----------
    h5_path : str
        Path to HDF5 file
    expected_n_volumes : int, optional
        Expected number of 3D confocal volumes/timepoints (e.g., 1200)
        NOT the number of z-slices per volume
    output_dir : str, optional
        Output directory
    min_peak_distance : int, optional
        Minimum distance between peaks
    min_peak_prominence : float, optional
        Minimum peak prominence
        
    Returns
    -------
    dict
        Dictionary of all output file paths
    """
    print("="*70)
    print("NIR-CONFOCAL SYNCHRONIZATION PIPELINE")
    print("="*70)
    
    # Run synchronization
    confocal_peaks, nir_starts, mapping, data = sync_nir_confocal(
        h5_path=h5_path,
        expected_n_volumes=expected_n_volumes,
        min_peak_distance=min_peak_distance,
        min_peak_prominence=min_peak_prominence
    )
    
    # Setup output directory
    if output_dir is None:
        output_dir = osp.dirname(h5_path)
    
    name = osp.basename(h5_path).split('.')[0]
    
    # Generate validation plots
    # Full view
    plot_path_full = osp.join(output_dir, f'{name}_sync_validation_full.png')
    plot_timing_validation(
        confocal_peaks, nir_starts, mapping, data,
        output_path=plot_path_full
    )
    
    # Zoomed view (first 10000 samples)
    plot_path_zoom = osp.join(output_dir, f'{name}_sync_validation_zoom.png')
    plot_timing_validation(
        confocal_peaks, nir_starts, mapping, data,
        output_path=plot_path_zoom,
        plot_range=(0, min(10000, len(data['daqmx_ai'])))
    )
    
    # Export aligned data
    output_files = export_aligned_data(
        h5_path=h5_path,
        mapping=mapping,
        output_dir=output_dir,
        save_nir=True,
        save_stage=True,
        save_metadata=True
    )
    
    # Add plot paths to output
    output_files['plot_full'] = plot_path_full
    output_files['plot_zoom'] = plot_path_zoom
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nOutput files:")
    for key, path in output_files.items():
        print(f"  {key}: {path}")
    
    return output_files


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Synchronize NIR and confocal data with validation'
    )
    parser.add_argument('--h5_path', type=str, required=True,
                       help='Path to HDF5 file')
    parser.add_argument('--expected_n_volumes', '--n_volumes', type=int, default=None,
                       help='Expected number of 3D confocal volumes/timepoints (e.g., 1200, NOT z-slices per volume)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as h5_path)')
    parser.add_argument('--min_peak_distance', type=int, default=100,
                       help='Minimum distance between peaks (default: 100)')
    parser.add_argument('--min_peak_prominence', type=float, default=1.0,
                       help='Minimum peak prominence (default: 1.0)')
    
    args = parser.parse_args()
    
    run_full_pipeline(
        h5_path=args.h5_path,
        expected_n_volumes=args.expected_n_volumes,
        output_dir=args.output_dir,
        min_peak_distance=args.min_peak_distance,
        min_peak_prominence=args.min_peak_prominence
    )


if __name__ == '__main__':
    main()
