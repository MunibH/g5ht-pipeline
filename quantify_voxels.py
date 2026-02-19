import h5py
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import glob
from skimage import measure
import os
from tqdm import tqdm
from functools import partial
import h5py


import matplotlib
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 15}
matplotlib.rc('font', **font)


def process_single_tif(args):
    """Process a single TIF file - designed for parallel execution."""
    tif_path, z_slices, h_binned, w_binned, binning_factor = args
    
    stack = tifffile.imread(tif_path).astype(np.float32).clip(min=0, max=4096)
    
    # Reshape to (Z, 2, H, W) if needed
    if stack.ndim == 3:
        stack = stack.reshape(z_slices, 2, stack.shape[1], stack.shape[2])
    
    # Perform binning on spatial dimensions
    z, c, h, w = stack.shape
    h_crop = h_binned * binning_factor
    w_crop = w_binned * binning_factor
    
    # More efficient binning using reshape and mean
    binned = stack[:, :, :h_crop, :w_crop].reshape(
        z, c, h_binned, binning_factor, w_binned, binning_factor
    ).mean(axis=(3, 5))
    
    return binned[:, 0], binned[:, 1]


def main():

    input_dir = sys.argv[1]
    reg_dir = sys.argv[2]
    binning_factor = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    baseline_window = sys.argv[4] if len(sys.argv) > 4 and isinstance(sys.argv[4], tuple) else (0, 60)
    fps = float(sys.argv[5]) if len(sys.argv) > 5 else 1/0.533
    
    registered_dir = os.path.join(input_dir, reg_dir)

    tif_paths = glob.glob(os.path.join(registered_dir, '*.tif'))
    # sort 
    tif_paths = sorted(tif_paths, key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    # get frame index from filename (assuming format like 'path/to/file/0001.tif')
    tif_fns = sorted(tif_paths, key=lambda x: int(os.path.basename(x).split('.')[0]))
    frame_vec = [int(os.path.basename(fn).split('.')[0]) for fn in tif_fns]
    # make a time vector
    time_vec = np.array(frame_vec) / fps
    
    # for each tif, we have a 3d stack of 2 channels. We want to quantify the intensity of channel 0 in each voxel after performing binning, normalized by the mean intensity of channel 1 in the same voxel
    # load each tif, and perform binning, create a (time, z, height, width) array
    
    # Load first stack to get dimensions
    first_stack = tifffile.imread(tif_paths[0])
    # Assuming first_stack is (Z*C, H, W), need to separate channels
    if first_stack.ndim == 3:
        # Stack is (Z*C, H, W), need to separate channels
        z_slices = first_stack.shape[0] // 2
        first_stack = first_stack.reshape(z_slices, 2, first_stack.shape[1], first_stack.shape[2])
    
    # Calculate binned dimensions
    z_slices, _, h, w = first_stack.shape
    h_binned = h // binning_factor
    w_binned = w // binning_factor
    
    # Initialize output array for GFP only: (T, Z, H_binned, W_binned)
    # Use float32 to reduce memory usage
    normalized_gfp = np.zeros((len(tif_paths), z_slices, h_binned, w_binned), dtype=np.float32)
    # For RFP, only store the running sum to compute mean (saves ~17 GiB)
    rfp_sum = np.zeros((z_slices, h_binned, w_binned), dtype=np.float64)  # Use float64 for accumulation precision
    
    # Process files sequentially (was doing this in parallel, but ran into memory issues)
    print(f"Processing {len(tif_paths)} files")
    
    for i, tif_path in enumerate(tqdm(tif_paths, desc="Processing stacks")):
        args = (tif_path, z_slices, h_binned, w_binned, binning_factor)
        gfp, rfp = process_single_tif(args)
        normalized_gfp[i] = gfp
        rfp_sum += rfp  # Accumulate RFP for mean calculation
    
    # Compute RFP mean from accumulated sum
    rfp_mean = (rfp_sum / len(tif_paths)).astype(np.float32)
    
    # Create mask for voxels that are 0 in GFP channel (should remain 0 throughout)
    zero_mask = normalized_gfp == 0

    # Ratiometric normalization: divide channel 0 by channel 1's mean across time
    normalized_data = np.divide(normalized_gfp, rfp_mean, out=np.zeros_like(normalized_gfp), where=rfp_mean!=0)
    # # baseline normalize by each voxel's mean over first 60 time points to get F/F_baseline
    baseline = normalized_data[baseline_window[0]:baseline_window[1]].mean(axis=0)
    normalized_data = np.divide(normalized_data, baseline, out=np.zeros_like(normalized_data), where=baseline!=0)
    # divide each voxel by its 10th percentile across time
    # F10 = np.percentile(normalized_data, 10, axis=0)
    # normalized_data = normalized_data / (F10 + 1e-6)
    
    # Ensure voxels that were originally 0 remain 0
    normalized_data[zero_mask] = 0

    r_rbaseline = normalized_data.copy()
    
    # Now normalized_data is (T, Z, H, W) array ready for further processing
    print(f"Processed data shape: {normalized_data.shape}")
    print(f"Binning factor: {binning_factor}")
    
    # # Save the normalized data, rfp_mean, gfp_mean, baseline in a npy file
    # print('Saving r/r0 to npy file...')
    # np.save(os.path.join(input_dir, 'r_r0.npy'), r_rbaseline)
    # np.save(os.path.join(input_dir, 'rfp_mean.npy'), rfp_mean)
    # np.save(os.path.join(input_dir, 'gfp_mean.npy'), normalized_gfp.mean(axis=0))
    # np.save(os.path.join(input_dir, 'baseline.npy'), baseline)
    # print(f"Saved r/r0 to {os.path.join(input_dir, 'r_r0.npy')}")

    # load mask with metadata (it was saved to tif as tifffile.imwrite(roi_pth, roi.astype(np.uint8), imagej=True, metadata={'Labels': roi_labels}))
    roi = tifffile.imread(os.path.join(input_dir, 'roi.tif'))
    with tifffile.TiffFile(os.path.join(input_dir, 'roi.tif')) as tif:
        labels = tif.imagej_metadata['Labels']

    # load fixed mask
    fixed_fn = glob.glob(os.path.join(input_dir, 'fixed_mask_[0-9][0-9][0-9][0-9]*.tif'))[0]
    fixed_mask = tifffile.imread(fixed_fn)

    # save all data in a .h5 file
    print('Saving r/r0, roi, and fixed mask to h5 file...')
    with h5py.File(os.path.join(input_dir, f'{os.path.basename(input_dir)}_processed_voxels.h5'), 'w') as f:
        f.create_dataset('r_r0', data=r_rbaseline, compression='gzip')
        f.create_dataset('rfp_mean', data=rfp_mean, compression='gzip')
        f.create_dataset('gfp_mean', data=normalized_gfp.mean(axis=0), compression='gzip')
        f.create_dataset('baseline', data=baseline, compression='gzip')
        f.create_dataset('time_vec', data=time_vec, compression='gzip')
        f.create_dataset('frame_vec', data=frame_vec, compression='gzip')
        f.create_dataset('binning_factor', data=binning_factor)
        f.create_dataset('fps', data=fps)
        f.create_dataset('baseline_window', data=baseline_window)
        f.create_dataset('roi', data=roi, compression='gzip')
        f.create_dataset('roi_labels', data=labels)
        f.create_dataset('fixed_mask', data=fixed_mask, compression='gzip')
    print("Done")