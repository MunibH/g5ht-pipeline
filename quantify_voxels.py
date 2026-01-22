import tifffile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import glob
from skimage import measure
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

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
    
    registered_dir = os.path.join(input_dir, reg_dir)

    tif_paths = glob.glob(os.path.join(registered_dir, '*.tif'))
    # sort 
    tif_paths = sorted(tif_paths, key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    
    
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
    
    # Initialize output array for all timepoints: (T, Z, H_binned, W_binned)
    normalized_gfp = np.zeros((len(tif_paths), z_slices, h_binned, w_binned))
    normalized_rfp = np.zeros((len(tif_paths), z_slices, h_binned, w_binned))
    
    # Prepare arguments for parallel processing
    process_args = [(tif_path, z_slices, h_binned, w_binned, binning_factor) 
                    for tif_path in tif_paths]
    
    # Use multiprocessing to parallelize file processing
    n_workers = min(cpu_count(), len(tif_paths))
    print(f"Processing {len(tif_paths)} files using {n_workers} workers...")
    
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_tif, process_args),
            total=len(tif_paths),
            desc="Processing stacks"
        ))
    
    # Unpack results
    for i, (gfp, rfp) in enumerate(results):
        normalized_gfp[i] = gfp
        normalized_rfp[i] = rfp
    
    # Create mask for voxels that are 0 in GFP channel (should remain 0 throughout)
    zero_mask = normalized_gfp == 0
        
    # now normalize data so that it is F/F10, where F10 is the 10th percentile of the entire time series for each voxel
    # F10 = np.percentile(normalized_data, 10, axis=0)
    # normalized_data = normalized_data / (F10 + 1e-6)

    # Ratiometric normalization: divide channel 0 by channel 1's mean across time
    rfp_mean = normalized_rfp.mean(axis=0)  # Mean across time for channel 1
    normalized_data = np.divide(normalized_gfp, rfp_mean, out=np.zeros_like(normalized_gfp), where=rfp_mean!=0)
    # # baseline normalize by each voxel's mean over first 60 time points to get F/F_baseline
    baseline = normalized_data[:60].mean(axis=0)
    normalized_data = np.divide(normalized_data, baseline, out=np.zeros_like(normalized_data), where=baseline!=0)
    # divide each voxel by its 10th percentile across time
    # F10 = np.percentile(normalized_data, 10, axis=0)
    # normalized_data = normalized_data / (F10 + 1e-6)
    
    # Ensure voxels that were originally 0 remain 0
    normalized_data[zero_mask] = 0
    
    # Now normalized_data is (T, Z, H, W) array ready for further processing
    print(f"Processed data shape: {normalized_data.shape}")
    print(f"Binning factor: {binning_factor}")
    
    # Save the normalized data, rfp_mean, gfp_mean, baseline in a npy file
    np.save(os.path.join(input_dir, 'normalized_voxels.npy'), normalized_data)
    np.save(os.path.join(input_dir, 'rfp_mean.npy'), rfp_mean)
    np.save(os.path.join(input_dir, 'gfp_mean.npy'), normalized_gfp.mean(axis=0))
    np.save(os.path.join(input_dir, 'baseline.npy'), baseline)
    print(f"Saved normalized data (ratiometric) to {os.path.join(input_dir, 'normalized_voxels.npy')}")
