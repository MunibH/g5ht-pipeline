import tifffile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import glob
from skimage import measure
import os
from tqdm import tqdm

import matplotlib
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 15}
matplotlib.rc('font', **font)


def main():

    input_dir = sys.argv[1]
    registered_dir = os.path.join(input_dir, 'registered_wholistic_smooth-0.200_patch-7')

    tif_paths = glob.glob(os.path.join(registered_dir, '*.tif'))
    tif_paths = sorted(tif_paths)[:]
    
    binning_factor = 2
    
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
    normalized_data = np.zeros((len(tif_paths), z_slices, h_binned, w_binned))
    
    for i, tif_path in enumerate(tqdm(tif_paths, desc="Processing stacks")):
        stack = tifffile.imread(tif_path)
        
        # Reshape to (Z, 2, H, W) if needed
        if stack.ndim == 3:
            z_slices = stack.shape[0] // 2
            stack = stack.reshape(z_slices, 2, stack.shape[1], stack.shape[2])
        
        # Perform binning on spatial dimensions: (Z, C, H, W) -> (Z, C, H_binned, W_binned)
        # Crop to multiple of binning_factor
        z, c, h, w = stack.shape
        h_crop = h_binned * binning_factor
        w_crop = w_binned * binning_factor
        
        binned = stack[:, :, :h_crop, :w_crop].reshape(
            z, c, h_binned, binning_factor, w_binned, binning_factor
        ).mean(axis=(3, 5))
        
        # Normalize channel 0 by channel 1 for each z-slice
        # binned is now (Z, C, H_binned, W_binned)
        # Add small epsilon to avoid division by zero
        normalized_data[i] = binned[:, 0] / (binned[:, 1] + 1e-10)
    
    # Now normalized_data is (T, Z, H, W) array ready for further processing
    print(f"Processed data shape: {normalized_data.shape}")
    print(f"Binning factor: {binning_factor}")
    
    # Save the normalized data
    np.save(os.path.join(input_dir, 'normalized_voxels.npy'), normalized_data)
    print(f"Saved normalized data to {os.path.join(input_dir, 'normalized_voxels.npy')}")

    # out = np.zeros((len(tif_paths), 3))
    # out[:] = np.nan
    # for i in tqdm(range(len(out))):
    #     stack = tifffile.imread(tif_paths[i])
    #     for j in range(3):
    #         denominator =  np.sum(stack[:, 1][mask == j + 1])
    #         if denominator > 0:
    #             out[i, j] = np.sum(stack[:, 0][mask == j + 1]) / denominator

    # t = np.arange(len(out)) * 0.533 / 60
    # # df = pd.DataFrame(out, index=t)
    # df = df.interpolate()
    # df.to_csv(os.path.join(input_dir, 'quantified.csv'))

    # plt.figure(figsize=(10, 4))
    # plt.plot(t, out[:, 0] / np.mean(out[:60, 0]), label='Dorsal nerve ring')
    # plt.plot(t, out[:, 1] / np.mean(out[:60, 1]), label='Ventral nerve ring')
    # plt.plot(t, out[:, 2] / np.mean(out[:60, 2]), label='Pharynx')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) 
    # plt.xlabel('Time (min)')
    # plt.ylabel(r'$F/F_{baseline}$')
    # plt.xlim(0, np.max(t))
    # plt.axhline(1, ls='--', c='k', zorder=0)
    # plt.tight_layout()
    # plt.savefig(os.path.join(input_dir, 'quantified.png'), dpi=300)
    # plt.show()

    # fixed = tifffile.imread(os.path.join(input_dir, 'fixed.tif'))
    # img = np.zeros((200, 500, 3), np.float32)
    # img[..., 0] = np.max(fixed[:, 1], axis=0)
    # img[..., 0] = np.clip(img[..., 0] / 400, 0, 1)
    # img = (img * 255).astype(np.ubyte)

    # plt.figure(figsize=(10, 4))
    # contours = measure.find_contours(np.max(mask == 1, axis=0), level=0.5)
    # for contour in contours:
    #     plt.plot(contour[:, 1], contour[:, 0], color='C0', linewidth=2)
    # contours = measure.find_contours(np.max(mask == 2, axis=0), level=0.5)
    # for contour in contours:
    #     plt.plot(contour[:, 1], contour[:, 0], color='C1', linewidth=2)
    # contours = measure.find_contours(np.max(mask == 3, axis=0), level=0.5)
    # for contour in contours:
    #     plt.plot(contour[:, 1], contour[:, 0], color='C2', linewidth=2)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig(os.path.join(input_dir , 'roi.png'), dpi=300)
    # plt.show()