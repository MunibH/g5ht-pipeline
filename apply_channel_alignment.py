from nd2reader import ND2Reader
import numpy as np
import itk
import sys
import os
import tifffile
import warnings; warnings.filterwarnings('ignore', category=UserWarning, module='itk')

import multiprocessing as mp
from functools import partial
import tqdm  # optional, for progress bar
import pandas as pd
import matplotlib.pyplot as plt

def plot_channel_alignment_params(input_nd2_pth, beads_alignment_file=None):
    """Plot histograms of channel alignment parameters for worm and/or beads recordings."""
    from plot_utils import default_plt_params
    default_plt_params()

    params = ['TransformParameter_0', 'TransformParameter_1', 'TransformParameter_2',
              'TransformParameter_3', 'TransformParameter_4', 'TransformParameter_5']
    labels = ['Rx', 'Ry', 'Rz', 'Tx', 'Ty', 'Tz']

    # Worm recording
    try:
        out_dir = os.path.splitext(input_nd2_pth)[0]
        df = pd.read_csv(os.path.join(out_dir, 'chan_align_params.csv'))

        plt.figure(figsize=(12, 8), tight_layout=True)
        for i, param in enumerate(params):
            plt.subplot(2, 3, i + 1)
            plt.hist(df[param], bins=30, color='red', alpha=0.6)
            median_value = df[param].median()
            plt.axvline(median_value, color='black', linestyle='dashed', linewidth=2)
            plt.xlabel(labels[i])
            plt.ylabel('Frequency')
            plt.title(f'Median: {np.round(median_value, 3)}', fontsize=14)
        plt.savefig(os.path.join(out_dir, 'chan_align_params_histograms.png'))
        plt.show()
    except FileNotFoundError:
        print("No chan_align_params.csv found for worm recording")

    # Beads recording
    if beads_alignment_file is not None:
        try:
            out_dir = os.path.splitext(beads_alignment_file)[0]
            df = pd.read_csv(os.path.join(out_dir, 'chan_align_params.csv'))

            plt.figure(figsize=(12, 8), tight_layout=True)
            for i, param in enumerate(params):
                plt.subplot(2, 3, i + 1)
                plt.hist(df[param], bins=30, color='blue', alpha=0.6)
                median_value = df[param].median()
                plt.axvline(median_value, color='black', linestyle='dashed', linewidth=2)
                plt.xlabel(labels[i])
                plt.ylabel('Frequency')
                plt.title(f'Median: {np.round(median_value, 3)}', fontsize=14)
            plt.savefig(os.path.join(out_dir, 'chan_align_params_histograms.png'))
            plt.show()
        except FileNotFoundError:
            print("No chan_align_params.csv found for beads recording")

def apply_transform(stack, parameter_object):
    gfp_3d = np.ascontiguousarray(stack[:, 0, :, :]).astype(np.float32)
    gfp_itk = itk.image_from_array(gfp_3d)
    gfp_aligned = itk.transformix_filter(gfp_itk, parameter_object)
    stack[:, 0, :, :] = np.asarray(gfp_aligned)
    return stack

def process_one(index, out_dir, save_dir, align_file, stack_shape=(39, 2, 512, 512)):
    """Process a single frame index (shear correction + channel alignment)."""
    try:
        # Create parameter object in each worker (ITK objects can't be pickled)
        parameter_object = itk.ParameterObject.New()
        parameter_object.ReadParameterFile(align_file)

        # load shear corrected
        shear_correct_pth = os.path.join(out_dir, "shear_corrected")
        shear_correct_fn = os.path.join(shear_correct_pth, f"{index:04d}.tif")
        shear_corrected = tifffile.imread(shear_correct_fn).astype(np.float32)

        # apply transform
        channel_aligned = apply_transform(shear_corrected, parameter_object)
        channel_aligned = np.clip(channel_aligned, 0, 4095).astype(np.uint16)

        # save channel aligned
        tif_path = os.path.join(save_dir, f"{index:04d}.tif")
        tifffile.imwrite(tif_path, channel_aligned, imagej=True)
    except Exception as e:
        raise RuntimeError(f"Frame {index}: {type(e).__name__}: {e}") from None

def main():
    """
    Main pipeline: process multiple frame indices in parallel.
    Usage:
        python preprocess.py <input_nd2> <start_index> <end_index> <noise_path> [n_workers]
    """
    input_nd2 = sys.argv[1]
    start_idx = int(sys.argv[2])
    end_idx = int(sys.argv[3])
    noise_pth = sys.argv[4]
    stack_length = int(sys.argv[5])
    n_workers = int(sys.argv[6]) if len(sys.argv) > 6 else mp.cpu_count()
    num_frames, height, width, num_channels = int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10])
    stack_shape = (stack_length-2,num_channels,height,width) 
    align_with_beads = int(sys.argv[11]) if len(sys.argv) > 11 else False
    beads_alignment_file = sys.argv[12] if len(sys.argv) > 12 else None
    
    out_dir = os.path.splitext(input_nd2)[0]
    save_dir = os.path.join(out_dir, 'channel_aligned')
    os.makedirs(save_dir, exist_ok=True)

    if align_with_beads:
        align_dir = os.path.splitext(beads_alignment_file)[0]
    else:
        align_dir = out_dir

    # Path to channel alignment parameter file (workers will load it themselves)
    align_file = os.path.join(align_dir, 'chan_align.txt')
    

    # --- prepare parallel job list ---
    indices = list(range(start_idx, end_idx + 1))
    print(f"Processing {len(indices)} stacks ({start_idx}-{end_idx}) using {n_workers} workers...")

    # Use 'spawn' context to avoid fork-related deadlocks with ITK on Linux
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(
                partial(process_one, out_dir=out_dir, save_dir=save_dir, align_file=align_file, stack_shape=stack_shape),
                indices),
                total=len(indices)):
            pass

    print("Parallel preprocessing complete.")





if __name__ == "__main__":
    main()
