from nd2reader import ND2Reader
import numpy as np
import itk
import sys
import os
import tifffile
import warnings; warnings.filterwarnings('ignore', category=UserWarning, module='itk')

from multiprocessing import Pool, cpu_count
from functools import partial
import tqdm  # optional, for progress bar


def apply_transform(stack, parameter_object):
    stack[:,0,:,:] = itk.transformix_filter(stack[:,0,:,:], parameter_object)
    return stack

def process_one(index, out_dir, save_dir, align_file, stack_shape=(39, 2, 512, 512)):
    """Process a single frame index (shear correction + channel alignment)."""

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
    n_workers = int(sys.argv[6]) if len(sys.argv) > 6 else cpu_count()
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

    with Pool(processes=n_workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(
                partial(process_one, out_dir=out_dir, save_dir=save_dir, align_file=align_file, stack_shape=stack_shape),
                indices),
                total=len(indices)):
            pass

    print("Parallel preprocessing complete.")


if __name__ == "__main__":
    main()
