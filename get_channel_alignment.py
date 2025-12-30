from nd2reader import ND2Reader
import numpy as np
import itk
import sys
import os
import tifffile
import warnings; warnings.filterwarnings('ignore', category=UserWarning, module='itk')

import utils

from multiprocessing import Pool, cpu_count
from functools import partial
import tqdm  # optional, for progress bar


# noise_path = '/home/albert_w/scripts/noise_042925.tif'
# noise_tif = tifffile.imread(noise_path)
# noise_stack = np.stack([noise_tif] * 41, axis=0).astype(np.float32)

def get_stack(input_nd2, index, noise_stack, stack_shape=(41, 2, 512, 512), trim=2):
    """Extracts and preprocesses a specific stack from the ND2 file, returns float32 array with trimmed z-slices."""

    if stack_shape[0]==1:
        noise_stack = np.mean(noise_stack,axis=0)
        noise_stack = noise_stack[np.newaxis,:,:,:]
    elif stack_shape[0] != 41:
        noise_stack = np.mean(noise_stack,axis=0)
        noise_stack = noise_stack[np.newaxis,:,:,:]

    stack = np.zeros(stack_shape, np.float32)
    frame_indices = np.arange(stack_shape[0] * index, stack_shape[0] * (index + 1))
    with ND2Reader(input_nd2) as f:
        for i, j in enumerate(frame_indices):
            stack[i] = f.get_frame_2D(0, j), f.get_frame_2D(1, j)
    denoised = np.clip(stack - noise_stack, 0, 4095)

    if stack_shape[0]==1:
        return denoised
    else:
        return denoised[:-trim]
    

def register(fixed, moving, parameter_object, threads=8):
    """Performs rigid registration between two images using ITK's elastix with binary masks."""
    #fixed_mask = itk.image_view_from_array((fixed > 0).astype(np.ubyte))
    #moving_mask = itk.image_view_from_array((moving > 0).astype(np.ubyte))
    return itk.elastix_registration_method(fixed, moving, #fixed_mask=fixed_mask, moving_mask=moving_mask,
                                           parameter_object=parameter_object, number_of_threads=threads)
                                           #log_to_file=True, log_file_name='test.log', output_directory='.')


def align_channels(stack, parameter_object):
    """Aligns GFP to RFP channel using max intensity slices."""

    gfp_moving, rfp_fixed = np.max(stack, axis=0)
    _, params = register(rfp_fixed, gfp_moving, parameter_object)


    # for i in range(stack.shape[0]):
    #     stack[i,0,:,:] = itk.transformix_filter(stack[i,0,:,:], params)
    # channel_aligned = stack.copy()

    return params

def process_one(index, input_nd2, noise_stack, out_dir, stack_shape=(41, 2, 512, 512), align_with_beads=False):
    """Process a single frame index (shear correction + channel alignment)."""

    

    if align_with_beads:
        beads_nd2 = os.path.splitext(input_nd2)[0] + '_chan_alignment.nd2'
        beads_pth = os.path.splitext(input_nd2)[0] + '_chan_alignment'

        os.makedirs(beads_pth, exist_ok=True)
        txt_pth = os.path.join(beads_pth, "txt")
        os.makedirs(txt_pth, exist_ok=True)
        txt_fn = os.path.join(txt_pth, f"{index:04d}.txt")

        # beads don't need to be shear corrected, just load the denoised stack
        stack = get_stack(beads_nd2, index, noise_stack, stack_shape=stack_shape)
        stack = np.clip(stack, 0, 4095).astype(np.float32)

        channel_align_parameter_object = itk.ParameterObject.New()
        channel_align_parameter_map = channel_align_parameter_object.GetDefaultParameterMap('rigid', 1)
        channel_align_parameter_object.AddParameterMap(channel_align_parameter_map)

        params = align_channels(stack, channel_align_parameter_object)
        params.WriteParameterFile(params, txt_fn) # save alignment parameters

    else:
        shear_correct_pth = os.path.join(out_dir, "shear_corrected")
        shear_correct_fn = os.path.join(shear_correct_pth, f"{index:04d}.tif")

        txt_pth = os.path.join(out_dir,"txt")
        os.makedirs(txt_pth, exist_ok=True)
        txt_pth =  os.path.join(txt_pth, f"{index:04d}.txt")

        channel_align_parameter_object = itk.ParameterObject.New()
        channel_align_parameter_map = channel_align_parameter_object.GetDefaultParameterMap('rigid', 1)
        channel_align_parameter_object.AddParameterMap(channel_align_parameter_map)

        shear_corrected = tifffile.imread(shear_correct_fn).astype(np.float32)

        params = align_channels(shear_corrected, channel_align_parameter_object)
        params.WriteParameterFile(params, txt_pth) # save alignment parameters


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
    stack_shape = (stack_length,num_channels,height,width)
    align_with_beads = int(sys.argv[11]) if len(sys.argv) > 11 else False

    out_dir = os.path.splitext(input_nd2)[0]

    noise_stack = utils.get_noise_stack(noise_pth)

    # --- prepare parallel job list ---
    indices = list(range(start_idx, end_idx + 1))
    print(f"Processing {len(indices)} stacks ({start_idx}-{end_idx}) using {n_workers} workers...")

    with Pool(processes=n_workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(
                partial(process_one, input_nd2=input_nd2, noise_stack=noise_stack, out_dir=out_dir, stack_shape=stack_shape, align_with_beads=align_with_beads),
                indices),
                total=len(indices)):
            pass

    print("Parallel preprocessing complete.")


if __name__ == "__main__":
    main()
