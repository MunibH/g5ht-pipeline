from nd2reader import ND2Reader
import numpy as np
import itk
import sys
import os
import tifffile
import warnings; warnings.filterwarnings('ignore', category=UserWarning, module='itk')

from utils import get_noise_stack

from multiprocessing import Pool, cpu_count
from functools import partial
import tqdm  # optional, for progress bar

'''
Shear correction module for 3D image stacks using ITK elastix registration.
Processes ND2 files, applies noise correction, and performs shear correction
Trimmed z-slices can be specified.
'''

# noise_path = '/home/albert_w/scripts/noise_042925.tif'
# noise_tif = tifffile.imread(noise_path)
# noise_stack = np.stack([noise_tif] * 41, axis=0).astype(np.float32)

def get_stack(input_nd2, index, noise_stack, stack_shape=(41, 2, 512, 512), zplane_to_keep=(2,-1)):
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
        start, end = zplane_to_keep

        if end == -1:
            return denoised[start:]
        else:
            return denoised[start:end+1]

def register(fixed, moving, parameter_object, threads=8):
    """Performs rigid registration between two images using ITK's elastix with binary masks."""
    #fixed_mask = itk.image_view_from_array((fixed > 0).astype(np.ubyte))
    #moving_mask = itk.image_view_from_array((moving > 0).astype(np.ubyte))
    return itk.elastix_registration_method(fixed, moving, #fixed_mask=fixed_mask, moving_mask=moving_mask,
                                           parameter_object=parameter_object, number_of_threads=threads)
                                           #log_to_file=True, log_file_name='test.log', output_directory='.')

def shear_correct(stack, parameter_object):
    """Performs shear correction using RFP channel as reference, propagating from max intensity slice."""
    rfp_means = np.mean(stack[:, 1], axis=(1, 2))
    max_rfp_z = np.argmax(rfp_means)
    output = stack.copy()

    for i in range(max_rfp_z - 1, -1, -1):
        rfp_fixed, rfp_moving = output[i+1, 1], output[i, 1]
        rfp_reg, params = register(rfp_fixed, rfp_moving, parameter_object)
        gfp_reg = itk.transformix_filter(output[i, 0], params)
        output[i] = gfp_reg, rfp_reg
    
    for i in range(max_rfp_z + 1, len(output)):
        rfp_fixed, rfp_moving = output[i-1, 1], output[i, 1]
        rfp_reg, params = register(rfp_fixed, rfp_moving, parameter_object)
        gfp_reg = itk.transformix_filter(output[i, 0], params)
        output[i] = gfp_reg, rfp_reg
    return output

def process_one(index, input_nd2, noise_stack, out_dir, stack_shape=(41, 2, 512, 512), zplane_to_keep=(2,-1), skip_shear_correction=False):
    """shear correct a single frame index"""
    tif_path = os.path.join(out_dir, f"{index:04d}.tif")

    # --- shear correction ---
    # if not os.path.exists(tif_path):
    shear_correct_parameter_object = itk.ParameterObject.New()
    shear_correct_parameter_map = shear_correct_parameter_object.GetDefaultParameterMap('rigid', 4)
    shear_correct_parameter_object.AddParameterMap(shear_correct_parameter_map)
    
    # # add bspline to parameter object for smoother transformations
    # shear_correct_parameter_map_bspline = shear_correct_parameter_object.GetDefaultParameterMap('bspline', 4)
    # shear_correct_parameter_map_bspline['FinalGridSpacingInVoxels'] = ['16']
    # shear_correct_parameter_object.AddParameterMap(shear_correct_parameter_map_bspline)

    stack = get_stack(input_nd2, index, noise_stack, stack_shape, zplane_to_keep)
    if skip_shear_correction:
        shear_corrected = stack
    else:
        shear_corrected = shear_correct(stack, shear_correct_parameter_object)

    shear_corrected = np.clip(shear_corrected, 0, 4095).astype(np.uint16)
    tifffile.imwrite(tif_path, shear_corrected, imagej=True)


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
    n_workers = int(sys.argv[6]) if sys.argv[6].lower() != 'all' else 10
    num_frames, height, width, num_channels = int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10])
    stack_shape = (stack_length,num_channels,height,width)
    z2keep = sys.argv[11] if len(sys.argv) > 11 else (2,-1)
    skip_shear_correction = sys.argv[12] if len(sys.argv) > 12 else False

    out_dir = os.path.join(os.path.splitext(input_nd2)[0], 'shear_corrected')
    # out_dir = os.path.join(os.path.splitext(input_nd2)[0], 'not_trimmed')
    os.makedirs(out_dir, exist_ok=True)

    noise_stack = get_noise_stack(noise_pth)

    # --- prepare parallel job list ---
    indices = list(range(start_idx, end_idx + 1))
    print(f"Processing {len(indices)} stacks ({start_idx}-{end_idx}) using {n_workers} workers...")

    with Pool(processes=n_workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(
                partial(process_one, input_nd2=input_nd2, noise_stack=noise_stack, out_dir=out_dir, stack_shape=stack_shape, zplane_to_keep=z2keep, skip_shear_correction=skip_shear_correction),
                indices),
                total=len(indices)):
            pass

    print("Parallel preprocessing complete.")


if __name__ == "__main__":
    main()
