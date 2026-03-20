# uses g5ht-pipeline conda env

from pathlib import PurePosixPath
import subprocess
from nd2reader import ND2Reader
import numpy as np
import os
import os.path as osp
import psutil
import re
import tifffile
import warnings; warnings.filterwarnings('ignore', category=UserWarning, module='itk')

# plotting utils in plot_utils.py

# utils for loading and preprocessing data

def get_noise_stack(noise_path, stack_length=41):
    noise_tif = tifffile.imread(noise_path)
    noise_stack = np.stack([noise_tif] * stack_length, axis=0).astype(np.float32)
    return noise_stack

def get_range_from_nd2(input_nd2, stack_length=41):
    """Returns a range object containing valid stack indices from the ND2 file."""
    with ND2Reader(input_nd2) as f:
        num_frames = (f.metadata['num_frames'] // stack_length)
        height = f.metadata['height']
        width = f.metadata['width']
        num_channels = len(f.metadata['channels'])
    return num_frames, height, width, num_channels

def get_stack_from_nd2(input_nd2, index, noise_stack, stack_shape=(41, 2, 512, 512), trim=2, denoise=True, make_third_channel=False):
    """Extracts and preprocesses a specific stack from the ND2 file, returns float32 array with trimmed z-slices.
    
    Arguments:
    make_third_channel: bool, add a third color channel of zeros so images can be plotted using matplotlib
    """
    stack = np.zeros(stack_shape, np.float32)
    frame_indices = np.arange(stack_shape[0] * index, stack_shape[0] * (index + 1))
    with ND2Reader(input_nd2) as f:
        for i, frame in enumerate(frame_indices):
            stack[i] = f.get_frame_2D(0, frame), f.get_frame_2D(1, frame)
    if denoise:
        denoised = np.clip(stack - noise_stack, 0, 4095)
    else:
        denoised = np.clip(stack, 0, 4095)

    if make_third_channel:
        new_channel = np.full(denoised[:, :1, :, :].shape, 0)
        denoised = np.concatenate([denoised, new_channel], axis=1)

    return denoised[:-trim]

def get_stack_z_coordinates(input_nd2, index, stack_shape=(41, 2, 512, 512), trim=2):
    # NOT COMPLETED
    """get piezo z-coordinates. in Albert's g5-ht recordings each z step is 1.08 um and voxels are 0.36 um^3
    The zstep value can be different in Munib's recordings"""
    stack = np.zeros(stack_shape, np.float32)
    frame_indices = np.arange(stack_shape[0] * index, stack_shape[0] * (index + 1))
    with ND2Reader(input_nd2) as f:
        zcoords = f.metadata['z_positions'][frame_indices]
        zstep = f.metadata['z_step_size']
    zcoords_um = (zcoords - np.min(zcoords)) / zstep * 0.36
    # zcoords_um = zcoords_um[:-trim]
    return zcoords, zcoords_um

def get_beads_alignment_file(input_nd2):
    """get beads alignment file"""
    out_dir = os.path.splitext(input_nd2)[0]
    # check if beads file exists
    beads_file = os.path.join(out_dir+'_chan_alignment.nd2')
    if not os.path.exists(beads_file):
        return None
    else:
        return beads_file
    
def parse_datasets(datasets_path, section='UNPROCESSED'):
    """Return nd2 paths listed under the given section header."""
    paths = []
    in_section = False
    with open(datasets_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(f'# {section}'):
                in_section = True
                continue
            if line.startswith('#'):
                if in_section:
                    break
                continue
            if in_section and line:
                nd2_path = re.sub(r'\s*\(.*\)\s*$', '', line)
                paths.append(nd2_path)
    # only keep paths that contain 'nd2' and exist
    paths = [p for p in paths if 'nd2' in p and os.path.exists(p)]
    return paths

def get_output_dir(nd2_path):
    """Create and return output directory named after the nd2 file."""
    dirname = osp.dirname(nd2_path)
    basename = osp.splitext(osp.basename(nd2_path))[0]
    output_dir = osp.join(dirname, basename)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def make_mip_rgb(stack):
    """Max-project z-stack and map GFP/RFP channels to green/red."""
    gfp, rfp = np.max(stack, axis=0)
    rgb = np.zeros((*rfp.shape, 3), np.uint8)
    rgb[..., 0] = np.clip(rfp / 500, 0, 1) * 255
    rgb[..., 1] = np.clip(gfp / 200, 0, 1) * 255
    return rgb


def get_optimal_cpu_count(cap=16, buffer_ratio=0.5):
    """
    Calculates the number of CPUs to use based on current system load.
    
    Args:
        cap (int): The absolute maximum number of cores to use.
        buffer_ratio (float): The fraction of available cores to claim (0.5 = 50%).
        
    Returns:
        int: Number of CPUs to utilize.
    """
    # detect total logical cores
    total_cpus = os.cpu_count() or 1
    
    # sample system usage over 1 second
    # A higher interval is more accurate but makes the function slower.
    usage_pct = psutil.cpu_percent(interval=1)
    
    # calculate "free" core equivalents
    busy_cpus = (usage_pct / 100) * total_cpus
    available_cpus = max(0, total_cpus - busy_cpus)
    
    # apply your logic: half of available, capped at 'cap'
    # we use max(1, ...) to ensure we always return at least one thread.
    n_cpus = max(1, int(available_cpus * buffer_ratio))
    final_count = min(n_cpus, cap)
    
    print(f"System Load: {usage_pct}% | Available: {available_cpus:.2f} | Allocating: {final_count}")
    
    return final_count