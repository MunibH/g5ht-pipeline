# uses g5ht-pipeline conda env

from pathlib import PurePosixPath
import subprocess
from nd2reader import ND2Reader
import numpy as np
import itk
import sys
import os
import tifffile
import warnings; warnings.filterwarnings('ignore', category=UserWarning, module='itk')
import matplotlib.pyplot as plt
import matplotlib
import glob

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
    
def scp_from_flvc(filename, data_dir_flvc, data_dir_local, flvc):
    
    # 1. Handle Local Path (Standard Windows style)
    date_str = filename.split('_')[0].split('-')[1]
    local_dir = data_dir_local / date_str
    local_dir.mkdir(parents=True, exist_ok=True)

    # 2. Handle Remote Path (Force Linux/Posix style)
    remote_path = PurePosixPath(data_dir_flvc) / date_str / filename

    # 3. Check if remote file exists
    ssh_command = f'ssh {flvc} "test -e {remote_path}"'
    print(ssh_command)
    
    result = subprocess.run(ssh_command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Remote file not found or error: {result.stderr}")
        return

    # 4. Transfer using scp (available on Windows via OpenSSH)
    print(f"Transferring {filename} to {local_dir}...")
    scp_command = f'scp "{flvc}:{remote_path}" "{local_dir}"'
    print(scp_command)
    subprocess.run(scp_command, shell=True, check=True)



# utils for plotting data

def default_plt_params():
    font = {'family' : 'Arial',
            'weight' : 'normal',
            'size'   : 15}
    matplotlib.rc('font', **font)

def pretty_plot(figsize=(6,4), tick_dir='out', tick_length=5, tick_width=1, spine_width=0.75, fontsize=20, top_border=False, right_border=False):
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.tick_params(direction=tick_dir, length=tick_length, width=tick_width)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    if not top_border:
        ax.spines['top'].set_visible(False)
    if not right_border:
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig, ax