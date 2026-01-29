import glob
import sys
import os
import pandas as pd
import numpy as np
import tifffile
from nd2reader import ND2Reader
import imageio
import matplotlib.pyplot as plt
import itk
import warnings; warnings.filterwarnings('ignore', category=UserWarning, module='itk')
from tqdm import tqdm
from scipy.ndimage import laplace




def extract_frame_from_path(path):
    # get the basename (filename with extension): '13845.tif'
    filename_with_ext = os.path.basename(path)
    filename_without_ext = os.path.splitext(filename_with_ext)[0]
    # convert the resulting string to an integer for numerical sorting
    # this step is needed for sorting '0837' correctly before '1675', etc.
    return int(filename_without_ext)

def get_range(input_nd2, stack_length=41):
    """Returns a range object containing valid stack indices from the ND2 file."""
    with ND2Reader(input_nd2) as f:
        stack_range = range(f.metadata['num_frames'] // stack_length)
    return stack_range

def check_files(out_dir, stack_range, extension):
    """Checks if all the expected files in the stack range are in the directory"""
    existing_files = set(glob.glob(os.path.join(out_dir, f'*.{extension}')))
    expected_files = {os.path.join(out_dir, f'{i:04d}.{extension}') for i in stack_range}
    if missing_files := expected_files - existing_files:
        stacks = sorted([int(os.path.splitext(os.path.basename(f))[0]) for f in missing_files])
        raise FileNotFoundError(f"Missing .{extension} files: {','.join([str(i) for i in stacks])}")

    return sorted(expected_files, key=extract_frame_from_path)

def get_transform_params(file, parameter_object):
    """Extracts transform parameters from a text file."""
    parameter_object.ReadParameterFile(file)
    return np.array(parameter_object.GetParameterMap(0)['TransformParameters'], float)

def write_alignment(out_dir, stack_range):
    """Creates an alignment file from the parameter DataFrame."""
    align_path = os.path.join(out_dir, 'align.txt')
    if not os.path.exists(align_path):
        parameter_object = itk.ParameterObject.New()
        alignment_files = check_files(out_dir, stack_range, 'txt')
        params = np.zeros((len(alignment_files), 3))
        for i, file in enumerate(alignment_files):
            params[i] = get_transform_params(file, parameter_object)
        param_df = pd.DataFrame(params, columns=['theta', 't_x', 't_y'])
        param_df.to_csv(os.path.join(out_dir,'params.csv'))

        median_params = np.median(param_df.to_numpy(), axis=0)
        parameter_object.ReadParameterFile(os.path.join(r'C:\Users\munib\POSTDOC\CODE\g5ht-pipeline','template.txt'))
        changed_param_map = parameter_object.GetParameterMap(0)
        changed_param_map['TransformParameters'] = [f'{i:.15f}' for i in median_params]
        parameter_object.SetParameterMap(0, changed_param_map)
        parameter_object.WriteParameterFile(parameter_object, align_path)

        plt.subplot(311)
        plt.hist(param_df['theta'], bins=100)
        plt.axvline(median_params[0], color='k', linestyle='--')  
        plt.title(r'$\theta$')
        plt.subplot(312)
        plt.hist(param_df['t_x'], bins=100)
        plt.axvline(median_params[1], color='k', linestyle='--')
        plt.title(r'$t_x$')
        plt.subplot(313)
        plt.hist(param_df['t_y'], bins=100)
        plt.axvline(median_params[2], color='k', linestyle='--')
        plt.title(r'$t_y$')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'align.png'))
        plt.close()

def check_focus(out_dir, tif_dir, stack_range):
    """Checks the focus of the recording"""

    sys.stdout.write('\r')
    sys.stdout.write('Checking focus... ')
    sys.stdout.flush()

    plot_path = os.path.join(out_dir, 'focus.png')
    tif_files = check_files(os.path.join(out_dir, tif_dir), stack_range, 'tif')
    # read first file to get shape
    stack_shape = tifffile.imread(tif_files[0]).shape
    rfp_z = np.zeros((len(tif_files), stack_shape[0] if len(stack_shape)==4 else 1))
    for i in tqdm(range(0, len(tif_files))):
        stack = tifffile.imread(tif_files[i])
        if len(stack.shape)==3: # 1 z slice
            stack = stack[np.newaxis,:,:,:]
        rfp = stack[:, 1]
        rfp_z[i, :] = np.mean(rfp, axis=(1, 2)) # mean over x,y for each z

    # plot rfp_z (focus check) as a heatmap
    plt.imshow(rfp_z.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Mean RFP intensity')
    plt.xlabel('Frame')
    plt.ylabel('Z-slice')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # save rfp_z as a CSV
    df = pd.DataFrame(rfp_z)
    df.to_csv(os.path.join(out_dir, 'focus_check.csv'), index=False, header=[f'Z{i}' for i in range(rfp_z.shape[1])])

def write_mip(out_dir, tif_dir, stack_range):
    """Combines multiple TIF files into a single output file using tifffile."""
    mip_path = os.path.join(out_dir,  'mip_' + tif_dir + '.tif')

    # tif_files = check_files(os.path.join(out_dir,'tif'), stack_range, 'tif')
    tif_files = check_files(os.path.join(out_dir,tif_dir), stack_range, 'tif')

    with tifffile.TiffWriter(mip_path, bigtiff=True, imagej=True) as tif:
        for tif_file in tif_files:
            stack = tifffile.imread(tif_file)
            if len(stack.shape)==3:
                stack = stack[np.newaxis,:,:,:]
            gfp, rfp = np.max(stack, axis=0).astype(np.float32)
            tif.write(np.stack([gfp, rfp], axis=0).clip(0, 4095).astype(np.uint16), contiguous=True)
            print(f'{os.path.basename(tif_file)} written to {os.path.basename(mip_path)}')

    mip = tifffile.imread(mip_path)
    gfp_means, rfp_means = np.mean(mip, axis=(2, 3)).T
    plt.subplot(211)
    plt.plot(gfp_means, c='C2')
    plt.xlim(0, len(mip))
    plt.title(f'Max GFP mean: index {np.argmax(gfp_means)}')
    plt.subplot(212)
    plt.plot(rfp_means, c='C3')
    plt.xlim(0, len(mip))
    plt.title(f'Max RFP mean: index {np.argmax(rfp_means)}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'means.png'))
    plt.close()

def make_rgb(frame, rmax, gmax, shape=(512, 512, 3)):
    """Creates an RGB image from a frame."""
    gfp, rfp = frame
    rgb = np.zeros(shape, np.ubyte)
    adjust = lambda frame, lo, hi: np.clip((frame.astype(np.float32) - lo) / (hi - lo), 0, 1)
    rgb[..., 0] = adjust(rfp, 0, rmax) * 255
    rgb[..., 1] = adjust(gfp, 0, gmax) * 255
    return rgb

def write_mp4(out_dir, tif_dir, rmax, gmax, fps, quality): # fps = 5/0.533
    """Creates an AVI file from a MIP TIF file."""
    mp4_path = os.path.join(out_dir , 'mip_' + tif_dir + '.mp4')
    # if not os.path.exists(mp4_path):
    mip = tifffile.imread(os.path.join(out_dir, 'mip_' + tif_dir + '.tif'))
    with imageio.get_writer(os.path.join(out_dir, 'mip_' + tif_dir + '.mp4'), fps=fps, codec='mjpeg', quality=quality, pixelformat='yuvj444p') as mp4:
        for frame in mip:
            # overlay frame number on top left of frame
            mp4.append_data(make_rgb(frame, rmax, gmax))

def main():
    """Main pipeline: create parameter DataFrame, save results."""
    input_nd2 = sys.argv[1]
    tif_dir = sys.argv[2]
    stack_length = sys.argv[3]
    num_frames = sys.argv[4] 
    fps = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    rmax = int(sys.argv[6]) if len(sys.argv) > 6 else 750
    gmax = int(sys.argv[7]) if len(sys.argv) > 7 else 100
    mp4_quality = int(sys.argv[8]) if len(sys.argv) > 8 else 5
    do_focus = bool(sys.argv[9]) if len(sys.argv) > 9 else False


    stack_range = range(num_frames) #get_range(input_nd2, stack_length)
    out_dir = os.path.splitext(input_nd2)[0]
    
    if do_focus:
        check_focus(out_dir, tif_dir, stack_range)

    ##### write_alignment(out_dir, stack_range) # no longer performed during MIP
    write_mip(out_dir, tif_dir, stack_range)
    write_mp4(out_dir, tif_dir, rmax, gmax, fps, mp4_quality)

if __name__ == '__main__':
    main()
