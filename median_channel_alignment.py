import glob
import os
import sys
import numpy as np
import pandas as pd
import itk
import warnings; warnings.filterwarnings('ignore', category=UserWarning, module='itk')

def extract_frame_from_path(path):
    # get the basename (filename with extension): '13845.tif'
    filename_with_ext = os.path.basename(path)
    filename_without_ext = os.path.splitext(filename_with_ext)[0]
    # convert the resulting string to an integer for numerical sorting
    # this step is needed for sorting '0837' correctly before '1675', etc.
    return int(filename_without_ext)

def check_files(align_dir, stack_range, extension):
    """Checks if all the expected files in the stack range are in the directory"""
    existing_files = set(glob.glob(os.path.join(align_dir, f'*.{extension}')))
    expected_files = {os.path.join(align_dir, f'{i:04d}.{extension}') for i in stack_range}
    if missing_files := expected_files - existing_files:
        stacks = sorted([int(os.path.splitext(os.path.basename(f))[0]) for f in missing_files])
        # raise FileNotFoundError(f"Missing .{extension} files: {','.join([str(i) for i in stacks])}")
        print(f"Warning: Missing .{extension} files: {','.join([str(i) for i in stacks])}")
        expected_files = existing_files

    return sorted(expected_files, key=extract_frame_from_path)

def get_transform_params(file, parameter_object):
    """Extracts transform parameters from a text file.
    
    Returns:
        dict with CenterOfRotationPoint, Size, and TransformParameters
    """
    parameter_object.ReadParameterFile(file)
    param_map = parameter_object.GetParameterMap(0)
    return {
        'CenterOfRotationPoint': np.array(param_map['CenterOfRotationPoint'], float),
        'Size': np.array(param_map['Size'], int),
        'TransformParameters': np.array(param_map['TransformParameters'], float)
    }

def write_median_alignment(align_dir, stack_range):
    """Creates an alignment file from the parameter DataFrame.
    
    Reads all .txt registration files, extracts CenterOfRotationPoint, Size, 
    and TransformParameters, saves them to params.csv, computes median values,
    and writes the final alignment file.
    """
    align_path = os.path.join(align_dir, 'chan_align.txt')
    if not os.path.exists(align_path):
        parameter_object = itk.ParameterObject.New()
        alignment_files = check_files(os.path.join(align_dir,'txt'), stack_range, 'txt')
        
        # Extract parameters from all registration files
        all_params = []
        for file in alignment_files:
            try:
                params = get_transform_params(file, parameter_object)
            except:
                'a'
            all_params.append(params)
        
        # Get the number of transform parameters (6 for 3D EulerTransform)
        n_transform_params = len(all_params[0]['TransformParameters'])
        n_center_params = len(all_params[0]['CenterOfRotationPoint'])
        n_size_params = len(all_params[0]['Size'])
        
        # Build DataFrame with all parameters
        columns = []
        columns.extend([f'CenterOfRotationPoint_{i}' for i in range(n_center_params)])
        columns.extend([f'Size_{i}' for i in range(n_size_params)])
        columns.extend([f'TransformParameter_{i}' for i in range(n_transform_params)])
        
        data = np.zeros((len(alignment_files), len(columns)))
        for i, params in enumerate(all_params):
            idx = 0
            for j in range(n_center_params):
                data[i, idx] = params['CenterOfRotationPoint'][j]
                idx += 1
            for j in range(n_size_params):
                data[i, idx] = params['Size'][j]
                idx += 1
            for j in range(n_transform_params):
                data[i, idx] = params['TransformParameters'][j]
                idx += 1
        
        param_df = pd.DataFrame(data, columns=columns)
        param_df.to_csv(os.path.join(align_dir, 'chan_align_params.csv'), index=False)
        
        # Compute median values
        median_center = np.median(np.array([p['CenterOfRotationPoint'] for p in all_params]), axis=0)
        median_size = np.median(np.array([p['Size'] for p in all_params]), axis=0).astype(int)
        median_transform = np.median(np.array([p['TransformParameters'] for p in all_params]), axis=0)
        
        # Load template and fill in median values
        parameter_object.ReadParameterFile(os.path.join(r'C:\Users\munib\POSTDOC\CODE\g5ht-pipeline', 'template_3d.txt'))
        changed_param_map = parameter_object.GetParameterMap(0)
        changed_param_map['CenterOfRotationPoint'] = [f'{v:.15f}' for v in median_center]
        changed_param_map['Size'] = [str(v) for v in median_size]
        changed_param_map['TransformParameters'] = [f'{v:.15f}' for v in median_transform]
        parameter_object.SetParameterMap(0, changed_param_map)
        parameter_object.WriteParameterFile(parameter_object, align_path)

def main():
    """
    Main pipeline: process multiple frame indices in parallel.
    Usage:
        python preprocess.py <input_nd2> <start_index> <end_index> <noise_path> [n_workers]
    """
    input_nd2 = sys.argv[1]
    start_idx = int(sys.argv[2])
    end_idx = int(sys.argv[3])

    out_dir = os.path.splitext(input_nd2)[0]

    stack_range = range(0, end_idx + 1)
    align_dir = out_dir

    write_median_alignment(align_dir, stack_range)

if __name__ == "__main__":
    main()
