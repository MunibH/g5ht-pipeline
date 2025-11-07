# INSTALLATION

### CLONE REPOSITORY
`git clone [repo url]`

### CREATING CONDA ENVIRONMENT

Assuming you already have anaconda/miniconda/miniforge installed...

#### On MIT's engaging cluster
-  `module load miniforge`
-  `conda create env -f environment.yml`
#### On local PC
-  `conda create env -f environment.yml`

# STEPS
## 1. PREPROCESS DATA

This step denoises data, stores each frame stack as a tif, performs shear correction, and aligns red and green channels

There are multiple ways you can preprocess your data on Engaging:

__BATCH PROCESS MULTIPLE EXPERIMENTS WITH JOB ARRAY__
1. make a text file with each line containing the fullfile to a .nd2 that needs preprocessing
    - see `datasets_to_preprocess.txt` as an example
2. change `LIST_FILE` in `batch_all_preprocess.sh` to the fullfile to the text file of datasets to preprocess
3. change `noise_path` in  `preprocess.py` to point to your `noise.tif` file
3. Run `sbatch /home/munib/CODE/g5ht-pipeline/preprocess/batch_all_preprocess.sh`

__JUST PROCESS ONE EXPERIMENT WITH JOB ARRAY__
1. change `noise_path` in  `preprocess.py` to point to your `noise.tif` file
2. Run `sbatch /home/munib/CODE/g5ht-pipeline/preprocess/submit_preprocess.sh "path/to/.nd2"` 

__BATCH PROCESS MULTIPLE EXPERIMENTS WITH ONE JOB PER EXPERIMENT & PARALLELIZATION__
1. make a text file with each line containing the fullfile to a .nd2 that needs preprocessing
    - see `datasets_to_preprocess.txt` as an example
2. change `LIST_FILE` in `batch_all_preprocess_parellel.sh` to the fullfile to the text file of datasets to preprocess
3. change `noise_path` in  `preprocess_parallel.py` to point to your `noise.tif` file
3. Run `sbatch /home/munib/CODE/g5ht-pipeline/preprocess/batch_all_preprocess_parellel.sh`

__JUST PROCESS ONE EXPERIMENT WITH ONE JOB & PARALLELIZATION__
1. change `noise_path` in  `preprocess_parallel.py` to point to your `noise.tif` file
2. Run `sbatch /home/munib/CODE/g5ht-pipeline/preprocess/submit_preprocess_parallel.sh "path/to/.nd2"` 

If you want to preprocess your data on a local PC (TODO, developing on Windows 11 PC):


#### OUTPUTS

- makes a new folder in a directory with the same name as your experiment's `.nd2` file
    - stores each frame stack as a `.tif` in the `tif` subdirectory
    - stores each frame stack's channel alignment parameters in the `txt` subdirectory

## 2. MAX INTENSITY PROJECTION

Computes max intensity projections for each frame stack, checks alignment, checks focus

There are multiple ways you can do this on Engaging:

__BATCH PROCESS MULTIPLE EXPERIMENTS__
1. make a text file with each line containing the fullfile to a .nd2 that needs preprocessing
    - see `datasets_to_mip.txt` as an example
2. change `LIST_FILE` in `batch_all_mip.sh` to the fullfile to the text file of datasets to preprocess
3. in `mip.py` change the following line to point to your `template.txt`:
    - `parameter_object.ReadParameterFile(os.path.join('/home/munib/CODE/g5ht-pipeline','template.txt'))`
3. Run `sbatch /home/munib/CODE/g5ht-pipeline/mip/batch_all_mip.sh`

__JUST PROCESS ONE EXPERIMENT__
1. in `mip.py` change the following line to point to your `template.txt`:
    - `parameter_object.ReadParameterFile(os.path.join('/home/munib/CODE/g5ht-pipeline','template.txt'))`
2. Run `sbatch /home/munib/CODE/g5ht-pipeline/mip/mip.sh "pth/to/.nd2"`

#### OUTPUTS

- in same output subdirectory from step1:
    - saves `align.png`, `align.txt`, `focus.png`, `means.png`, `mip.mp4`, `mip.tif`

# TODO
- `noise_path` in `preprocess.py` should be pulled out and put in `batch_all_processes.sh`
- validation that all frames tifs and txts were created, and create a validation output (like a txt file saysing 'preprocessed' or something)
- is it possible to 
- write local pc pipeline
- make engaging pipeline full batch script (do all steps in one batch script)
- incorporate NIR cam processing
    -https://github.com/flavell-lab/BehaviorDataNIR.jl/tree/main
- simple gui that allows you to view .nd2 or .tif images (before or after preprocessing)
    - run simple preprocessing steps to see how it changes data
    - show position data
    - show other behavior data