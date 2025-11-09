# INSTALLATION

### CLONE REPOSITORY
`git clone [repo url]`

### CREATING CONDA ENVIRONMENTS

Assuming you already have anaconda/miniconda/miniforge installed...

#### On MIT's engaging cluster
-  `module load miniforge`
-  `conda create env -f environment.yml`
    - created `g5ht-pipeline` environment, used for all steps other than segmentation
-  `conda create env -f segment/segment_torch_environment.yml` or `segment/segment_environment.yml`
    - creates `segment-torch` or `eval` conda environments, used for segmentation
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

## 3. SEGMENTATION

Uses a deeplabv3p_resnet50 network to segement the worm's body from background. 

There are multiple ways you can do this on Engaging. The pytorch framework is newer, and I mostly wrote it since it's the only way I could get something to work on a local windows pc.
The two frameworks perform identically, so up to you which one you want to use.

__WITH PYTORCH FRAMEWORK__
1. install `segement-torch` conda environment (see above)
2. make a text file with each line containing the fullfile to a .nd2 that needs segmentation (`mip.tif` is the thing that will be used for segmentation)
    - see `datasets_to_segment.txt` as an example
3. change `LIST_FILE` in `batch_segment_torch.sh` to the fullfile to the text file of datasets to preprocess
4. change paths in `batch_segment_torch.sh` and `segment_torch.sh`
5. in `segment_torch.py` update the `CHECKPOINT` variable file path to your `.pth` weights file:
6. Run `sbatch /home/munib/CODE/g5ht-pipeline/segment/batch_segment_torch.sh`
   - alternatively, can directly call `sbatch /home/munib/CODE/g5ht-pipeline/segment/segment_torch.sh "pth/to/mip/"`

__WITH TENSORFLOW FRAMEWORK__

TODO

#### OUTPUTS

- in same output subdirectory from step1:
    - saves `label.tif`, containing a mask for each frame


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