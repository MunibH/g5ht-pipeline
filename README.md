# INSTALLATION

### CLONE REPOSITORY

```bash
git clone [repo url]
```

### CREATING CONDA ENVIRONMENTS

Assuming you already have Anaconda/Miniconda/Miniforge installed.

#### On MIT's Engaging Cluster

* Load Miniforge:

```bash
module load miniforge
```

* Create main environment:

```bash
conda create env -f environment.yml
```

* Creates `g5ht-pipeline` environment, used for all steps except segmentation.
* Create segmentation environment:

```bash
conda create env -f segment/segment_torch_environment.yml  # PyTorch framework
conda create env -f segment/segment_environment.yml        # TensorFlow framework
```

#### On Local PC

```bash
conda create env -f environment.yml
```

---

# STEPS

## 1. PREPROCESS DATA

This step denoises data, stores frame stacks as `.tif`, performs shear correction, and aligns red and green channels.

### BATCH PROCESS MULTIPLE EXPERIMENTS WITH JOB ARRAY

1. Create a text file listing `.nd2` files (one per line). Example: `datasets_to_preprocess.txt`.
2. Update `LIST_FILE` in `batch_all_preprocess.sh` to this file.
3. Update `noise_path` in `preprocess.py` to point to your `noise.tif`.
4. Run:

```bash
sbatch /home/munib/CODE/g5ht-pipeline/preprocess/batch_all_preprocess.sh
```

### JUST PROCESS ONE EXPERIMENT WITH JOB ARRAY

1. Update `noise_path` in `preprocess.py`.
2. Run:

```bash
sbatch /home/munib/CODE/g5ht-pipeline/preprocess/submit_preprocess.sh "path/to/.nd2"
```

### BATCH PROCESS MULTIPLE EXPERIMENTS WITH PARALLELIZATION

1. Create a text file listing `.nd2` files.
2. Update `LIST_FILE` in `batch_all_preprocess_parallel.sh`.
3. Update `noise_path` in `preprocess_parallel.py`.
4. Run:

```bash
sbatch /home/munib/CODE/g5ht-pipeline/preprocess/batch_all_preprocess_parallel.sh
```

### JUST PROCESS ONE EXPERIMENT WITH PARALLELIZATION

1. Update `noise_path` in `preprocess_parallel.py`.
2. Run:

```bash
sbatch /home/munib/CODE/g5ht-pipeline/preprocess/submit_preprocess_parallel.sh "path/to/.nd2"
```

> **Note:** Local PC preprocessing is under development (Windows 11).

#### OUTPUTS

| Directory | Contents                     |
| --------- | ---------------------------- |
| `tif/`    | Frame stacks as `.tif`       |
| `txt/`    | Channel alignment parameters |

---

## 2. MAX INTENSITY PROJECTION

Generates max intensity projections, checks alignment and focus.

### BATCH PROCESS MULTIPLE EXPERIMENTS

1. Create a text file with `.nd2` paths (example: `datasets_to_mip.txt`).
2. Update `LIST_FILE` in `batch_all_mip.sh`.
3. Update `template.txt` path in `mip.py`:

```python
parameter_object.ReadParameterFile(os.path.join('/home/munib/CODE/g5ht-pipeline','template.txt'))
```

4. Run:

```bash
sbatch /home/munib/CODE/g5ht-pipeline/mip/batch_all_mip.sh
```

### JUST PROCESS ONE EXPERIMENT

1. Update `template.txt` path in `mip.py`.
2. Run:

```bash
sbatch /home/munib/CODE/g5ht-pipeline/mip/mip.sh "path/to/.nd2"
```

#### OUTPUTS

| File        | Description                    |
| ----------- | ------------------------------ |
| `align.png` | Alignment image                |
| `align.txt` | Alignment parameters           |
| `focus.png` | Focus check                    |
| `means.png` | Mean intensity image           |
| `mip.mp4`   | Max intensity projection video |
| `mip.tif`   | Max intensity projection stack |

---

## 3. SEGMENTATION

Uses `deeplabv3p_resnet50` to segment worm body from background.

> **Tip:** PyTorch framework is recommended for local Windows PCs. TensorFlow also works; both produce similar results.

### PYTORCH FRAMEWORK

1. Install `segment-torch` conda environment.
2. Create a text file listing `.nd2` files (`mip.tif` used for segmentation).
3. Update `LIST_FILE` in `batch_segment_torch.sh`.
4. Update paths in `batch_segment_torch.sh` and `segment_torch.sh`.
5. Update `CHECKPOINT` in `segment_torch.py` with `.pth` weights path.
6. Run:

```bash
sbatch /home/munib/CODE/g5ht-pipeline/segment/batch_segment_torch.sh
```

* Or directly:

```bash
sbatch /home/munib/CODE/g5ht-pipeline/segment/segment_torch.sh "path/to/mip/"
```

### TENSORFLOW FRAMEWORK

* TODO

#### OUTPUTS

* Saves `label.tif` containing segmentation masks in the same output folder from Step 1.

---

# TODO

* Move `noise_path` in `preprocess.py` to `batch_all_processes.sh`.
* Validate creation of all `.tif` and `.txt` files; generate summary.
* Develop local PC pipeline.
* Create full Engaging batch pipeline (all steps in one script).
* Incorporate NIR camera processing: [BehaviorDataNIR.jl](https://github.com/flavell-lab/BehaviorDataNIR.jl/tree/main)
* Build a simple GUI to preview `.nd2` or `.tif` images before/after preprocessing, including:

  * Preprocessing effects
  * Position data
  * Behavioral data
