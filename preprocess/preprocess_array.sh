#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=2GB
#SBATCH --time=0:10:00
#SBATCH --output=/home/munib/OUTPUT/g5ht-pipeline/preprocess/%A/%a.out
#SBATCH --array=0-0

# --array will get overwritten

module load miniforge
conda activate g5ht-pipeline
# make conda env: `conda create env -f environment.yaml`

ND2=$1

python -u /home/munib/CODE/g5ht-pipeline/preprocess/preprocess.py "$ND2" "$SLURM_ARRAY_TASK_ID"