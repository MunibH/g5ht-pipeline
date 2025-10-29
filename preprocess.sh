#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=2GB
#SBATCH --time=0:10:00
#SBATCH --output=/home/albert_w/outputs/slurm-%A_%a.out
#SBATCH --array=0-200

python -u /home/albert_w/scripts/preprocess.py $1 $SLURM_ARRAY_TASK_ID
