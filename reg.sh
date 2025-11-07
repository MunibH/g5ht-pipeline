#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4GB
#SBATCH --time=0:10:00
#SBATCH --output=/home/albert_w/outputs/slurm-%A_%a.out
#SBATCH --array=0-1199

python -u /home/albert_w/scripts/reg.py $1 $SLURM_ARRAY_TASK_ID
