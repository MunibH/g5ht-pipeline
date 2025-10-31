#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=2GB
#SBATCH --time=0:10:00
#SBATCH --output=/home/albert_w/outputs/slurm-%A_%a.out
#SBATCH --array=0-1200

python -u /home/albert_w/scripts/preprocess.py $1 $SLURM_ARRAY_TASK_ID

# $1 is the nd2 fullfile
# $SLURM_ARRAY_TASK_ID must be the frame index???

# the above job is requesting 1 node, 1 task, 8 cores, 2GB of memory (is that per core?), for 10 mins, and a job array with index values between 0 and 1200