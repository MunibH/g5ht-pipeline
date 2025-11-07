#!/bin/bash
#SBATCH --job-name=batch-segment
#SBATCH --output=/home/munib/OUTPUT/g5ht-pipeline/segment/%j.out
#SBATCH -n 1
#SBATCH -p ou_bcs_normal
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G

module load miniforge
conda activate eval
python -u /home/munib/CODE/g5ht-pipeline/segment/eval.py $1
