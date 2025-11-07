#!/bin/bash
#SBATCH -n 1
#SBATCH -p ou_bcs_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=/home/albert_w/outputs/slurm-%j.out

module load miniforge
conda activate eval
python -u /home/albert_w/scripts/eval.py $1
