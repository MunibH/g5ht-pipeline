#!/bin/bash
#SBATCH -p ou_bcs_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/albert_w/outputs/slurm-%j.out

module load miniforge
conda activate g5ht-pipeline
python -u /home/albert_w/scripts/make-noise/make_noise.py
