#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=2GB
#SBATCH --time=0:10:00
#SBATCH --output=/home/albert_w/outputs/slurm-%j.out

python -u /home/albert_w/scripts/mip.py $1
