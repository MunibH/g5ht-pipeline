#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=512MB
#SBATCH --time=0:05:00
#SBATCH --output=/home/albert_w/outputs/slurm-%j.out

python -u /home/albert_w/scripts/quantify.py $1
