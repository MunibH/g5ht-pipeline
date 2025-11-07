#!/bin/bash
#SBATCH --job-name=mip
#SBATCH --output=/home/munib/OUTPUT/g5ht-pipeline/mip/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=2GB
#SBATCH --time=0:20:00

module load miniforge
conda activate g5ht-pipeline
# make conda env: `conda create env -f environment.yaml`

echo "running mip.py for $1"
python -u /home/munib/CODE/g5ht-pipeline/mip/mip.py $1

echo "Finished"