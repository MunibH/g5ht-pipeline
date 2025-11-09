#!/bin/bash
#SBATCH --job-name=segment
#SBATCH --output=/home/munib/OUTPUT/g5ht-pipeline/segment/segment_%j.out
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB

echo "Segmenting worm: $1"

module load miniforge
conda activate segment-torch
python -u /home/munib/CODE/g5ht-pipeline/segment/segment_torch.py $1

echo "Segmentation completed for $1"
