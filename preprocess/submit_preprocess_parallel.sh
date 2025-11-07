#!/bin/bash
#SBATCH --job-name=preprocess_parallel
#SBATCH --output=/home/munib/OUTPUT/g5ht-pipeline/preprocess/%x_%j.out
#SBATCH --cpus-per-task=16       # allocate however many parallel workers you want
#SBATCH --mem=32G
#SBATCH --time=01:00:00

module load miniforge
conda activate g5ht-pipeline

ND2="$1"

# --- determine number of frames automatically ---
echo "Determining number of frames for $ND2 ..."

N_FRAMES=$(python - <<PYCODE
from nd2reader import ND2Reader
path = "$ND2"
STACK_LENGTH=41
try:
    with ND2Reader(path) as f:
        print(f.metadata['num_frames'] // STACK_LENGTH)
except Exception as e:
    print(1)
PYCODE
)

echo "Detected frames: $N_FRAMES"


# Compute last index (0-based)
END_IDX=$((N_FRAMES - 1))
N_CPUS=$SLURM_CPUS_PER_TASK

# --- run parallel preprocessing ---
python -u /home/munib/CODE/g5ht-pipeline/preprocess/preprocess_parallel.py "$ND2" 0 "$END_IDX" "$N_CPUS"