#!/bin/bash
#SBATCH --job-name=submit-preprocess
#SBATCH --output=/home/munib/OUTPUT/g5ht-pipeline/preprocess/submit_preprocess_%j.out
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB

ND2="$1"
MAX_JOBS_PER_BATCH=400
CHUNK_CONCURRENCY=399

module load miniforge
conda activate g5ht-pipeline

# --- check file path ---
if [ ! -f "$ND2" ]; then
    echo "ND2 file not found: $ND2"
    exit 1
fi

# --- determine total frames automatically ---
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

# --- compute batches ---
N_BATCHES=$(( (N_FRAMES + MAX_JOBS_PER_BATCH - 1) / MAX_JOBS_PER_BATCH ))
echo "Will submit $N_BATCHES batch(es)."

for ((b=0; b<N_BATCHES; b++)); do
    start=$(( b * MAX_JOBS_PER_BATCH ))
    end=$(( (start + MAX_JOBS_PER_BATCH - 1) < (N_FRAMES - 1) ? (start + MAX_JOBS_PER_BATCH - 1) : (N_FRAMES - 1) ))

    echo "Submitting batch $((b+1))/$N_BATCHES: frames $start-$end"
    jid=$(sbatch --parsable --array=${start}-${end}%${CHUNK_CONCURRENCY} /home/munib/CODE/g5ht-pipeline/preprocess/preprocess_array.sh "$ND2")
    echo "  Submitted job ID: $jid"

    echo "Waiting for job $jid to finish..."
    while squeue -j "$jid" >/dev/null 2>&1 && squeue -j "$jid" | grep -q "$jid"; do
        sleep 60
    done
    echo "  Job $jid finished. Continuing..."
done

echo "All batches completed for $ND2"
