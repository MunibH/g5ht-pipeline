#!/bin/bash
#SBATCH --job-name=batch-all_parallel
#SBATCH --output=/home/munib/OUTPUT/g5ht-pipeline/preprocess/batch_preprocess_parallel_%j.out
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB

LIST_FILE="/home/munib/CODE/g5ht-pipeline/preprocess/datasets_to_preprocess.txt"
MAX_ACTIVE=1  # how many ND2s to run concurrently, for parallel processing might be able to make this more than one (NEEDS TESTING THO)

# SEARCH_DIR="/home/munib/orcd/pool/g5-HT-free"
# if [ -f "$LIST_FILE" ]; then
#     mapfile -t ND2_LIST < "$LIST_FILE"
# else
#     mapfile -t ND2_LIST < <(find "$SEARCH_DIR" -type f -name "*.nd2" | sort)
# fi

ND2_LIST=$(cat "$LIST_FILE")

for ND2 in "${ND2_LIST[@]}"; do
    echo "Launching preprocessing for: $ND2"
    jid=$(sbatch --parsable /home/munib/CODE/g5ht-pipeline/preprocess/submit_preprocess_parallel.sh "$ND2")
    echo "  Submitted job ID $jid"

    # Throttle total concurrent ND2 jobs
    while (( $(squeue -u $USER | grep -c preprocess_parallel) >= MAX_ACTIVE )); do
        echo "  Waiting for running jobs to finish..."
        sleep 300
    done
done

echo "All ND2 datasets submitted."