#!/bin/bash
#SBATCH --job-name=batch-all
#SBATCH --output=/home/munib/OUTPUT/g5ht-pipeline/preprocess/batch_preprocess_%j.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB

# Choose how to find ND2 files
LIST_FILE="/home/munib/CODE/g5ht-pipeline/preprocess/datasets_to_preprocess.txt"

MAX_ACTIVE=1  # how many dataset jobs can be active at once

SEARCH_DIR="/home/munib/orcd/pool/g5-HT-free" # commented out searching for datasets to process for now
# # Prefer list file if it exists; otherwise scan directory
# if [ -f "$LIST_FILE" ]; then
#     ND2_LIST=$(cat "$LIST_FILE")
# else
#     ND2_LIST=$(find "$SEARCH_DIR" -type f -name "*.nd2" | sort)
# fi

ND2_LIST=$(cat "$LIST_FILE")

for ND2 in $ND2_LIST; do
    [ -z "$ND2" ] && continue
    echo "Launching submit_preprocess for: $ND2"

    jid=$(sbatch --parsable /home/munib/CODE/g5ht-pipeline/preprocess/submit_preprocess.sh "$ND2")
    echo "  Submitted dataset job $jid"

    # Throttle to avoid exceeding job submission limit
    while (( $(squeue -u $USER | grep -c "submit-preprocess") >= MAX_ACTIVE )); do
        echo "  Too many active dataset jobs — waiting 5 min..."
        sleep 300
    done
done

echo "All dataset preprocessing jobs submitted."
