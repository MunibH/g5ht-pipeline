#!/bin/bash
#SBATCH --job-name=batch-mip
#SBATCH --output=/home/munib/OUTPUT/g5ht-pipeline/mip/batch_mip_%j.out
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB

LIST_FILE="/home/munib/CODE/g5ht-pipeline/mip/datasets_to_mip.txt"
MAX_ACTIVE=1  # how many datasets to run concurrently, NEEDS TESTING, but should be many at once is fine

ND2_LIST=$(cat "$LIST_FILE")

for ND2 in "${ND2_LIST[@]}"; do
    echo "Launching preprocessing for: $ND2"
    jid=$(sbatch --parsable /home/munib/CODE/g5ht-pipeline/mip/mip.sh "$ND2")
    echo "  Submitted job ID $jid"

    # Throttle total concurrent ND2 jobs
    while (( $(squeue -u $USER | grep -c mip) >= MAX_ACTIVE )); do
        echo "  Waiting for running jobs to finish..."
        sleep 300
    done
done

echo "All ND2 datasets submitted."