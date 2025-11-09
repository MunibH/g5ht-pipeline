#!/bin/bash
#SBATCH --job-name=batch-segment
#SBATCH --output=/home/munib/OUTPUT/g5ht-pipeline/segment/batch_segment_%j.out
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB

LIST_FILE="/home/munib/CODE/g5ht-pipeline/segment/datasets_to_segment.txt"
MAX_ACTIVE=1  # how many worms to run concurrently, might be able to make this more than one (NEEDS TESTING THO)

# Read each line of LIST_FILE into an array element
mapfile -t ND2_LIST < "$LIST_FILE"

for ND2 in "${ND2_LIST[@]}"; do
    base="${ND2%.nd2}"     # strip .nd2 suffix, should be standard results output directory

    echo "Launching segmentation for: $base"

    jid=$(sbatch --parsable \
        /home/munib/CODE/g5ht-pipeline/segment/segment_torch.sh "$base")

    echo "  Submitted job ID $jid"

    while (( $(squeue -u "$USER" | grep -c segment) >= MAX_ACTIVE )); do
        sleep 100
    done
done

echo "All ND2 datasets submitted."
