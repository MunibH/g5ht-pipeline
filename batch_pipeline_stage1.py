#!/usr/bin/env python
"""Batch pipeline script that processes all UNPROCESSED datasets from datasets.txt.

Only performs up until drift estimation (step 5). If output of step 5 (z_selection.csv) already exists, it will skip that dataset unless --force is used.

Usage:
    uv run python /home/munib/code/g5ht-pipeline/batch_pipeline_stage1.py                    # process all unprocessed datasets
    uv run python /home/munib/code/g5ht-pipeline/batch_pipeline_stage1.py --force            # re-process even if outputs exist
    uv run python /home/munib/code/g5ht-pipeline/batch_pipeline_stage1.py --steps 1 2 3 4 5  # run only specific steps
    uv run python /home/munib/code/g5ht-pipeline/batch_pipeline_stage1.py --dry-run          # list datasets without processing
    
Running with tmux:

tmux (Terminal Multiplexer) creates a persistent session on the server. 
You can "detach" from it, go home, and "re-attach" later to see the exact same screen.
Start a session:
    tmux new -s pipeline
Run your code:
    uv run python /home/munib/code/g5ht-pipeline/batch_pipeline_stage1.py
Detach: Press Ctrl+B, then let go and press D. 
You can now safely close your terminal or turn off your computer.
Re-attach later:
    tmux attach -t pipeline
Once done running:
    Press Ctrl+B, then let go and press D to detach again.
    To kill the session:
        tmux kill-session -t pipeline
"""

import argparse
import importlib
import os
import sys
import traceback
from datetime import datetime

import utils

# pipeline modules
import shear_correct
import get_channel_alignment
import median_channel_alignment
import apply_channel_alignment
import bleach_correct
import mip
import drift_estimation

NOISE_PTH = '/home/munib/code/g5ht-pipeline/noise/noise_111125.tif'
DATASETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets.txt')


def is_already_processed(nd2_path):
    """Check if a dataset has already been fully processed (drift estimation output exists)."""
    pth = os.path.splitext(nd2_path)[0]
    return os.path.exists(os.path.join(pth, 'z_selection.csv'))


def process_dataset(nd2_path, steps, force=False):
    """Run stage 1 of the pipeline on a single nd2 file.

    Args:
        nd2_path: Path to the .nd2 file.
        steps: Set of step numbers to run (1-5).
        force: If True, run even if outputs already exist.
    """
    print(f"\n{'='*80}")
    print(f"Processing: {nd2_path}")
    print(f"{'='*80}\n")

    if not os.path.exists(nd2_path):
        print(f"Skipping (not found): {nd2_path}")
        return False

    if not force and is_already_processed(nd2_path):
        print(f"Skipping (already processed): {nd2_path}")
        return True

    INPUT_ND2_PTH = nd2_path
    INPUT_ND2 = os.path.basename(nd2_path)
    date_str = INPUT_ND2.split('_')[0].split('-')[1]

    OUT_DIR = utils.get_output_dir(INPUT_ND2_PTH)

    STACK_LENGTH = 41 if 'immo' not in INPUT_ND2 else 122

    # z-slice range depends on recording date
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    if date_obj < datetime(2025, 12, 1):
        z2keep = (0, STACK_LENGTH - 2)
    else:
        z2keep = (2, STACK_LENGTH)

    # get noise stack and metadata from nd2
    noise_stack = utils.get_noise_stack(NOISE_PTH, STACK_LENGTH)
    num_frames, height, width, num_channels = utils.get_range_from_nd2(INPUT_ND2_PTH, stack_length=STACK_LENGTH)
    beads_alignment_file = utils.get_beads_alignment_file(INPUT_ND2_PTH)

    start_index = "0"
    end_index = str(num_frames - 1)
    ncpu = str(utils.get_optimal_cpu_count())

    # ---- Step 1: Shear Correction ----
    if 1 in steps:
        print("\n--- Step 1: Shear Correction ---")
        importlib.reload(shear_correct)
        skip_shear_correction = False
        sys.argv = ["", INPUT_ND2_PTH, start_index, end_index, NOISE_PTH, STACK_LENGTH, ncpu,
                    num_frames, height, width, num_channels, z2keep, skip_shear_correction]
        shear_correct.main()

    # ---- Step 2a: Get Channel Alignment ----
    if 2 in steps:
        print("\n--- Step 2a: Get Channel Alignment ---")
        importlib.reload(get_channel_alignment)
        importlib.reload(median_channel_alignment)

        every_other = True
        if beads_alignment_file is not None:
            align_with_beads = True
            every_other = False
            num_frames_beads, _, _, _ = utils.get_range_from_nd2(beads_alignment_file, stack_length=STACK_LENGTH)
            sys.argv = ["", beads_alignment_file, start_index, end_index, NOISE_PTH, STACK_LENGTH, ncpu,
                        num_frames_beads, height, width, num_channels, every_other, align_with_beads]
        else:
            align_with_beads = False
            sys.argv = ["", INPUT_ND2_PTH, start_index, end_index, NOISE_PTH, STACK_LENGTH, ncpu,
                        num_frames, height, width, num_channels, every_other, align_with_beads]

        get_channel_alignment.main()
        median_channel_alignment.main()

        # ---- Step 2b: Apply Channel Alignment ----
        print("\n--- Step 2b: Apply Channel Alignment ---")
        importlib.reload(apply_channel_alignment)

        if beads_alignment_file is not None:
            align_with_beads = True
            num_frames_beads, _, _, _ = utils.get_range_from_nd2(beads_alignment_file, stack_length=STACK_LENGTH)
            sys.argv = ["", INPUT_ND2_PTH, start_index, end_index, NOISE_PTH, STACK_LENGTH, ncpu,
                        num_frames, height, width, num_channels, align_with_beads, beads_alignment_file]
        else:
            align_with_beads = False
            sys.argv = ["", INPUT_ND2_PTH, start_index, end_index, NOISE_PTH, STACK_LENGTH, ncpu,
                        num_frames, height, width, num_channels, align_with_beads]

        apply_channel_alignment.main()

    # ---- Step 3: Bleach Correction ----
    if 3 in steps:
        print("\n--- Step 3: Bleach Correction ---")
        importlib.reload(bleach_correct)

        PTH = os.path.splitext(INPUT_ND2_PTH)[0]
        REG_DIR = 'channel_aligned'
        channels = 1  # 0-gfp, 1-rfp
        method = 'block'
        mode = 'total'
        output_dir = os.path.join(PTH, 'bleach_corrected')

        bleach_correct.correct_bleaching(os.path.join(PTH, REG_DIR), output_dir=output_dir,
                                         channels=channels, method=method, fbc=0.04, intensity_mode=mode)

    # ---- Step 4: MIP ----
    if 4 in steps:
        print("\n--- Step 4: MIP ---")
        importlib.reload(mip)

        framerate = 8
        tif_dir = 'bleach_corrected'
        rmax = 850
        gmax = 150
        mp4_quality = 10
        do_focus = True
        sys.argv = ["", INPUT_ND2_PTH, tif_dir, STACK_LENGTH, num_frames, framerate, rmax, gmax, mp4_quality, do_focus]
        mip.main()

    # ---- Step 5: Drift Estimation ----
    if 5 in steps:
        print("\n--- Step 5: Drift Estimation ---")
        importlib.reload(drift_estimation)

        tif_dir = 'bleach_corrected'
        sys.argv = ["", INPUT_ND2_PTH, tif_dir, STACK_LENGTH, num_frames]
        drift_estimation.main()

    print(f"\nCompleted: {nd2_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Batch processing pipeline for g5ht datasets.')
    parser.add_argument('--force', action='store_true',
                        help='Re-process datasets even if outputs already exist.')
    parser.add_argument('--steps', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        help='Pipeline steps to run (1=shear, 2=channel_align, 3=bleach, 4=mip, 5=drift). Default: all.')
    parser.add_argument('--dry-run', action='store_true',
                        help='List datasets without processing.')
    parser.add_argument('--datasets', default=DATASETS_PATH,
                        help='Path to datasets.txt file.')
    args = parser.parse_args()

    steps = set(args.steps)
    nd2_paths = utils.parse_datasets(args.datasets, section='UNPROCESSED')

    if not nd2_paths:
        print("No unprocessed datasets found in datasets.txt")
        return

    print(f"Found {len(nd2_paths)} unprocessed dataset(s):")
    for p in nd2_paths:
        status = ""
        if not os.path.exists(p):
            status = " [NOT FOUND]"
        elif is_already_processed(p):
            status = " [ALREADY DONE]"
        print(f"  {p}{status}")

    if args.dry_run:
        return

    print(f"\nRunning steps: {sorted(steps)}")
    print(f"Force re-processing: {args.force}\n")

    succeeded = []
    failed = []
    skipped = []

    for i, nd2_path in enumerate(nd2_paths):
        print(f"\n[{i+1}/{len(nd2_paths)}]")
        try:
            result = process_dataset(nd2_path, steps, force=args.force)
            if result:
                succeeded.append(nd2_path)
            else:
                skipped.append(nd2_path)
        except Exception as e:
            print(f"\nERROR processing {nd2_path}: {e}")
            traceback.print_exc()
            failed.append(nd2_path)
            continue

    # summary
    print(f"\n{'='*80}")
    print("BATCH SUMMARY")
    print(f"{'='*80}")
    print(f"  Succeeded: {len(succeeded)}")
    print(f"  Skipped:   {len(skipped)}")
    print(f"  Failed:    {len(failed)}")
    if failed:
        print("\nFailed datasets:")
        for p in failed:
            print(f"  {p}")


if __name__ == '__main__':
    main()
