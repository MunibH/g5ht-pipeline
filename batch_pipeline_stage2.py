#!/usr/bin/env python
"""Batch pipeline stage 2: orient + warp for all UNPROCESSED datasets.

Requires stage 1 to be completed (spline.json, dilated.tif, bleach_corrected/ must exist)
and orient_nose.csv to be created (via the interactive nose annotator in the pipeline.ipynb).

Usage:
    uv run python batch_pipeline_stage2.py                        # process all unprocessed datasets
    uv run python batch_pipeline_stage2.py --force                # re-process even if outputs exist
    uv run python batch_pipeline_stage2.py --steps 8 9            # run only specific steps
    uv run python batch_pipeline_stage2.py --dry-run              # list datasets without processing
    uv run python batch_pipeline_stage2.py | tee "/home/munib/code/g5ht-pipeline/processing_logs/stage2_log_$(date +'%Y%m%d_%H%M%S').log"
    
Running with tmux:

tmux (Terminal Multiplexer) creates a persistent session on the server. 
You can "detach" from it, go home, and "re-attach" later to see the exact same screen.
Start a session:
    tmux new -s pipeline
Run your code:
    uv run python batch_pipeline_stage2.py
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

from tqdm import tqdm

import utils

import orient   # 8
import warp     # 9

DATASETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets.txt')

# stage 1 outputs that must exist before stage 2 can run
STAGE1_REQUIRED = {
    'spline.json': 'Step 7 (spline)',
    'dilated.tif': 'Step 7 (spline)',
    'bleach_corrected': 'Step 3 (bleach correction)',
}


def check_stage1(pth):
    """Return list of missing stage 1 outputs."""
    missing = []
    for name, step in STAGE1_REQUIRED.items():
        full = os.path.join(pth, name)
        if not os.path.exists(full):
            missing.append(f'{name} ({step})')
    return missing


def is_already_processed(pth):
    """Check if stage 2 is already complete (warped directory with files)."""
    warped_dir = os.path.join(pth, 'warped')
    return os.path.isdir(warped_dir) and len(os.listdir(warped_dir)) > 0


def process_dataset(nd2_path, steps, force=False):
    """Run stage 2 (orient + warp) on a single dataset.

    Returns True if processing succeeded or was skipped (already done),
    False if skipped due to missing prerequisites.
    """
    print(f"\n{'='*80}")
    print(f"Processing: {nd2_path}")
    print(f"{'='*80}\n")

    if not os.path.exists(nd2_path):
        print(f"  SKIP — nd2 file not found: {nd2_path}")
        return False

    INPUT_ND2 = os.path.basename(nd2_path)
    pth = os.path.splitext(nd2_path)[0]
    STACK_LENGTH = 41 if 'immo' not in INPUT_ND2 else 122

    if not force and is_already_processed(pth):
        print(f"  SKIP — already processed (warped/ exists): {nd2_path}")
        return True

    # verify stage 1 completion
    missing = check_stage1(pth)
    if missing:
        print(f"  SKIP — stage 1 incomplete. Missing outputs:")
        for m in missing:
            print(f"    - {m}")
        return False

    # verify orient_nose.csv exists
    orient_csv = os.path.join(pth, 'orient_nose.csv')
    if not os.path.exists(orient_csv):
        print(f"  SKIP — orient_nose.csv not found at {orient_csv}")
        print(f"         Run the nose annotator in pipeline.ipynb first.")
        return False

    num_frames, _, _, _ = utils.get_range_from_nd2(nd2_path, stack_length=STACK_LENGTH)

    # ---- Step 8: Orient ----
    if 8 in steps:
        print("\n--- Step 8: Orient ---")
        importlib.reload(orient)
        spline_pth = os.path.join(pth, 'spline.json')
        # orient.py reads orient_nose.csv automatically when no nose args provided
        sys.argv = ["", spline_pth]
        orient.main()

    # ---- Step 9: Warp ----
    if 9 in steps:
        print("\n--- Step 9: Warp ---")
        importlib.reload(warp)
        for i in tqdm(range(num_frames), desc='Warping'):
            sys.argv = ["", pth, i]
            warp.main()

    print(f"\nCompleted: {nd2_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Batch pipeline stage 2: orient + warp.')
    parser.add_argument('--force', action='store_true',
                        help='Re-process datasets even if outputs already exist.')
    parser.add_argument('--steps', nargs='+', type=int, default=[8, 9],
                        help='Steps to run (8=orient, 9=warp). Default: both.')
    parser.add_argument('--dry-run', action='store_true',
                        help='List datasets and readiness without processing.')
    parser.add_argument('--datasets', default=DATASETS_PATH,
                        help='Path to datasets.txt file.')
    args = parser.parse_args()

    steps = set(args.steps)
    nd2_paths = utils.parse_datasets(args.datasets, section='UNPROCESSED')

    if not nd2_paths:
        print("No unprocessed datasets found in datasets.txt")
        return

    print(f"Found {len(nd2_paths)} unprocessed dataset(s):\n")
    for p in nd2_paths:
        pth = os.path.splitext(p)[0]
        status_parts = []
        if not os.path.exists(p):
            status_parts.append("NOT FOUND")
        else:
            missing = check_stage1(pth)
            if missing:
                status_parts.append("STAGE1 INCOMPLETE")
            if not os.path.exists(os.path.join(pth, 'orient_nose.csv')):
                status_parts.append("NO NOSE CSV")
            if is_already_processed(pth):
                status_parts.append("ALREADY DONE")
            if not status_parts:
                status_parts.append("READY")
        status = ", ".join(status_parts)
        print(f"  [{status}] {p}")

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

    print(f"\n{'='*80}")
    print("BATCH STAGE 2 SUMMARY")
    print(f"{'='*80}")
    print(f"  Succeeded: {len(succeeded)}")
    print(f"  Skipped:   {len(skipped)}")
    print(f"  Failed:    {len(failed)}")
    if failed:
        print("\nFailed datasets:")
        for p in failed:
            print(f"  {p}")
    if skipped:
        print("\nSkipped datasets:")
        for p in skipped:
            print(f"  {p}")


if __name__ == '__main__':
    main()
