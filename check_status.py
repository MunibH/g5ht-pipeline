"""
Check processing status of all datasets under a data root directory.

Walks every date subdirectory, finds .nd2 files, inspects their output
folders, and writes a CSV report with one row per dataset.

Usage:
    uv run python check_status.py                          # defaults
    uv run python check_status.py --data-root /path/to/data
    uv run python check_status.py --output status.csv
    uv run python check_status.py --skip-disk-usage        # faster
"""

import argparse
import json
import os
import os.path as osp
import re
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT = '/store1/shared/g5ht/data/'


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def find_datasets(data_root):
    """Walk date subdirectories and return (nd2_path, output_dir) for every .nd2 file.

    Date directories are identified by names matching 7-8 digits (e.g. 20260320).
    The output directory for an nd2 is the same path with the extension stripped.
    """
    datasets = []
    for entry in sorted(os.scandir(data_root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        if not re.fullmatch(r'\d{7,8}', entry.name):
            continue
        for f in sorted(os.scandir(entry.path), key=lambda e: e.name):
            if f.is_file() and f.name.endswith('.nd2'):
                nd2_path = f.path
                output_dir = osp.splitext(nd2_path)[0]
                datasets.append((nd2_path, output_dir))
    return datasets


# ---------------------------------------------------------------------------
# Check registry — each entry is (column_name, check_function).
#
# To add a new check, write a function with signature:
#     def my_check(nd2_path: str, output_dir: str) -> str | int | float | bool | None
# then append ('column_name', my_check) to CHECK_REGISTRY.
# ---------------------------------------------------------------------------

CHECK_REGISTRY = []


def register(column_name):
    """Decorator that registers a check function under *column_name*."""
    def decorator(fn):
        CHECK_REGISTRY.append((column_name, fn))
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dir_exists_and_nonempty(path):
    """Return True if *path* is a directory containing at least one entry."""
    if not osp.isdir(path):
        return False
    try:
        return next(os.scandir(path), None) is not None
    except PermissionError:
        return False


def _count_files(dirpath, ext=None):
    """Count files in *dirpath* (non-recursive). Optionally filter by extension."""
    if not osp.isdir(dirpath):
        return 0
    count = 0
    for entry in os.scandir(dirpath):
        if entry.is_file():
            if ext is None or entry.name.endswith(ext):
                count += 1
    return count


def _load_metadata(output_dir):
    """Load metadata.json from *output_dir*, returning dict or None."""
    meta_path = osp.join(output_dir, 'metadata.json')
    if not osp.isfile(meta_path):
        return None
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _parse_nd2_filename(nd2_path):
    """Extract date, strain, condition, worm_id from the nd2 filename.

    Filename convention (some fields optional):
        date-YYYYMMDD[_time-HHMM]_strain-{NAME}_condition-{COND}_worm{NNN}.nd2
    Falls back to the parent directory name for date.
    """
    basename = osp.splitext(osp.basename(nd2_path))[0]
    info = {}

    m = re.search(r'date-(\d{8})', basename)
    info['date'] = m.group(1) if m else osp.basename(osp.dirname(nd2_path))

    m = re.search(r'strain-([^_]+)', basename)
    info['strain'] = m.group(1) if m else None

    m = re.search(r'condition-([^_]+)', basename)
    info['condition'] = m.group(1) if m else None

    m = re.search(r'worm(\d+)', basename)
    info['worm_id'] = int(m.group(1)) if m else None

    return info


def _dir_size_gb(path):
    """Return total size of *path* in GiB using a recursive walk."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for fname in filenames:
            fp = osp.join(dirpath, fname)
            try:
                total += osp.getsize(fp)
            except OSError:
                pass
    return round(total / (1024 ** 3), 2)


# ---------------------------------------------------------------------------
# Identity / metadata checks
# ---------------------------------------------------------------------------

@register('dataset')
def _dataset_name(nd2_path, output_dir):
    return osp.splitext(osp.basename(nd2_path))[0]

@register('date')
def _date(nd2_path, output_dir):
    return _parse_nd2_filename(nd2_path)['date']

@register('strain')
def _strain(nd2_path, output_dir):
    return _parse_nd2_filename(nd2_path)['strain']

@register('condition')
def _condition(nd2_path, output_dir):
    return _parse_nd2_filename(nd2_path)['condition']

@register('worm_id')
def _worm_id(nd2_path, output_dir):
    return _parse_nd2_filename(nd2_path)['worm_id']


# ---------------------------------------------------------------------------
# ND2 file checks
# ---------------------------------------------------------------------------

@register('nd2_exists')
def _nd2_exists(nd2_path, output_dir):
    return osp.isfile(nd2_path)

@register('nd2_size_gb')
def _nd2_size(nd2_path, output_dir):
    if not osp.isfile(nd2_path):
        return None
    return round(osp.getsize(nd2_path) / (1024 ** 3), 2)


# ---------------------------------------------------------------------------
# Metadata.json checks
# ---------------------------------------------------------------------------

@register('metadata_json')
def _has_metadata(nd2_path, output_dir):
    return osp.isfile(osp.join(output_dir, 'metadata.json'))

@register('num_frames')
def _num_frames(nd2_path, output_dir):
    meta = _load_metadata(output_dir)
    return meta.get('nframes') if meta else None

@register('fps')
def _fps(nd2_path, output_dir):
    meta = _load_metadata(output_dir)
    if meta and 'fps' in meta:
        return round(meta['fps'], 3)
    return None

@register('bad_frames')
def _bad_frames(nd2_path, output_dir):
    """Number of bad frames (from metadata.json)."""
    meta = _load_metadata(output_dir)
    if meta and 'bad_frames' in meta:
        return len(meta['bad_frames'])
    return None

@register('encounter_frame')
def _encounter_frame(nd2_path, output_dir):
    meta = _load_metadata(output_dir)
    return meta.get('encounter_frame') if meta else None


# ---------------------------------------------------------------------------
# Stage 1 per-step checks
# ---------------------------------------------------------------------------

@register('shear_corrected')
def _shear_corrected(nd2_path, output_dir):
    return _dir_exists_and_nonempty(osp.join(output_dir, 'shear_corrected'))

@register('chan_align_params')
def _chan_align(nd2_path, output_dir):
    return osp.isfile(osp.join(output_dir, 'chan_align_params.csv'))

@register('channel_aligned')
def _channel_aligned(nd2_path, output_dir):
    return _dir_exists_and_nonempty(osp.join(output_dir, 'channel_aligned'))

@register('bleach_corrected')
def _bleach_corrected(nd2_path, output_dir):
    return _dir_exists_and_nonempty(osp.join(output_dir, 'bleach_corrected'))

@register('mip')
def _mip(nd2_path, output_dir):
    return osp.isfile(osp.join(output_dir, 'mip_bleach_corrected.tif'))

@register('z_selection')
def _z_selection(nd2_path, output_dir):
    return osp.isfile(osp.join(output_dir, 'z_selection.csv'))

@register('segmentation')
def _segmentation(nd2_path, output_dir):
    return osp.isfile(osp.join(output_dir, 'label.tif'))

@register('spline')
def _spline(nd2_path, output_dir):
    return osp.isfile(osp.join(output_dir, 'spline.json'))

@register('stage1_status')
def _stage1_status(nd2_path, output_dir):
    """Summarize Stage 1 progress."""
    if not osp.isdir(output_dir):
        return 'NOT STARTED'

    steps = [
        ('shear_corrected', lambda: _dir_exists_and_nonempty(osp.join(output_dir, 'shear_corrected'))),
        ('channel_alignment', lambda: osp.isfile(osp.join(output_dir, 'chan_align_params.csv'))),
        ('channel_aligned', lambda: _dir_exists_and_nonempty(osp.join(output_dir, 'channel_aligned'))),
        ('bleach_corrected', lambda: _dir_exists_and_nonempty(osp.join(output_dir, 'bleach_corrected'))),
        ('mip', lambda: osp.isfile(osp.join(output_dir, 'mip_bleach_corrected.tif'))),
        ('z_selection', lambda: osp.isfile(osp.join(output_dir, 'z_selection.csv'))),
        ('segmentation', lambda: osp.isfile(osp.join(output_dir, 'label.tif'))),
        ('spline', lambda: osp.isfile(osp.join(output_dir, 'spline.json'))),
    ]

    last_complete = -1
    for i, (name, check) in enumerate(steps):
        if check():
            last_complete = i
        else:
            break

    if last_complete == len(steps) - 1:
        return 'COMPLETE'
    if last_complete == -1:
        # check if any step is done (non-sequential progress)
        any_done = any(check() for _, check in steps)
        return 'IN PROGRESS' if any_done else 'NOT STARTED'
    return f'INCOMPLETE (after {steps[last_complete][0]})'


# ---------------------------------------------------------------------------
# Stage 2 per-step checks
# ---------------------------------------------------------------------------

@register('nose_annotated')
def _nose_annotated(nd2_path, output_dir):
    return osp.isfile(osp.join(output_dir, 'orient_nose.csv'))

@register('oriented')
def _oriented(nd2_path, output_dir):
    return osp.isfile(osp.join(output_dir, 'oriented.json'))

@register('warped')
def _warped(nd2_path, output_dir):
    return _dir_exists_and_nonempty(osp.join(output_dir, 'warped'))

@register('fixed_frame_selected')
def _fixed_frame(nd2_path, output_dir):
    return osp.isfile(osp.join(output_dir, 'fixed_frame_candidates.csv'))

@register('registered')
def _registered(nd2_path, output_dir):
    reg_dir = osp.join(output_dir, 'registered_elastix')
    return _count_files(reg_dir) > 50

@register('num_registered_files')
def _num_registered(nd2_path, output_dir):
    return _count_files(osp.join(output_dir, 'registered_elastix'))

@register('stage2_status')
def _stage2_status(nd2_path, output_dir):
    """Summarize Stage 2 progress."""
    if not osp.isfile(osp.join(output_dir, 'spline.json')):
        return 'STAGE1 NOT COMPLETE'

    has_nose = osp.isfile(osp.join(output_dir, 'orient_nose.csv'))
    has_oriented = osp.isfile(osp.join(output_dir, 'oriented.json'))
    has_warped = _dir_exists_and_nonempty(osp.join(output_dir, 'warped'))
    has_registered = _count_files(osp.join(output_dir, 'registered_elastix')) > 50

    if has_registered:
        return 'COMPLETE'
    if has_warped:
        return 'INCOMPLETE (after warp)'
    if has_oriented:
        return 'INCOMPLETE (after orient)'
    if not has_nose:
        return 'NEEDS NOSE ANNOTATION'
    return 'IN PROGRESS'


# ---------------------------------------------------------------------------
# Post-processing checks
# ---------------------------------------------------------------------------

@register('quantified')
def _quantified(nd2_path, output_dir):
    return osp.isfile(osp.join(output_dir, 'quantified.csv'))

@register('roi')
def _roi(nd2_path, output_dir):
    return osp.isfile(osp.join(output_dir, 'roi.tif'))

@register('benchmark')
def _benchmark(nd2_path, output_dir):
    return osp.isfile(osp.join(output_dir, 'zncc_timeseries.npy'))

@register('posture_embedding')
def _posture_embedding(nd2_path, output_dir):
    return osp.isfile(osp.join(output_dir, 'posture_embedding.csv'))


# ---------------------------------------------------------------------------
# Overall status
# ---------------------------------------------------------------------------

@register('overall_status')
def _overall_status(nd2_path, output_dir):
    """Single-word summary of the dataset's processing state."""
    if not osp.isdir(output_dir):
        return 'RAW'

    has_spline = osp.isfile(osp.join(output_dir, 'spline.json'))
    has_nose = osp.isfile(osp.join(output_dir, 'orient_nose.csv'))
    has_registered = _count_files(osp.join(output_dir, 'registered_elastix')) > 50
    has_quantified = osp.isfile(osp.join(output_dir, 'quantified.csv'))

    if has_quantified:
        return 'POST_PROCESSED'
    if has_registered:
        return 'STAGE2_COMPLETE'
    if has_spline and has_nose:
        return 'STAGE2_IN_PROGRESS'
    if has_spline and not has_nose:
        return 'NEEDS_NOSE'
    if has_spline:
        return 'STAGE1_COMPLETE'

    # check if any processing has started
    any_output = any(
        osp.exists(osp.join(output_dir, f))
        for f in ('shear_corrected', 'bleach_corrected', 'mip_bleach_corrected.tif',
                   'label.tif', 'chan_align_params.csv')
    )
    return 'STAGE1_IN_PROGRESS' if any_output else 'RAW'


# ---------------------------------------------------------------------------
# Disk usage (optional, can be slow)
# ---------------------------------------------------------------------------

def _add_disk_usage_check():
    """Register the disk-usage check only when --skip-disk-usage is NOT set."""

    @register('disk_usage_gb')
    def _disk_usage(nd2_path, output_dir):
        if not osp.isdir(output_dir):
            return 0.0
        return _dir_size_gb(output_dir)


# ---------------------------------------------------------------------------
# Future checks — behavior / NIR (stubs)
# ---------------------------------------------------------------------------
# When behavior analysis is added, register new checks here. Example:
#
# @register('behavior_csv')
# def _behavior(nd2_path, output_dir):
#     return osp.isfile(osp.join(output_dir, 'behavior.csv'))
#
# @register('nir_aligned')
# def _nir(nd2_path, output_dir):
#     return osp.isfile(osp.join(output_dir, 'nir', '*_nir_aligned.tif'))
#     # or use glob for pattern matching


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data_root, output_csv, skip_disk_usage=False):
    """Discover datasets, run all checks, write CSV report."""
    if not skip_disk_usage:
        _add_disk_usage_check()

    datasets = find_datasets(data_root)
    if not datasets:
        print(f'No .nd2 files found under {data_root}')
        sys.exit(1)

    print(f'Found {len(datasets)} dataset(s) under {data_root}')

    rows = []
    for nd2_path, output_dir in datasets:
        row = {}
        for col_name, check_fn in CHECK_REGISTRY:
            try:
                row[col_name] = check_fn(nd2_path, output_dir)
            except Exception as exc:
                row[col_name] = f'ERROR: {exc}'
        rows.append(row)
        shortname = osp.basename(nd2_path)
        status = row.get('overall_status', '?')
        print(f'  {shortname:70s} {status}')

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f'\nReport written to {output_csv}')

    # brief summary
    if 'overall_status' in df.columns:
        print('\n--- Summary ---')
        counts = df['overall_status'].value_counts()
        for status, n in counts.items():
            print(f'  {status:25s} {n}')
    print(f'  {"TOTAL":25s} {len(df)}')


def main():
    parser = argparse.ArgumentParser(
        description='Check processing status of all datasets under a data root.'
    )
    parser.add_argument(
        '--data-root', default=DEFAULT_DATA_ROOT,
        help=f'Root directory containing date subdirectories (default: {DEFAULT_DATA_ROOT})'
    )
    parser.add_argument(
        '--output', default=None,
        help='Path for the output CSV (default: <data-root>/processing_status.csv)'
    )
    parser.add_argument(
        '--skip-disk-usage', action='store_true',
        help='Skip computing disk usage per dataset (faster)'
    )
    args = parser.parse_args()

    output_csv = args.output or osp.join(args.data_root, 'processing_status.csv')
    run(args.data_root, output_csv, skip_disk_usage=args.skip_disk_usage)


if __name__ == '__main__':
    main()