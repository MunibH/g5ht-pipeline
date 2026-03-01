"""
quantify_voxels.py — Ratiometric voxel quantification for dual-channel 5-HT imaging.

Reads registered TIF stacks (GFP + RFP channels), computes the ratiometric
signal R = GFP / <RFP>_t, NaN-fills bad frames, and saves everything needed
for downstream analyses (baseline normalization, F/F20, latency, PCA, etc.)
into a single consolidated .h5 file.

Usage:
    python quantify_voxels.py <input_dir> <reg_dir> [binning_factor]

Arguments:
    input_dir       Path to the worm data directory (must contain metadata.json,
                    roi.tif, and fixed_mask_*.tif).
    reg_dir         Subdirectory containing registered TIF stacks (e.g.
                    'registered_elastix').
    binning_factor  Spatial binning factor (default: 1, no binning).

Output:
    <input_dir>/<basename>_processed_voxels.h5 with datasets:

    Data arrays
        ratio           (T, Z, H, W) float32  — R = GFP / <RFP>_t.
                         Bad frames are NaN-filled.
        rfp_mean        (Z, H, W)    float32  — time-averaged RFP channel.
        gfp_mean        (Z, H, W)    float32  — time-averaged GFP channel.
        baseline        (Z, H, W)    float32  — mean of R over baseline_window.
                         Absent when baseline_window is not set.
        f20             (Z, H, W)    float32  — 20th percentile of R across
                         time (ignoring NaN / bad frames).

    Masks
        roi             (Z, H, W)    uint8    — ROI label mask (1-indexed).
        roi_labels      (N,)         str      — ROI label names.
        fixed_mask      (H, W)       uint8    — 2D worm binary mask.

    Timing / metadata
        time_vec        (T,)         float64  — time in seconds.
        frame_index     (T,)         int      — original frame numbers from
                         TIF filenames.
        fps             scalar       float    — frames per second.
        binning_factor  scalar       int      — spatial binning factor.
        baseline_window (2,)         int      — (start, end) frame indices.
                         (-1, -1) when not set.
        encounter_frame scalar       int      — frame of food encounter.
                         -1 when not set.
        bad_frames      (N,)         int      — indices of bad frames (can be
                         empty).
        nframes         scalar       int      — total number of frames.
"""

import json
import sys
import os
import glob

import h5py
import tifffile
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _process_single_tif(tif_path, z_slices, h_binned, w_binned, binning_factor):
    """Load a registered TIF and return spatially-binned (GFP, RFP) arrays.

    Parameters
    ----------
    tif_path : str
        Path to a single registered TIF file (Z*C, H, W).
    z_slices : int
        Number of Z slices (C=2 channels assumed).
    h_binned, w_binned : int
        Target spatial dimensions after binning.
    binning_factor : int
        Spatial binning factor.

    Returns
    -------
    gfp : ndarray (Z, H_binned, W_binned) float32
    rfp : ndarray (Z, H_binned, W_binned) float32
    """
    stack = tifffile.imread(tif_path).astype(np.float32).clip(min=0, max=4096)

    # Reshape to (Z, 2, H, W) if needed
    if stack.ndim == 3:
        stack = stack.reshape(z_slices, 2, stack.shape[1], stack.shape[2])

    # Spatial binning via reshape + mean
    z, c, h, w = stack.shape
    h_crop = h_binned * binning_factor
    w_crop = w_binned * binning_factor

    binned = (
        stack[:, :, :h_crop, :w_crop]
        .reshape(z, c, h_binned, binning_factor, w_binned, binning_factor)
        .mean(axis=(3, 5))
    )

    return binned[:, 0], binned[:, 1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # Parse arguments
    # ------------------------------------------------------------------
    if len(sys.argv) < 3:
        print("Usage: python quantify_voxels.py <input_dir> <reg_dir> [require_metadata] [binning_factor]")
        sys.exit(1)

    input_dir = sys.argv[1]
    reg_dir = sys.argv[2]
    require_metadata = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True
    binning_factor = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    registered_dir = os.path.join(input_dir, reg_dir)

    # ------------------------------------------------------------------
    # Discover registered TIFs (sorted by frame number)
    # ------------------------------------------------------------------
    tif_paths = sorted(
        glob.glob(os.path.join(registered_dir, '*.tif')),
        key=lambda x: int(os.path.basename(x).split('.')[0]),
    )
    if not tif_paths:
        raise FileNotFoundError(f"No .tif files found in {registered_dir}")

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------
    metadata_path = os.path.join(input_dir, 'metadata.json')

    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as fj:
            meta = json.load(fj)

        bad_frames = np.array(meta['bad_frames'], dtype=int)
        frame_index = np.array(meta['frame_index'], dtype=int)
        fps = float(meta['fps'])
        nframes = int(meta['nframes'])
        encounter_frame = meta.get('encounter_frame')  # int or None

        bsf = meta.get('baseline_start_frame')
        bef = meta.get('baseline_end_frame')
        has_baseline = (bsf is not None) and (bef is not None)
        baseline_window = (int(bsf), int(bef)) if has_baseline else None

        time_vec = frame_index / fps
        print(f"Loaded metadata from {metadata_path}")
    else:
        if require_metadata:
            print(f"metadata.json not found in {input_dir} and require_metadata is True")
            return
        print(f"WARNING: metadata.json not found in {input_dir}")
        print("  Using defaults — create metadata.json for accurate processing.")
        fps = 1 / 0.533
        baseline_window = (0, 60)
        has_baseline = True
        frame_index = np.array(
            [int(os.path.basename(p).split('.')[0]) for p in tif_paths],
            dtype=int,
        )
        time_vec = frame_index / fps
        bad_frames = np.array([], dtype=int)
        encounter_frame = None
        nframes = len(tif_paths)

    print(f"  fps={fps:.4f}, nframes={nframes}, binning_factor={binning_factor}")
    print(f"  baseline_window={baseline_window}, encounter_frame={encounter_frame}")
    print(f"  bad_frames: {len(bad_frames)} frames")

    # ------------------------------------------------------------------
    # Determine stack dimensions from first TIF
    # ------------------------------------------------------------------
    first_stack = tifffile.imread(tif_paths[0])
    if first_stack.ndim == 3:
        z_slices = first_stack.shape[0] // 2
        first_stack = first_stack.reshape(z_slices, 2, first_stack.shape[1], first_stack.shape[2])

    z_slices, _, h, w = first_stack.shape
    h_binned = h // binning_factor
    w_binned = w // binning_factor
    del first_stack

    # ------------------------------------------------------------------
    # Process all TIFs → GFP array + RFP running sum
    # ------------------------------------------------------------------
    T = len(tif_paths)
    gfp_all = np.zeros((T, z_slices, h_binned, w_binned), dtype=np.float32)
    rfp_sum = np.zeros((z_slices, h_binned, w_binned), dtype=np.float64)

    print(f"Processing {T} registered stacks …")
    for i, tif_path in enumerate(tqdm(tif_paths, desc="Loading stacks")):
        gfp, rfp = _process_single_tif(
            tif_path, z_slices, h_binned, w_binned, binning_factor,
        )
        gfp_all[i] = gfp
        rfp_sum += rfp

    rfp_mean = (rfp_sum / T).astype(np.float32)
    gfp_mean = gfp_all.mean(axis=0)
    del rfp_sum

    # ------------------------------------------------------------------
    # Ratiometric normalization: R = GFP / <RFP>_t
    # ------------------------------------------------------------------
    # Mask voxels that are identically 0 in GFP — these should stay 0.
    zero_mask = gfp_all == 0

    ratio = np.divide(
        gfp_all, rfp_mean,
        out=np.zeros_like(gfp_all),
        where=rfp_mean != 0,
    )
    ratio[zero_mask] = 0.0
    del gfp_all, zero_mask

    # ------------------------------------------------------------------
    # NaN-fill bad frames (preserves T dimension & frame alignment)
    # ------------------------------------------------------------------
    if len(bad_frames) > 0:
        valid_bad = bad_frames[bad_frames < T]
        if len(valid_bad) > 0:
            ratio[valid_bad] = np.nan
            print(f"NaN-filled {len(valid_bad)} bad frames in ratio array")

    # ------------------------------------------------------------------
    # Compute normalization references (saved but NOT applied)
    # ------------------------------------------------------------------
    # 1. Baseline mean of R over baseline_window (only if baseline is set)
    if has_baseline:
        baseline = ratio[baseline_window[0]:baseline_window[1]]
        baseline = np.nanmean(baseline, axis=0).astype(np.float32)
        print(f"Computed baseline mean over frames {baseline_window}")
    else:
        baseline = None
        print("No baseline_window set — skipping baseline computation")

    # 2. F20: 20th percentile of R across time (ignoring NaN)
    f20 = np.nanpercentile(ratio, 20, axis=0).astype(np.float32)
    print("Computed F20 (20th percentile across time)")

    print(f"Ratio array shape: {ratio.shape}")

    # ------------------------------------------------------------------
    # Load ROI and fixed mask
    # ------------------------------------------------------------------
    # roi_path = os.path.join(input_dir, 'roi.tif')
    # roi = tifffile.imread(roi_path) # shape (Z,H,W), uint8 labels
    # with tifffile.TiffFile(roi_path) as tif:
    #     roi_labels = tif.imagej_metadata['Labels'] # should include ['procorpus', 'metacorpus', 'isthmus', 'terminal_bulb', 'nerve_ring', 'ventral_nerve_cord'] and sometimes also include ['dorsal_nerve_cord']

    fixed_fns = glob.glob(os.path.join(input_dir, 'fixed_mask_[0-9][0-9][0-9][0-9]*.tif'))
    if not fixed_fns:
        raise FileNotFoundError(f"No fixed_mask_*.tif found in {input_dir}")
    fixed_mask = tifffile.imread(fixed_fns[0]) # shape (H,W)

    # ------------------------------------------------------------------
    # Save consolidated .h5
    # ------------------------------------------------------------------
    out_name = f"{os.path.basename(input_dir)}_processed_data.h5"
    out_path = os.path.join(input_dir, out_name)
    print(f"Saving to {out_path} …")

    with h5py.File(out_path, 'w') as f:
        # --- Data arrays ---
        f.create_dataset('ratio', data=ratio, compression='gzip')
        f.create_dataset('rfp_mean', data=rfp_mean, compression='gzip')
        f.create_dataset('gfp_mean', data=gfp_mean, compression='gzip')
        f.create_dataset('f20', data=f20, compression='gzip')
        if baseline is not None:
            f.create_dataset('baseline', data=baseline, compression='gzip')

        # --- Masks ---
        # f.create_dataset('roi', data=roi, compression='gzip') # will append rois later
        # f.create_dataset('roi_labels', data=roi_labels)
        f.create_dataset('fixed_mask', data=fixed_mask, compression='gzip')

        # --- Timing ---
        f.create_dataset('time_vec', data=time_vec, compression='gzip')
        f.create_dataset('frame_index', data=frame_index, compression='gzip')
        f.create_dataset('fps', data=fps)

        # --- Processing parameters ---
        f.create_dataset('binning_factor', data=binning_factor)
        f.create_dataset('nframes', data=nframes)

        # --- Metadata (use -1 sentinel for unset values) ---
        bw = np.array(baseline_window, dtype=int) if has_baseline else np.array([-1, -1], dtype=int)
        f.create_dataset('baseline_window', data=bw)
        f.create_dataset('encounter_frame', data=encounter_frame if encounter_frame is not None else -1)
        f.create_dataset('bad_frames', data=bad_frames, compression='gzip')

    print(f"Done — saved {out_path}")


if __name__ == '__main__':
    main()