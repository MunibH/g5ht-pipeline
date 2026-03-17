"""Semi-automatic fixed/reference frame selection for registration.

Ranks all warped frames by a composite score (straightness, sharpness,
RFP representativeness, z-coverage, temporal bias) and copies top-N
candidates to a `potential_fixed/` folder with a z-slice montage.

Usage from pipeline.ipynb:
    import select_fixed
    select_fixed.main(PTH)

Or from command line:
    uv run python select_fixed.py /path/to/dataset_dir
"""

import json
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_straightness(oriented_path):
    """Compute straightness ratio (end-to-end / arc-length) per frame.

    Returns dict {frame_int: straightness_float}.
    """
    with open(oriented_path) as f:
        oriented = json.load(f)

    straightness = {}
    for k, pts in oriented.items():
        frame = int(k)
        arr = np.asarray(pts, dtype=np.float64)
        if arr.ndim != 2 or len(arr) < 2:
            straightness[frame] = 0.0
            continue
        diffs = np.diff(arr, axis=0)
        arc_length = np.sum(np.linalg.norm(diffs, axis=1))
        end_to_end = np.linalg.norm(arr[-1] - arr[0])
        straightness[frame] = end_to_end / arc_length if arc_length > 0 else 0.0
    return straightness


def compute_mean_rfp_mip(warped_dir, num_frames, subsample=10):
    """Compute temporal-mean RFP MIP from warped volumes.

    Subsamples every `subsample`-th frame for speed.
    Returns (mean_mip, frame_indices_used).
    """
    frame_indices = list(range(0, num_frames, subsample))
    running_sum = None
    count = 0

    for idx in tqdm(frame_indices, desc="Computing mean RFP MIP"):
        path = os.path.join(warped_dir, f"{idx:04d}.tif")
        if not os.path.exists(path):
            continue
        vol = tifffile.imread(path)  # (Z, 2, 200, 500)
        rfp = vol[:, 1].astype(np.float64)
        mip = rfp.max(axis=0)  # (200, 500)
        if running_sum is None:
            running_sum = mip
        else:
            running_sum += mip
        count += 1

    if count == 0:
        raise RuntimeError(f"No warped TIFs found in {warped_dir}")
    mean_mip = running_sum / count
    return mean_mip, frame_indices


def compute_zncc(image, reference):
    """Zero-normalized cross-correlation between two 2D arrays."""
    a = image.ravel().astype(np.float64)
    b = reference.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_rfp_zncc(warped_dir, num_frames, mean_mip):
    """Compute ZNCC of each frame's RFP MIP vs temporal-mean MIP.

    Returns dict {frame_int: zncc_float}.
    """
    zncc_scores = {}
    for idx in tqdm(range(num_frames), desc="Computing RFP ZNCC"):
        path = os.path.join(warped_dir, f"{idx:04d}.tif")
        if not os.path.exists(path):
            zncc_scores[idx] = 0.0
            continue
        vol = tifffile.imread(path)
        rfp = vol[:, 1].astype(np.float64)
        mip = rfp.max(axis=0)
        zncc_scores[idx] = compute_zncc(mip, mean_mip)
    return zncc_scores


def load_sharpness(sharpness_csv):
    """Load sharpness CSV and return mean sharpness per frame.

    Returns dict {frame_int: mean_sharpness_float}.
    """
    df = pd.read_csv(sharpness_csv, index_col="frame")
    means = df.mean(axis=1)
    return {int(frame): float(val) for frame, val in means.items()}


def load_z_coverage(z_selection_csv):
    """Load z_selection CSV and return n_usable per frame.

    Returns dict {frame_int: n_usable_int}.
    """
    df = pd.read_csv(z_selection_csv, index_col="frame")
    return {int(frame): int(row["n_usable"]) for frame, row in df.iterrows()}


def compute_temporal_bias(num_frames):
    """Gaussian prior centered at the middle of the recording.

    sigma = num_frames / 6  (middle third ~1σ)
    Returns dict {frame_int: weight_float} in [0, 1].
    """
    center = num_frames / 2.0
    sigma = num_frames / 6.0
    bias = {}
    for i in range(num_frames):
        bias[i] = float(np.exp(-0.5 * ((i - center) / sigma) ** 2))
    return bias


def minmax_normalize(values):
    """Min-max normalize a 1D array to [0, 1]."""
    arr = np.asarray(values, dtype=np.float64)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.ones_like(arr)
    return (arr - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def rank_frames(
    dataset_dir,
    *,
    n_candidates=10,
    subsample_mean=10,
    weights=None,
    sharpness_floor_pct=10,
):
    """Score and rank all warped frames. Returns a DataFrame sorted by score.

    Args:
        dataset_dir: Path to the dataset output directory (e.g. .../date-YYMMDD_.../). 
                     Must contain oriented.json, sharpness.csv, z_selection.csv, warped/.
        n_candidates: Number of top candidates to return.
        subsample_mean: Every N-th frame used for mean RFP MIP computation.
        weights: Dict with keys {straightness, sharpness, zncc, z_coverage, temporal}.
                 Values should sum to 1. Defaults provided if None.
        sharpness_floor_pct: Percentile below which frames are excluded before scoring.

    Returns:
        DataFrame with columns: frame, composite_score, straightness, sharpness,
        zncc, n_usable, temporal_weight, and normalized versions.
    """
    if weights is None:
        weights = {
            "straightness": 0.25,
            "sharpness": 0.25,
            "zncc": 0.25,
            "z_coverage": 0.15,
            "temporal": 0.10,
        }

    oriented_path = os.path.join(dataset_dir, "oriented.json")
    sharpness_csv = os.path.join(dataset_dir, "sharpness.csv")
    z_selection_csv = os.path.join(dataset_dir, "z_selection.csv")
    warped_dir = os.path.join(dataset_dir, "warped")

    for p, label in [
        (oriented_path, "oriented.json"),
        (sharpness_csv, "sharpness.csv"),
        (z_selection_csv, "z_selection.csv"),
        (warped_dir, "warped/"),
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file/directory not found: {p} ({label})")

    # 1. Straightness
    print("Computing straightness...")
    straightness = compute_straightness(oriented_path)

    # figure out frame count from oriented.json keys
    frames = sorted(straightness.keys())
    num_frames = len(frames)
    print(f"  {num_frames} frames")

    # 2. Sharpness
    print("Loading sharpness...")
    sharpness_scores = load_sharpness(sharpness_csv)

    # 3. Z-coverage
    print("Loading z-coverage...")
    z_coverage = load_z_coverage(z_selection_csv)

    # 4. Mean RFP MIP + ZNCC
    print("Computing mean RFP MIP (subsampled)...")
    mean_mip, _ = compute_mean_rfp_mip(warped_dir, num_frames, subsample=subsample_mean)
    print("Computing per-frame RFP ZNCC...")
    zncc_scores = compute_rfp_zncc(warped_dir, num_frames, mean_mip)

    # 5. Temporal bias
    temporal = compute_temporal_bias(num_frames)

    # Assemble into arrays (aligned by frame index)
    raw = {
        "frame": frames,
        "straightness": [straightness.get(f, 0.0) for f in frames],
        "sharpness": [sharpness_scores.get(f, 0.0) for f in frames],
        "zncc": [zncc_scores.get(f, 0.0) for f in frames],
        "n_usable": [z_coverage.get(f, 0) for f in frames],
        "temporal_weight": [temporal.get(f, 0.0) for f in frames],
    }
    df = pd.DataFrame(raw)

    # Pre-filter: exclude frames below sharpness floor
    sharpness_arr = df["sharpness"].values
    floor = np.percentile(sharpness_arr, sharpness_floor_pct)
    mask = sharpness_arr >= floor
    # also exclude frames with empty splines (straightness == 0)
    mask &= df["straightness"].values > 0
    df = df[mask].copy()

    if len(df) == 0:
        raise RuntimeError("All frames were filtered out. Try lowering sharpness_floor_pct.")

    # Normalize each metric to [0, 1]
    for col in ["straightness", "sharpness", "zncc", "n_usable", "temporal_weight"]:
        df[f"{col}_norm"] = minmax_normalize(df[col].values)

    # Composite score
    df["composite_score"] = (
        weights["straightness"] * df["straightness_norm"]
        + weights["sharpness"] * df["sharpness_norm"]
        + weights["zncc"] * df["zncc_norm"]
        + weights["z_coverage"] * df["n_usable_norm"]
        + weights["temporal"] * df["temporal_weight_norm"]
    )

    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    return df.head(n_candidates)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_montage(dataset_dir, candidates_df, n_zslices=9, output_name="fixed_frame_montage.png"):
    """Generate a z-slice montage for top candidate frames.

    Each row = one candidate frame, each column = evenly-spaced z-slice (RFP channel).
    """
    warped_dir = os.path.join(dataset_dir, "warped")
    n_candidates = len(candidates_df)

    fig, axes = plt.subplots(
        n_candidates, n_zslices,
        figsize=(n_zslices * 3, n_candidates * 1.8),
        squeeze=False,
    )

    for row_idx, (_, row) in enumerate(candidates_df.iterrows()):
        frame = int(row["frame"])
        score = row["composite_score"]
        path = os.path.join(warped_dir, f"{frame:04d}.tif")
        vol = tifffile.imread(path)  # (Z, 2, 200, 500)
        rfp = vol[:, 1]
        n_z = rfp.shape[0]
        z_indices = np.linspace(0, n_z - 1, n_zslices, dtype=int)

        for col_idx, z_idx in enumerate(z_indices):
            ax = axes[row_idx, col_idx]
            ax.imshow(rfp[z_idx], cmap="gray", vmin=0, vmax=np.percentile(rfp, 99.5))
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(f"#{row['rank']}: f{frame}\n({score:.3f})", fontsize=8)
            if row_idx == 0:
                ax.set_title(f"z={z_idx}", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(dataset_dir, output_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Montage saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Copy candidates
# ---------------------------------------------------------------------------

def copy_candidates(dataset_dir, candidates_df):
    """Copy warped volumes and masks for candidate frames to potential_fixed/."""
    warped_dir = os.path.join(dataset_dir, "warped")
    masks_dir = os.path.join(dataset_dir, "masks")
    out_dir = os.path.join(dataset_dir, "potential_fixed")
    os.makedirs(out_dir, exist_ok=True)

    for _, row in candidates_df.iterrows():
        frame = int(row["frame"])
        fn = f"{frame:04d}.tif"
        newfn = f"fixed_{fn}"
        newmaskfn = f"fixed_mask_{fn}"
        # copy warped volume
        src = os.path.join(warped_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, newfn))
        # copy mask
        mask_src = os.path.join(masks_dir, fn)
        if os.path.exists(mask_src):
            shutil.copy2(mask_src, os.path.join(out_dir, newmaskfn))

    print(f"Copied {len(candidates_df)} candidates to: {out_dir}")
    return out_dir


def set_fixed_frame(dataset_dir, frame_index):
    """Copy chosen candidate as the official fixed frame + mask for registration.

    Copies:
        potential_fixed/{frame:04d}.tif      -> fixed_{frame:04d}.tif
        potential_fixed/mask_{frame:04d}.tif  -> fixed_mask_{frame:04d}.tif
    """
    fn = f"{frame_index:04d}.tif"
    potential_dir = os.path.join(dataset_dir, "potential_fixed")
    vol_src = os.path.join(potential_dir, fn)
    mask_src = os.path.join(potential_dir, f"mask_{fn}")

    if not os.path.exists(vol_src):
        # fall back to warped/ directly
        vol_src = os.path.join(dataset_dir, "warped", fn)
    if not os.path.exists(mask_src):
        mask_src = os.path.join(dataset_dir, "masks", fn)

    vol_dst = os.path.join(dataset_dir, f"fixed_{fn}")
    mask_dst = os.path.join(dataset_dir, f"fixed_mask_{fn}")

    shutil.copy2(vol_src, vol_dst)
    shutil.copy2(mask_src, mask_dst)
    print(f"Fixed frame set: {vol_dst}")
    print(f"Fixed mask set:  {mask_dst}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(
    dataset_dir=None,
    *,
    n_candidates=10,
    subsample_mean=10,
    weights=None,
    sharpness_floor_pct=10,
):
    """Run the full fixed-frame selection pipeline.

    Args:
        dataset_dir: Path to dataset output directory. If None, reads from sys.argv[1].
        n_candidates: Number of top candidates to output.
        subsample_mean: Subsample rate for mean RFP MIP computation.
        weights: Scoring weights dict. See rank_frames() for details.
        sharpness_floor_pct: Exclude frames below this sharpness percentile.

    Returns:
        DataFrame of ranked candidates.
    """
    if dataset_dir is None:
        dataset_dir = sys.argv[1]

    print(f"Dataset: {dataset_dir}")
    print(f"Candidates: {n_candidates} | Subsample: {subsample_mean}")

    # Score and rank
    df = rank_frames(
        dataset_dir,
        n_candidates=n_candidates,
        subsample_mean=subsample_mean,
        weights=weights,
        sharpness_floor_pct=sharpness_floor_pct,
    )

    # Save CSV
    csv_path = os.path.join(dataset_dir, "fixed_frame_candidates.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCandidates saved: {csv_path}")

    # Print top candidates
    display_cols = ["rank", "frame", "composite_score", "straightness", "sharpness", "zncc", "n_usable", "temporal_weight"]
    print(f"\nTop {n_candidates} candidates:")
    print(df[display_cols].to_string(index=False))

    # Montage
    make_montage(dataset_dir, df)

    # Copy to potential_fixed/
    copy_candidates(dataset_dir, df)

    print(f"\n>> Recommended fixed frame: {int(df.iloc[0]['frame'])}")
    print("   Review the montage and potential_fixed/ folder, then run:")
    print(f"   select_fixed.set_fixed_frame('{dataset_dir}', <chosen_frame>)")

    return df


if __name__ == "__main__":
    main()
