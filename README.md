# G5HT-PIPELINE

Image processing pipeline for volumetric calcium imaging data of freely moving *C. elegans*. Takes raw `.nd2` recordings (dual-channel GFP/RFP, 3D volumetric time series) and produces registered, straightened volumes suitable for quantitative analysis.

The main entry point is the notebook `run_pipeline_flvc_transfer.ipynb`, which orchestrates each step and handles data transfer to/from a remote Linux server (FLVC).

---

## Quick Reference

| Step | Script(s) | Inputs | Outputs | Approx. Time | Conda Env |
|------|-----------|--------|---------|---------------|-----------|
| 0. Metadata | `utils.py` | `.nd2` file | frame count, dimensions, channel count | seconds | `g5ht-pipeline` |
| 1. Shear Correction | `shear_correct.py` | `.nd2`, noise `.tif` | `shear_corrected/*.tif` | ~1 hr | `g5ht-pipeline` |
| 2a. Get Channel Alignment | `get_channel_alignment.py` | `shear_corrected/*.tif` or beads `.nd2` | `txt/*.txt` | ~30 min | `g5ht-pipeline` |
| 2b. Median Channel Alignment | `median_channel_alignment.py` | `txt/*.txt` | `chan_align_params.csv`, `chan_align.txt` | seconds | `g5ht-pipeline` |
| 2c. Apply Channel Alignment | `apply_channel_alignment.py` | `shear_corrected/*.tif`, `chan_align.txt` | `channel_aligned/*.tif` | ~30 min | `g5ht-pipeline` |
| 3. Bleach Correction | `bleach_correct.py` | `channel_aligned/*.tif` | `bleach_corrected/*.tif` | ~5 min | `g5ht-pipeline` |
| 4. MIP | `mip.py` | `bleach_corrected/*.tif` | `mip_*.tif`, `mip_*.mp4`, `means.png`, `focus_check.csv` | minutes | `g5ht-pipeline` |
| 5. Drift Estimation | `drift_estimation.py` | `bleach_corrected/*.tif` | `z_selection.csv`, `sharpness.csv`, `z_selection_diagnostics.png` | minutes | `g5ht-pipeline` |
| 6. Segment | `segment/segment_torch.py` | `mip_*.tif` | `label.tif` | ~1 min | `segment-torch` / `torchcu129` |
| 7. Spline | `spline.py` | `label.tif` | `spline.json`, `spline.tif`, `dilated.tif` | minutes | `g5ht-pipeline` |
| 8. Orient | `orient.py` | `spline.json` | `oriented.json`, `oriented.png`, `oriented_stack.tif` | minutes | `g5ht-pipeline` |
| 9. Warp | `warp.py` | `bleach_corrected/*.tif`, `oriented.json`, `dilated.tif` | `warped/*.tif`, `masks/*.tif` | 1-2 hrs | `g5ht-pipeline` |
| 10. Register | `reg.py` | `warped/*.tif`, `masks/*.tif`, `fixed_XXXX.tif`, `fixed_mask_XXXX.tif` | `registered_elastix/*.tif` | ~5 hrs | `g5ht-pipeline` |

**Storage requirement:** ~200 GB per recording. After registration, ~150 GB of intermediates (shear_corrected, channel_aligned, warped) can be deleted.

---

## Conda Environments

- **`g5ht-pipeline`** — used for all steps except segmentation. Install via `environment.yml`.
- **`segment-torch`** / **`torchcu129`** — used only for step 6 (segmentation). Requires PyTorch and `segmentation_models_pytorch`.

---

## Detailed Step Descriptions

### 0. Metadata Extraction

**Script:** `utils.py` (functions `get_range_from_nd2`, `get_noise_stack`, `get_beads_alignment_file`)

Reads the `.nd2` file header to extract recording metadata:
- **Number of frames (volumes):** Total 2D frames divided by the stack length (typically 41 z-slices per volume).
- **Image dimensions:** Height and width of each 2D frame (typically 512×512).
- **Number of channels:** Typically 2 (GFP at channel 0, RFP at channel 1).
- **Noise stack:** A pre-recorded noise calibration image (`noise_XXXXXX.tif`) is loaded and tiled to match the stack depth. This represents the camera's fixed-pattern noise floor.
- **Beads alignment file:** Checks for a companion `*_chan_alignment.nd2` file containing bead recordings used for chromatic aberration correction.

### 1. Shear Correction

**Script:** `shear_correct.py`

Corrects intra-volume motion artifacts caused by the worm moving during the ~0.5 s acquisition time of a single volume.

**Algorithm:**
1. **Stack extraction & denoising:** Each volume is extracted from the `.nd2` file as a 4D array `(Z, C, H, W)`. The pre-recorded noise stack is subtracted and values are clipped to `[0, 4095]` (12-bit range). Unstable z-slices at the beginning or end of each volume are trimmed based on the recording date.
2. **Reference slice selection:** The z-slice with the highest mean RFP intensity is identified as the reference (most signal, least likely to be affected by edge artifacts).
3. **Bidirectional rigid registration:** Starting from the reference slice, adjacent slices are sequentially registered using ITK Elastix rigid registration (4-resolution pyramid). Registration propagates outward in both directions (toward z=0 and toward z=max). For each slice:
   - The RFP channel of the adjacent (already-corrected) slice serves as the fixed image.
   - The RFP channel of the current slice is the moving image.
   - The resulting rigid transform (translation + rotation) is also applied to the GFP channel via `itk.transformix_filter`.
4. **Output:** Each corrected volume is saved as a `uint16` TIFF file (`XXXX.tif`) in the `shear_corrected/` directory.
5. **Parallelization:** Volumes are processed in parallel using Python's `multiprocessing.Pool`.

### 2. Channel Alignment

Corrects chromatic aberration / misalignment between the GFP and RFP channels. This is a three-sub-step process.

#### 2a. Get Channel Alignment Parameters

**Script:** `get_channel_alignment.py`

Estimates the 3D rigid transform (EulerTransform with 6 DOF: 3 rotations Rx, Ry, Rz + 3 translations Tx, Ty, Tz) that aligns the GFP channel to the RFP channel.

**Algorithm:**
1. **Input selection:** If a separate bead recording (`*_chan_alignment.nd2`) exists, it is used for alignment (preferred, as beads provide point-source calibration). Otherwise, the worm recording itself is used.
2. **Per-frame 3D registration:** For each volume, the full 3D GFP volume is registered to the 3D RFP volume using ITK Elastix with:
   - `EulerTransform` (rigid 3D: Rx, Ry, Rz, Tx, Ty, Tz)
   - `AdvancedMattesMutualInformation` metric (robust to intensity differences between channels)
   - B-spline interpolation (order 3)
   - 3-level multi-resolution pyramid
   - 1024 max iterations, 4096 spatial samples
3. **Output:** Per-frame transform parameters are saved as `.txt` files in the `txt/` subdirectory.
4. **Parallelization:** Frames are processed in parallel.

#### 2b. Compute Median Channel Alignment

**Script:** `median_channel_alignment.py`

Aggregates per-frame alignment parameters into a single robust estimate.

**Algorithm:**
1. All per-frame `.txt` parameter files are read. For each, the `CenterOfRotationPoint`, `Size`, and `TransformParameters` (6 values) are extracted.
2. Parameters are saved to `chan_align_params.csv` for inspection.
3. The **element-wise median** across all frames is computed for each of the 6 transform parameters, the center of rotation, and the image size. The median is robust to outlier frames.
4. The median parameters are written into a template Elastix parameter file (`chan_align.txt`) using `template_3d.txt` as the base.

#### 2c. Apply Channel Alignment

**Script:** `apply_channel_alignment.py`

Applies the median alignment transform to every volume.

**Algorithm:**
1. For each shear-corrected volume, the GFP channel (all z-slices) is transformed using `itk.transformix_filter` with the median alignment parameters from `chan_align.txt`.
2. The RFP channel is kept unchanged.
3. Output volumes `(Z, C, H, W)` are saved as `uint16` TIFFs in the `channel_aligned/` directory.
4. Parallelized across frames.

### 3. Bleach Correction

**Script:** `bleach_correct.py`

Corrects for photobleaching — the gradual decrease in fluorescence intensity over the course of the recording.

**Algorithm (block method, default):**
1. **Signal estimation:** For each volume, the total (or median) pixel intensity of the specified channel is computed, yielding a 1D signal trace over time.
2. **Smoothing:** A 1D median filter (kernel size 251) is applied to the signal trace to obtain a smooth bleaching profile.
3. **Block segmentation:** The smooth profile is divided into blocks based on fractional intensity drops (`fbc`, default 4%). Block boundaries are placed where the cumulative intensity has dropped by each `fbc` fraction of the total intensity change.
4. **Multiplicative correction:** For each block, the mean signal is computed. A correction factor is calculated as `(first block mean) / (current block mean)`. Every voxel in every volume within that block is multiplied by the block's correction factor.
5. **Output:** Corrected volumes are saved in `bleach_corrected/`. Diagnostic plots (`bleach_diagnostics_*.png`, `bleach_profile_*.png`) are saved showing raw vs. corrected signals, block boundaries, and correction factors.

**Alternative method (exponential):**
- Fits an exponential decay model `y = a * exp(-b * t) + c` to the signal trace.
- Correction factor per frame: `fitted(0) / fitted(t)`.

### 4. Maximum Intensity Projection (MIP)

**Script:** `mip.py`

Creates 2D maximum intensity projections from the 3D volumes for visualization and downstream 2D processing (segmentation, spline fitting).

**Algorithm:**
1. **MIP computation:** For each volume `(Z, C, H, W)`, the maximum value along the z-axis is taken independently for each channel, producing a 2D `(C, H, W)` frame.
2. **TIFF stack:** All MIP frames are concatenated into a single BigTIFF file (`mip_bleach_corrected.tif`).
3. **MP4 video:** An RGB video is generated where the red channel displays RFP and the green channel displays GFP, with configurable intensity scaling (`rmax`, `gmax`).
4. **Mean intensity plot:** Per-frame mean GFP and RFP intensities are plotted and saved as `means.png`.
5. **Focus check (optional):** For each frame and each z-slice, the mean RFP intensity is computed. The resulting `(frames × z)` matrix is saved as `focus_check.csv` and plotted as a heatmap (`focus.png`) to visualize focus drift.

### 5. Drift Estimation

**Script:** `drift_estimation.py`

Estimates axial (z) focus drift over time and selects consistent z-slices for downstream analysis.

**Algorithm:**
1. **Sharpness computation:** For every z-slice of every frame, the Laplacian variance (variance of the Laplacian-filtered image) is computed on the RFP channel. Higher values indicate sharper, more in-focus images.
2. **Peak detection:** The z-slice with maximum sharpness is identified per frame (`z_peak_raw`).
3. **Sub-slice refinement:** Parabolic interpolation on the sharpness values around the peak provides sub-pixel precision (`z_peak_refined`).
4. **Drift tracking:** A causal exponential weighted moving average (EWMA, α=0.15) smooths the refined peak positions to track gradual drift without being affected by frame-to-frame noise.
5. **Z-slice selection:** For each frame, a window of `n_slices` (default 24) consecutive z-slices centered on the tracked focus is selected. Slices below a sharpness threshold (30th percentile) are discarded and replaced with zero-padding if necessary.
6. **Output:** `z_selection.csv` (selected z-indices, padding flags, tracked focus, drift per frame), `sharpness.csv` (full sharpness matrix), and `z_selection_diagnostics.png` (4-panel diagnostic plot with sharpness heatmap, drift curve, usable slices per frame, and selected z-range).

### 6. Segmentation

**Script:** `segment/segment_torch.py`

Segments the worm body in each MIP frame using a deep learning model.

**Algorithm:**
1. **Model:** DeepLabV3+ with a ResNet-50 backbone (`segmentation_models_pytorch`). Pre-trained weights are loaded from a checkpoint file.
2. **Preprocessing:** Each MIP frame's RFP channel is z-scored (zero mean, unit variance). The single-channel image is replicated to 3 channels (to match the model's expected RGB input).
3. **Inference:** The model outputs per-pixel logits for 2 classes (background, worm). The argmax of the logits produces a binary segmentation mask.
4. **Batched processing:** Frames are processed in batches of 4 on GPU (CUDA) for efficiency.
5. **Output:** `label.tif` — a `(T, H, W)` boolean TIFF stack where `True` pixels belong to the worm body.

### 7. Spline Fitting

**Script:** `spline.py`

Extracts the worm's midline (centerline / spline) from each segmentation mask.

**Algorithm:**
1. **Mask cleanup:** The segmentation mask is processed with connected component analysis. Only the largest connected component is retained. The mask is then dilated by `r_dilation` = 8 pixels (isotropic dilation).
2. **Skeletonization:** The dilated mask is eroded by `r_erosion + r_dilation` = 40 pixels, then skeletonized (`morphology.skeletonize`) to produce a 1-pixel-wide centerline.
3. **Graph construction:** The skeleton pixels are converted to a graph where each pixel is a node and 8-connected neighbors are edges.
4. **Pruning:** All branch points (nodes with degree > 2) are removed. The largest remaining connected component (a simple path) is selected as the main centerline.
5. **Ordering:** Starting from one endpoint (degree-1 node), a BFS traversal produces an ordered sequence of `(y, x)` coordinates from head to tail (or tail to head — orientation is resolved in the next step).
6. **Output:**
   - `spline.json`: Dictionary mapping frame index → ordered list of `(y, x)` centerline points.
   - `dilated.tif`: Dilated segmentation masks (used later for warping masks).
   - `spline.tif`: Visualization of the spline overlaid on the dilated mask.

### 8. Orientation

**Script:** `orient.py`

Determines head-to-tail orientation of the spline for each frame, ensuring temporal consistency.

**Algorithm:**
1. **Initialization:** The user provides the `(y, x)` coordinates of the nose (head) in the first frame, identified by inspecting `spline.tif`.
2. **Sequential orientation:** For each frame, the algorithm computes:
   - Distance from the first spline point to the previous frame's nose position.
   - Distance from the last spline point to the previous frame's nose position.
   - If the last point is closer, the spline is reversed.
3. **Truncation:** Each oriented spline is truncated to 350 points for consistency.
4. **Constraint support:** Optional `(frame, nose_y, nose_x)` triplets can be specified to override the orientation at specific frames where tracking might fail.
5. **Output:**
   - `oriented.json`: Dictionary with consistently oriented spline coordinates.
   - `oriented.png`: Overlay of all oriented splines color-coded by time.
   - `oriented_stack.tif`: Per-frame visualization of the oriented spline on the dilated mask.

### 9. Warping (Straightening)

**Script:** `warp.py`

Straightens ("uncoils") each 3D volume from the worm's curved body coordinate system into a canonical rectangular frame aligned along the anterior-posterior axis.

**Algorithm:**
1. **Spline parameterization:** The oriented spline points are fit with a parametric B-spline (`scipy.interpolate.splprep`). The arc length is computed to set parameterization bounds: `[-100, +400]` pixels from the nose along the body axis.
2. **Control point grid:** At 51 evenly spaced positions along the spline, perpendicular transects of ±100 pixels (21 sample points each) are computed using the spline tangent normals. This creates a curvilinear grid of `51 × 21 = 1071` control points mapping from the straightened coordinate system to the original image coordinates.
3. **Piecewise affine warping:** A `PiecewiseAffineTransform` (scikit-image) is estimated from the control points. Each z-slice of each channel is warped into a canonical `(200, 500)` output frame using cubic interpolation (order 3).
4. **Mask warping:** The dilated segmentation mask is also warped (nearest-neighbor interpolation, order 0) for use as a registration mask.
5. **Parallelization:** Z-slices within each volume are warped in parallel using `joblib`.
6. **Output:** `warped/XXXX.tif` (straightened volumes, `(Z, C, 200, 500)`) and `masks/XXXX.tif` (straightened masks).

### 10. Registration

**Script:** `reg.py`

Registers all straightened volumes to a single reference frame, compensating for residual between-frame deformations.

**Algorithm:**
1. **Fixed frame selection:** The user selects a representative frame (e.g., frame 450) and copies its warped volume and mask as `fixed_XXXX.tif` and `fixed_mask_XXXX.tif` in the main output directory.
2. **Multi-stage registration:** Each moving volume's RFP channel is registered to the fixed volume's RFP channel using ITK Elastix with a cascade of transforms:
   - **Rigid** (4-resolution pyramid): Global rotation and translation.
   - **Affine** (4-resolution pyramid): Adds scaling and shearing.
   - **B-spline deformable** at grid spacings of 128, 64, and 32 voxels (4-resolution each): Progressively finer non-rigid deformations.
3. **Mask guidance:** Both fixed and moving masks (replicated across z) constrain the registration to focus on the worm body, ignoring background.
4. **Z-axis zoom (optional):** Both fixed and moving stacks can be upsampled along z by a zoom factor to improve registration quality for sparse z-sampling.
5. **Transform application:** The transform estimated from RFP registration is applied to the GFP channel via `itk.transformix_filter`, ensuring both channels are consistently registered.
6. **Output:** `registered_elastix/XXXX.tif` — registered volumes `(Z, 2, 200, 500)` in `uint16`.

### Post-Processing

After registration, additional steps are available:

- **Behavior extraction:** NIR behavioral video is extracted from the companion `.h5` file and saved as `nir_video.mp4`.
- **Transfer to FLVC:** Results are synced back to the remote Linux server using `rsync` (via MSYS on Windows), excluding large intermediate directories (`channel_aligned/`, `shear_corrected/`).

---

## File Organization

Each recording produces a directory named after the dataset (without extension), e.g.:

```
date-YYYYMMDD_strain-XXX_condition-YYY_wormNNN/
├── shear_corrected/          # Step 1 output (deletable after step 2c)
│   ├── 0000.tif
│   ├── 0001.tif
│   └── ...
├── txt/                      # Step 2a output (alignment parameter files)
│   ├── 0000.txt
│   └── ...
├── channel_aligned/          # Step 2c output (deletable after step 3)
│   ├── 0000.tif
│   └── ...
├── bleach_corrected/         # Step 3 output
│   ├── 0000.tif
│   └── ...
├── warped/                   # Step 9 output (deletable after step 10)
│   ├── 0000.tif
│   └── ...
├── masks/                    # Step 9 output
│   ├── 0000.tif
│   └── ...
├── registered_elastix/       # Step 10 output (final result)
│   ├── 0000.tif
│   └── ...
├── chan_align_params.csv      # Step 2b
├── chan_align.txt             # Step 2b (median alignment parameters)
├── mip_bleach_corrected.tif   # Step 4
├── mip_bleach_corrected.mp4   # Step 4
├── means.png                  # Step 4
├── focus_check.csv            # Step 4
├── label.tif                  # Step 6
├── spline.json                # Step 7
├── spline.tif                 # Step 7
├── dilated.tif                # Step 7
├── oriented.json              # Step 8
├── oriented.png               # Step 8
├── oriented_stack.tif         # Step 8
├── z_selection.csv            # Step 5
├── sharpness.csv              # Step 5
├── z_selection_diagnostics.png # Step 5
├── fixed_XXXX.tif             # Step 10 (user-selected reference)
├── fixed_mask_XXXX.tif        # Step 10
├── bleach_diagnostics_*.png   # Step 3
└── bleach_profile_*.png       # Step 3
```

---

## TODO
- Auto or manual pick fixed frame and mask, save which ones used
- Save input to orient.py (approximate pixel location of nose)
- Labelme, label ROIs, quantify.py
- Heatmap of each voxel over time, cluster to find ROIs
- MIP for xy, xz, zy planes and for several slices
- Parallelize warp and registration steps