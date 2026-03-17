# Some of these functions were adapted from `deepreg/model/kernel.py` in the 'DeepReg' repository
# on GitHub. The original implementation can be found at:
# https://github.com/flavell-lab/DeepReg/deepreg/loss
# The code is used under the MIT License.
from __future__ import annotations

import glob
import os
import re
from typing import Literal, Tuple

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import tifffile
import torch
from tqdm import tqdm

import benchmark_utils as utils


EPS = 1.0e-5


def _ensure_5d_single_channel(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure a tensor is shaped (B, 1, D, H, W).

    Accepts:
      - (B, D, H, W)
      - (B, D, H, W, 1)  channels-last
      - (B, 1, D, H, W)  channels-first
    """
    if x.dim() == 4:
        return x.unsqueeze(1)
    if x.dim() == 5:
        if x.shape[1] == 1:
            return x
        if x.shape[-1] == 1:
            return x.permute(0, 4, 1, 2, 3).contiguous()
        raise ValueError(
            "Only single-channel inputs are supported. Provide x shaped "
            "(B,D,H,W), (B,D,H,W,1), or (B,1,D,H,W)."
        )
    raise ValueError(f"Expected a 4D or 5D tensor, got shape {tuple(x.shape)}")


######################
##### Label loss #####
######################
def centroid_dist_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    smooth_nr: float = EPS,
    smooth_dr: float = EPS,
) -> torch.Tensor:
    """
    Compute the average distance between corresponding points, ignoring points
    that are marked as invalid with -1 in all coordinate dimensions.

    This is a functional PyTorch rewrite of the original `CentroidDistScore` loss.

    :param y_true: shape (B, N, C) where C is typically 3
    :param y_pred: shape (B, N, C)
    :return: shape (B,)
    """
    y_true = y_true.to(dtype=torch.float32)
    y_pred = y_pred.to(dtype=torch.float32)

    mask_true = torch.all(y_true == -1.0, dim=-1)  # (B,N)
    mask_pred = torch.all(y_pred == -1.0, dim=-1)  # (B,N)
    mask = mask_true | mask_pred  # (B,N)

    displacement = torch.where(mask.unsqueeze(-1), torch.zeros_like(y_pred), y_pred - y_true)
    distance = torch.linalg.norm(displacement, dim=-1)  # (B,N)

    valid = (~mask).to(dtype=torch.float32)
    return (distance.sum(dim=-1) + smooth_nr) / (valid.sum(dim=-1) + smooth_dr)


######################
##### Image loss #####
#####################
def local_normalized_cross_correlation(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    kernel_size: int = 9,
    kernel_type: Literal["gaussian", "rectangular", "triangular"] = "rectangular",
    smooth_dr: float = EPS,
) -> torch.Tensor:
    """
    Local squared zero-normalized cross-correlation (LNCC), returned per-batch.

    Denote y_true as t and y_pred as p. Consider a window having n elements.
    Each position in the window corresponds a weight w_i for i=1:n.

    The local squared ZNCC is:

        E[(t-E[t])*(p-E[p])] ** 2 / V[t] / V[p]

    where expectations/variances are computed locally using a separable kernel.

    Notes:
      - Only single-channel inputs are supported (matches original code).
      - Accepts input shapes (B,D,H,W), (B,D,H,W,1) or (B,1,D,H,W).

    :return: tensor of shape (B,)
    """
    kernel_fn = {
        "gaussian": utils.gaussian_kernel1d,
        "rectangular": utils.rectangular_kernel1d,
        "triangular": utils.triangular_kernel1d,
    }.get(kernel_type)
    if kernel_fn is None:
        raise ValueError(f"Wrong kernel_type {kernel_type!r}. Choose from gaussian/rectangular/triangular.")

    y_true_5d = _ensure_5d_single_channel(y_true).to(dtype=torch.float32)
    y_pred_5d = _ensure_5d_single_channel(y_pred).to(dtype=torch.float32)

    kernel = kernel_fn(kernel_size=kernel_size, device=y_true_5d.device, dtype=y_true_5d.dtype)
    kernel_vol = (kernel.sum()) ** 3  # separable kernel volume

    # t = y_true, p = y_pred
    t2 = y_true_5d * y_true_5d
    p2 = y_pred_5d * y_pred_5d
    tp = y_true_5d * y_pred_5d

    # sums over kernel (still scaled by E[1])
    t_sum = utils.separable_filter(y_true_5d, kernel=kernel)  # E[t] * E[1]
    p_sum = utils.separable_filter(y_pred_5d, kernel=kernel)  # E[p] * E[1]
    t2_sum = utils.separable_filter(t2, kernel=kernel)        # E[t^2] * E[1]
    p2_sum = utils.separable_filter(p2, kernel=kernel)        # E[p^2] * E[1]
    tp_sum = utils.separable_filter(tp, kernel=kernel)        # E[tp] * E[1]

    t_avg = t_sum / kernel_vol
    p_avg = p_sum / kernel_vol

    cross = tp_sum - p_avg * t_sum
    t_var = t2_sum - t_avg * t_sum
    p_var = p2_sum - p_avg * p_sum

    # ensure variance >= 0
    t_var = torch.clamp(t_var, min=0.0)
    p_var = torch.clamp(p_var, min=0.0)

    ncc = (cross * cross) / (t_var * p_var + smooth_dr)

    # mean over (C,D,H,W) => per-batch
    # return ncc.mean(dim=(1, 2, 3, 4))
    return ncc


def global_normalized_cross_correlation(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Global squared zero-normalized cross-correlation (GNCC), returned per-batch.

    Accepts any shape (B, ...). Computes statistics across all dims except batch.

    :return: tensor of shape (B,)
    """
    if y_true.dim() < 2:
        raise ValueError("y_true must have at least a batch dimension.")
    if y_pred.shape != y_true.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {tuple(y_true.shape)} vs {tuple(y_pred.shape)}")

    y_true = y_true.to(dtype=torch.float32)
    y_pred = y_pred.to(dtype=torch.float32)

    dims = tuple(range(1, y_true.dim()))
    mu_pred = y_pred.mean(dim=dims, keepdim=True)
    mu_true = y_true.mean(dim=dims, keepdim=True)

    var_pred = y_pred.var(dim=dims, unbiased=False)
    var_true = y_true.var(dim=dims, unbiased=False)

    numerator = torch.abs(((y_pred - mu_pred) * (y_true - mu_true)).mean(dim=dims))
    return (numerator * numerator) / (var_pred * var_true + eps)


###########################################
##### Deformation/Regularization loss #####
###########################################
def nonrigid_penalty(
    ddf: torch.Tensor,
    *,
    img_size: Tuple[int, int, int],
    l1: bool = False,
) -> torch.Tensor:
    """
    Penalize non-rigid deformations by comparing local spatial derivatives of the DDF
    against the spatial derivatives of a reference grid (matches the original Layer).

    :param ddf: shape (B, D, H, W, 3)
    :param img_size: tuple (D,H,W) used to build the reference grid
    :param l1: if True, L1-like version; else L2-like version
    :return: tensor of shape (B,)
    """
    if ddf.dim() != 5 or ddf.shape[-1] != 3:
        raise ValueError(f"Expected ddf shaped (B,D,H,W,3), got {tuple(ddf.shape)}")

    grid_ref = utils.get_reference_grid(grid_size=img_size, device=ddf.device, dtype=ddf.dtype).unsqueeze(0)
    ddf_ref = -grid_ref  # (1,D,H,W,3)

    delta = ddf - ddf_ref
    dfdx = utils.gradient_dxyz(delta, utils.gradient_dx)
    dfdy = utils.gradient_dxyz(delta, utils.gradient_dy)
    dfdz = utils.gradient_dxyz(delta, utils.gradient_dz)

    if l1:
        norms = torch.abs(utils.stable_f(torch.abs(dfdx) + torch.abs(dfdy) + torch.abs(dfdz)) - 2.0)
    else:
        norms = torch.abs(utils.stable_f(dfdx**2 + dfdy**2 + dfdz**2) - 2.0)

    return norms.mean(dim=(1, 2, 3, 4))


def difference_norm(
    ddf: torch.Tensor,
    *,
    l1: bool = False,
) -> torch.Tensor:
    """
    Average displacement magnitude of a DDF using elementwise L1 or L2.

    :param ddf: shape (B, D, H, W, 3)
    :param l1: if True -> mean(abs(ddf)); else -> mean(ddf^2)
    :return: tensor of shape (B,)
    """
    if ddf.dim() != 5 or ddf.shape[-1] != 3:
        raise ValueError(f"Expected ddf shaped (B,D,H,W,3), got {tuple(ddf.shape)}")

    norms = torch.abs(ddf) if l1 else (ddf * ddf)
    return norms.mean(dim=(1, 2, 3, 4))


#############################
##### Utility metric (np) ###
#############################
def calculate_ncc(moving: np.ndarray, fixed: np.ndarray) -> float:
    """
    Computes the NCC (Normalized Cross-Correlation) of two image arrays
    `moving` and `fixed` corresponding to a registration.

    This mirrors the original numpy implementation.
    """
    if fixed.shape != moving.shape:
        raise ValueError("Fixed and moving images must have the same shape.")

    med_f = np.median(np.max(fixed, axis=2))
    med_m = np.median(np.max(moving, axis=2))

    fixed_new = np.maximum(fixed - med_f, 0)
    moving_new = np.maximum(moving - med_m, 0)

    mu_f = np.mean(fixed_new)
    mu_m = np.mean(moving_new)

    fixed_new = (fixed_new / mu_f) - 1
    moving_new = (moving_new / mu_m) - 1

    numerator = np.sum(fixed_new * moving_new)
    denominator = np.sqrt(np.sum(fixed_new**2) * np.sum(moving_new**2))

    return float(numerator / denominator)


###############################################
##### Registration ZNCC time-series utils #####
###############################################

def zncc_2d(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Zero-normalized cross-correlation between two 2-D arrays."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    num = np.sum(a_centered * b_centered)
    den = np.sqrt(np.sum(a_centered ** 2) * np.sum(b_centered ** 2)) + eps
    return float(num / den)


def zncc_per_zslice(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    """
    Per-z-slice ZNCC between *ref* and *mov*, both shaped (Z, H, W).

    Returns an array of shape (Z,) with scalar ZNCC per slice.
    """
    if ref.shape != mov.shape:
        raise ValueError(
            f"ref and mov must have the same shape, got {ref.shape} vs {mov.shape}"
        )
    Z = ref.shape[0]
    return np.array([zncc_2d(ref[z], mov[z]) for z in range(Z)])


def parse_zoom_factor(reg_dir: str) -> int:
    """
    Extract the integer zoom factor from a directory name containing 'zoom<N>'.

    Returns 1 if 'zoom' is not present.
    """
    m = re.search(r"zoom(\d+)", reg_dir)
    if m:
        return int(m.group(1))
    return 1


def load_registered_paths(reg_dir: str):
    """
    Return a sorted list of .tif paths inside *reg_dir*.
    """
    paths = glob.glob(os.path.join(reg_dir, "*.tif"))
    paths = sorted(paths, key=lambda p: int(os.path.basename(p).split(".")[0]))
    return paths


def compute_zncc_timeseries(
    fixed_img: np.ndarray,
    reg_dir: str,
    channel: int = 1,
) -> np.ndarray:
    """
    Compute per-z-slice ZNCC between a fixed/reference image and every frame
    in a registered time-series directory.

    Parameters
    ----------
    fixed_img : np.ndarray
        Reference volume of shape (Z_fixed, C, H, W).
    reg_dir : str
        Directory containing registered .tif files named 0000.tif, 0001.tif, ...
        Each .tif has shape (Z_reg, C, H, W).
        If the directory name contains 'zoom<N>', the fixed image is zoomed
        along Z by that factor so its Z dimension matches the registered images.
    channel : int
        Channel index to correlate (default 1, i.e. the 2nd channel).

    Returns
    -------
    zncc : np.ndarray
        Array of shape (T, Z_reg) with the per-z-slice ZNCC at each time point.
    """
    reg_paths = load_registered_paths(reg_dir)
    T = len(reg_paths)
    if T == 0:
        raise ValueError(f"No .tif files found in {reg_dir}")

    # Determine zoom factor from directory name
    zoom_factor = parse_zoom_factor(os.path.basename(reg_dir))
    zoom_factor = 1

    # Extract the channel from the fixed image and zoom if needed
    ref = fixed_img[:, channel, :, :]  # (Z_fixed, H, W)
    if zoom_factor > 1:
        ref = ndi.zoom(ref, zoom=(zoom_factor, 1, 1), order=1)

    # Read first registered frame to confirm Z matches
    first_frame = tifffile.imread(reg_paths[0])
    Z_reg = first_frame.shape[0]
    if ref.shape[0] != Z_reg:
        raise ValueError(
            f"After zoom, reference Z={ref.shape[0]} does not match "
            f"registered Z={Z_reg}. Check zoom factor."
        )

    zncc = np.zeros((T, Z_reg))
    for i in tqdm(range(T), desc="ZNCC time-series"):
        mov = tifffile.imread(reg_paths[i])[:, channel, :, :]  # (Z_reg, H, W)
        zncc[i] = zncc_per_zslice(ref, mov)

    return zncc


def plot_zncc_timeseries(
    zncc: np.ndarray,
    *,
    ax=None,
    show_mean: bool = False,
    alpha_slices: float = 0.5,
    label: str = "mean ZNCC",
    title: str = "ZNCC over time",
    cmap: str = "viridis",
):
    """
    Plot ZNCC vs frame index.

    Parameters
    ----------
    zncc : np.ndarray
        Shape (T, Z).  Per-z-slice ZNCC at each time point.
    ax : matplotlib Axes, optional
        If None, a new figure is created.
    show_mean : bool
        If True, the mean ZNCC line is drawn.
    alpha_slices : float
        Opacity for individual z-slice traces.
    label : str
        Legend label for the mean line.
    title : str
        Axes title.
    cmap : str
        Matplotlib colormap name used to color individual z-slice traces.

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    T, Z = zncc.shape
    frames = np.arange(T)
    mean_zncc = zncc.mean(axis=1)
    sem_zncc = zncc.std(axis=1) / np.sqrt(Z)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    else:
        fig = ax.figure

    colors = plt.get_cmap(cmap)(np.linspace(0, 1, Z))
    for z in range(Z):
        ax.plot(frames, zncc[:, z], alpha=alpha_slices, lw=0.5, color=colors[z])

    # ax.fill_between(frames, mean_zncc - sem_zncc, mean_zncc + sem_zncc,
    #                  alpha=0.3, color="k", label="±SEM")
    if show_mean:
        ax.plot(frames, mean_zncc, color="k", lw=1.5, label=label)

    ax.set_xlabel("Frame")
    ax.set_ylabel("ZNCC")
    ax.set_title(title)
    # ax.legend()
    return fig, ax


###############################################
##### Additional registration quality metrics #
###############################################

def ssim_per_zslice(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    """
    Per-z-slice SSIM between *ref* and *mov*, both shaped (Z, H, W).

    Returns an array of shape (Z,).
    """
    from skimage.metrics import structural_similarity
    if ref.shape != mov.shape:
        raise ValueError(
            f"ref and mov must have the same shape, got {ref.shape} vs {mov.shape}"
        )
    Z = ref.shape[0]
    data_range = max(ref.max() - ref.min(), mov.max() - mov.min(), 1e-8)
    return np.array([
        structural_similarity(ref[z], mov[z], data_range=data_range)
        for z in range(Z)
    ])


def rmse_per_zslice(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    """
    Per-z-slice RMSE between *ref* and *mov*, both shaped (Z, H, W).

    Returns an array of shape (Z,).
    """
    if ref.shape != mov.shape:
        raise ValueError(
            f"ref and mov must have the same shape, got {ref.shape} vs {mov.shape}"
        )
    Z = ref.shape[0]
    return np.array([
        np.sqrt(np.mean((ref[z].astype(np.float64) - mov[z].astype(np.float64)) ** 2))
        for z in range(Z)
    ])


def dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Dice coefficient between two binary masks."""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.sum(a & b)
    total = np.sum(a) + np.sum(b)
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def compute_metric_timeseries(
    fixed_img: np.ndarray,
    reg_dir: str,
    channel: int = 1,
    metric_fn=None,
) -> np.ndarray:
    """
    Compute a per-z-slice metric between a fixed image and every frame
    in a registered time-series directory.

    Parameters
    ----------
    fixed_img : np.ndarray
        Reference volume of shape (Z, C, H, W).
    reg_dir : str
        Directory containing registered .tif files.
    channel : int
        Channel index (default 1 = RFP).
    metric_fn : callable
        Function (ref_ZHW, mov_ZHW) -> np.ndarray of shape (Z,).

    Returns
    -------
    result : np.ndarray
        Array of shape (T, Z) with the per-z-slice metric at each time point.
    """
    if metric_fn is None:
        metric_fn = zncc_per_zslice

    reg_paths = load_registered_paths(reg_dir)
    T = len(reg_paths)
    if T == 0:
        raise ValueError(f"No .tif files found in {reg_dir}")

    ref = fixed_img[:, channel, :, :]  # (Z, H, W)
    first_frame = tifffile.imread(reg_paths[0])
    Z_reg = first_frame.shape[0]
    if ref.shape[0] != Z_reg:
        raise ValueError(
            f"Reference Z={ref.shape[0]} does not match registered Z={Z_reg}."
        )

    result = np.zeros((T, Z_reg))
    for i in tqdm(range(T), desc=f"{metric_fn.__name__} time-series"):
        mov = tifffile.imread(reg_paths[i])[:, channel, :, :]
        result[i] = metric_fn(ref, mov)

    return result


def mask_overlap_timeseries(dataset_dir: str) -> np.ndarray:
    """
    Dice coefficient between the fixed mask and each moving mask.

    Returns an array of shape (T,) with Dice at each time point.
    """
    fixed_mask_fn = glob.glob(os.path.join(dataset_dir, 'fixed_mask_[0-9][0-9][0-9][0-9]*.tif'))
    if not fixed_mask_fn:
        raise FileNotFoundError(f"No fixed_mask_*.tif found in {dataset_dir}")
    fixed_mask = tifffile.imread(fixed_mask_fn[0]).astype(bool)

    masks_dir = os.path.join(dataset_dir, 'masks')
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"masks/ directory not found in {dataset_dir}")

    mask_paths = sorted(glob.glob(os.path.join(masks_dir, '*.tif')),
                        key=lambda p: int(os.path.basename(p).split('.')[0]))
    T = len(mask_paths)
    dice_arr = np.zeros(T)
    for i in tqdm(range(T), desc="Dice overlap"):
        mov_mask = tifffile.imread(mask_paths[i]).astype(bool)
        dice_arr[i] = dice_coefficient(fixed_mask, mov_mask)

    return dice_arr


###############################################
##### High-level benchmark entry points #######
###############################################

def _find_fixed_and_regdir(dataset_dir: str):
    """Locate the fixed frame .tif and registered_elastix directory."""
    fixed_fn = glob.glob(os.path.join(dataset_dir, 'fixed_[0-9][0-9][0-9][0-9]*.tif'))
    if not fixed_fn:
        raise FileNotFoundError(f"No fixed_*.tif found in {dataset_dir}")
    fixed_path = fixed_fn[0]

    reg_dir = os.path.join(dataset_dir, 'registered_elastix')
    if not os.path.isdir(reg_dir):
        raise FileNotFoundError(f"registered_elastix/ not found in {dataset_dir}")

    return fixed_path, reg_dir


def benchmark_registration(
    dataset_dir: str,
    channel: int = 1,
) -> np.ndarray:
    """
    Compute per-z-slice ZNCC between the fixed frame and every registered
    frame, save results, and produce a plot.

    Outputs saved to *dataset_dir*:
        - zncc_timeseries.npy   (T, Z)
        - zncc_summary.csv      frame, mean_zncc, std_zncc, min_zncc, median_zncc
        - zncc_timeseries.png

    Returns the (T, Z) ZNCC array.
    """
    import matplotlib.pyplot as plt

    fixed_path, reg_dir = _find_fixed_and_regdir(dataset_dir)
    fixed_img = tifffile.imread(fixed_path).astype(np.float32)

    zncc = compute_zncc_timeseries(fixed_img, reg_dir, channel=channel)

    # save npy
    np.save(os.path.join(dataset_dir, 'zncc_timeseries.npy'), zncc)

    # save summary csv
    T = zncc.shape[0]
    summary = pd.DataFrame({
        'frame': np.arange(T),
        'mean_zncc': zncc.mean(axis=1),
        'std_zncc': zncc.std(axis=1),
        'min_zncc': zncc.min(axis=1),
        'median_zncc': np.median(zncc, axis=1),
    })
    summary.to_csv(os.path.join(dataset_dir, 'zncc_summary.csv'), index=False)

    # plot
    fig, ax = plot_zncc_timeseries(zncc, show_mean=True, title="Registration ZNCC over time")
    fig.savefig(os.path.join(dataset_dir, 'zncc_timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: zncc_timeseries.npy, zncc_summary.csv, zncc_timeseries.png")
    return zncc


def compute_all_benchmarks(
    dataset_dir: str,
    channel: int = 1,
) -> dict:
    """
    Run ZNCC, SSIM, RMSE, and Dice benchmarks on a registered dataset.

    Saves per-metric .npy arrays and a combined summary CSV.
    Returns a dict of {metric_name: np.ndarray}.
    """
    fixed_path, reg_dir = _find_fixed_and_regdir(dataset_dir)
    fixed_img = tifffile.imread(fixed_path).astype(np.float32)

    results = {}

    # ZNCC
    print("Computing ZNCC...")
    zncc = compute_zncc_timeseries(fixed_img, reg_dir, channel=channel)
    np.save(os.path.join(dataset_dir, 'zncc_timeseries.npy'), zncc)
    results['zncc'] = zncc

    # SSIM
    print("Computing SSIM...")
    ssim = compute_metric_timeseries(fixed_img, reg_dir, channel=channel,
                                     metric_fn=ssim_per_zslice)
    np.save(os.path.join(dataset_dir, 'ssim_timeseries.npy'), ssim)
    results['ssim'] = ssim

    # RMSE
    print("Computing RMSE...")
    rmse = compute_metric_timeseries(fixed_img, reg_dir, channel=channel,
                                     metric_fn=rmse_per_zslice)
    np.save(os.path.join(dataset_dir, 'rmse_timeseries.npy'), rmse)
    results['rmse'] = rmse

    # Dice (mask overlap — not per-z-slice, just per frame)
    print("Computing Dice overlap...")
    try:
        dice = mask_overlap_timeseries(dataset_dir)
        np.save(os.path.join(dataset_dir, 'dice_timeseries.npy'), dice)
        results['dice'] = dice
    except FileNotFoundError as e:
        print(f"  Skipping Dice: {e}")

    # Combined summary CSV
    T = zncc.shape[0]
    summary = pd.DataFrame({
        'frame': np.arange(T),
        'mean_zncc': zncc.mean(axis=1),
        'mean_ssim': ssim.mean(axis=1),
        'mean_rmse': rmse.mean(axis=1),
    })
    if 'dice' in results:
        summary['dice'] = dice
    summary.to_csv(os.path.join(dataset_dir, 'registration_benchmark.csv'), index=False)

    print(f"Saved benchmark results to {dataset_dir}")
    return results


def plot_benchmark_summary(dataset_dir: str, save: bool = True):
    """
    Multi-panel figure of all saved benchmark metrics.

    Loads .npy files from *dataset_dir* and produces a 2x2 plot.
    """
    import matplotlib.pyplot as plt

    zncc_path = os.path.join(dataset_dir, 'zncc_timeseries.npy')
    ssim_path = os.path.join(dataset_dir, 'ssim_timeseries.npy')
    rmse_path = os.path.join(dataset_dir, 'rmse_timeseries.npy')
    dice_path = os.path.join(dataset_dir, 'dice_timeseries.npy')

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=150)

    # ZNCC
    if os.path.exists(zncc_path):
        zncc = np.load(zncc_path)
        ax = axes[0, 0]
        mean_z = zncc.mean(axis=1)
        ax.plot(mean_z, 'k', lw=0.8)
        ax.fill_between(np.arange(len(mean_z)),
                        zncc.min(axis=1), zncc.max(axis=1),
                        alpha=0.2, color='steelblue')
        ax.set_ylabel('ZNCC')
        ax.set_title('ZNCC (mean ± range)')
    else:
        axes[0, 0].set_visible(False)

    # SSIM
    if os.path.exists(ssim_path):
        ssim = np.load(ssim_path)
        ax = axes[0, 1]
        mean_s = ssim.mean(axis=1)
        ax.plot(mean_s, 'k', lw=0.8)
        ax.fill_between(np.arange(len(mean_s)),
                        ssim.min(axis=1), ssim.max(axis=1),
                        alpha=0.2, color='coral')
        ax.set_ylabel('SSIM')
        ax.set_title('SSIM (mean ± range)')
    else:
        axes[0, 1].set_visible(False)

    # RMSE
    if os.path.exists(rmse_path):
        rmse = np.load(rmse_path)
        ax = axes[1, 0]
        mean_r = rmse.mean(axis=1)
        ax.plot(mean_r, 'k', lw=0.8)
        ax.fill_between(np.arange(len(mean_r)),
                        rmse.min(axis=1), rmse.max(axis=1),
                        alpha=0.2, color='seagreen')
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE (mean ± range)')
    else:
        axes[1, 0].set_visible(False)

    # Dice
    if os.path.exists(dice_path):
        dice = np.load(dice_path)
        ax = axes[1, 1]
        ax.plot(dice, 'k', lw=0.8)
        ax.set_ylabel('Dice')
        ax.set_title('Mask Overlap (Dice)')
    else:
        axes[1, 1].set_visible(False)

    for ax in axes.flat:
        if ax.get_visible():
            ax.set_xlabel('Frame')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save:
        out_path = os.path.join(dataset_dir, 'registration_benchmark.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")

    return fig, axes
