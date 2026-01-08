# These functions were adapted from `deepreg/model/kernel.py` in the 'DeepReg' repository
# on GitHub. The original implementation can be found at:
# https://github.com/flavell-lab/DeepReg/deepreg/loss
# The code is used under the MIT License.
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import torch

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
