# These functions were adapted from `deepreg/model/kernel.py` in the 'DeepReg' repository
# on GitHub. The original implementation can be found at:
# https://github.com/flavell-lab/DeepReg/deepreg/loss
# The code is used under the MIT License.
from __future__ import annotations

from typing import Callable, List, Tuple, Union

import math
import torch
import torch.nn.functional as F


def get_reference_grid(
    grid_size: Union[Tuple[int, ...], List[int]],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate a 3D grid with given size.

    :param grid_size: list or tuple of size 3, [dim1, dim2, dim3]
    :return: shape = (dim1, dim2, dim3, 3),
             grid[i, j, k, :] = [i j k]
    """
    if len(grid_size) != 3:
        raise ValueError(f"grid_size must have length 3, got {grid_size!r}")

    r0 = torch.arange(grid_size[0], device=device)
    r1 = torch.arange(grid_size[1], device=device)
    r2 = torch.arange(grid_size[2], device=device)

    # torch.meshgrid supports indexing='ij' in modern torch versions.
    try:
        mesh_grid = torch.meshgrid(r0, r1, r2, indexing="ij")
    except TypeError:  # pragma: no cover
        mesh_grid = torch.meshgrid(r0, r1, r2)

    grid = torch.stack(mesh_grid, dim=-1).to(dtype=dtype)  # (d1,d2,d3,3)
    return grid


def gaussian_kernel1d(
    kernel_size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Return the 1D Gaussian kernel for LocalNormalizedCrossCorrelation.

    :param kernel_size: scalar, size of the 1-D kernel
    :return: filters, of shape (kernel_size, )
    """
    mean = (kernel_size - 1) / 2.0
    sigma = kernel_size / 3.0

    grid = torch.arange(0, kernel_size, device=device, dtype=dtype)
    filters = torch.exp(-((grid - mean) ** 2) / (2.0 * (sigma**2)))
    return filters


def triangular_kernel1d(
    kernel_size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Return the 1D triangular kernel for LocalNormalizedCrossCorrelation.

    This mirrors the TensorFlow implementation which:
      1) builds a 1D "boxy" kernel (zeros at ends, ones in the middle),
      2) smooths it by convolving with a ones filter using SAME padding.

    :param kernel_size: scalar, size of the 1-D kernel (must be odd and >= 3)
    :return: kernel_weights, of shape (kernel_size, )
    """
    if kernel_size < 3 or (kernel_size % 2) == 0:
        raise ValueError("kernel_size must be odd and >= 3")

    padding = kernel_size // 2
    base = (
        [0] * math.ceil(padding / 2)
        + [1] * (kernel_size - padding)
        + [0] * math.floor(padding / 2)
    )
    kernel = torch.tensor(base, device=device, dtype=dtype).view(1, 1, -1)  # (1,1,L)

    filt_len = kernel_size - padding
    filt = torch.ones((1, 1, filt_len), device=device, dtype=dtype)

    # TensorFlow SAME padding can be asymmetric for even kernel lengths.
    total_pad = filt_len - 1
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left
    kernel_padded = F.pad(kernel, (pad_left, pad_right))

    out = F.conv1d(kernel_padded, filt, stride=1, padding=0)  # (1,1,L)
    return out.view(-1)


def rectangular_kernel1d(
    kernel_size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Return the 1D rectangular kernel for LocalNormalizedCrossCorrelation.

    :param kernel_size: scalar, size of the 1-D kernel
    :return: kernel_weights, of shape (kernel_size, )
    """
    return torch.ones((kernel_size,), device=device, dtype=dtype)


def _pad_3d(
    x: torch.Tensor,
    *,
    pad_d: tuple[int, int] = (0, 0),
    pad_h: tuple[int, int] = (0, 0),
    pad_w: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    # F.pad for 5D uses (w_left, w_right, h_left, h_right, d_left, d_right)
    return F.pad(x, (pad_w[0], pad_w[1], pad_h[0], pad_h[1], pad_d[0], pad_d[1]))


def _same_pad_1d(k: int) -> tuple[int, int]:
    # For stride=1 SAME padding: total padding = k-1, left = floor, right = ceil
    total = k - 1
    left = total // 2
    right = total - left
    return left, right


def separable_filter(tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Create a 3D separable filter using three 1D convolutions.

    TensorFlow code expects:
      - tensor: (batch, dim1, dim2, dim3, 1)  (channels-last)

    This PyTorch version accepts either:
      - (batch, dim1, dim2, dim3, 1)  (channels-last), or
      - (batch, 1, dim1, dim2, dim3)  (channels-first, preferred)

    and returns the same layout as the input.

    :param tensor: shape = (B, D, H, W, 1) or (B, 1, D, H, W)
    :param kernel: shape = (K,)
    :return: filtered tensor with the same shape/layout as input
    """
    if tensor.dim() != 5:
        raise ValueError(f"tensor must be 5D, got shape {tuple(tensor.shape)}")

    channels_last = (tensor.shape[-1] == 1) and (tensor.shape[1] != 1)
    if channels_last:
        x = tensor.permute(0, 4, 1, 2, 3).contiguous()  # (B,1,D,H,W)
    else:
        if tensor.shape[1] != 1:
            raise ValueError(
                "Expected a single channel. Provide tensor shaped (B,1,D,H,W) "
                "or (B,D,H,W,1)."
            )
        x = tensor

    k = kernel.to(device=x.device, dtype=x.dtype).view(-1)
    K = int(k.numel())
    w_d = k.view(1, 1, K, 1, 1)
    w_h = k.view(1, 1, 1, K, 1)
    w_w = k.view(1, 1, 1, 1, K)

    pd = _same_pad_1d(K)
    ph = _same_pad_1d(K)
    pw = _same_pad_1d(K)

    x = F.conv3d(_pad_3d(x, pad_d=pd), w_d, stride=1, padding=0)
    x = F.conv3d(_pad_3d(x, pad_h=ph), w_h, stride=1, padding=0)
    x = F.conv3d(_pad_3d(x, pad_w=pw), w_w, stride=1, padding=0)

    if channels_last:
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B,D,H,W,1)
    return x


def gradient_dx(fx: torch.Tensor) -> torch.Tensor:
    """
    Calculate gradients on x-axis of a 3D tensor using central finite difference.

    :param fx: shape = (batch, m_dim1, m_dim2, m_dim3)
    :return: shape = (batch, m_dim1-2, m_dim2-2, m_dim3-2)
    """
    return (fx[:, 2:, 1:-1, 1:-1] - fx[:, :-2, 1:-1, 1:-1]) / 2.0


def gradient_dy(fy: torch.Tensor) -> torch.Tensor:
    """
    Calculate gradients on y-axis of a 3D tensor using central finite difference.

    :param fy: shape = (batch, m_dim1, m_dim2, m_dim3)
    :return: shape = (batch, m_dim1-2, m_dim2-2, m_dim3-2)
    """
    return (fy[:, 1:-1, 2:, 1:-1] - fy[:, 1:-1, :-2, 1:-1]) / 2.0


def gradient_dz(fz: torch.Tensor) -> torch.Tensor:
    """
    Calculate gradients on z-axis of a 3D tensor using central finite difference.

    :param fz: shape = (batch, m_dim1, m_dim2, m_dim3)
    :return: shape = (batch, m_dim1-2, m_dim2-2, m_dim3-2)
    """
    return (fz[:, 1:-1, 1:-1, 2:] - fz[:, 1:-1, 1:-1, :-2]) / 2.0


def gradient_dxyz(fxyz: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """
    Calculate gradients on x,y,z-axis of a tensor using central finite difference.

    :param fxyz: shape = (..., 3) (typically (B,D,H,W,3))
    :param fn: one of gradient_dx/gradient_dy/gradient_dz
    :return: shape = (..., 3) after differentiation (typically (B,D-2,H-2,W-2,3))
    """
    return torch.stack([fn(fxyz[..., i]) for i in (0, 1, 2)], dim=-1)


def stable_f(x: torch.Tensor, min_value: float = 1e-6) -> torch.Tensor:
    """
    Perform the operation f(x) = x + 1/x in a numerically stable way.

    Intended to penalize growing and shrinking equally.

    :param x: Input tensor.
    :param min_value: The minimum value to which x will be clamped.
    :return: The result of the operation.
    """
    max_value = torch.finfo(x.dtype).max if x.is_floating_point() else float("inf")
    x_clamped = torch.clamp(x, min=min_value, max=max_value)
    return x_clamped + 1.0 / x_clamped
