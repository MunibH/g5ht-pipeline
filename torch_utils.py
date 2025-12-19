# uses torchcu129 conda env

from typing import Union
import torch

import numpy as np
from numpy.lib.stride_tricks import as_strided

from typing import Union, Tuple

def cross_correlation_torch(
    reference: torch.Tensor,
    template: torch.Tensor,
    stride: Union[tuple, int] = 1,
    normalize: bool = True,
    center: bool = True,
) -> torch.Tensor:
    """Compute the zero-normalized cross-correlation between a reference and a template.

    The output tensor is the result of the batched sliding cross-correlation between
    a multi-channel reference matrix and a template matrix:
     - (normalize = False, center = False): Standard cross-correlation;
     - (normalize = True, center = False): Normalized cross-correlation (NCC);
     - (normalize = False, center = True): Zero cross-correlation (ZCC), generally performs as good as ZNCC, but is faster;
     - (normalize = True, center = True): Zero-normalized cross-correlation (ZNCC);

    Args:
        reference (torch.Tensor): Reference tensor, must be of shape (N, C_in, H_ref, W_ref).
        template (torch.Tensor): Template tensor, must be of shape (C_out, C_in, H_t, W_t).
        stride (Union[tuple, int]): Stride of the sliding window. Default to 1.
        normalize (bool): If True, the output is normalized by the Standard Deviation of the reference and template patches. Default to True.
        center (bool): If True, the output is normalized by the Mean of the reference and template patches. Default to True.

    Returns:
        torch.Tensor: Normalized cross correlation (N, C_out, H_ref, W_ref).

    Raises:
        ValueError: If the input dimensions are not correct.

    Example:
        >>> image = PIL.Image.open(requests.get("https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png", stream=True).raw)
        >>> reference = torchvision.transforms.ToTensor()(image).to(torch.float32)
        >>> template = reference[..., 240:280, 240:300]
        >>> matching = normalized_cross_correlation(reference.unsqueeze(dim=0), template.unsqueeze(dim=0))
        >>> plt.imshow(matching[0].numpy())

    References:
        https://en.wikipedia.org/w/index.php?title=Cross-correlation#Zero-normalized_cross-correlation_(ZNCC)
    """ 

    # Check the input dimensions
    if not (reference.dim() == template.dim() == 4):
        raise ValueError("Reference and template must have 4 dimensions.")

    if reference.shape[1] != template.shape[1]:
        raise ValueError("Reference and template must have the same number of channel (C_in).")

    if (reference.shape[2] < template.shape[2]) or (reference.shape[3] < template.shape[3]):
        raise ValueError(
            "Reference matrix must be larger than template matrix (H_ref>=H_t & W_ref>=W_t)."
        )

    if isinstance(stride, int):
        stride_h = stride_w = stride
    elif isinstance(stride, tuple):
        if len(stride) != 2:
            raise ValueError("Stride must be a 2 dimensional tuple.")
        stride_h, stride_w = stride
    else:
        raise ValueError("Stride must be an integer or a 2 dimensional tuple.")

    # Get the template size
    template_h, template_w = template.shape[-2:]

    # 2D unfold of the batch of reference matrix
    # (N, C_in, H_ref, W_ref) -> (N, C_in, H_out, W_out, H_t, W_t)
    reference = reference.contiguous()
    reference_stride = reference.stride()
    reference_patches = reference.as_strided(
        size=(
            reference.shape[0],
            reference.shape[1],
            (reference.shape[2] - template_h) // stride_h + 1,
            (reference.shape[3] - template_w) // stride_w + 1,
            template_h,
            template_w,
        ),
        stride=(
            reference_stride[0],
            reference_stride[1],
            stride_h * reference_stride[2],
            stride_w * reference_stride[3],
            reference_stride[2],
            reference_stride[3],
        ),
    )

    # Rearange dimensions to ease Standard Deviation and Mean computation
    # (N, C_in, H_out, W_out, H_t, W_t) -> (N, H_out, W_out, C_in, H_t, W_t)
    reference_patches = reference_patches.permute(0, 2, 3, 1, 4, 5)

    if normalize:
        # Compute the Standard Deviation and Mean of the reference and template patches
        # (N, H_out, W_out, C_in, H_t, W_t) -> (N, H_out, W_out, C_in)
        reference_std, reference_mean = torch.std_mean(
            reference_patches, dim=(-1, -2), keepdim=False
        )
        # (C_out, C_in, H_out, W_out) -> (C_out, C_in)
        template_std, template_mean = torch.std_mean(template, dim=(-1, -2), keepdim=False)
    elif center:
        # Compute the Mean of the reference and template patches
        # (N, H_out, W_out, C_in, H_t, W_t) -> (N, H_out, W_out, C_in)
        reference_mean = torch.mean(reference_patches, dim=(-1, -2), keepdim=False)
        # (C_out, C_in, H_out, W_out) -> (C_out, C_in)
        template_mean = torch.mean(template, dim=(-1, -2), keepdim=False)

    if normalize or center:
        # Add empty dimensions to the template and reference mean to ease subsequent operations
        # (N, H_out, W_out, C_in) -> (N, H_out, W_out, C_in, 1, 1)
        reference_mean = reference_mean[..., None, None]
        # (C_out, C_in) -> (C_out, C_in, 1, 1)
        template_mean = template_mean[..., None, None]

    # Compute the covariance between the reference and template patches
    # Add a dimension to the reference patches to allow broadcasting
    # (N, H_out, W_out, C_in, H_t, W_t) x (C_out, C_in, H_t, W_t) -> (N, H_out, W_out, C_out, C_in)
    # Translated using Einstein Summation: nxychw,mchw->nxymc
    if center:
        matching = torch.einsum(
            'nxychw,mchw->nxymc',
            (reference_patches - reference_mean),
            (template - template_mean)
        )
        # The above operation is equivalent to the following one.
        # However, the `einsum` lazy evaluation avoid the creation of an extremely large new temporary tensor.
        # matching = (
        #     (reference_patches - reference_mean)[:, :, :, None, ...] * (template - template_mean)
        # ).sum(dim=(-1, -2))

    else:
        # TODO: Implement lazy computation similar to the one above
        matching = (reference_patches[:, :, :, None, ...] * template).sum(dim=(-1, -2))

    # Place C_out as first dimension to perform the normalization
    # (N, H_out, W_out, C_out, C_in) -> (C_out, N, H_out, W_out, C_in)
    matching = matching.permute( 3, 0, 1, 2, 4)

    if normalize:
        # Normalize by the Standard Deviation of the reference and template patches
        # (C_out, N, H_out, W_out, C_in) / (N, H_out, W_out, C_in) -> (C_out, N, H_out, W_out, C_in)
        matching.div_(reference_std)
        # (C_out, N, H_out, W_out, C_in) / (C_out, 1, 1, 1, C_in) -> (C_out, N, H_out, W_out, C_in)
        matching.div_(template_std[:, None, None, None, :])

    # Sum the matching score over the input channels
    # (C_out, N, H_out, W_out, C_in) -> (C_out, N, H_out, W_out)
    matching = matching.sum(dim=-1)

    # Place N as first dimension
    # (C_out, N, H_out, W_out) -> (N, C_out, H_out, W_out)
    matching = matching.permute(1, 0, 2, 3)
    
    return matching.contiguous()

def cross_correlation_np(
    reference: np.ndarray,
    template: np.ndarray,
    stride: Union[Tuple[int, int], int] = 1,
    normalize: bool = True,
    center: bool = True,
    ddof: int = 0,  # torch's default may differ across versions; ddof=0 is common for NCC/ZNCC
) -> np.ndarray:
    """
    Numpy version of the provided torch cross_correlation.

    Args:
        reference: (N, C_in, H_ref, W_ref)
        template:  (C_out, C_in, H_t, W_t)
        stride:    int or (stride_h, stride_w)
        normalize: if True, divide by std of reference patches and template
        center:    if True, subtract means (ZCC / ZNCC)
        ddof:      std degrees-of-freedom (0 = population std)

    Returns:
        matching: (N, C_out, H_out, W_out)
    """
    reference = np.asarray(reference)
    template = np.asarray(template)

    if not (reference.ndim == template.ndim == 4):
        raise ValueError("Reference and template must have 4 dimensions.")

    if reference.shape[1] != template.shape[1]:
        raise ValueError("Reference and template must have the same number of channels (C_in).")

    if (reference.shape[2] < template.shape[2]) or (reference.shape[3] < template.shape[3]):
        raise ValueError("Reference must be larger than template (H_ref>=H_t & W_ref>=W_t).")

    if isinstance(stride, int):
        stride_h = stride_w = stride
    elif isinstance(stride, tuple):
        if len(stride) != 2:
            raise ValueError("Stride must be a 2 dimensional tuple.")
        stride_h, stride_w = stride
    else:
        raise ValueError("Stride must be an integer or a 2 dimensional tuple.")

    # Work in float for mean/std and correlation
    out_dtype = np.result_type(reference, template, np.float32)
    reference = np.ascontiguousarray(reference, dtype=out_dtype)
    template  = np.asarray(template, dtype=out_dtype)

    N, C_in, H_ref, W_ref = reference.shape
    C_out, _, H_t, W_t = template.shape

    H_out = (H_ref - H_t) // stride_h + 1
    W_out = (W_ref - W_t) // stride_w + 1

    # (N, C_in, H_ref, W_ref) -> view (N, C_in, H_out, W_out, H_t, W_t)
    sN, sC, sH, sW = reference.strides
    patches = as_strided(
        reference,
        shape=(N, C_in, H_out, W_out, H_t, W_t),
        strides=(sN, sC, stride_h * sH, stride_w * sW, sH, sW),
        writeable=False,
    )
    # -> (N, H_out, W_out, C_in, H_t, W_t)
    patches = patches.transpose(0, 2, 3, 1, 4, 5)

    # Precompute stats as needed
    if normalize:
        ref_mean = patches.mean(axis=(-1, -2))                      # (N, H_out, W_out, C_in)
        ref_std  = patches.std(axis=(-1, -2), ddof=ddof)            # (N, H_out, W_out, C_in)
        tpl_mean = template.mean(axis=(-1, -2))                     # (C_out, C_in)
        tpl_std  = template.std(axis=(-1, -2), ddof=ddof)           # (C_out, C_in)
    elif center:
        ref_mean = patches.mean(axis=(-1, -2))                      # (N, H_out, W_out, C_in)
        tpl_mean = template.mean(axis=(-1, -2))                     # (C_out, C_in)

    # Core matching: output (N, H_out, W_out, C_out, C_in)
    if center:
        # Compute sum((P - mp)*(T - mt)) without materializing (P-mp) or (T-mt):
        # sum(P*T) - mt*sum(P) - mp*sum(T) + mp*mt*(H_t*W_t)
        sum_PT = np.einsum("nxychw,mchw->nxymc", patches, template, optimize=True)
        sum_P  = patches.sum(axis=(-1, -2))                         # (N, H_out, W_out, C_in)
        sum_T  = template.sum(axis=(-1, -2))                        # (C_out, C_in)

        # Broadcast to (N, H_out, W_out, C_out, C_in)
        mt = tpl_mean[None, None, None, :, :]                       # (1,1,1,C_out,C_in)
        mp = ref_mean[:, :, :, None, :]                             # (N,H_out,W_out,1,C_in)

        matching = (
            sum_PT
            - mt * sum_P[:, :, :, None, :]
            - mp * sum_T[None, None, None, :, :]
            + mp * mt * (H_t * W_t)
        )
    else:
        matching = np.einsum("nxychw,mchw->nxymc", patches, template, optimize=True)

    # (N, H_out, W_out, C_out, C_in) -> (C_out, N, H_out, W_out, C_in)
    matching = matching.transpose(3, 0, 1, 2, 4)

    if normalize:
        # Divide by stds (broadcasted)
        matching = matching / ref_std[None, :, :, :, :]             # (1,N,H_out,W_out,C_in)
        matching = matching / tpl_std[:, None, None, None, :]       # (C_out,1,1,1,C_in)

    # Sum over input channels: (C_out, N, H_out, W_out, C_in) -> (C_out, N, H_out, W_out)
    matching = matching.sum(axis=-1)

    # (C_out, N, H_out, W_out) -> (N, C_out, H_out, W_out)
    matching = matching.transpose(1, 0, 2, 3)

    return np.ascontiguousarray(matching)
