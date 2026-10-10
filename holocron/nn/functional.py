# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections.abc import Callable
from math import ceil, floor
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "add2d",
    "complement_cross_entropy",
    "concat_downsample2d",
    "dice_loss",
    "dropblock2d",
    "focal_loss",
    "hard_mish",
    "multilabel_cross_entropy",
    "mutual_channel_loss",
    "nl_relu",
    "norm_conv2d",
    "poly_loss",
    "z_pool",
]


def hard_mish(x: Tensor, inplace: bool = False) -> Tensor:
    """Implements the HardMish activation function

    Args:
        x: input tensor
        inplace: whether the operation should be conducted inplace

    Returns:
        output tensor
    """
    if inplace:
        return x.mul_(0.5 * (x + 2).clamp(min=0, max=2))
    return 0.5 * x * (x + 2).clamp(min=0, max=2)


def nl_relu(x: Tensor, beta: float = 1.0, inplace: bool = False) -> Tensor:
    """Implements the natural logarithm ReLU activation function

    Args:
        x: input tensor
        beta: beta used for NReLU
        inplace: whether the operation should be performed inplace

    Returns:
        output tensor
    """
    if inplace:
        return torch.log(F.relu_(x).mul_(beta).add_(1), out=x)
    return torch.log(1 + beta * F.relu(x))


def focal_loss(
    x: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    gamma: float = 2.0,
) -> Tensor:
    """Implements the focal loss from
    ["Focal Loss for Dense Object Detection"](https://arxiv.org/pdf/1708.02002.pdf)

    Args:
        x: input tensor of shape [N, K, ...]
        target: hard target tensor of shape [N, ...]
        weight: manual rescaling of each class of shape [K]
        ignore_index: specifies target value that is ignored and do not contribute to gradient
        reduction: reduction method
        gamma: gamma parameter of focal loss

    Returns:
        loss reduced with `reduction` method
    """
    # Accumulate dense losses in float32 for low-precision logits.
    if x.dtype in {torch.float16, torch.bfloat16}:
        x = x.float()
    # log(P[class]) = log_softmax(score)[class]
    logpt = F.log_softmax(x, dim=1)

    # Compute pt and logpt only for target classes (the remaining will have a 0 coefficient)
    valid_idxs = target.reshape(-1) != ignore_index
    safe_target = target.reshape(-1).masked_fill(~valid_idxs, 0)
    logpt = logpt.transpose(1, 0).flatten(1).gather(0, safe_target.unsqueeze(0)).squeeze(0)

    # Get P(class)
    pt = logpt.exp()

    # Weight
    if isinstance(weight, Tensor):
        # Tensor type
        if weight.type() != x.data.type():
            weight = weight.type_as(x.data)
        logpt = weight.gather(0, safe_target) * logpt

    # Loss
    loss = cast(Tensor, -1 * (1 - pt) ** gamma * logpt).masked_fill(~valid_idxs, 0)

    # Loss reduction
    if reduction == "sum":
        loss = loss[valid_idxs].sum()
    elif reduction == "mean":
        loss = loss.sum() / valid_idxs.sum().clamp_min(1)
    else:
        # if no reduction, reshape tensor like target
        loss = loss.view(*target.shape)

    return loss


def concat_downsample2d(x: Tensor, scale_factor: int) -> Tensor:
    """Implements a loss-less downsampling operation described in
    ["YOLO9000: Better, Faster, Stronger"](https://pjreddie.com/media/files/papers/YOLO9000.pdf)
    by stacking adjacent information on the channel dimension.

    Args:
        x: input tensor of shape [N, C, H, W]
        scale_factor: spatial scaling factor

    Returns:
        downsampled tensor of shape [N, scale_factor ** 2 * C, H / scale_factor, W / scale_factor]

    Raises:
        AssertionError: if the spatial size of the input tensor is not multiples of `scale_factor`
    """
    b, c, h, w = x.shape

    if (h % scale_factor != 0) or (w % scale_factor != 0):
        raise AssertionError("Spatial size of input tensor must be multiples of `scale_factor`")

    # N * C * H * W --> N * C * (H/scale_factor) * scale_factor * (W/scale_factor) * scale_factor
    x = x.view(b, c, h // scale_factor, scale_factor, w // scale_factor, scale_factor)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
    return x.view(b, c * int(scale_factor**2), h // scale_factor, w // scale_factor)


def z_pool(x: Tensor, dim: int) -> Tensor:
    """Z-pool layer from ["Rotate to Attend: Convolutional Triplet Attention Module"](https://arxiv.org/pdf/2010.03045.pdf).

    Args:
        x: input tensor
        dim: dimension to pool

    Returns:
        z-pooled tensor
    """
    return torch.cat([x.max(dim, keepdim=True).values, x.mean(dim, keepdim=True)], dim=dim)


def multilabel_cross_entropy(
    x: Tensor, target: Tensor, weight: Tensor | None = None, ignore_index: int = -100, reduction: str = "mean"
) -> Tensor:
    """Implements the cross entropy loss for multi-label targets

    Args:
        x: input tensor of shape [N, K, ...]
        target: target tensor of shape [N, K, ...]
        weight: manual rescaling of each class of shape [K]
        ignore_index: specifies target value that is ignored and do not contribute to gradient
        reduction: reduction method

    Returns:
        loss reduced with `reduction` method
    """
    # log(P[class]) = log_softmax(score)[class]
    logpt = F.log_softmax(x, dim=1)

    # Ignore index (set loss contribution to 0)
    valid_idxs = torch.ones(logpt.shape[1], dtype=torch.bool, device=x.device)
    if ignore_index >= 0 and ignore_index < x.shape[1]:
        valid_idxs[ignore_index] = False

    # Weight
    if weight is not None:
        # Tensor type
        if weight.type() != x.data.type():
            weight = weight.type_as(x.data)
        logpt = logpt * weight.view(1, -1, *([1] * (x.ndim - 2)))

    # CE Loss
    loss = -target * logpt

    # Loss reduction
    if reduction == "sum":
        loss = loss[:, valid_idxs].sum()
    else:
        loss = loss[:, valid_idxs].sum(dim=1)
        if reduction == "mean":
            loss = loss.mean()

    return loss


def complement_cross_entropy(
    x: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    gamma: float = -1,
) -> Tensor:
    """Implements the complement cross entropy loss from
    ["Imbalanced Image Classification with Complement Cross Entropy"](https://arxiv.org/pdf/2009.02189.pdf)

    Args:
        x: input tensor of shape [N, K, ...]
        target: target tensor of shape [N, ...]
        weight: manual rescaling of each class of shape [K]
        ignore_index: specifies target value that is ignored and do not contribute to gradient
        reduction: reduction method
        gamma: complement factor

    Returns:
        loss reduced with `reduction` method
    """
    ce_loss = F.cross_entropy(x, target, weight, ignore_index=ignore_index, reduction=reduction)

    if gamma == 0:
        return ce_loss

    # log(P[class]) = log_softmax(score)[class]
    # logpt = F.log_softmax(x, dim=1)

    pt = F.softmax(x, dim=1)
    pt = pt / (1 - pt.transpose(0, 1).gather(0, target.unsqueeze(0)).transpose(0, 1))

    loss = -1 / (x.shape[1] - 1) * pt * torch.log(pt)

    # Nullify contributions to the loss
    # TODO: vectorize or write CUDA extension
    for class_idx in torch.unique(target):
        loss[:, class_idx][target == class_idx] = 0.0

    # Ignore index (set loss contribution to 0)
    valid_idxs = torch.ones(loss.shape[1], dtype=torch.bool, device=x.device)
    if ignore_index >= 0 and ignore_index < x.shape[1]:
        valid_idxs[ignore_index] = False

    # Weight
    if weight is not None:
        # Tensor type
        if weight.type() != x.data.type():
            weight = weight.type_as(x.data)
        loss = loss * weight.view(1, -1, *([1] * (x.ndim - 2)))

    # Loss reduction
    if reduction == "sum":
        loss = loss[:, valid_idxs].sum()
    else:
        loss = loss[:, valid_idxs].sum(dim=1)
        if reduction == "mean":
            loss = loss.mean()

    # Smooth the labels
    return ce_loss + gamma * loss


def mutual_channel_loss(
    x: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    xi: int = 2,
    alpha: float = 1.0,
    training: bool = True,
) -> Tensor:
    """Implements the mutual channel loss from
    ["The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification"](https://arxiv.org/pdf/2002.04264.pdf).

    Args:
        x: input tensor of shape [N, K, ...]
        target: target tensor of shape [N, ...]
        weight: manual rescaling of each class of shape [K]
        ignore_index: specifies target value that is ignored and do not contribute to gradient
        reduction: reduction method
        xi: num of features per class
        alpha: diversity factor
        training: whether to randomly mask channels; evaluation uses all channels

    Returns:
        loss reduced with `reduction` method

    Raises:
        ValueError: if the number of channels is not a multiple of a positive xi
    """
    # Accumulate dense losses in float32 for low-precision logits.
    if x.dtype in {torch.float16, torch.bfloat16}:
        x = x.float()
    # Flatten spatial dimension
    b, c = x.shape[:2]
    spatial_dims = x.shape[2:]
    if xi < 1 or c % xi != 0:
        raise ValueError("The number of channels must be a multiple of a positive xi")
    cnum = c // xi
    x = x.reshape(b, cnum, xi, -1)
    valid = target != ignore_index

    # CWA
    discr_out = x
    if training:
        base_mask = torch.zeros(xi, device=x.device)
        base_mask[: ceil(xi / 2)] = 1
        chan_mask = torch.zeros((cnum, xi), device=x.device)
        for idx in range(cnum):
            chan_mask[idx] = base_mask[torch.randperm(xi, device=x.device)]
        discr_out = x.masked_fill(chan_mask.view(1, cnum, xi, 1) == 0, torch.finfo(x.dtype).min)
    # CCMP
    discr_out = discr_out.max(dim=2).values
    discr_out = discr_out.view(b, cnum, *spatial_dims)
    # Weight casting
    if isinstance(weight, torch.Tensor) and weight.type() != x.data.type():
        weight = weight.type_as(x.data)

    if valid.any():
        discr_loss = F.cross_entropy(discr_out, target, weight, ignore_index=ignore_index, reduction=reduction)
    else:
        discr_loss = (discr_out * 0).sum(dim=1)
        if reduction != "none":
            discr_loss = discr_loss.sum()

    # Softmax
    spatial_valid = valid.reshape(b, 1, 1, -1)
    div_out = F.softmax(x.masked_fill(~spatial_valid, torch.finfo(x.dtype).min), dim=-1)
    # CCMP
    div_out = div_out.max(dim=2).values

    diversity_loss = div_out.mean(dim=1).masked_fill(~valid.reshape(b, -1), 0)

    if reduction == "sum":
        diversity_loss = diversity_loss.sum()
    elif reduction == "mean":
        diversity_loss = diversity_loss.sum() / valid.sum().clamp_min(1)
    else:
        diversity_loss = diversity_loss.view(b, *spatial_dims)

    return discr_loss - alpha * diversity_loss


def _xcorr2d(
    fn: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    normalize_slices: bool = False,
    eps: float = 1e-14,
) -> Tensor:
    """Implements cross-correlation operation"""  # noqa: DOC201
    # Reshape input Tensor into properly sized slices
    h, w = x.shape[-2:]
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    x = F.unfold(x, weight.shape[-2:], dilation=dilation, padding=padding, stride=stride)
    x = x.transpose(1, 2)
    # Normalize the slices
    if normalize_slices:
        unfold_scale = (x.var(-1, unbiased=False, keepdim=True) + eps).rsqrt()
        x -= x.mean(-1, keepdim=True)
        x *= unfold_scale.expand_as(x)

    # Perform common convolutions
    x = fn(x, weight)
    if bias is not None:
        x += bias
    x = x.transpose(1, 2)

    # Check output shape
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    h = floor((h + (2 * padding[0]) - (dilation[0] * (weight.shape[-2] - 1)) - 1) / stride[0] + 1)
    w = floor((w + (2 * padding[1]) - (dilation[1] * (weight.shape[-1] - 1)) - 1) / stride[1] + 1)

    return x.view(-1, weight.shape[0], h, w)


def _convNd(x: Tensor, weight: Tensor) -> Tensor:
    """Implements inner cross-correlation operation over slices

    Args:
        x: input tensor of shape [N, num_slices, Cin * K1 * ...]
        weight: filters of shape [Cout, Cin, K1, ...]

    Returns:
        output tensor
    """
    return x @ weight.view(weight.size(0), -1).t()


def norm_conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    eps: float = 1e-14,
) -> Tensor:
    """Implements a normalized convolution operations in 2D. Based on the [implementation](https://github.com/kimdongsuk1/NormalizedCNN)
    by the paper's author.
    See [`holocron.nn.NormConv2d`][holocron.nn.NormConv2d] for details and output shape.

    Args:
        x: input tensor of shape [N, in_channels, H, W]
        weight: filters of shape [out_channels, in_channels, Kh, Kw]
        bias: optional bias tensor of shape [out_channels]
        stride: the stride of the convolving kernel. Can be a single number or a tuple `(sH, sW)`.
        padding: implicit paddings on both sides of the input. Can be a single number or a tuple `(padH, padW)`.
        dilation: the spacing between kernel elements. Can be a single number or a tuple `(dH, dW)`.
        groups: split input into groups, in_channels should be divisible by the number of groups.
        eps: a value added to the denominator for numerical stability.

    Returns:
        output tensor

    Example:
        ```python
        # With square kernels and equal stride
        filters = torch.randn(8,4,3,3)
        inputs = torch.randn(1,4,5,5)
        F.norm_conv2d(inputs, filters, padding=1)
        ```
    """
    return _xcorr2d(_convNd, x, weight, bias, stride, padding, dilation, groups, True, eps)


def _addNd(x: Tensor, weight: Tensor) -> Tensor:
    """Implements inner adder operation over slices

    Args:
        x: input tensor of shape [N, num_slices, Cin * K1 * ...]
        weight: filters of shape [Cout, Cin, K1, ...]

    Returns:
        output tensor
    """
    return -(x.unsqueeze(2) - weight.view(weight.size(0), -1)).abs().sum(-1)


def add2d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    normalize_slices: bool = False,
    eps: float = 1e-14,
) -> Tensor:
    """Implements an adder operation in 2D from ["AdderNet: Do We Really Need Multiplications in Deep Learning?"](https://arxiv.org/pdf/1912.13200.pdf).
    See [`holocron.nn.Add2d`][holocron.nn.Add2d] for details and output shape.

    Args:
        x: input tensor of shape [N, in_channels, H, W]
        weight: filters of shape [out_channels, in_channels, Kh, Kw]
        bias: optional bias tensor of shape [out_channels]
        stride: the stride of the convolving kernel. Can be a single number or a tuple `(sH, sW)`.
        padding: implicit paddings on both sides of the input. Can be a single number or a tuple `(padH, padW)`.
        dilation: the spacing between kernel elements. Can be a single number or a tuple `(dH, dW)`.
        groups: split input into groups, in_channels should be divisible by the number of groups.
        normalize_slices: whether input slices should be normalized
        eps: a value added to the denominator for numerical stability.

    Returns:
        output tensor

    Example:
        ```python
        filters = torch.randn(8,4,3,3)
        inputs = torch.randn(1,4,5,5)
        F.norm_conv2d(inputs, filters, padding=1)
        ```
    """
    return _xcorr2d(_addNd, x, weight, bias, stride, padding, dilation, groups, normalize_slices, eps)


def dropblock2d(x: Tensor, drop_prob: float, block_size: int, inplace: bool = False, training: bool = True) -> Tensor:
    """Implements the dropblock operation from ["DropBlock: A regularization method for convolutional networks"](https://arxiv.org/pdf/1810.12890.pdf).

    Args:
        x: input tensor of shape [N, C, H, W]
        drop_prob: probability of dropping activation value
        block_size: size of each block that is expended from the sampled mask
        inplace: whether the operation should be done inplace
        training: whether the input should be processed in training mode

    Returns:
        output tensor
    """
    if not training or drop_prob == 0:
        return x

    # cf. Eq (1) from the paper
    gamma = drop_prob / block_size**2

    # Sample a mask for the centers of blocks that will be dropped
    mask = (torch.rand((x.shape[0], *x.shape[2:]), device=x.device) <= gamma).to(dtype=x.dtype)

    # Expand zero positions to block size
    mask = 1 - F.max_pool2d(mask, kernel_size=(block_size, block_size), stride=(1, 1), padding=block_size // 2)

    # Avoid NaNs
    one_count = mask.sum()
    if inplace:
        x *= mask.unsqueeze(1)
        if one_count > 0:
            x *= mask.numel() / one_count
        return x

    out = x * mask.unsqueeze(1)
    if one_count > 0:
        out *= mask.numel() / one_count

    return out


def dice_loss(
    x: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    gamma: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Implements the dice loss from ["V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"](https://arxiv.org/pdf/1606.04797.pdf)

    Args:
        x: predicted probability of shape [N, K, ...] with at least one spatial dimension
        target: target probability of shape [N, K, ...]
        weight: manual rescaling of each class of shape [K]
        gamma: controls the balance between recall (gamma > 1) and precision (gamma < 1)
        eps: epsilon to balance the loss and avoids division by zero

    Returns:
        loss reduced with `reduction` method
    """
    inter = gamma * (x * target).flatten(2).sum((0, 2))
    cardinality = (x + gamma * target).flatten(2).sum((0, 2))

    dice_coeff = (inter + eps) / (cardinality + eps)

    # Weight
    if weight is None:
        loss = 1 - (1 + 1 / gamma) * dice_coeff.mean()
    else:
        # Tensor type
        if weight.type() != x.data.type():
            weight = weight.type_as(x.data)
        loss = 1 - (1 + 1 / gamma) * (weight * dice_coeff).sum() / weight.sum()

    return loss


def poly_loss(
    x: Tensor,
    target: Tensor,
    eps: float = 2.0,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> Tensor:
    """Implements the Poly1 loss from ["PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions"](https://arxiv.org/pdf/2204.12511.pdf).

    Args:
        x: raw logits of shape [N, K, ...]; log-softmax is applied internally
        target: hard class indices of shape [N, ...] with dtype torch.int64, or floating-point
            class probabilities with the same shape as x
        eps: epsilon 1 from the paper
        weight: manual rescaling of each class of shape [K]
        ignore_index: class index excluded from sum/mean reduction; for hard targets it must be in
            [0, K) if present in target. Out-of-range hard targets, including -100, raise RuntimeError
        reduction: reduction method; with hard targets, "none" returns a flat tensor and does not
            zero ignored positions

    Returns:
        loss reduced with `reduction` method

    Raises:
        TypeError: if hard class-index targets do not have dtype torch.int64
        ValueError: if the target shape is invalid
    """
    # log(P[class]) = log_softmax(score)[class]
    # Shape (N, K, ...)
    logpt = F.log_softmax(x, dim=1)

    # Hard label
    if target.ndim == x.ndim - 1:
        if target.dtype != torch.long:
            raise TypeError("target dtype is expected to be torch.int64")
        # Compute pt and logpt only for target classes (the remaining will have a 0 coefficient)
        # Shape (N, ...)
        logpt = logpt.transpose(1, 0).flatten(1).gather(0, target.view(1, -1)).squeeze(0)
    else:
        if target.ndim != x.ndim or target.shape[0] != x.shape[0] or target.shape[1] != x.shape[1]:
            raise ValueError("invalid target shape")
        # Shape (N, K, ...)
        logpt = target * logpt

    # Get P(class)
    loss = -1 * logpt + eps * (1 - logpt.exp())

    # Weight
    if isinstance(weight, Tensor):
        # Tensor type
        if weight.type() != x.data.type():
            weight = weight.type_as(x.data)
        if target.ndim == x.ndim - 1:
            loss = weight.gather(0, target.data.view(-1)) * loss
        else:
            loss = weight.reshape(1, -1) * loss

    # Ignore index (set loss contribution to 0)
    if target.ndim == x.ndim - 1:
        valid_idxs = torch.ones(target.view(-1).shape[0], dtype=torch.bool, device=x.device)
        if ignore_index >= 0 and ignore_index < x.shape[1]:
            valid_idxs[target.view(-1) == ignore_index] = False
    else:
        valid_idxs = torch.ones(target.shape[1], dtype=torch.bool, device=x.device)
        if ignore_index >= 0 and ignore_index < x.shape[1]:
            valid_idxs[ignore_index] = False

    # Loss reduction
    if reduction == "sum":
        loss = loss[valid_idxs].sum() if target.ndim == x.ndim - 1 else loss[:, valid_idxs].sum()
    elif reduction == "mean":
        loss = loss[valid_idxs].mean() if target.ndim == x.ndim - 1 else loss[:, valid_idxs].sum(1).mean()
    # if no reduction, reshape tensor like target
    # (N, ...)
    elif target.ndim == x.ndim:
        loss = loss[:, valid_idxs].sum(1)

    return loss
