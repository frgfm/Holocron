# Copyright (C) 2019-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from math import ceil, floor
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "hard_mish",
    "nl_relu",
    "focal_loss",
    "multilabel_cross_entropy",
    "complement_cross_entropy",
    "mutual_channel_loss",
    "norm_conv2d",
    "add2d",
    "dropblock2d",
    "z_pool",
    "concat_downsample2d",
    "dice_loss",
    "poly_loss",
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
    weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    gamma: float = 2.0,
) -> Tensor:
    """Implements the focal loss from
    `"Focal Loss for Dense Object Detection" <https://arxiv.org/pdf/1708.02002.pdf>`_

    Args:
        x (torch.Tensor[N, K, ...]): input tensor
        target (torch.Tensor[N, ...]): hard target tensor
        weight (torch.Tensor[K], optional): manual rescaling of each class
        ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
        reduction (str, optional): reduction method
        gamma (float, optional): gamma parameter of focal loss

    Returns:
        torch.Tensor: loss reduced with `reduction` method
    """

    # log(P[class]) = log_softmax(score)[class]
    logpt = F.log_softmax(x, dim=1)

    # Compute pt and logpt only for target classes (the remaining will have a 0 coefficient)
    logpt = logpt.transpose(1, 0).flatten(1).gather(0, target.view(1, -1)).squeeze()
    # Ignore index (set loss contribution to 0)
    valid_idxs = torch.ones(target.view(-1).shape[0], dtype=torch.bool, device=x.device)
    if ignore_index >= 0 and ignore_index < x.shape[1]:
        valid_idxs[target.view(-1) == ignore_index] = False

    # Get P(class)
    pt = logpt.exp()

    # Weight
    if weight is not None:
        # Tensor type
        if weight.type() != x.data.type():
            weight = weight.type_as(x.data)
        logpt = weight.gather(0, target.data.view(-1)) * logpt

    # Loss
    loss = -1 * (1 - pt) ** gamma * logpt

    # Loss reduction
    if reduction == "sum":
        loss = loss[valid_idxs].sum()
    elif reduction == "mean":
        loss = loss[valid_idxs].mean()
    else:
        # if no reduction, reshape tensor like target
        loss = loss.view(*target.shape)

    return loss


def concat_downsample2d(x: Tensor, scale_factor: int) -> Tensor:
    """Implements a loss-less downsampling operation described in
    `"YOLO9000: Better, Faster, Stronger" <https://pjreddie.com/media/files/papers/YOLO9000.pdf>`_
    by stacking adjacent information on the channel dimension.

    Args:
        x (torch.Tensor[N, C, H, W]): input tensor
        scale_factor (int): spatial scaling factor

    Returns:
        torch.Tensor[N, scale_factor ** 2 * C, H / scale_factor, W / scale_factor]: downsampled tensor
    """

    b, c, h, w = x.shape

    if (h % scale_factor != 0) or (w % scale_factor != 0):
        raise AssertionError("Spatial size of input tensor must be multiples of `scale_factor`")

    # N * C * H * W --> N * C * (H/scale_factor) * scale_factor * (W/scale_factor) * scale_factor
    x = x.view(b, c, h // scale_factor, scale_factor, w // scale_factor, scale_factor)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
    x = x.view(b, int(c * scale_factor**2), h // scale_factor, w // scale_factor)

    return x


def z_pool(x: Tensor, dim: int) -> Tensor:
    """Z-pool layer from `"Rotate to Attend: Convolutional Triplet Attention Module"
    <https://arxiv.org/pdf/2010.03045.pdf>`_.

    Args:
        x: input tensor
        dim: dimension to pool
    """

    return torch.cat([x.max(dim, keepdim=True).values, x.mean(dim, keepdim=True)], dim=dim)


def multilabel_cross_entropy(
    x: Tensor, target: Tensor, weight: Optional[Tensor] = None, ignore_index: int = -100, reduction: str = "mean"
) -> Tensor:
    """Implements the cross entropy loss for multi-label targets

    Args:
        x (torch.Tensor[N, K, ...]): input tensor
        target (torch.Tensor[N, K, ...]): target tensor
        weight (torch.Tensor[K], optional): manual rescaling of each class
        ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
        reduction (str, optional): reduction method

    Returns:
        torch.Tensor: loss reduced with `reduction` method
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
    weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    gamma: float = -1,
) -> Tensor:
    """Implements the complement cross entropy loss from
    `"Imbalanced Image Classification with Complement Cross Entropy" <https://arxiv.org/pdf/2009.02189.pdf>`_

    Args:
        x (torch.Tensor[N, K, ...]): input tensor
        target (torch.Tensor[N, ...]): target tensor
        weight (torch.Tensor[K], optional): manual rescaling of each class
        ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
        reduction (str, optional): reduction method
        gamma (float, optional): complement factor

    Returns:
        torch.Tensor: loss reduced with `reduction` method
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
    weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    xi: int = 2,
    alpha: float = 1.0,
) -> Tensor:
    """Implements the mutual channel loss from
    `"The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification"
    <https://arxiv.org/pdf/2002.04264.pdf>`_.

    Args:
        x (torch.Tensor[N, K, ...]): input tensor
        target (torch.Tensor[N, ...]): target tensor
        weight (torch.Tensor[K], optional): manual rescaling of each class
        ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
        reduction (str, optional): reduction method
        xi (int, optional): num of features per class
        alpha (float, optional): diversity factor

    Returns:
        torch.Tensor: loss reduced with `reduction` method
    """

    # Flatten spatial dimension
    b, c = x.shape[:2]
    spatial_dims = x.shape[2:]
    cnum = c // xi
    x = x.view(b, cnum, xi, -1)

    # CWA
    base_mask = torch.zeros(xi, device=x.device)
    base_mask[: ceil(xi / 2)] = 1
    chan_mask = torch.zeros((cnum, xi), device=x.device)
    for idx in range(cnum):
        chan_mask[idx] = base_mask[torch.randperm(xi)]
    discr_out = x * chan_mask.view(1, cnum, xi, 1)
    # CCMP
    discr_out = discr_out.max(dim=2).values
    discr_out = discr_out.view(b, cnum, *spatial_dims)
    # Weight casting
    if isinstance(weight, torch.Tensor) and weight.type() != x.data.type():
        weight = weight.type_as(x.data)

    discr_loss = F.cross_entropy(discr_out, target, weight, ignore_index=ignore_index, reduction=reduction)

    # Softmax
    div_out = F.softmax(x, dim=-1)
    # CCMP
    div_out = div_out.max(dim=2).values

    diversity_loss = div_out.mean(dim=1)

    if reduction == "sum":
        diversity_loss = diversity_loss.sum()
    elif reduction == "mean":
        diversity_loss = diversity_loss.mean()
    else:
        diversity_loss = diversity_loss.view(b, *spatial_dims)

    return discr_loss - alpha * diversity_loss


def _xcorr2d(
    fn: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    normalize_slices: bool = False,
    eps: float = 1e-14,
) -> Tensor:
    """Implements cross-correlation operation"""

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

    x = x.view(-1, weight.shape[0], h, w)

    return x


def _convNd(x: Tensor, weight: Tensor) -> Tensor:
    """Implements inner cross-correlation operation over slices

    Args:
        x (torch.Tensor[N, num_slices, Cin * K1 * ...]): input Tensor
        weight (torch.Tensor[Cout, Cin, K1, ...]): filters
    """

    return x @ weight.view(weight.size(0), -1).t()


def norm_conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    eps: float = 1e-14,
) -> Tensor:
    """Implements a normalized convolution operations in 2D. Based on the `implementation
    <https://github.com/kimdongsuk1/NormalizedCNN>`_ by the paper's author.
    See :class:`~holocron.nn.NormConv2d` for details and output shape.

    Args:
        x (torch.Tensor[N, in_channels, H, W]): input tensor
        weight (torch.Tensor[out_channels, in_channels, Kh, Kw]): filters
        bias (torch.Tensor[out_channels], optional): optional bias tensor of shape (out_channels).
          Default: ``None``
        stride (int, optional): the stride of the convolving kernel. Can be a single number or a
          tuple `(sH, sW)`. Default: 1
        padding (int, optional): implicit paddings on both sides of the input. Can be a
          single number or a tuple `(padH, padW)`. Default: 0
        dilation (int, optional): the spacing between kernel elements. Can be a single number or
          a tuple `(dH, dW)`. Default: 1
        groups (int, optional): split input into groups, in_channels should be divisible by the
          number of groups. Default: 1
        eps (float, optional): a value added to the denominator for numerical stability.
            Default: 1e-14
    Examples::
        >>> # With square kernels and equal stride
        >>> filters = torch.randn(8,4,3,3)
        >>> inputs = torch.randn(1,4,5,5)
        >>> F.norm_conv2d(inputs, filters, padding=1)
    """

    return _xcorr2d(_convNd, x, weight, bias, stride, padding, dilation, groups, True, eps)


def _addNd(x: Tensor, weight: Tensor) -> Tensor:
    """Implements inner adder operation over slices

    Args:
        x (torch.Tensor[N, num_slices, Cin * K1 * ...]): input Tensor
        weight (torch.Tensor[Cout, Cin, K1, ...]): filters
    """

    return -(x.unsqueeze(2) - weight.view(weight.size(0), -1)).abs().sum(-1)


def add2d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    normalize_slices: bool = False,
    eps: float = 1e-14,
) -> Tensor:
    """Implements an adder operation in 2D from `"AdderNet: Do We Really Need Multiplications in Deep Learning?"
    <https://arxiv.org/pdf/1912.13200.pdf>`_. See :class:`~holocron.nn.Add2d` for details and output shape.

    Args:
        x (torch.Tensor[N, in_channels, H, W]): input tensor
        weight (torch.Tensor[out_channels, in_channels, Kh, Kw]): filters
        bias (torch.Tensor[out_channels], optional): optional bias tensor of shape (out_channels).
          Default: ``None``
        stride (int, optional): the stride of the convolving kernel. Can be a single number or a
          tuple `(sH, sW)`. Default: 1
        padding (int, optional): implicit paddings on both sides of the input. Can be a
          single number or a tuple `(padH, padW)`. Default: 0
        dilation (int, optional): the spacing between kernel elements. Can be a single number or
          a tuple `(dH, dW)`. Default: 1
        groups (int, optional): split input into groups, in_channels should be divisible by the
          number of groups. Default: 1
        normalize_slices (bool, optional): whether input slices should be normalized
        eps (float, optional): a value added to the denominator for numerical stability.
            Default: 1e-14
    Examples::
        >>> # With square kernels and equal stride
        >>> filters = torch.randn(8,4,3,3)
        >>> inputs = torch.randn(1,4,5,5)
        >>> F.norm_conv2d(inputs, filters, padding=1)
    """

    return _xcorr2d(_addNd, x, weight, bias, stride, padding, dilation, groups, normalize_slices, eps)


def dropblock2d(x: Tensor, drop_prob: float, block_size: int, inplace: bool = False, training: bool = True) -> Tensor:
    """Implements the dropblock operation from `"DropBlock: A regularization method for convolutional networks"
    <https://arxiv.org/pdf/1810.12890.pdf>`_

    Args:
        x (torch.Tensor): input tensor of shape (N, C, H, W)
        drop_prob (float): probability of dropping activation value
        block_size (int): size of each block that is expended from the sampled mask
        inplace (bool, optional): whether the operation should be done inplace
        training (bool, optional): whether the input should be processed in training mode
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
    weight: Optional[Tensor] = None,
    gamma: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Implements the dice loss from `"V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image
    Segmentation" <https://arxiv.org/pdf/1606.04797.pdf>`_

    Args:
        x (torch.Tensor[N, K, ...]): predicted probability
        target (torch.Tensor[N, K, ...]): target probability
        weight (torch.Tensor[K], optional): manual rescaling of each class
        gamma (float, optional): controls the balance between recall (gamma > 1) and precision (gamma < 1)
        eps (float, optional): epsilon to balance the loss and avoids division by zero

    Returns:
        torch.Tensor: loss reduced with `reduction` method
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
    weight: Optional[Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> Tensor:
    """Implements the Poly1 loss from `"PolyLoss: A Polynomial Expansion Perspective of Classification Loss
    Functions" <https://arxiv.org/pdf/2204.12511.pdf>`_.

    Args:
        x (torch.Tensor[N, K, ...]): predicted probability
        target (torch.Tensor[N, K, ...]): target probability
        eps (float, optional): epsilon 1 from the paper
        weight (torch.Tensor[K], optional): manual rescaling of each class
        ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
        reduction (str, optional): reduction method

    Returns:
        torch.Tensor: loss reduced with `reduction` method
    """

    # log(P[class]) = log_softmax(score)[class]
    logpt = F.log_softmax(x, dim=1)

    # Compute pt and logpt only for target classes (the remaining will have a 0 coefficient)
    logpt = logpt.transpose(1, 0).flatten(1).gather(0, target.view(1, -1)).squeeze()
    # Ignore index (set loss contribution to 0)
    valid_idxs = torch.ones(target.view(-1).shape[0], dtype=torch.bool, device=x.device)
    if ignore_index >= 0 and ignore_index < x.shape[1]:
        valid_idxs[target.view(-1) == ignore_index] = False

    # Get P(class)
    loss = -1 * logpt + eps * (1 - logpt.exp())

    # Weight
    if weight is not None:
        # Tensor type
        if weight.type() != x.data.type():
            weight = weight.type_as(x.data)
        logpt = weight.gather(0, target.data.view(-1)) * logpt

    # Loss reduction
    if reduction == "sum":
        loss = loss[valid_idxs].sum()
    elif reduction == "mean":
        loss = loss[valid_idxs].mean()
    else:
        # if no reduction, reshape tensor like target
        loss = loss.view(*target.shape)

    return loss
