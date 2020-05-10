# -*- coding: utf-8 -*-

'''
Functional interface
'''

import torch
import torch.nn.functional as F


__all__ = ['mish', 'nl_relu', 'focal_loss']


def mish(x):
    """Implements the Mish activation function

    Args:
        x (torch.Tensor): input tensor
    Returns:
        torch.Tensor[x.size()]: output tensor
    """

    return x * torch.tanh(F.softplus(x))


def nl_relu(x, beta=1., inplace=False):
    """Implements the natural logarithm ReLU activation function

    Args:
        x (torch.Tensor): input tensor
        beta (float): beta used for NReLU
        inplace (bool): whether the operation should be performed inplace
    Returns:
        torch.Tensor[x.size()]: output tensor
    """

    if inplace:
        return torch.log(F.relu_(x).mul_(beta).add_(1), out=x)
    else:
        return torch.log(1 + beta * F.relu(x))


def focal_loss(x, target, weight=None, ignore_index=-100, reduction='mean', gamma=2):
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

    # Compute pt and logpt only for target classes (the remaining will have a 0 coefficient)
    logpt = logpt.transpose(1, 0).flatten(1).gather(0, target.view(1, -1)).squeeze()
    # Ignore index (set loss contribution to 0)
    if ignore_index >= 0:
        logpt[target.view(-1) == ignore_index] = 0

    # Get P(class)
    pt = logpt.exp()

    # Weight
    if weight is not None:
        # Tensor type
        if weight.type() != x.data.type():
            weight = weight.type_as(x.data)
        at = weight.gather(0, target.data.view(-1))
        logpt *= at

    # Loss
    loss = -1 * (1 - pt) ** gamma * logpt

    # Loss reduction
    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        # Ignore contribution to the loss if target is `ignore_index`
        if ignore_index >= 0:
            loss = loss[target.view(-1) != ignore_index]
        loss = loss.mean()
    else:
        # if no reduction, reshape tensor like target
        loss = loss.view(*target.shape)

    return loss


def concat_downsample2d(x, scale_factor):
    """Implements a loss-less downsampling operation described in
    `"YOLO9000: Better, Faster, Stronger" <https://pjreddie.com/media/files/papers/YOLO9000.pdf>`_
    by stacking adjacent information on the channel dimension.

    Args:
        x (torch.Tensor): input tensor
        scale_factor (int): spatial scaling factor

    Returns:
        torch.Tensor: downsampled tensor
    """

    b, c, h, w = x.shape

    if (h % scale_factor != 0) or (w % scale_factor != 0):
        raise AssertionError("Spatial size of input tensor must be multiples of `scale_factor`")
    new_h, new_w = h // scale_factor, w // scale_factor

    # N * C * H * W --> N * C * (H/scale_factor) * scale_factor * (W/scale_factor) * scale_factor
    out = x.view(b, c, new_h, scale_factor, new_w, scale_factor)
    # Move extra axes to last position to flatten them with channel dimension
    out = out.permute(0, 2, 4, 1, 3, 5).flatten(3)
    # Reorder all axes
    out = out.permute(0, 3, 1, 2)

    return out
