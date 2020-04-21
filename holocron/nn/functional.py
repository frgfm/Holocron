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


def focal_loss(input, target, weight=None, ignore_index=-100, reduction='mean', gamma=2):
    """Implements the focal loss from https://arxiv.org/pdf/1708.02002.pdf

    Args:
        x (torch.Tensor): input tensor
        target (torch.Tensor): target tensor
        weight (torch.Tensor, optional): manual rescaling of each class
        ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
        reduction (str, optional): reduction method


    Returns:
        torch.Tensor: loss reduced with `reduction` method
    """

    # Non-reduced CE-Loss = -log(pt)
    ce_loss = F.cross_entropy(input, target, ignore_index=ignore_index, reduction='none')
    # Use it to get pt
    pt = (-ce_loss).exp()

    # Get focal loss
    loss = (1 - pt) ** gamma * ce_loss

    # Class rescaling
    if weight is not None:
        if weight.type() != x.data.type():
            weight = weight.type_as(x.data)
        at = weight.view(1, -1, *([1] * (logpt.ndim - 2)))
        logpt = logpt * at

    # Loss reduction
    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()

    return loss
