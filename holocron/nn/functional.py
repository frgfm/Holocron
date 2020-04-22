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

    # log(P[class]) = log_softmax(score)[class]
    logpt = F.log_softmax(x, dim=1)

    # Compute pt and logpt only for target classes (the remaining will have a 0 coefficient)
    logpt = logpt.transpose(1, -1).reshape(-1, x.shape[1]).index_select(-1, target.view(-1)).diag()
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
    if reduction == 'mean':
        # Ignore contribution to the loss if target is `ignore_index`
        if ignore_index >= 0:
            loss = loss[target.view(-1) != ignore_index]
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        # if no reduction, reshape tensor like target
        loss = loss.view(*target.shape)

    return loss
