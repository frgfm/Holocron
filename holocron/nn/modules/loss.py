# -*- coding: utf-8 -*-

"""
Loss implementations
"""

import torch
import torch.nn as nn
from .. import functional as F

__all__ = ['FocalLoss', 'LabelSmoothingCrossEntropy']


class FocalLoss(nn.Module):
    """Implementation of Focal Loss as described in
    `"Focal Loss for Dense Object Detection" <https://arxiv.org/pdf/1708.02002.pdf>`_

    Args:
        gamma (float): exponent parameter of the focal loss
        weight (torch.Tensor[K], optional): class weight for loss computation
        ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
        reduction (str, optional): type of reduction to apply to the final loss
    """

    def __init__(self, gamma=2, weight=None, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        # Cast class weights if possible
        if isinstance(weight, (float, int)):
            self.weight = torch.Tensor([weight, 1 - weight])
        elif isinstance(weight, list):
            self.weight = torch.Tensor(weight)
        self.ignore_index = ignore_index
        # Set the reduction method
        if reduction not in ['none', 'mean', 'sum']:
            raise NotImplementedError(f"argument reduction received an incorrect input")
        else:
            self.reduction = reduction

    def forward(self, x, target):
        return F.focal_loss(x, target, self.weight, self.ignore_index, self.reduction, self.gamma)

    def __repr__(self):
        return f"{self.__class__.__name__}(gamma={self.gamma}, reduction='{self.reduction}')"


class LabelSmoothingCrossEntropy(nn.Module):
    """Implementation of the cross-entropy loss with label smoothing on hard target as described in
    `"Attention Is All You Need" <https://arxiv.org/pdf/1706.03762.pdf>`_

    Args:
        eps (float): smoothing factor
        weight (torch.Tensor[K], optional): class weight for loss computation
        ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
        reduction (str, optional): type of reduction to apply to the final loss
    """

    def __init__(self, eps=2, weight=None, ignore_index=-100, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.weight = weight
        # Cast class weights if possible
        if isinstance(weight, (float, int)):
            self.weight = torch.Tensor([weight, 1 - weight])
        elif isinstance(weight, list):
            self.weight = torch.Tensor(weight)
        self.ignore_index = ignore_index
        # Set the reduction method
        if reduction not in ['none', 'mean', 'sum']:
            raise NotImplementedError(f"argument reduction received an incorrect input")
        else:
            self.reduction = reduction

    def focal_loss(self, x, target):
        return F.ls_cross_entropy(x, target, self.weight, self.ignore_index, self.reduction, self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(eps={self.eps}, reduction='{self.reduction}')"
