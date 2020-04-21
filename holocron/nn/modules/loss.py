# -*- coding: utf-8 -*-

"""
Loss implementations
"""

import torch
import torch.nn as nn
from .. import functional as F

__all__ = ['FocalLoss']


class FocalLoss(nn.Module):
    """Implementation of Focal Loss as described in https://arxiv.org/pdf/1708.02002.pdf

    Args:
        gamma (float): exponent parameter of the focal loss
        weight (torch.Tensor): class weight for loss computation
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
