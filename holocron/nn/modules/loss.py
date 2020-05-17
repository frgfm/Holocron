# -*- coding: utf-8 -*-

"""
Loss implementations
"""

import torch
import torch.nn as nn
from .. import functional as F

__all__ = ['FocalLoss', 'MultiLabelCrossEntropy', 'LabelSmoothingCrossEntropy', 'MixupLoss']


class _Loss(nn.Module):

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.weight = weight
        # Cast class weights if possible
        if isinstance(weight, (float, int)):
            self.weight = torch.Tensor([weight, 1 - weight])
        elif isinstance(weight, list):
            self.weight = torch.Tensor(weight)
        self.ignore_index = ignore_index
        # Set the reduction method
        if reduction not in ['none', 'mean', 'sum']:
            raise NotImplementedError("argument reduction received an incorrect input")
        else:
            self.reduction = reduction


class FocalLoss(_Loss):
    """Implementation of Focal Loss as described in
    `"Focal Loss for Dense Object Detection" <https://arxiv.org/pdf/1708.02002.pdf>`_.

    While the weighted cross-entropy is described by:

    .. math::
        CE(p_t) = -\\alpha_t log(p_t)

    where :math:`\\alpha_t` is the loss weight of class :math:`t`,
    and :math:`p_t` is the predicted probability of class :math:`t`.

    the focal loss introduces a modulating factor

    .. math::
        FL(p_t) = -\\alpha_t (1 - p_t)^\\gamma log(p_t)

    where :math:`\\gamma` is a positive focusing parameter.

    Args:
        gamma (float, optional): exponent parameter of the focal loss
        weight (torch.Tensor[K], optional): class weight for loss computation
        ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
        reduction (str, optional): type of reduction to apply to the final loss
    """

    def __init__(self, gamma=2, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def forward(self, x, target):
        return F.focal_loss(x, target, self.weight, self.ignore_index, self.reduction, self.gamma)

    def __repr__(self):
        return f"{self.__class__.__name__}(gamma={self.gamma}, reduction='{self.reduction}')"


class MultiLabelCrossEntropy(_Loss):
    """Implementation of the cross-entropy loss for multi-label targets

    Args:
        weight (torch.Tensor[K], optional): class weight for loss computation
        ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
        reduction (str, optional): type of reduction to apply to the final loss
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, target):
        return F.multilabel_cross_entropy(x, target, self.weight, self.ignore_index, self.reduction)

    def __repr__(self):
        return f"{self.__class__.__name__}(reduction='{self.reduction}')"


class LabelSmoothingCrossEntropy(_Loss):
    """Implementation of the cross-entropy loss with label smoothing on hard target as described in
    `"Attention Is All You Need" <https://arxiv.org/pdf/1706.03762.pdf>`_

    Args:
        eps (float, optional): smoothing factor
        weight (torch.Tensor[K], optional): class weight for loss computation
        ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
        reduction (str, optional): type of reduction to apply to the final loss
    """

    def __init__(self, eps=0.1, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def forward(self, x, target):
        return F.ls_cross_entropy(x, target, self.weight, self.ignore_index, self.reduction, self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(eps={self.eps}, reduction='{self.reduction}')"


class MixupLoss(_Loss):
    """Implements a Mixup wrapper as described in
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/pdf/1710.09412.pdf>`_

    Args:
        criterion (callable): initial criterion to be used on normal sample & targets
    """
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, x, target_a, target_b, lam):
        """Computes the mixed-up loss

        Args:
            x (torch.Tensor): predictions
            target_a (torch.Tensor): target for first sample
            target_b (torch.Tensor): target for second sample
            lam (float): lambda factor
        Returns:
            torch.Tensor: loss
        """
        return lam * self.criterion(x, target_a) + (1 - lam) * self.criterion(x, target_b)

    def __repr__(self):
        return f"Mixup_{self.criterion.__repr__()}"
