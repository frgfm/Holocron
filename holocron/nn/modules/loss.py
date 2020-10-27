import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Union, Any, List
from .. import functional as F

__all__ = ['FocalLoss', 'MultiLabelCrossEntropy', 'LabelSmoothingCrossEntropy', 'ComplementCrossEntropy',
           'MixupLoss', 'ClassBalancedWrapper', 'MutualChannelLoss']


class _Loss(nn.Module):

    def __init__(
        self,
        weight: Optional[Union[float, List[float], Tensor]] = None,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ) -> None:
        super().__init__()
        # Cast class weights if possible
        self.weight: Optional[Tensor]
        if isinstance(weight, (float, int)):
            self.weight = torch.Tensor([weight, 1 - weight])  # type: ignore[assignment]
        elif isinstance(weight, list):
            self.weight = torch.Tensor(weight)  # type: ignore[assignment]
        else:
            self.weight = weight
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

    def __init__(self, gamma: float = 2., **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gamma = gamma

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return F.focal_loss(x, target, self.weight, self.ignore_index, self.reduction, self.gamma)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gamma={self.gamma}, reduction='{self.reduction}')"


class MultiLabelCrossEntropy(_Loss):
    """Implementation of the cross-entropy loss for multi-label targets

    Args:
        weight (torch.Tensor[K], optional): class weight for loss computation
        ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
        reduction (str, optional): type of reduction to apply to the final loss
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return F.multilabel_cross_entropy(x, target, self.weight, self.ignore_index, self.reduction)

    def __repr__(self) -> str:
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

    def __init__(self, eps: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.eps = eps

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return F.ls_cross_entropy(x, target, self.weight, self.ignore_index, self.reduction, self.eps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.eps}, reduction='{self.reduction}')"


class ComplementCrossEntropy(_Loss):
    """Implements the complement cross entropy loss from
    `"Imbalanced Image Classification with Complement Cross Entropy" <https://arxiv.org/pdf/2009.02189.pdf>`_

    Args:
        gamma (float, optional): smoothing factor
        weight (torch.Tensor[K], optional): class weight for loss computation
        ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
        reduction (str, optional): type of reduction to apply to the final loss
    """

    def __init__(self, gamma: float = -1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gamma = gamma

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return F.complement_cross_entropy(x, target, self.weight, self.ignore_index, self.reduction, self.gamma)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gamma={self.gamma}, reduction='{self.reduction}')"


class MixupLoss(_Loss):
    """Implements a Mixup wrapper as described in
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/pdf/1710.09412.pdf>`_

    Args:
        criterion (callable): initial criterion to be used on normal sample & targets
    """
    def __init__(self, criterion: nn.Module) -> None:
        super().__init__()
        self.criterion = criterion

    def forward(self, x: Tensor, target_a: Tensor, target_b: Tensor, lam: float) -> Tensor:
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

    def __repr__(self) -> str:
        return f"Mixup_{self.criterion.__repr__()}"


class ClassBalancedWrapper(nn.Module):
    """Implementation of the class-balanced loss as described in `"Class-Balanced Loss Based on Effective Number
    of Samples" <https://arxiv.org/pdf/1901.05555.pdf>`_.

    Given a loss function :math:`\\mathcal{L}`, the class-balanced loss is described by:

    .. math::
        CB(p, y) = \\frac{1 - \\beta}{1 - \\beta^{n_y}} \\mathcal{L}(p, y)

    where :math:`p` is the predicted probability for class :math:`y`, :math:`n_y` is the number of training
    samples for class :math:`y`, and :math:`\\beta` is exponential factor.

    Args:
        criterion (torch.nn.Module): loss module
        num_samples (torch.Tensor[K]): number of samples for each class
        beta (float, optional): rebalancing exponent
    """

    def __init__(self, criterion: nn.Module, num_samples: Tensor, beta: float = 0.99) -> None:
        super().__init__()
        self.criterion = criterion
        self.beta = beta
        cb_weights = (1 - beta) / (1 - beta ** num_samples)
        if self.criterion.weight is None:
            self.criterion.weight = cb_weights
        else:
            self.criterion.weight *= cb_weights.to(device=self.criterion.weight.device)

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return self.criterion.forward(x, target)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.criterion.__repr__()}, beta={self.beta})"


class MutualChannelLoss(_Loss):
    """Implements the mutual channel loss from
    `"The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification"
    <https://arxiv.org/pdf/2002.04264.pdf>`_.

    Args:
        weight (torch.Tensor[K], optional): class weight for loss computation
        ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
        reduction (str, optional): type of reduction to apply to the final loss
        chi (in, optional): num of features per class
        alpha (float, optional): diversity factor
    """

    def __init__(
        self,
        weight: Optional[Union[float, List[float], Tensor]] = None,
        ignore_index: int = -100,
        reduction: str = 'mean',
        chi: int = 2,
        alpha: float = 1,
    ) -> None:
        super().__init__(weight, ignore_index, reduction)
        self.chi = chi
        self.alpha = alpha

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return F.mutual_channel_loss(x, target, self.weight, self.ignore_index, self.reduction, self.chi, self.alpha)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reduction='{self.reduction}', chi={self.chi}, alpha={self.alpha})"
