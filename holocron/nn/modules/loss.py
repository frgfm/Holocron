# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import Any, cast

import torch
from torch import Tensor, nn

from .. import functional as F

__all__ = [
    "ClassBalancedWrapper",
    "ComplementCrossEntropy",
    "DiceLoss",
    "FocalLoss",
    "Loss",
    "MultiLabelCrossEntropy",
    "MutualChannelLoss",
    "PolyLoss",
]


class Loss(nn.Module):
    """Base loss class.

    Args:
        weight: class weight for loss computation
        ignore_index: specifies target value that is ignored and do not contribute to gradient
        reduction: type of reduction to apply to the final loss

    Raises:
        NotImplementedError: if the reduction method is not supported
    """

    def __init__(
        self,
        weight: float | list[float] | Tensor | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        # Cast class weights if possible
        self.weight: Tensor | None
        if isinstance(weight, (float, int)):
            self.register_buffer("weight", torch.Tensor([weight, 1 - weight]))
        elif isinstance(weight, list):
            self.register_buffer("weight", torch.Tensor(weight))
        elif isinstance(weight, Tensor):
            self.register_buffer("weight", weight)
        else:
            self.weight: Tensor | None = None
        self.ignore_index: int = ignore_index
        # Set the reduction method
        if reduction not in {"none", "mean", "sum"}:
            raise NotImplementedError("argument reduction received an incorrect input")
        self.reduction: str = reduction


class FocalLoss(Loss):
    r"""Implementation of Focal Loss as described in
    ["Focal Loss for Dense Object Detection"](https://arxiv.org/pdf/1708.02002.pdf).

    While the weighted cross-entropy is described by:

    $$
    CE(p_t) = -\alpha_t log(p_t)
    $$

    where $\alpha_t$ is the loss weight of class $t$,
    and $p_t$ is the predicted probability of class $t$.

    the focal loss introduces a modulating factor

    $$
    FL(p_t) = -\alpha_t (1 - p_t)^\gamma log(p_t)
    $$

    where $\gamma$ is a positive focusing parameter.

    Args:
        gamma: exponent parameter of the focal loss
        **kwargs: keyword args of [`Loss`][holocron.nn.modules.loss.Loss]
    """

    def __init__(self, gamma: float = 2.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gamma: float = gamma

    def forward(self, x: Tensor, target: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        return F.focal_loss(x, target, self.weight, self.ignore_index, self.reduction, self.gamma)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gamma={self.gamma}, reduction='{self.reduction}')"


class MultiLabelCrossEntropy(Loss):
    """Implementation of the cross-entropy loss for multi-label targets

    Args:
        *args: args of [`Loss`][holocron.nn.modules.loss.Loss]
        **kwargs: keyword args of [`Loss`][holocron.nn.modules.loss.Loss]
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor, target: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        return F.multilabel_cross_entropy(x, target, self.weight, self.ignore_index, self.reduction)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reduction='{self.reduction}')"


class ComplementCrossEntropy(Loss):
    """Implements the complement cross entropy loss from
    ["Imbalanced Image Classification with Complement Cross Entropy"](https://arxiv.org/pdf/2009.02189.pdf)

    Args:
        gamma: smoothing factor
        **kwargs: keyword args of [`Loss`][holocron.nn.modules.loss.Loss]
    """

    def __init__(self, gamma: float = -1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gamma: float = gamma

    def forward(self, x: Tensor, target: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        return F.complement_cross_entropy(x, target, self.weight, self.ignore_index, self.reduction, self.gamma)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gamma={self.gamma}, reduction='{self.reduction}')"


class ClassBalancedWrapper(nn.Module):
    r"""Implementation of the class-balanced loss as described in ["Class-Balanced Loss Based on Effective Number
    of Samples"](https://arxiv.org/pdf/1901.05555.pdf).

    Given a loss function $\mathcal{L}$, the class-balanced loss is described by:

    $$
    CB(p, y) = \frac{1 - \beta}{1 - \beta^{n_y}} \mathcal{L}(p, y)
    $$

    where $p$ is the predicted probability for class $y$, $n_y$ is the number of training
    samples for class $y$, and $\beta$ is exponential factor.

    Args:
        criterion: loss module
        num_samples: number of samples for each class
        beta: rebalancing exponent
    """

    def __init__(self, criterion: nn.Module, num_samples: Tensor, beta: float = 0.99) -> None:
        super().__init__()
        self.criterion = criterion
        self.beta: float = beta
        cb_weights = (1 - beta) / (1 - beta**num_samples)
        if self.criterion.weight is None:
            self.criterion.weight: Tensor | None = cb_weights
        else:
            self.criterion.weight *= cb_weights.to(device=self.criterion.weight.device)  # ty: ignore[invalid-argument-type]

    def forward(self, x: Tensor, target: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        return cast(Tensor, self.criterion.forward(x, target))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.criterion.__repr__()}, beta={self.beta})"


class MutualChannelLoss(Loss):
    """Implements the mutual channel loss from
    ["The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification"](https://arxiv.org/pdf/2002.04264.pdf).

    Args:
        weight: class weight for loss computation
        ignore_index: specifies target value that is ignored and do not contribute to gradient
        reduction: type of reduction to apply to the final loss
        xi: num of features per class
        alpha: diversity factor
    """

    def __init__(
        self,
        weight: float | list[float] | Tensor | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        xi: int = 2,
        alpha: float = 1,
    ) -> None:
        super().__init__(weight, ignore_index, reduction)
        self.xi: int = xi
        self.alpha: float = alpha

    def forward(self, x: Tensor, target: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        return F.mutual_channel_loss(
            x, target, self.weight, self.ignore_index, self.reduction, self.xi, self.alpha, training=self.training
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reduction='{self.reduction}', xi={self.xi}, alpha={self.alpha})"


class DiceLoss(Loss):
    """Implements the dice loss from ["V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image
    Segmentation"](https://arxiv.org/pdf/1606.04797.pdf).

    Args:
        weight: class weight for loss computation
        gamma: recall/precision control param
        eps: small value added to avoid division by zero
    """

    def __init__(
        self,
        weight: float | list[float] | Tensor | None = None,
        gamma: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(weight)
        self.gamma: float = gamma
        self.eps: float = eps

    def forward(self, x: Tensor, target: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        return F.dice_loss(x, target, self.weight, self.gamma, self.eps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reduction='{self.reduction}', gamma={self.gamma}, eps={self.eps})"


class PolyLoss(Loss):
    """Implements the Poly1 loss from ["PolyLoss: A Polynomial Expansion Perspective of Classification Loss
    Functions"](https://arxiv.org/pdf/2204.12511.pdf).

    Pass raw logits with shape `(N, K, ...)`, where `K` is the number of classes.
    Hard targets have shape `(N, ...)` and dtype `torch.int64`, with values in
    `[0, K)`. Soft targets are floating-point class probabilities with the same
    shape as the logits. Do not apply softmax to the input first.

    Example:
        >>> import torch
        >>> from holocron.nn import PolyLoss
        >>> logits = torch.randn(4, 10, requires_grad=True)
        >>> target = torch.tensor([0, 9, 3, 1], dtype=torch.int64)
        >>> loss = PolyLoss()(logits, target)
        >>> loss.backward()

    Note:
        Out-of-range hard targets, including the default `ignore_index=-100`,
        raise `RuntimeError` if present in the target. Only an in-range class index
        can be ignored for sum/mean reduction. With `reduction="none"`, ignored
        positions are not zeroed and hard-target losses are returned as a flat
        tensor. See the [loss input guide](https://frgfm.github.io/holocron/reference/nn/#loss-functions)
        for a loss that supports out-of-range ignored targets.

    Args:
        *args: positional weight, ignore_index and reduction arguments of
            [`Loss`][holocron.nn.modules.loss.Loss]
        eps: epsilon 1 from the paper (default: 2.0)
        **kwargs: weight, ignore_index and reduction keyword arguments of
            [`Loss`][holocron.nn.modules.loss.Loss]
    """

    def __init__(
        self,
        *args: Any,
        eps: float = 2.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.eps: float = eps

    def forward(self, x: Tensor, target: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        return F.poly_loss(x, target, self.eps, self.weight, self.ignore_index, self.reduction)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.eps}, reduction='{self.reduction}')"
