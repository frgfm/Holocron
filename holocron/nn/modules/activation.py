# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import ClassVar

import torch
from torch import Tensor, nn

from .. import functional as F

__all__ = ["FReLU", "HardMish", "NLReLU"]


class _Activation(nn.Module):
    __constants__: ClassVar[list[str]] = ["inplace"]

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace: bool = inplace

    def extra_repr(self) -> str:
        return "inplace=True" if self.inplace else ""


class HardMish(_Activation):
    r"""Implements the Hard Mish activation module from ["H-Mish"](https://github.com/digantamisra98/H-Mish).

    This activation is computed as follows:

    $$
    f(x) = \frac{x}{2} \cdot \min(2, \max(0, x + 2))
    $$

    Args:
        inplace: should the operation be performed inplace
    """

    def forward(self, x: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        return F.hard_mish(x, inplace=self.inplace)


class NLReLU(_Activation):
    r"""Implements the Natural-Logarithm ReLU activation module from ["Natural-Logarithm-Rectified Activation
    Function in Convolutional Neural Networks"](https://arxiv.org/pdf/1908.03682.pdf).

    This activation is computed as follows:

    $$
    f(x) = ln(1 + \beta \cdot max(0, x))
    $$

    Args:
        beta: beta used for NReLU
        inplace: should the operation be performed inplace
    """

    def __init__(self, beta: float = 1.0, inplace: bool = False) -> None:
        super().__init__(inplace)
        self.beta: float = beta

    def forward(self, x: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        return F.nl_relu(x, beta=self.beta, inplace=self.inplace)


class FReLU(nn.Module):
    r"""Implements the Funnel activation module from ["Funnel Activation for Visual Recognition"](https://arxiv.org/pdf/2007.11824.pdf).

    This activation is computed as follows:

    $$
    f(x) = max(\mathbb{T}(x), x)
    $$

    where the $\mathbb{T}$ is the spatial contextual feature extraction. It is a convolution filter of size
    `kernel_size`, same padding and groups equal to the number of input channels, followed by a batch normalization.

    Args:
        in_channels: number of input channels
        kernel_size: size of the convolution filter
    """

    def __init__(self, in_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2, groups=in_channels)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        out = self.conv(x)
        out = self.bn(out)
        return torch.max(x, out)
