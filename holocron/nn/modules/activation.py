# Copyright (C) 2019-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from .. import functional as F

__all__ = ["HardMish", "NLReLU", "FReLU"]


class _Activation(nn.Module):

    __constants__: List[str] = ["inplace"]

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class HardMish(_Activation):
    r"""Implements the Had Mish activation module from `"H-Mish" <https://github.com/digantamisra98/H-Mish>`_

    This activation is computed as follows:

    .. math::
        f(x) = \frac{x}{2} \cdot \min(2, \max(0, x + 2))
    """

    def forward(self, x: Tensor) -> Tensor:
        return F.hard_mish(x, inplace=self.inplace)


class NLReLU(_Activation):
    r"""Implements the Natural-Logarithm ReLU activation module from `"Natural-Logarithm-Rectified Activation
    Function in Convolutional Neural Networks" <https://arxiv.org/pdf/1908.03682.pdf>`_

    This activation is computed as follows:

    .. math::
        f(x) = ln(1 + \beta \cdot max(0, x))

    Args:
        inplace (bool): should the operation be performed inplace
    """

    def forward(self, x: Tensor) -> Tensor:
        return F.nl_relu(x, inplace=self.inplace)


class FReLU(nn.Module):
    r"""Implements the Funnel activation module from `"Funnel Activation for Visual Recognition"
    <https://arxiv.org/pdf/2007.11824.pdf>`_

    This activation is computed as follows:

    .. math::
        f(x) = max(\mathbb{T}(x), x)

    where the :math:`\mathbb{T}` is the spatial contextual feature extraction. It is a convolution filter of size
    `kernel_size`, same padding and groups equal to the number of input channels, followed by a batch normalization.

    Args:
        inplace (bool): should the operation be performed inplace
    """

    def __init__(self, in_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2, groups=in_channels)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn(out)
        x = torch.max(x, out)
        return x
