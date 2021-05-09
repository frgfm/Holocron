# Copyright (C) 2019-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from torch import Tensor
import torch.nn as nn
from .. import functional as F

__all__ = ['StackUpsample2d']


class StackUpsample2d(nn.Module):
    """Implements a loss-less upsampling operation described in `"Real-Time Single Image and Video Super-Resolution
    Using an Efficient Sub-Pixel Convolutional Neural Network" <https://arxiv.org/pdf/1609.05158.pdf>`_
    by unstacking the channel axis into adjacent information.

    .. image:: https://docs.fast.ai/images/pixelshuffle.png
        :align: center

    Args:
        scale_factor (int): spatial scaling factor
    """

    def __init__(self, scale_factor: int) -> None:
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: Tensor) -> Tensor:

        return F.stack_upsample2d(x, self.scale_factor)

    def extra_repr(self) -> str:
        return f"scale_factor={self.scale_factor}"