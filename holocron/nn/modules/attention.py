# Copyright (C) 2019-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import torch
import torch.nn as nn
from torch import Tensor

from .downsample import ZPool

__all__ = ["SAM", "TripletAttention"]


class SAM(nn.Module):
    """SAM layer from `"CBAM: Convolutional Block Attention Module" <https://arxiv.org/pdf/1807.06521.pdf>`_
    modified in `"YOLOv4: Optimal Speed and Accuracy of Object Detection" <https://arxiv.org/pdf/2004.10934.pdf>`_.

    Args:
        in_channels (int): input channels
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(self.conv(x))


class DimAttention(nn.Module):
    """Attention layer across a specific dimension

    Args:
        dim: dimension to compute attention on
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.compress = nn.Sequential(
            ZPool(dim=1),
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01),
            nn.Sigmoid(),
        )
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        if self.dim != 1:
            x = x.transpose(self.dim, 1).contiguous()
        out = x * self.compress(x)
        if self.dim != 1:
            out = out.transpose(self.dim, 1).contiguous()
        return out


class TripletAttention(nn.Module):
    """Triplet attention layer from `"Rotate to Attend: Convolutional Triplet Attention Module"
    <https://arxiv.org/pdf/2010.03045.pdf>`_. This implementation is based on the
    `one <https://github.com/LandskapeAI/triplet-attention/blob/master/MODELS/triplet_attention.py>`_
    from the paper's authors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.c_branch = DimAttention(dim=1)
        self.h_branch = DimAttention(dim=2)
        self.w_branch = DimAttention(dim=3)

    def forward(self, x: Tensor) -> Tensor:
        x_c = self.c_branch(x)
        x_h = self.h_branch(x)
        x_w = self.w_branch(x)

        out = (x_c + x_h + x_w) / 3
        return out
