# Copyright (C) 2019-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .. import functional as F

__all__ = ["ConcatDownsample2d", "ConcatDownsample2dJit", "GlobalAvgPool2d", "BlurPool2d", "SPP", "ZPool"]


class ConcatDownsample2d(nn.Module):
    """Implements a loss-less downsampling operation described in `"YOLO9000: Better, Faster, Stronger"
    <https://pjreddie.com/media/files/papers/YOLO9000.pdf>`_ by stacking adjacent information on the channel dimension.

    Args:
        scale_factor (int): spatial scaling factor
    """

    def __init__(self, scale_factor: int) -> None:
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: Tensor) -> Tensor:

        return F.concat_downsample2d(x, self.scale_factor)


@torch.jit.script  # type: ignore[attr-defined]
class ConcatDownsample2dJit(object):
    """Implements a loss-less downsampling operation described in `"YOLO9000: Better, Faster, Stronger"
    <https://pjreddie.com/media/files/papers/YOLO9000.pdf>`_ by stacking adjacent information on the channel dimension.

    Args:
        scale_factor (int): spatial scaling factor
    """

    def __init__(self, scale_factor: int) -> None:
        self.scale_factor = scale_factor

    def __call__(self, x: Tensor) -> Tensor:

        return F.concat_downsample2d(x, self.scale_factor)


class GlobalAvgPool2d(nn.Module):
    """Fast implementation of global average pooling from `"TResNet: High Performance GPU-Dedicated Architecture"
    <https://arxiv.org/pdf/2003.13630.pdf>`_

    Args:
        flatten (bool, optional): whether spatial dimensions should be squeezed
    """

    def __init__(self, flatten: bool = False) -> None:
        super().__init__()
        self.flatten = flatten

    def forward(self, x: Tensor) -> Tensor:
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)

    def extra_repr(self) -> str:
        inplace_str = "flatten=True" if self.flatten else ""
        return inplace_str


def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class BlurPool2d(nn.Module):
    """Ross Wightman's `implementation
    <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/blur_pool.py>`_ of blur pooling
    module as described in `"Making Convolutional Networks Shift-Invariant Again"
    <https://arxiv.org/pdf/1904.11486.pdf>`_.

    .. image:: https://github.com/frgfm/Holocron/releases/download/v0.1.3/blurpool.png
        :align: center

    Args:
        channels (int): Number of input channels
        kernel_size (int, optional): binomial filter size for blurring. currently supports 3 (default) and 5.
        stride (int, optional): downsampling filter stride
    Returns:
        torch.Tensor: the transformed tensor.
    """

    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 2) -> None:
        super().__init__()
        self.channels = channels
        if kernel_size <= 1:
            raise AssertionError
        self.kernel_size = kernel_size
        self.stride = stride
        pad_size = [get_padding(kernel_size, stride, dilation=1)] * 4
        self.padding = nn.ReflectionPad2d(pad_size)  # type: ignore[arg-type]
        self._coeffs = torch.tensor((np.poly1d((0.5, 0.5)) ** (self.kernel_size - 1)).coeffs)  # for torchscript compat
        self.kernel: Dict[str, Tensor] = {}  # lazy init by device for DataParallel compat

    def _create_filter(self, like: Tensor) -> Tensor:
        blur_filter = (self._coeffs[:, None] * self._coeffs[None, :]).to(dtype=like.dtype, device=like.device)
        return blur_filter[None, None, :, :].repeat(self.channels, 1, 1, 1)

    def _apply(self, fn: Callable[[nn.Module], None]) -> None:
        # override nn.Module _apply, reset filter cache if used
        self.kernel = {}
        super()._apply(fn)

    def forward(self, input_tensor: Tensor) -> Tensor:
        blur_filter = self.kernel.get(str(input_tensor.device), self._create_filter(input_tensor))
        return nn.functional.conv2d(
            self.padding(input_tensor), blur_filter, stride=self.stride, groups=input_tensor.shape[1]
        )

    def extra_repr(self) -> str:
        return f"{self.channels}, kernel_size={self.kernel_size}, stride={self.stride}"


class SPP(nn.ModuleList):
    """SPP layer from `"Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition"
    <https://arxiv.org/pdf/1406.4729.pdf>`_.

    Args:
        kernel_sizes (list<int>): kernel sizes of each pooling
    """

    def __init__(self, kernel_sizes: List[int]) -> None:
        super().__init__([nn.MaxPool2d(k_size, stride=1, padding=k_size // 2) for k_size in kernel_sizes])

    def forward(self, x):
        feats = [x] + [pool_layer(x) for pool_layer in self]
        return torch.cat(feats, dim=1)


class ZPool(nn.Module):
    """Z-pool layer from `"Rotate to Attend: Convolutional Triplet Attention Module"
    <https://arxiv.org/pdf/2010.03045.pdf>`_.

    Args:
        dim: dimension to pool
    """

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return F.z_pool(x, self.dim)
