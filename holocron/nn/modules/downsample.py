# -*- coding: utf-8 -*-

'''
Downsampling modules
'''

import numpy as np
import torch
import torch.nn as nn
from .. import functional as F

__all__ = ['ConcatDownsample2d', 'ConcatDownsample2dJit', 'GlobalAvgPool2d', 'BlurPool2d']


class ConcatDownsample2d(nn.Module):
    """Implements a loss-less downsampling operation described in `"YOLO9000: Better, Faster, Stronger"
    <https://pjreddie.com/media/files/papers/YOLO9000.pdf>`_ by stacking adjacent information on the channel dimension.

    Args:
        scale_factor (int): spatial scaling factor
    """

    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):

        return F.concat_downsample2d(x, self.scale_factor)


@torch.jit.script
class ConcatDownsample2dJit(object):
    """Implements a loss-less downsampling operation described in `"YOLO9000: Better, Faster, Stronger"
    <https://pjreddie.com/media/files/papers/YOLO9000.pdf>`_ by stacking adjacent information on the channel dimension.

    Args:
        scale_factor (int): spatial scaling factor
    """

    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, x):

        return F.concat_downsample2d(x, self.scale_factor)


class GlobalAvgPool2d(nn.Module):
    """Fast implementation of global average pooling from `"TResNet: High Performance GPU-Dedicated Architecture"
    <https://arxiv.org/pdf/2003.13630.pdf>`_

    Args:
        flatten (bool, optional): whether spatial dimensions should be squeezed
    """
    def __init__(self, flatten=False):
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)

    def extra_repr(self):
        inplace_str = 'flatten=True' if self.flatten else ''
        return inplace_str


def get_padding(kernel_size, stride=1, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class BlurPool2d(nn.Module):
    """Ross Wightman's `implementation
    <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/blur_pool.py>`_ of blur pooling
    module as described in `"Making Convolutional Networks Shift-Invariant Again"
    <https://arxiv.org/pdf/1904.11486.pdf>`_.

    Args:
        channels (int): Number of input channels
        kernel_size (int, optional): binomial filter size for blurring. currently supports 3 (default) and 5.
        stride (int, optional): downsampling filter stride
    Returns:
        torch.Tensor: the transformed tensor.
    """

    def __init__(self, channels, kernel_size=3, stride=2):
        super().__init__()
        self.channels = channels
        if not kernel_size > 1:
            raise AssertionError
        self.kernel_size = kernel_size
        self.stride = stride
        pad_size = [get_padding(kernel_size, stride, dilation=1)] * 4
        self.padding = nn.ReflectionPad2d(pad_size)
        self._coeffs = torch.tensor((np.poly1d((0.5, 0.5)) ** (self.kernel_size - 1)).coeffs)  # for torchscript compat
        self.kernel = {}  # lazy init by device for DataParallel compat

    def _create_filter(self, like):
        blur_filter = (self._coeffs[:, None] * self._coeffs[None, :]).to(dtype=like.dtype, device=like.device)
        return blur_filter[None, None, :, :].repeat(self.channels, 1, 1, 1)

    def _apply(self, fn):
        # override nn.Module _apply, reset filter cache if used
        self.kernel = {}
        super()._apply(fn)

    def forward(self, input_tensor):
        blur_filter = self.kernel.get(str(input_tensor.device), self._create_filter(input_tensor))
        return nn.functional.conv2d(
            self.padding(input_tensor), blur_filter, stride=self.stride, groups=input_tensor.shape[1])

    def extra_repr(self):
        return f"{self.channels}, kernel_size={self.kernel_size}, stride={self.stride}"
