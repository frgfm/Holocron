# -*- coding: utf-8 -*-

'''
Downsampling modules
'''

import torch.nn as nn
from .. import functional as F

__all__ = ['ConcatDownsample2d', 'GlobalAvgPool2d']


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
