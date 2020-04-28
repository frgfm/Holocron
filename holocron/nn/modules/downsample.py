# -*- coding: utf-8 -*-

'''
Downsampling modules
'''

import torch.nn as nn
from .. import functional as F

__all__ = ['ConcatDownsample2d']


class ConcatDownsample2d(nn.Module):
    """Implements a loss-less downsampling operation described in https://pjreddie.com/media/files/papers/YOLO9000.pdf
    by stacking adjacent information on the channel dimension.

    Args:
        scale_factor (int): spatial scaling factor
    """

    def __init__(self, scale_factor):

        super(ConcatDownsample2d, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):

        return F.concat_downsample2d(x, self.scale_factor)
