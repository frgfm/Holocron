# -*- coding: utf-8 -*-

'''
Regularization modules
'''

import torch
import torch.nn as nn
from .. import functional as F

__all__ = ['DropBlock2d']


class DropBlock2d(nn.Module):
    """Implements the DropBlock module from `"DropBlock: A regularization method for convolutional networks"
    <https://arxiv.org/pdf/1810.12890.pdf>`_

    Args:
        p (float): probability of dropping activation value
        block_size (int): size of each block that is expended from the sampled mask
        inplace (bool, optional): whether the operation should be done inplace
    """

    def __init__(self, p, block_size, inplace=False):
        super().__init__()
        self.p = p
        self.block_size = block_size
        self.inplace = inplace

    @property
    def drop_prob(self):
        return self.p / self.block_size ** 2

    def forward(self, x):
        return F.dropblock2d(x, self.drop_prob, self.block_size, self.inplace)

    def extra_repr(self):
        return f"p={self.p}, block_size={self.block_size}, inplace={self.inplace}"
