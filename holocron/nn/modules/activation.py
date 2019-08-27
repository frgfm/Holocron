#!usr/bin/python
# -*- coding: utf-8 -*-

'''
Activation modules
'''

import torch.nn as nn
from .. import functional as F


class Mish(nn.Module):
    """Implements the Mish activation module from https://arxiv.org/pdf/1908.08681.pdf

    Args:
        inplace (bool): should the operation be performed inplace
    """
    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(Mish, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.mish(input, inplace=self.inplace)

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str