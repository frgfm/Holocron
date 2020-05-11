# -*- coding: utf-8 -*-

'''
Activation modules
'''

import torch.nn as nn
from .. import functional as F

__all__ = ['Mish', 'NLReLU']


class Mish(nn.Module):
    """Implements the Mish activation module from `"Mish: A Self Regularized Non-Monotonic Neural Activation Function"
    <https://arxiv.org/pdf/1908.08681.pdf>`_"""

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, input):
        return F.mish(input)


class NLReLU(nn.Module):
    """Implements the Natural-Logarithm ReLU activation module from `"Natural-Logarithm-Rectified Activation
    Function in Convolutional Neural Networks" <https://arxiv.org/pdf/1908.03682.pdf>`_

    Args:
        inplace (bool): should the operation be performed inplace
    """
    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(NLReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.nl_relu(input, inplace=self.inplace)

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
