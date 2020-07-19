# -*- coding: utf-8 -*-

'''
Activation modules
'''

import torch.nn as nn
from .. import functional as F

__all__ = ['Swish', 'Mish', 'NLReLU']


class _Activation(nn.Module):

    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class Swish(nn.Module):
    """Implements the Swish activation module from `"Searching for Activation Functions"
    <https://arxiv.org/pdf/1710.05941.pdf>`_

    This activation is computed as follows:

    .. math::
        f(x) = x \\cdot \\sigma(\\beta \\cdot x)
    """
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, input):
        return F.swish(input, self.beta)


class Mish(nn.Module):
    """Implements the Mish activation module from `"Mish: A Self Regularized Non-Monotonic Neural Activation Function"
    <https://arxiv.org/pdf/1908.08681.pdf>`_

    This activation is computed as follows:

    .. math::
        f(x) = x \\cdot \\tanh(ln(1 + e^x))
    """
    def forward(self, input):
        return F.mish(input)


class NLReLU(_Activation):
    """Implements the Natural-Logarithm ReLU activation module from `"Natural-Logarithm-Rectified Activation
    Function in Convolutional Neural Networks" <https://arxiv.org/pdf/1908.03682.pdf>`_

    This activation is computed as follows:

    .. math::
        f(x) = ln(1 + \\beta \\cdot max(0, x))

    Args:
        inplace (bool): should the operation be performed inplace
    """
    def forward(self, input):
        return F.nl_relu(input, inplace=self.inplace)
