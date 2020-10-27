import torch
from torch import Tensor
import torch.nn as nn
from .. import functional as F
from typing import List

__all__ = ['SiLU', 'Mish', 'HardMish', 'NLReLU', 'FReLU']


class _Activation(nn.Module):

    __constants__: List[str] = ['inplace']

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class _SiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return F.silu(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sig = torch.sigmoid(x)
        return grad_output * sig * (1 + x * (1 - sig))


class SiLU(nn.Module):
    """Implements the SiLU activation from `"Gaussian Error Linear Units (GELUs)"
    <https://arxiv.org/pdf/1606.08415.pdf>`_ (also known as Swish).

    This activation is computed as follows:

    .. math::
        f(x) = x \\cdot \\sigma(x)
    """
    def forward(self, x: Tensor) -> Tensor:
        return _SiLU.apply(x)


class Mish(nn.Module):
    """Implements the Mish activation module from `"Mish: A Self Regularized Non-Monotonic Neural Activation Function"
    <https://arxiv.org/pdf/1908.08681.pdf>`_

    This activation is computed as follows:

    .. math::
        f(x) = x \\cdot \\tanh(ln(1 + e^x))
    """
    def forward(self, x: Tensor) -> Tensor:
        return F.mish(x)


class HardMish(_Activation):
    """Implements the Had Mish activation module from `"H-Mish" <https://github.com/digantamisra98/H-Mish>`_

    This activation is computed as follows:

    .. math::
        f(x) = \\frac{x}{2} \\cdot \\min(2, \\max(0, x + 2))
    """
    def forward(self, x: Tensor) -> Tensor:
        return F.hard_mish(x, inplace=self.inplace)


class NLReLU(_Activation):
    """Implements the Natural-Logarithm ReLU activation module from `"Natural-Logarithm-Rectified Activation
    Function in Convolutional Neural Networks" <https://arxiv.org/pdf/1908.03682.pdf>`_

    This activation is computed as follows:

    .. math::
        f(x) = ln(1 + \\beta \\cdot max(0, x))

    Args:
        inplace (bool): should the operation be performed inplace
    """
    def forward(self, x: Tensor) -> Tensor:
        return F.nl_relu(x, inplace=self.inplace)


class FReLU(nn.Module):
    """Implements the Funnel activation module from `"Funnel Activation for Visual Recognition"
    <https://arxiv.org/pdf/2007.11824.pdf>`_

    This activation is computed as follows:

    .. math::
        f(x) = max(\\mathbb{T}(x), x)

    where the :math:`\\mathbb{T}` is the spatial contextual feature extraction. It is a convolution filter of size
    `kernel_size`, same padding and groups equal to the number of input channels, followed by a batch normalization.

    Args:
        inplace (bool): should the operation be performed inplace
    """
    def __init__(self, in_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2, groups=in_channels)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn(out)
        x = torch.max(x, out)
        return x
