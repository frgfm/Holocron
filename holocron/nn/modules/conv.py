# -*- coding: utf-8 -*-

'''
Convolutional modules
'''

from torch.nn.modules.conv import _ConvNd
from .. import functional as F

__all__ = ['NormConv2d']


class _NormConvNd(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, eps=1e-14):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, transposed, output_padding,
                         groups, bias, padding_mode)
        self.eps = eps


class NormConv2d(_NormConvNd):
    """Implements the normalized convolution module from `"Normalized Convolutional Neural Network"
    <https://arxiv.org/pdf/2005.05274v2.pdf>`_

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, H, W)` and output :math:`(N, C_{out}, H_{out}, W_{out})`
    can be precisely described as:

    .. math::
        out(N_i, C_{out_j}) = bias(C_{out_j}) +
        \sum_{k = 0}^{C_{in} - 1} weight(C_{out_j}, k) \star
        \\frac{input(N_i, k) - \mu(N_i, k)}{\sqrt{\sigma^2(N_i, k) + \epsilon}}

    where :math:`\star` is the valid 2D cross-correlation operator,
    :math:`\mu(N_i, k)` and :math:`\sigma²(N_i, k)` are the mean and variance of :math:`input(N_i, k)` over all slices,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
        eps: a value added to the denominator for numerical stability.
            Default: 1e-14
    """

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.norm_conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                 weight, self.bias, self.stride, _pair(0),
                                 self.dilation, self.groups, self.eps)
        return F.norm_conv2d(input, weight, self.bias, self.stride, self.padding,
                             self.dilation, self.groups, self.eps)
