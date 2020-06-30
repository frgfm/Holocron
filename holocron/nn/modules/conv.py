# -*- coding: utf-8 -*-

'''
Convolutional modules
'''

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.functional import pad
from .. import functional as F

__all__ = ['NormConv2d', 'Add2d', 'SlimConv2d']


class _NormConvNd(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, normalize_slices=False, eps=1e-14):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, transposed, output_padding,
                         groups, bias, padding_mode)
        self.normalize_slices = normalize_slices
        self.eps = eps


class NormConv2d(_NormConvNd):
    """Implements the normalized convolution module from `"Normalized Convolutional Neural Network"
    <https://arxiv.org/pdf/2005.05274v2.pdf>`_.

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
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        eps (float, optional): a value added to the denominator for numerical stability.
            Default: 1e-14
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', eps=1e-14):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode, False, eps)

    def forward(self, input):
        if self.padding_mode != 'zeros':
            return F.norm_conv2d(pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                 self.weight, self.bias, self.stride, _pair(0),
                                 self.dilation, self.groups, self.eps)
        return F.norm_conv2d(input, self.weight, self.bias, self.stride, self.padding,
                             self.dilation, self.groups, self.eps)


class Add2d(_NormConvNd):
    """Implements the adder module from `"AdderNet: Do We Really Need Multiplications in Deep Learning?"
    <https://arxiv.org/pdf/1912.13200.pdf>`_.

    In the simplest case, the output value of the layer at position :math:`(m, n)` in channel :math:`c`
    with filter F of spatial size :math:`(d, d)`, intput size :math:`(C_{in}, H, W)` and output :math:`(C_{out}, H, W)`
    can be precisely described as:

    .. math::
        out(m, n, c) = - \\sum\\limits_{i=0}^d \\sum\\limits_{j=0}^d \\sum\\limits_{k=0}^{C_{in}}
        |X(m + i, n + j, k) - F(i, j, k, c)|

    where :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        normalize_slices (bool, optional): whether slices should be normalized before performing cross-correlation.
            Default: False
        eps (float, optional): a value added to the denominator for numerical stability.
            Default: 1e-14
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', normalize_slices=False, eps=1e-14):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode, normalize_slices, eps)

    def forward(self, input):
        if self.padding_mode != 'zeros':
            return F.add2d(pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                           self.weight, self.bias, self.stride, _pair(0),
                           self.dilation, self.groups, self.normalize_slices, self.eps)
        return F.add2d(input, self.weight, self.bias, self.stride, self.padding,
                       self.dilation, self.groups, self.normalize_slices, self.eps)


class SlimConv2d(nn.Module):
    """Implements the convolution module from `"SlimConv: Reducing Channel Redundancy in Convolutional Neural Networks
    by Weights Flipping" <https://arxiv.org/pdf/2003.07469.pdf>`_.

    First, we compute channel-wise weights as follows:

    .. math::
        z(c) = \\frac{1}{H \\cdot W} \\sum\\limits_{i=1}^H \\sum\\limits_{j=1}^W X_{c,i,j}

    where :math:`X \\in \\mathbb{R}^{C \\times H \\times W}` is the input tensor,
    :math:`H` is height in pixels, and :math:`W` is
    width in pixels.

    .. math::
        w = \\sigma(F_{fc2}(\\delta(F_{fc1}(z))))

    where :math:`z \\in \\mathbb{R}^{C}` contains channel-wise statistics,
    :math:`\\sigma` refers to the sigmoid function,
    :math:`\\delta` refers to the ReLU function,
    :math:`F_{fc1}` is a convolution operation with kernel of size :math:`(1, 1)`
    with :math:`max(C/r, L)` output channels followed by batch normalization,
    and :math:`F_{fc2}` is a plain convolution operation with kernel of size :math:`(1, 1)`
    with :math:`C` output channels.

    We then proceed with reconstructing and transforming both pathways:

    .. math::
        X_{top} = X \\odot w

    .. math::
        X_{bot} = X \\odot \\check{w}

    where :math:`\\odot` refers to the element-wise multiplication and :math:`\\check{w}` is
    the channel-wise reverse-flip of :math:`w`.

    .. math::
        T_{top} = F_{top}(X_{top}^{(1)} + X_{top}^{(2)})

    .. math::
        T_{bot} = F_{bot}(X_{bot}^{(1)} + X_{bot}^{(2)})

    where :math:`X^{(1)}` and :math:`X^{(2)}` are the channel-wise first and second halves of :math:`X`,
    :math:`F_{top}` is a convolution of kernel size :math:`(3, 3)`,
    and :math:`F_{bot}` is a convolution of kernel size :math:`(1, 1)` reducing channels by half,
    followed by a convolution of kernel size :math:`(3, 3)`.

    Finally we fuse both pathways to yield the output:

    .. math::
        Y = T_{top} \\oplus T_{bot}

    where :math:`\\oplus` is the channel-wise concatenation.


    Args:
        in_channels (int): Number of channels in the input image
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        r (int, optional): squeezing divider. Default: 32
        L (int, optional): minimum squeezed channels. Default: 8
    """

    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', r=32, L=2):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, max(in_channels // r, L), 1)
        self.bn = nn.BatchNorm2d(max(in_channels // r, L))
        self.fc2 = nn.Conv2d(max(in_channels // r, L), in_channels, 1)
        self.conv_top = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size, stride, padding,
                                  dilation, groups, bias, padding_mode)
        self.conv_bot1 = nn.Conv2d(in_channels // 2, in_channels // 4, 1)
        self.conv_bot2 = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size, stride, padding,
                                   dilation, groups, bias, padding_mode)

    def forward(self, x):
        # Channel-wise weights
        z = x.mean(dim=(2, 3), keepdims=True)
        z = self.bn(self.fc1(z))
        z = self.fc2(torch.relu(z))
        w = torch.sigmoid(z)

        # Compression
        X_w = x * w
        X_top = X_w[:, :x.shape[1] // 2] + X_w[:, x.shape[1] // 2:]
        X_w = x * w.flip(dims=(1,))
        X_bot = X_w[:, :x.shape[1] // 2] + X_w[:, x.shape[1] // 2:]

        # Transform
        X_top = self.conv_top(X_top)
        X_bot = self.conv_bot2(self.conv_bot1(X_bot))

        # Fuse
        return torch.cat((X_top, X_bot), dim=1)
