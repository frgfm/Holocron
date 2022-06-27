# Copyright (C) 2019-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import math
from typing import Any, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import pad
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

from .. import functional as F

__all__ = ["NormConv2d", "Add2d", "SlimConv2d", "PyConv2d", "Involution2d"]


class _NormConvNd(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        transposed: bool,
        output_padding: int,
        groups: int,
        bias: bool,
        padding_mode: str,
        normalize_slices=False,
        eps=1e-14,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,  # type: ignore[arg-type]
            stride,  # type: ignore[arg-type]
            padding,  # type: ignore[arg-type]
            dilation,  # type: ignore[arg-type]
            transposed,
            output_padding,  # type: ignore[arg-type]
            groups,
            bias,
            padding_mode,
        )
        self.normalize_slices = normalize_slices
        self.eps = eps


class NormConv2d(_NormConvNd):
    r"""Implements the normalized convolution module from `"Normalized Convolutional Neural Network"
    <https://arxiv.org/pdf/2005.05274v2.pdf>`_.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, H, W)` and output :math:`(N, C_{out}, H_{out}, W_{out})`
    can be precisely described as:

    .. math::
        out(N_i, C_{out_j}) = bias(C_{out_j}) +
        \sum_{k = 0}^{C_{in} - 1} weight(C_{out_j}, k) \star
        \frac{input(N_i, k) - \mu(N_i, k)}{\sqrt{\sigma^2(N_i, k) + \epsilon}}

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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        eps: float = 1e-14,
    ) -> None:
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            False,
            eps,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.padding_mode != "zeros":
            return F.norm_conv2d(
                pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                self.weight,
                self.bias,
                self.stride,  # type: ignore[arg-type]
                _pair(0),
                self.dilation,  # type: ignore[arg-type]
                self.groups,
                self.eps,
            )
        return F.norm_conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.dilation,  # type: ignore[arg-type]
            self.groups,
            self.eps,
        )


class Add2d(_NormConvNd):
    r"""Implements the adder module from `"AdderNet: Do We Really Need Multiplications in Deep Learning?"
    <https://arxiv.org/pdf/1912.13200.pdf>`_.

    In the simplest case, the output value of the layer at position :math:`(m, n)` in channel :math:`c`
    with filter F of spatial size :math:`(d, d)`, intput size :math:`(C_{in}, H, W)` and output :math:`(C_{out}, H, W)`
    can be precisely described as:

    .. math::
        out(m, n, c) = - \sum\limits_{i=0}^d \sum\limits_{j=0}^d \sum\limits_{k=0}^{C_{in}}
        |X(m + i, n + j, k) - F(i, j, k, c)|

    where :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    .. image:: https://github.com/frgfm/Holocron/releases/download/v0.1.3/add2d.png
        :align: center
        :alt: Add2D schema

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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        normalize_slices: bool = False,
        eps: float = 1e-14,
    ) -> None:
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            normalize_slices,
            eps,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.padding_mode != "zeros":
            return F.add2d(
                pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                self.weight,
                self.bias,
                self.stride,  # type: ignore[arg-type]
                _pair(0),
                self.dilation,  # type: ignore[arg-type]
                self.groups,
                self.normalize_slices,
                self.eps,
            )
        return F.add2d(
            x,
            self.weight,
            self.bias,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.dilation,  # type: ignore[arg-type]
            self.groups,
            self.normalize_slices,
            self.eps,
        )


class SlimConv2d(nn.Module):
    r"""Implements the convolution module from `"SlimConv: Reducing Channel Redundancy in Convolutional Neural Networks
    by Weights Flipping" <https://arxiv.org/pdf/2003.07469.pdf>`_.

    First, we compute channel-wise weights as follows:

    .. math::
        z(c) = \frac{1}{H \cdot W} \sum\limits_{i=1}^H \sum\limits_{j=1}^W X_{c,i,j}

    where :math:`X \in \mathbb{R}^{C \times H \times W}` is the input tensor,
    :math:`H` is height in pixels, and :math:`W` is
    width in pixels.

    .. math::
        w = \sigma(F_{fc2}(\delta(F_{fc1}(z))))

    where :math:`z \in \mathbb{R}^{C}` contains channel-wise statistics,
    :math:`\sigma` refers to the sigmoid function,
    :math:`\delta` refers to the ReLU function,
    :math:`F_{fc1}` is a convolution operation with kernel of size :math:`(1, 1)`
    with :math:`max(C/r, L)` output channels followed by batch normalization,
    and :math:`F_{fc2}` is a plain convolution operation with kernel of size :math:`(1, 1)`
    with :math:`C` output channels.

    We then proceed with reconstructing and transforming both pathways:

    .. math::
        X_{top} = X \odot w

    .. math::
        X_{bot} = X \odot \check{w}

    where :math:`\odot` refers to the element-wise multiplication and :math:`\check{w}` is
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
        Y = T_{top} \oplus T_{bot}

    where :math:`\oplus` is the channel-wise concatenation.

    .. image:: https://github.com/frgfm/Holocron/releases/download/v0.1.3/slimconv2d.png
        :align: center
        :alt: SlimConv2D schema


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

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        r: int = 32,
        L: int = 2,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, max(in_channels // r, L), 1)
        self.bn = nn.BatchNorm2d(max(in_channels // r, L))
        self.fc2 = nn.Conv2d(max(in_channels // r, L), in_channels, 1)
        self.conv_top = nn.Conv2d(
            in_channels // 2, in_channels // 2, kernel_size, stride, padding, dilation, groups, bias, padding_mode
        )
        self.conv_bot1 = nn.Conv2d(in_channels // 2, in_channels // 4, 1)
        self.conv_bot2 = nn.Conv2d(
            in_channels // 4, in_channels // 4, kernel_size, stride, padding, dilation, groups, bias, padding_mode
        )

    def forward(self, x: Tensor) -> Tensor:
        # Channel-wise weights
        z = x.mean(dim=(2, 3), keepdim=True)
        z = self.bn(self.fc1(z))
        z = self.fc2(torch.relu(z))
        w = torch.sigmoid(z)

        # Compression
        X_w = x * w
        X_top = X_w[:, : x.shape[1] // 2] + X_w[:, x.shape[1] // 2 :]
        X_w = x * w.flip(dims=(1,))
        X_bot = X_w[:, : x.shape[1] // 2] + X_w[:, x.shape[1] // 2 :]

        # Transform
        X_top = self.conv_top(X_top)
        X_bot = self.conv_bot2(self.conv_bot1(X_bot))

        # Fuse
        return torch.cat((X_top, X_bot), dim=1)


class PyConv2d(nn.ModuleList):
    """Implements the convolution module from `"Pyramidal Convolution: Rethinking Convolutional Neural Networks for
    Visual Recognition" <https://arxiv.org/pdf/2006.11538.pdf>`_.

    .. image:: https://github.com/frgfm/Holocron/releases/download/v0.1.3/pyconv2d.png
        :align: center
        :alt: PyConv2D schema

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the convolving kernel
        num_levels (int, optional): number of stacks in the pyramid
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        groups (list(int), optional): Number of blocked connections from input
            channels to output channels. Default: 1
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_levels: int = 2,
        padding: int = 0,
        groups: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> None:

        if num_levels == 1:
            super().__init__(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=padding,
                        groups=groups[0] if isinstance(groups, list) else 1,
                        **kwargs,
                    )
                ]
            )
        else:
            exp2 = int(math.log2(num_levels))
            reminder = num_levels - 2**exp2
            out_chans = [out_channels // 2 ** (exp2 + 1)] * (2 * reminder) + [out_channels // 2**exp2] * (
                num_levels - 2 * reminder
            )

            k_sizes = [kernel_size + 2 * idx for idx in range(num_levels)]
            if groups is None:
                groups = [1] + [
                    min(2 ** (2 + idx), out_chan) for idx, out_chan in zip(range(num_levels - 1), out_chans[1:])
                ]
            elif not isinstance(groups, list) or len(groups) != num_levels:
                raise ValueError("The argument `group` is expected to be a list of integer of size `num_levels`.")
            paddings = [padding + idx for idx in range(num_levels)]

            super().__init__(
                [
                    nn.Conv2d(in_channels, out_chan, k_size, padding=padding, groups=group, **kwargs)
                    for out_chan, k_size, padding, group in zip(out_chans, k_sizes, paddings, groups)
                ]
            )
        self.num_levels = num_levels

    def forward(self, x):

        if self.num_levels == 1:
            return self[0].forward(x)
        return torch.cat([conv(x) for conv in self], dim=1)


class Involution2d(nn.Module):
    """Implements the convolution module from `"Involution: Inverting the Inherence of Convolution for Visual
    Recognition" <https://arxiv.org/pdf/2103.06255.pdf>`_, adapted from the proposed PyTorch implementation in
    the paper.

    .. image:: https://github.com/frgfm/Holocron/releases/download/v0.1.3/involutions.png
        :align: center
        :alt: Involution2d schema

    Args:
        in_channels (int): Number of channels in the input image
        kernel_size (int): Size of the convolving kernel
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        stride: Stride of the convolution. Default: 1
        groups: Number of blocked connections from input channels to output channels. Default: 1
        dilation: Spacing between kernel elements. Default: 1
        reduction_ratio: reduction ratio of the channels to generate the kernel
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        reduction_ratio: float = 1,
    ) -> None:

        super().__init__()

        self.groups = groups
        self.k_size = kernel_size

        self.pool = nn.AvgPool2d(stride, stride) if stride > 1 else None
        self.reduce = nn.Conv2d(in_channels, int(in_channels // reduction_ratio), 1)
        self.span = nn.Conv2d(int(in_channels // reduction_ratio), kernel_size**2 * groups, 1)
        self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)

    def forward(self, x):

        # Kernel generation
        # (N, C, H, W) --> (N, C, H // s, W // s)
        kernel = self.pool(x) if isinstance(self.pool, nn.Module) else x
        # --> (N, C // r, H // s, W // s)
        kernel = self.reduce(kernel)
        # --> (N, K * K * G, H // s, W // s)
        kernel = self.span(kernel)
        # --> (N, G, 1, K ** 2, H // s, W // s)
        kernel = kernel.view(x.shape[0], self.groups, 1, self.k_size**2, *kernel.shape[-2:])

        # --> (N, C * K ** 2, H * W // s ** 2)
        x_unfolded = self.unfold(x)
        # --> (N, G, C // G, K ** 2, H // s, W // s)
        x_unfolded = x_unfolded.reshape(x.shape[0], self.groups, x.shape[1] // self.groups, -1, *kernel.shape[-2:])

        # Multiply-Add operation
        # --> (N, C, H // s, W // s)
        out = (kernel * x_unfolded).sum(dim=3).view(*x.shape[:2], *kernel.shape[-2:])

        return out
