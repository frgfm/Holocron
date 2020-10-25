import sys
from math import ceil
from collections import OrderedDict
import torch.nn as nn
from holocron.nn import SiLU, init, GlobalAvgPool2d
from .utils import conv_sequence, load_pretrained_params
from typing import Dict, Any


__all__ = ['SEBlock', 'ReXBlock', 'ReXNet', 'rexnet1_0x', 'rexnet1_3x', 'rexnet1_5x', 'rexnet2_0x', 'rexnet2_2x']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'rexnet1_0x': {'width_mult': 1.0, 'depth_mult': 1.0,
                   'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/rexnet1_0x_224-ab7b9733.pth'},
    'rexnet1_3x': {'width_mult': 1.3, 'depth_mult': 1.0,
                   'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/rexnet1_3x_224-95479104.pth'},
    'rexnet1_5x': {'width_mult': 1.5, 'depth_mult': 1.0,
                   'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/rexnet1_5x_224-c42a16ac.pth'},
    'rexnet2_0x': {'width_mult': 2.0, 'depth_mult': 1.0,
                   'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/rexnet2_0x_224-c8802402.pth'},
    'rexnet2_2x': {'width_mult': 2.2, 'depth_mult': 1.0,
                   'url': None},
}


class SEBlock(nn.Module):

    def __init__(self, channels, se_ratio=12, act_layer=None, norm_layer=None, drop_layer=None):
        super().__init__()
        self.pool = GlobalAvgPool2d(flatten=False)
        self.conv = nn.Sequential(
            *conv_sequence(channels, channels // se_ratio, act_layer, norm_layer, drop_layer,
                           kernel_size=1, stride=1, bias=False),
            *conv_sequence(channels // se_ratio, channels, nn.Sigmoid(), None, drop_layer,
                           kernel_size=1, stride=1))

    def forward(self, x):

        y = self.pool(x)
        y = self.conv(y)
        return x * y


class ReXBlock(nn.Module):
    def __init__(self, in_channels, channels, t, stride, use_se=True, se_ratio=12,
                 act_layer=None, norm_layer=None, drop_layer=None):
        super().__init__()

        if act_layer is None:
            act_layer = nn.ReLU6(inplace=True)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels
        self.out_channels = channels

        _layers = []
        if t != 1:
            dw_channels = in_channels * t
            _layers.extend(conv_sequence(in_channels, dw_channels, SiLU(), norm_layer, drop_layer, kernel_size=1,
                                         stride=1, bias=False))
        else:
            dw_channels = in_channels

        _layers.extend(conv_sequence(dw_channels, dw_channels, None, norm_layer, drop_layer, kernel_size=3,
                                     stride=stride, padding=1, bias=False, groups=dw_channels))

        if use_se:
            _layers.append(SEBlock(dw_channels, se_ratio, act_layer, norm_layer, drop_layer))

        _layers.append(act_layer)
        _layers.extend(conv_sequence(dw_channels, channels, None, norm_layer, drop_layer, kernel_size=1,
                                     stride=1, bias=False))
        self.conv = nn.Sequential(*_layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_shortcut:
            out[:, :self.in_channels] += x

        return out


class ReXNet(nn.Sequential):
    def __init__(self, width_mult=1.0, depth_mult=1.0, num_classes=1000, in_channels=3, in_planes=16, final_planes=180,
                 use_se=True, se_ratio=12, dropout_ratio=0.2, bn_momentum=0.9,
                 act_layer=None, norm_layer=None, drop_layer=None):
        """Mostly adapted from https://github.com/clovaai/rexnet/blob/master/rexnetv1.py"""
        super().__init__()

        if act_layer is None:
            act_layer = SiLU()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        num_blocks = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]
        num_blocks = [ceil(element * depth_mult) for element in num_blocks]
        strides = sum([[element] + [1] * (num_blocks[idx] - 1) for idx, element in enumerate(strides)], [])
        depth = sum(num_blocks)

        stem_channel = 32 / width_mult if width_mult < 1.0 else 32
        inplanes = in_planes / width_mult if width_mult < 1.0 else in_planes

        # The following channel configuration is a simple instance to make each layer become an expand layer
        chans = [int(round(width_mult * stem_channel))]
        chans.extend([int(round(width_mult * (inplanes + idx * final_planes / depth))) for idx in range(depth)])

        ses = [False] * (num_blocks[0] + num_blocks[1]) + [use_se] * sum(num_blocks[2:])

        _layers = conv_sequence(in_channels, chans[0], act_layer, norm_layer, drop_layer,
                                kernel_size=3, stride=2, padding=1, bias=False)

        t = 1
        for in_c, c, s, se in zip(chans[:-1], chans[1:], strides, ses):
            _layers.append(ReXBlock(in_channels=in_c, channels=c, t=t, stride=s, use_se=se, se_ratio=se_ratio))
            t = 6

        pen_channels = int(width_mult * 1280)
        _layers.extend(conv_sequence(chans[-1], pen_channels, act_layer, norm_layer, drop_layer,
                                     kernel_size=1, stride=1, padding=0, bias=False))

        super().__init__(OrderedDict([
            ('features', nn.Sequential(*_layers)),
            ('pool', GlobalAvgPool2d(flatten=True)),
            ('head', nn.Sequential(nn.Dropout(dropout_ratio), nn.Linear(pen_channels, num_classes)))]))

        # Init all layers
        init.init_module(self, nonlinearity='relu')


def _rexnet(arch, pretrained, progress, **kwargs):

    # Build the model
    model = ReXNet(default_cfgs[arch]['width_mult'], default_cfgs[arch]['depth_mult'], **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'], progress)

    return model


def rexnet1_0x(pretrained=False, progress=True, **kwargs):
    """ReXNet-1.0x from
    `"ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"
    <https://arxiv.org/pdf/2007.00992.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _rexnet('rexnet1_0x', pretrained, progress, **kwargs)


def rexnet1_3x(pretrained=False, progress=True, **kwargs):
    """ReXNet-1.3x from
    `"ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"
    <https://arxiv.org/pdf/2007.00992.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _rexnet('rexnet1_3x', pretrained, progress, **kwargs)


def rexnet1_5x(pretrained=False, progress=True, **kwargs):
    """ReXNet-1.5x from
    `"ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"
    <https://arxiv.org/pdf/2007.00992.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _rexnet('rexnet1_5x', pretrained, progress, **kwargs)


def rexnet2_0x(pretrained=False, progress=True, **kwargs):
    """ReXNet-2.0x from
    `"ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"
    <https://arxiv.org/pdf/2007.00992.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _rexnet('rexnet2_0x', pretrained, progress, **kwargs)


def rexnet2_2x(pretrained=False, progress=True, **kwargs):
    """ReXNet-2.2x from
    `"ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"
    <https://arxiv.org/pdf/2007.00992.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _rexnet('rexnet2_2x', pretrained, progress, **kwargs)
