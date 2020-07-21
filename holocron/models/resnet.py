# -*- coding: utf-8 -*-

"""
Implementations of ResNet variations
"""

import sys
import logging
from math import ceil
from collections import OrderedDict
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from holocron.nn import SiLU, init


__all__ = ['BasicBlock', 'Bottleneck', 'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnext50_32x4d', 'resnext101_32x8d',
           'SEBlock', 'ReXBlock', 'ReXNet', 'rexnet1_0x', 'rexnet1_3x', 'rexnet1_5x', 'rexnet2_0x', 'rexnet2_2x']


default_cfgs = {
    'resnet18': {'block': 'BasicBlock', 'num_blocks': [2, 2, 2, 2],
                 'url': None},
    'resnet34': {'block': 'BasicBlock', 'num_blocks': [3, 4, 6, 3],
                 'url': None},
    'resnet50': {'block': 'Bottleneck', 'num_blocks': [3, 4, 6, 3],
                 'url': None},
    'resnet101': {'block': 'Bottleneck', 'num_blocks': [3, 4, 23, 3],
                  'url': None},
    'resnet152': {'block': 'Bottleneck', 'num_blocks': [3, 8, 86, 3],
                  'url': None},
    'resnext50_32x4d': {'block': 'Bottleneck', 'num_blocks': [3, 4, 6, 3],
                        'url': None},
    'resnext101_32x8d': {'block': 'Bottleneck', 'num_blocks': [3, 4, 23, 3],
                         'url': None},
    'rexnet1_0x': {'width_mult': 1.0, 'depth_mult': 1.0,
                   'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/rexnet1_0_224-a120bf73.pth'},
    'rexnet1_3x': {'width_mult': 1.3, 'depth_mult': 1.0,
                   'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/rexnet1_3_224-191b60f1.pth'},
    'rexnet1_5x': {'width_mult': 1.5, 'depth_mult': 1.0,
                   'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/rexnet1_5_224-30ce6260.pth'},
    'rexnet2_0x': {'width_mult': 2.0, 'depth_mult': 1.0,
                   'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/rexnet2_0_224-e5243878.pth'},
    'rexnet2_2x': {'width_mult': 2.2, 'depth_mult': 1.0,
                   'url': None},
}


def _conv_sequence(in_channels, out_channels, act_layer=None, norm_layer=None, drop_layer=None, **kwargs):

    conv_seq = [nn.Conv2d(in_channels, out_channels, **kwargs)]

    if callable(norm_layer):
        conv_seq.append(norm_layer(out_channels))
    if callable(drop_layer):
        conv_seq.append(drop_layer(p=0.1, block_size=3, inplace=True))
    if callable(act_layer):
        conv_seq.append(act_layer)

    return conv_seq


class _ResBlock(nn.Module):

    def __init__(self, convs, downsample=None, act_layer=None):
        super().__init__()

        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        # Main branch
        self.conv = nn.Sequential(*convs)
        # Shortcut connection
        self.downsample = downsample
        self.activation = act_layer

    def forward(self, x):
        identity = x

        out = self.conv(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class BasicBlock(_ResBlock):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 dilation=1, norm_layer=None, act_layer=None, drop_layer=None):
        super().__init__(
            [*_conv_sequence(inplanes, planes, act_layer, norm_layer, drop_layer, kernel_size=3, stride=stride,
                             padding=dilation, groups=groups, bias=False, dilation=dilation),
             *_conv_sequence(planes, planes, None, norm_layer, drop_layer, kernel_size=3, stride=1,
                             padding=dilation, groups=groups, bias=False, dilation=dilation)],
            downsample, act_layer)


class Bottleneck(_ResBlock):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, act_layer=None, drop_layer=None):

        width = int(planes * (base_width / 64.)) * groups
        super().__init__(
            [*_conv_sequence(inplanes, width, act_layer, norm_layer, drop_layer, kernel_size=1,
                             stride=1, bias=False),
             *_conv_sequence(width, width, act_layer, norm_layer, drop_layer, kernel_size=3, stride=stride,
                             padding=dilation, groups=groups, bias=False, dilation=dilation),
             *_conv_sequence(width, planes * self.expansion, None, norm_layer, drop_layer, kernel_size=1,
                             stride=1, bias=False)],
            downsample, act_layer)


class ResNet(nn.Sequential):
    def __init__(self, block, num_blocks, planes, num_classes=10, in_channels=3, zero_init_residual=False,
                 groups=1, width_per_group=64, conv_layer=None, norm_layer=None, act_layer=None, drop_layer=None):

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        in_planes = 64
        # Stem
        _layers = [*_conv_sequence(in_channels, in_planes, act_layer, norm_layer, drop_layer,
                                   kernel_size=7, stride=2, padding=3, bias=False),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        # Consecutive convolutional blocks
        stride = 1
        for _num_blocks, _planes in zip(num_blocks, planes):
            _layers.append(self._res_layer(block, _num_blocks, in_planes, _planes, stride,
                                           norm_layer=norm_layer, act_layer=act_layer, drop_layer=drop_layer))
            in_planes = block.expansion * _planes
            stride = 2

        super().__init__(OrderedDict([
            ('features', nn.Sequential(*_layers)),
            ('pool', nn.AdaptiveAvgPool2d((1, 1))),
            ('flatten', nn.Flatten(1)),
            ('head', nn.Linear(in_planes, num_classes))]))

        # Init all layers
        init.init_module(self, nonlinearity='relu')

        # Init shortcut
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    m.convs[2][1].weight.data.zero_()
                elif isinstance(m, BasicBlock):
                    m.convs[1][1].weight.data.zero_()

    @staticmethod
    def _res_layer(block, num_blocks, in_planes, planes, stride=1, groups=1, width_per_group=64,
                   norm_layer=None, act_layer=None, drop_layer=None):

        downsample = None
        if stride != 1 or in_planes != planes * block.expansion:
            downsample = nn.Sequential(*_conv_sequence(in_planes, planes * block.expansion, None, norm_layer,
                                                       drop_layer, kernel_size=1, stride=stride, bias=False))
        layers = [block(in_planes, planes, stride, downsample, groups, width_per_group,
                        norm_layer=norm_layer, act_layer=act_layer, drop_layer=drop_layer)]

        for _ in range(num_blocks - 1):
            layers.append(block(block.expansion * planes, planes, 1, None, groups, width_per_group,
                                norm_layer=norm_layer, act_layer=act_layer, drop_layer=drop_layer))

        return nn.Sequential(*layers)


class SEBlock(nn.Module):

    def __init__(self, channels, se_ratio=12, act_layer=None, norm_layer=None, drop_layer=None):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            *_conv_sequence(channels, channels // se_ratio, act_layer, norm_layer, drop_layer,
                            kernel_size=1, stride=1),
            *_conv_sequence(channels // se_ratio, channels, nn.Sigmoid(), None, drop_layer,
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
            _layers.extend(_conv_sequence(in_channels, dw_channels, SiLU(), norm_layer, drop_layer, kernel_size=1,
                                          stride=1, bias=False))
        else:
            dw_channels = in_channels

        _layers.extend(_conv_sequence(dw_channels, dw_channels, None, norm_layer, drop_layer, kernel_size=3,
                                      stride=stride, padding=1, bias=False, groups=dw_channels))

        if use_se:
            _layers.append(SEBlock(dw_channels, se_ratio, act_layer, norm_layer, drop_layer))

        _layers.append(act_layer)
        _layers.extend(_conv_sequence(dw_channels, channels, None, norm_layer, drop_layer, kernel_size=1,
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

        _layers = _conv_sequence(in_channels, chans[0], act_layer, norm_layer, drop_layer,
                                 kernel_size=3, stride=2, padding=1, bias=False)

        t = 1
        for in_c, c, s, se in zip(chans[:-1], chans[1:], strides, ses):
            _layers.append(ReXBlock(in_channels=in_c, channels=c, t=t, stride=s, use_se=se, se_ratio=se_ratio))
            t = 6

        pen_channels = int(width_mult * 1280)
        _layers.extend(_conv_sequence(chans[-1], pen_channels, act_layer, norm_layer, drop_layer,
                                      kernel_size=1, stride=1, padding=0, bias=False))

        super().__init__(OrderedDict([
            ('features', nn.Sequential(*_layers)),
            ('pool', nn.AdaptiveAvgPool2d((1, 1))),
            ('flatten', nn.Flatten(1)),
            ('head', nn.Sequential(nn.Dropout(dropout_ratio), nn.Linear(pen_channels, num_classes)))]))


def _resnet(arch, pretrained, progress, **kwargs):

    # Retrieve the correct block type
    block = sys.modules[__name__].__dict__[default_cfgs[arch]['block']]

    # Build the model
    model = ResNet(block, default_cfgs[arch]['num_blocks'], [64, 128, 256, 512], **kwargs)
    # Load pretrained parameters
    if pretrained:
        if default_cfgs[arch]['url'] is None:
            logging.warning(f"Invalid model URL for {arch}, using default initialization.")
        else:
            state_dict = load_state_dict_from_url(default_cfgs[arch]['url'],
                                                  progress=progress)
            model.load_state_dict(state_dict)

    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """ResNet-18 from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _resnet('resnet18', pretrained, progress, **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    """ResNet-34 from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _resnet('resnet34', pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """ResNet-50 from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _resnet('resnet50', pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    """ResNet-101 from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _resnet('resnet101', pretrained, progress, **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    """ResNet-152 from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _resnet('resnet152', pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    """ResNeXt-50 from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    """ResNeXt-101 from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', pretrained, progress, **kwargs)


def _rexnet(arch, pretrained, progress, **kwargs):

    # Build the model
    model = ReXNet(default_cfgs[arch]['width_mult'], default_cfgs[arch]['depth_mult'], **kwargs)
    # Load pretrained parameters
    if pretrained:
        if default_cfgs[arch]['url'] is None:
            logging.warning(f"Invalid model URL for {arch}, using default initialization.")
        else:
            state_dict = load_state_dict_from_url(default_cfgs[arch]['url'],
                                                  progress=progress)
            model.load_state_dict(state_dict)

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
