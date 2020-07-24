# -*- coding: utf-8 -*-

"""
Implementations of ResNet variations
"""

import sys
import logging
from collections import OrderedDict
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from holocron.nn import init
from .utils import conv_sequence


__all__ = ['BasicBlock', 'Bottleneck', 'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnext50_32x4d', 'resnext101_32x8d', 'resnet50d']


default_cfgs = {
    'resnet18': {'block': 'BasicBlock', 'num_blocks': [2, 2, 2, 2],
                 'url': None},
    'resnet34': {'block': 'BasicBlock', 'num_blocks': [3, 4, 6, 3],
                 'url': None},
    'resnet50': {'block': 'Bottleneck', 'num_blocks': [3, 4, 6, 3],
                 'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/resnet50_256-5e6206e0.pth'},
    'resnet101': {'block': 'Bottleneck', 'num_blocks': [3, 4, 23, 3],
                  'url': None},
    'resnet152': {'block': 'Bottleneck', 'num_blocks': [3, 8, 86, 3],
                  'url': None},
    'resnext50_32x4d': {'block': 'Bottleneck', 'num_blocks': [3, 4, 6, 3],
                        'url': None},
    'resnext101_32x8d': {'block': 'Bottleneck', 'num_blocks': [3, 4, 23, 3],
                         'url': None},
    'resnet50d': {'block': 'Bottleneck', 'num_blocks': [3, 4, 6, 3],
                  'url': None},
}


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

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None, **kwargs):
        super().__init__(
            [*conv_sequence(inplanes, planes, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3,
                            stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation, **kwargs),
             *conv_sequence(planes, planes, None, norm_layer, drop_layer, conv_layer, kernel_size=3,
                            stride=1, padding=dilation, groups=groups, bias=False, dilation=dilation, **kwargs)],
            downsample, act_layer)


class Bottleneck(_ResBlock):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None, **kwargs):

        width = int(planes * (base_width / 64.)) * groups
        super().__init__(
            [*conv_sequence(inplanes, width, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=1,
                            stride=1, bias=False, **kwargs),
             *conv_sequence(width, width, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3,
                            stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation, **kwargs),
             *conv_sequence(width, planes * self.expansion, None, norm_layer, drop_layer, conv_layer, kernel_size=1,
                            stride=1, bias=False, **kwargs)],
            downsample, act_layer)


class ChannelRepeat(nn.Module):
    def __init__(self, chan_repeats=1):
        super().__init__()
        self.chan_repeats = chan_repeats

    def forward(self, x):
        repeats = [1] * x.ndim
        # Repeat the tensor along the channel dimension
        repeats[1] = self.chan_repeats
        return x.repeat(*repeats)


class ResNet(nn.Sequential):
    def __init__(self, block, num_blocks, planes, num_classes=10, in_channels=3, zero_init_residual=False,
                 width_per_group=64,
                 conv_layer=None, act_layer=None, norm_layer=None, drop_layer=None, deep_stem=False, stem_pool=True,
                 avg_downsample=False, num_repeats=1, block_args=None):

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)
        self.dilation = 1

        in_planes = 64
        # Deep stem from ResNet-C
        if deep_stem:
            _layers = [*conv_sequence(in_channels, in_planes // 2, act_layer, norm_layer, drop_layer, conv_layer,
                                      kernel_size=3, stride=2, padding=1, bias=False),
                       *conv_sequence(in_planes // 2, in_planes // 2, act_layer, norm_layer, drop_layer, conv_layer,
                                      kernel_size=3, stride=1, padding=1, bias=False),
                       *conv_sequence(in_planes // 2, in_planes, act_layer, norm_layer, drop_layer, conv_layer,
                                      kernel_size=3, stride=1, padding=1, bias=False)]
        else:
            _layers = conv_sequence(in_channels, in_planes, act_layer, norm_layer, drop_layer, conv_layer,
                                    kernel_size=7, stride=2, padding=3, bias=False)
        if stem_pool:
            _layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Optional tensor repetitions along channel axis (mainly for TridentNet)
        if num_repeats > 1:
            _layers.append(ChannelRepeat(num_repeats))

        # Consecutive convolutional blocks
        stride = 1
        # Block args
        if block_args is None:
            block_args = dict(groups=1)
        if not isinstance(block_args, list):
            block_args = [block_args] * len(num_blocks)
        for _num_blocks, _planes, _block_args in zip(num_blocks, planes, block_args):
            _layers.append(self._make_layer(block, _num_blocks, in_planes, _planes, stride, width_per_group,
                                            act_layer=act_layer, norm_layer=norm_layer, drop_layer=drop_layer,
                                            avg_downsample=avg_downsample, num_repeats=num_repeats,
                                            block_args=_block_args))
            in_planes = block.expansion * _planes
            stride = 2

        super().__init__(OrderedDict([
            ('features', nn.Sequential(*_layers)),
            ('pool', nn.AdaptiveAvgPool2d((1, 1))),
            ('flatten', nn.Flatten(1)),
            ('head', nn.Linear(num_repeats * in_planes, num_classes))]))

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
    def _make_layer(block, num_blocks, in_planes, planes, stride=1, width_per_group=64,
                    act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None,
                    avg_downsample=False, num_repeats=1, block_args=None):

        downsample = None
        if stride != 1 or in_planes != planes * block.expansion:
            # Downsampling from ResNet-D
            if avg_downsample:
                downsample = nn.Sequential(nn.AvgPool2d(stride, ceil_mode=True, count_include_pad=False),
                                           *conv_sequence(num_repeats * in_planes,
                                                          num_repeats * planes * block.expansion,
                                                          None, norm_layer, drop_layer, conv_layer,
                                                          kernel_size=1, stride=1, bias=False))
            else:
                downsample = nn.Sequential(*conv_sequence(num_repeats * in_planes,
                                                          num_repeats * planes * block.expansion,
                                                          None, norm_layer, drop_layer, conv_layer,
                                                          kernel_size=1, stride=stride, bias=False))
        if block_args is None:
            block_args = {}
        layers = [block(in_planes, planes, stride, downsample, base_width=width_per_group,
                        act_layer=act_layer, norm_layer=norm_layer, drop_layer=drop_layer, **block_args)]

        for _ in range(num_blocks - 1):
            layers.append(block(block.expansion * planes, planes, 1, None, base_width=width_per_group,
                                act_layer=act_layer, norm_layer=norm_layer, drop_layer=drop_layer, **block_args))

        return nn.Sequential(*layers)


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


def resnet50d(pretrained=False, progress=True, **kwargs):
    """ResNet-50-D from
    `"Bag of Tricks for Image Classification with Convolutional Neural Networks"
    <https://arxiv.org/pdf/1812.01187.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _resnet('resnet50d', pretrained, progress, deep_stem=True, avg_downsample=True, **kwargs)


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

    kwargs['width_per_group'] = 4
    block_args = dict(groups=32)
    return _resnet('resnext50_32x4d', pretrained, progress, block_args=block_args, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    """ResNeXt-101 from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    kwargs['width_per_group'] = 8
    block_args = dict(groups=32)
    return _resnet('resnext101_32x8d', pretrained, progress, block_args=block_args, **kwargs)
