# -*- coding: utf-8 -*-

"""
Implementations of ResNet variations
"""

import sys
import logging
import torch.nn as nn
from holocron.nn import init


__all__ = ['BasicBlock', 'Bottleneck', 'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnext50_32x4d', 'resnext101_32x8d']


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
}


def _conv_sequence(in_channels, out_channels, act_layer=None, norm_layer=None, drop_layer=None, **kwargs):

    conv_seq = [nn.Conv2d(in_channels, out_channels, **kwargs)]

    if callable(norm_layer):
        conv_seq.append(norm_layer(out_channels))
    if callable(drop_layer):
        conv_seq.append(drop_layer(p=0.1, block_size=3, inplace=True))
    if callable(act_layer):
        conv_seq.append(act_layer)

    return nn.Sequential(*conv_seq)


class _ResBlock(nn.Module):

    def __init__(self, convs, downsample=None, act_layer=None):
        super().__init__()

        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        # Main branch
        self.convs = nn.Sequential(*convs)
        # Shortcut connection
        self.downsample = downsample
        self.activation = act_layer

    def forward(self, x):
        identity = x

        out = self.convs(x)

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
            [_conv_sequence(inplanes, planes, act_layer, norm_layer, drop_layer, kernel_size=3, stride=stride,
                            padding=dilation, groups=groups, bias=False, dilation=dilation),
             _conv_sequence(planes, planes, None, norm_layer, drop_layer, kernel_size=3, stride=1,
                            padding=dilation, groups=groups, bias=False, dilation=dilation)],
            downsample, act_layer)


class Bottleneck(_ResBlock):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, act_layer=None, drop_layer=None):

        width = int(planes * (base_width / 64.)) * groups
        super().__init__(
            [_conv_sequence(inplanes, width, act_layer, norm_layer, drop_layer, kernel_size=1,
                            stride=1, bias=False),
             _conv_sequence(width, width, act_layer, norm_layer, drop_layer, kernel_size=3, stride=stride,
                            padding=dilation, groups=groups, bias=False, dilation=dilation),
             _conv_sequence(width, planes * self.expansion, None, norm_layer, drop_layer, kernel_size=1,
                            stride=1, bias=False)],
            downsample, act_layer)


class ResNet(nn.Sequential):

    def __init__(self, block, num_blocks, planes, num_classes=1000, in_channels=3, zero_init_residual=False,
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
        _layers = [_conv_sequence(in_channels, in_planes, act_layer, norm_layer, drop_layer,
                                  kernel_size=7, stride=2, padding=3, bias=False),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        # Consecutive convolutional blocks
        stride = 1
        for _num_blocks, _planes in zip(num_blocks, planes):
            _layers.append(self._res_layer(block, _num_blocks, in_planes, _planes, stride,
                                           norm_layer=norm_layer, act_layer=act_layer, drop_layer=drop_layer))
            in_planes = block.expansion * _planes
            stride = 2

        _layers.extend([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(in_planes, num_classes)])

        super().__init__(*_layers)

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
            downsample = _conv_sequence(in_planes, planes * block.expansion, None, norm_layer, drop_layer,
                                        kernel_size=1, stride=stride, bias=False)
        layers = [block(in_planes, planes, stride, downsample, groups, width_per_group,
                        norm_layer=norm_layer, act_layer=act_layer, drop_layer=drop_layer)]

        for _ in range(num_blocks - 1):
            layers.append(block(block.expansion * planes, planes, 1, None, groups, width_per_group,
                                norm_layer=norm_layer, act_layer=act_layer, drop_layer=drop_layer))

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

    return _resnet('resnet18', pretrained, progress, **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):

    return _resnet('resnet34', pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):

    return _resnet('resnet50', pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):

    return _resnet('resnet101', pretrained, progress, **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):

    return _resnet('resnet152', pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):

    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):

    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', pretrained, progress, **kwargs)
