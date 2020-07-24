# -*- coding: utf-8 -*-

"""
Implementation of Res2Net
based on https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/res2net.py
"""

import logging
import math
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from .resnet import _ResBlock, ResNet
from .utils import conv_sequence


__all__ = ['Bottle2neck', 'res2net50_26w_4s']


default_cfgs = {
    'res2net50_26w_4s': {'num_blocks': [3, 4, 6, 3], 'width_per_group': 26, 'scale': 4,
                         'url': None},
}


class ScaleConv2d(nn.Module):
    def __init__(self, scale, planes, kernel_size, stride=1, groups=1, downsample=False,
                 act_layer=None, norm_layer=None, drop_layer=None):
        super().__init__()

        self.scale = scale
        self.width = planes // scale
        self.conv = nn.ModuleList([nn.Sequential(*conv_sequence(self.width, self.width,
                                                                act_layer, norm_layer, drop_layer,
                                                                kernel_size=3, stride=stride, padding=1,
                                                                groups=groups, bias=False))
                                   for _ in range(max(1, scale - 1))])

        if downsample:
            self.downsample = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        else:
            self.downsample = None

    def forward(self, x):

        # Split the channel dimension into groups of self.width channels
        split_x = torch.split(x, self.width, 1)
        out = []
        _res = split_x[0]
        for idx, layer in enumerate(self.conv):
            # If downsampled, don't add previous branch
            if idx == 0 or self.downsample is not None:
                _res = split_x[idx]
            else:
                _res += split_x[idx]
            _res = layer(_res)
            out.append(_res)
        # Use the last chunk as shortcut connection
        if self.scale > 1:
            # If the convs were strided, the shortcut needs to be downsampled
            if self.downsample is not None:
                out.append(self.downsample(split_x[-1]))
            else:
                out.append(split_x[-1])
        out = torch.cat(out, 1)

        return out


class Bottle2neck(_ResBlock):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=26, dilation=1, act_layer=None, norm_layer=None, drop_layer=None,
                 scale=4):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        # Check if ScaleConv2d needs to downsample the identity branch
        _downsample = stride > 1 or downsample is not None

        width = int(math.floor(planes * (base_width / 64.))) * groups
        super().__init__(
            [*conv_sequence(inplanes, width * scale, act_layer, norm_layer, drop_layer, kernel_size=1,
                            stride=1, bias=False),
             ScaleConv2d(scale, width * scale, 3, stride, groups, _downsample, act_layer, norm_layer, drop_layer),
             *conv_sequence(width * scale, planes * self.expansion, None, norm_layer, drop_layer, kernel_size=1,
                            stride=1, bias=False)],
            downsample, act_layer)


def _res2net(arch, pretrained, progress, **kwargs):
    # Build the model
    model = ResNet(Bottle2neck, default_cfgs[arch]['num_blocks'], [64, 128, 256, 512],
                   width_per_group=default_cfgs[arch]['width_per_group'],
                   block_args=dict(scale=default_cfgs[arch]['scale']), **kwargs)
    # Load pretrained parameters
    if pretrained:
        if default_cfgs[arch]['url'] is None:
            logging.warning(f"Invalid model URL for {arch}, using default initialization.")
        else:
            state_dict = load_state_dict_from_url(default_cfgs[arch]['url'],
                                                  progress=progress)
            model.load_state_dict(state_dict)

    return model


def res2net50_26w_4s(pretrained=False, progress=True, **kwargs):
    """Res2Net-50 26wx4s from
    `"Res2Net: A New Multi-scale Backbone Architecture" <https://arxiv.org/pdf/1904.01169.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _res2net('res2net50_26w_4s', pretrained, progress, **kwargs)
