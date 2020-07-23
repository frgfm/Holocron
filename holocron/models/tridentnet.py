# -*- coding: utf-8 -*-

"""
Implementation of TridentNet
"""

import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models.utils import load_state_dict_from_url
from .resnet import _ResBlock, ResNet
from .utils import conv_sequence


__all__ = ['Tridentneck', 'tridentnet50']

default_cfgs = {
    'tridentnet50': {'block': 'Bottleneck', 'num_blocks': [3, 4, 6, 3],
                     'url': None},
}


class TridentConv2d(nn.Conv2d):

    num_branches = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.dilation[0] != 1 and self.dilation[0] != self.num_branches:
            raise ValueError(f"expected dilation to either be 1 or {self.num_branches}.")

    def forward(self, x):
        if x.shape[1] % self.num_branches != 0:
            raise ValueError("expected number of channels of input tensor to be a multiple of `num_branches`.")

        # Dilation for each chunk
        if self.dilation[0] == 1:
            dilations = [1] * self.num_branches
        else:
            dilations = [1 + idx for idx in range(self.num_branches)]

        out = []
        # Use shared weight to apply the convolution
        for _x, dilation in zip(torch.chunk(x, self.num_branches, 1), dilations):
            out.append(F.conv2d(_x, self.weight, self.bias, self.stride,
                                tuple(dilation * p for p in self.padding),
                                (dilation,) * len(self.dilation), self.groups))

        out = torch.cat(out, 1)

        return out


class Tridentneck(_ResBlock):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=3, act_layer=None, norm_layer=None, drop_layer=None):

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        width = int(planes * (base_width / 64.)) * groups
        # Concatenate along the channel axis and enlarge BN to leverage parallelization
        super().__init__(
            [*conv_sequence(inplanes, width, act_layer, norm_layer, drop_layer, TridentConv2d, bn_channels=3 * width,
                             kernel_size=1, stride=1, bias=False, dilation=1),
             *conv_sequence(width, width, act_layer, norm_layer, drop_layer, TridentConv2d, bn_channels=3 * width,
                             kernel_size=3, stride=stride,
                             padding=1, groups=groups, bias=False, dilation=3),
             *conv_sequence(width, planes * self.expansion, None, norm_layer, drop_layer, TridentConv2d,
                             bn_channels=3 * planes * self.expansion,
                             kernel_size=1, stride=1, bias=False, dilation=1)],
            downsample, act_layer)


def _tridentnet(arch, pretrained, progress, **kwargs):
    # Build the model
    model = ResNet(Tridentneck, default_cfgs[arch]['num_blocks'], [64, 128, 256, 512],
                   num_repeats=3, **kwargs)
    # Load pretrained parameters
    if pretrained:
        if default_cfgs[arch]['url'] is None:
            logging.warning(f"Invalid model URL for {arch}, using default initialization.")
        else:
            state_dict = load_state_dict_from_url(default_cfgs[arch]['url'],
                                                  progress=progress)
            model.load_state_dict(state_dict)

    return model


def tridentnet50(pretrained=False, progress=True, **kwargs):
    """TridentNet-50 from
    `"Scale-Aware Trident Networks for Object Detection" <https://arxiv.org/pdf/1901.01892.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _tridentnet('tridentnet50', pretrained, progress, **kwargs)
