# -*- coding: utf-8 -*-

"""
Implementation of PyConvResNet
"""

import logging
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from holocron.nn import PyConv2d
from .resnet import ResNet, _ResBlock
from .utils import conv_sequence


__all__ = ['PyBottleneck', 'pyconvresnet50']


default_cfgs = {
    'pyconvresnet50': {'block': 'Bottleneck', 'num_blocks': [3, 4, 6, 3],
                       'url': None},
}


class PyBottleneck(_ResBlock):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 act_layer=None, norm_layer=None, drop_layer=None, num_levels=2, **kwargs):

        width = int(planes * (base_width / 64.)) * groups

        super().__init__(
            [*conv_sequence(inplanes, width, act_layer, norm_layer, drop_layer, kernel_size=1,
                            stride=1, bias=False, **kwargs),
             *conv_sequence(width, width, act_layer, norm_layer, drop_layer, conv_layer=PyConv2d, kernel_size=3,
                            stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation,
                            num_levels=num_levels, **kwargs),
             *conv_sequence(width, planes * self.expansion, None, norm_layer, drop_layer, kernel_size=1,
                            stride=1, bias=False, **kwargs)],
            downsample, act_layer)


def _pyconvresnet(arch, pretrained, progress, **kwargs):
    # Build the model
    model = ResNet(PyBottleneck, default_cfgs[arch]['num_blocks'], [64, 128, 256, 512], stem_pool=False,
                   block_args=[dict(num_levels=levels) for levels in [4, 3, 2, 1]], **kwargs)
    # Load pretrained parameters
    if pretrained:
        if default_cfgs[arch]['url'] is None:
            logging.warning(f"Invalid model URL for {arch}, using default initialization.")
        else:
            state_dict = load_state_dict_from_url(default_cfgs[arch]['url'],
                                                  progress=progress)
            model.load_state_dict(state_dict)

    return model


def pyconvresnet50(pretrained=False, progress=True, **kwargs):
    """PyConvResNet-50 from `"Pyramidal Convolution: Rethinking Convolutional Neural Networks
    for Visual Recognition" <https://arxiv.org/pdf/2006.11538.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _pyconvresnet('pyconvresnet50', pretrained, progress, **kwargs)
