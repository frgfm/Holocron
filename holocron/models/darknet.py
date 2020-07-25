# -*- coding: utf-8 -*-

"""
Implementation of DarkNet
"""

import sys
import logging
from collections import OrderedDict
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

from ..nn.init import init_module
from .utils import conv_sequence
from .resnet import _ResBlock


__all__ = ['DarknetV1', 'DarknetV2', 'DarknetV3', 'darknet24', 'darknet19', 'darknet53']


default_cfgs = {
    'darknet24': {'arch': 'DarknetV1',
                  'layout': [[128, 256, 256, 512], [*([256, 512] * 4), 512, 1024], [512, 1024] * 2],
                  'url': None},
    'darknet19': {'arch': 'DarknetV2',
                  'layout': [(128, 1), (256, 1), (512, 2), (1024, 2)],
                  'url': None},
    'darknet53': {'arch': 'DarknetV3',
                  'layout': [(1, 64, 128), (2, 128, 256), (8, 256, 512), (8, 512, 1024), (4, 1024, None)],
                  'url': None},
}


class DarkBlockV1(nn.Sequential):
    def __init__(self, planes, act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):
        _layers = []
        k1 = True
        for in_planes, out_planes in zip(planes[:-1], planes[1:]):
            _layers.extend(conv_sequence(in_planes, out_planes, act_layer, norm_layer, drop_layer, conv_layer,
                                         kernel_size=3 if out_planes > in_planes else 1,
                                         padding=1 if out_planes > in_planes else 0, bias=False))
            k1 = not k1

        super().__init__(*_layers)


class DarknetBodyV1(nn.Sequential):
    def __init__(self, layout, in_channels=3,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):

        if act_layer is None:
            act_layer = nn.LeakyReLU(0.1, inplace=True)

        super().__init__(
            *conv_sequence(in_channels, 64, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=7, padding=3, stride=2, bias=False),
            nn.MaxPool2d(2),
            *conv_sequence(64, 192, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=1, stride=1, bias=False),
            nn.MaxPool2d(2),
            DarkBlockV1([192] + layout[0], act_layer, norm_layer, drop_layer, conv_layer),
            nn.MaxPool2d(2),
            DarkBlockV1(layout[0][-1:] + layout[1], act_layer, norm_layer, drop_layer, conv_layer),
            nn.MaxPool2d(2),
            DarkBlockV1(layout[1][-1:] + layout[2], act_layer, norm_layer, drop_layer, conv_layer))


class DarknetV1(nn.Sequential):
    def __init__(self, layout, num_classes=10, in_channels=3,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):
        super().__init__(OrderedDict([
            ('features', DarknetBodyV1(layout, in_channels, act_layer, norm_layer, drop_layer, conv_layer)),
            ('global_pool', nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())),
            ('classifier', nn.Linear(layout[2][-1], num_classes))]))

        init_module(self, 'leaky_relu')


class DarkBlockV2(nn.Sequential):
    def __init__(self, in_planes, out_planes, nb_compressions=0,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):

        layers = conv_sequence(in_planes, out_planes, act_layer, norm_layer, drop_layer, conv_layer,
                               kernel_size=3, padding=1, stride=1, bias=False)
        for _ in range(nb_compressions):
            layers.extend(conv_sequence(out_planes, in_planes, act_layer, norm_layer, drop_layer, conv_layer,
                                        kernel_size=1, padding=0, stride=1, bias=False))
            layers.extend(conv_sequence(in_planes, out_planes, act_layer, norm_layer, drop_layer, conv_layer,
                                        kernel_size=3, padding=1, stride=1, bias=False))

        super().__init__(*layers)


class DarknetBodyV2(nn.Sequential):

    passthrough = None

    def __init__(self, layout, in_channels=3,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):

        if act_layer is None:
            act_layer = nn.LeakyReLU(0.1, inplace=True)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        super().__init__(
            *conv_sequence(in_channels, 32, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=1, padding=0, stride=1, bias=False),
            nn.MaxPool2d(2),
            *conv_sequence(32, 64, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=1, stride=1, bias=False),
            nn.MaxPool2d(2),
            DarkBlockV2(64, *layout[0], act_layer, norm_layer, drop_layer, conv_layer),
            nn.MaxPool2d(2),
            DarkBlockV2(layout[0][0], *layout[1], act_layer, norm_layer, drop_layer, conv_layer),
            nn.MaxPool2d(2),
            DarkBlockV2(layout[1][0], *layout[2], act_layer, norm_layer, drop_layer, conv_layer),
            nn.MaxPool2d(2),
            DarkBlockV2(layout[2][0], *layout[3], act_layer, norm_layer, drop_layer, conv_layer))


class DarknetV2(nn.Sequential):
    def __init__(self, layout, num_classes=10, in_channels=3,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):

        super().__init__(OrderedDict([
            ('features', DarknetBodyV2(layout, in_channels, act_layer, norm_layer, drop_layer, conv_layer)),
            ('classifier', nn.Conv2d(layout[-1][0], num_classes, 1)),
            ('global_pool', nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()))]))

        init_module(self, 'leaky_relu')


class DarkBlockV3(_ResBlock):

    def __init__(self, planes, mid_planes, act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):
        super().__init__(
            [*conv_sequence(planes, mid_planes, act_layer, norm_layer, drop_layer, conv_layer,
                            kernel_size=1, bias=False),
             *conv_sequence(mid_planes, planes, act_layer, norm_layer, drop_layer, conv_layer,
                            kernel_size=3, padding=1, bias=False)],
            None, act_layer)


class DarknetBodyV3(nn.Sequential):

    def __init__(self, layout, in_channels=3,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):

        if act_layer is None:
            act_layer = nn.LeakyReLU(0.1, inplace=True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        super().__init__(
            *conv_sequence(in_channels, 32, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=1, bias=False),
            *conv_sequence(32, 64, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=2, stride=2, bias=False),
            self._make_layer(*layout[0], act_layer, norm_layer, drop_layer, conv_layer),
            self._make_layer(*layout[1], act_layer, norm_layer, drop_layer, conv_layer),
            self._make_layer(*layout[2], act_layer, norm_layer, drop_layer, conv_layer),
            self._make_layer(*layout[3], act_layer, norm_layer, drop_layer, conv_layer),
            self._make_layer(*layout[4], act_layer, norm_layer, drop_layer, conv_layer)
        )

    @staticmethod
    def _make_layer(num_blocks, in_planes, out_planes=None,
                    act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):

        layers = [DarkBlockV3(in_planes, in_planes // 2, act_layer, norm_layer, drop_layer, conv_layer)
                  for _ in range(num_blocks)]
        if isinstance(out_planes, int):
            layers.extend(conv_sequence(in_planes, out_planes, act_layer, norm_layer, drop_layer, conv_layer,
                                        kernel_size=3, padding=1, stride=2, bias=False))

        return nn.Sequential(*layers)


class DarknetV3(nn.Sequential):
    def __init__(self, layout, num_classes=10, in_channels=3,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):

        super().__init__(OrderedDict([
            ('features', DarknetBodyV3(layout, in_channels, act_layer, norm_layer, drop_layer, conv_layer)),
            ('global_pool', nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())),
            ('classifier', nn.Linear(layout[-1][-2], num_classes))]))

        init_module(self, 'leaky_relu')


def _darknet(arch, pretrained, progress, **kwargs):

    # Retrieve the correct Darknet layout type
    darknet_type = sys.modules[__name__].__dict__[default_cfgs[arch]['arch']]
    # Build the model
    model = darknet_type(default_cfgs[arch]['layout'], **kwargs)
    # Load pretrained parameters
    if pretrained:
        if default_cfgs[arch]['url'] is None:
            logging.warning(f"Invalid model URL for {arch}, using default initialization.")
        else:
            state_dict = load_state_dict_from_url(default_cfgs[arch]['url'],
                                                  progress=progress)
            model.load_state_dict(state_dict)

    return model


def darknet24(pretrained=False, progress=True, **kwargs):
    """Darknet-24 from
    `"You Only Look Once: Unified, Real-Time Object Detection" <https://pjreddie.com/media/files/papers/yolo_1.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _darknet('darknet24', pretrained, progress, **kwargs)


def darknet19(pretrained=False, progress=True, **kwargs):
    """Darknet-19 from
    `"YOLO9000: Better, Faster, Stronger" <https://pjreddie.com/media/files/papers/YOLO9000.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _darknet('darknet19', pretrained, progress, **kwargs)


def darknet53(pretrained=False, progress=True, **kwargs):
    """Darknet-53 from
    `"YOLOv3: An Incremental Improvement" <https://pjreddie.com/media/files/papers/YOLOv3.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _darknet('darknet53', pretrained, progress, **kwargs)
