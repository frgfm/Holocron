# -*- coding: utf-8 -*-

"""
Implementation of DarkNet
"""

import sys
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

from ..nn.init import init_module
from .utils import conv_sequence
from .resnet import _ResBlock
from holocron.nn import Mish, DropBlock2d


__all__ = ['DarknetV1', 'DarknetV2', 'DarknetV3', 'DarknetV4', 'darknet24', 'darknet19', 'darknet53', 'cspdarknet53']


default_cfgs = {
    'darknet24': {'arch': 'DarknetV1',
                  'layout': [[192], [128, 256, 256, 512], [*([256, 512] * 4), 512, 1024], [512, 1024] * 2],
                  'url': None},
    'darknet19': {'arch': 'DarknetV2',
                  'layout': [(64, 0), (128, 1), (256, 1), (512, 2), (1024, 2)],
                  'url': None},
    'darknet53': {'arch': 'DarknetV3',
                  'layout': [(64, 1), (128, 2), (256, 8), (512, 8), (1024, 4)],
                  'url': None},
    'cspdarknet53': {'arch': 'DarknetV4',
                     'layout': [(64, 1), (128, 2), (256, 8), (512, 8), (1024, 4)],
                     'url': None},
}


class DarknetBodyV1(nn.Sequential):
    def __init__(self, layout, in_channels=3, stem_channels=64,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):

        if act_layer is None:
            act_layer = nn.LeakyReLU(0.1, inplace=True)

        in_chans = [stem_channels] + [_layout[-1] for _layout in layout[:-1]]

        super().__init__(OrderedDict([
            ('stem', nn.Sequential(*conv_sequence(in_channels, stem_channels,
                                                  act_layer, norm_layer, drop_layer, conv_layer,
                                                  kernel_size=7, padding=3, stride=2, bias=False))),
            ('layers', nn.Sequential(*[self._make_layer([_in_chans] + planes,
                                                        act_layer, norm_layer, drop_layer, conv_layer)
                                       for _in_chans, planes in zip(in_chans, layout)]))])
        )

    @staticmethod
    def _make_layer(planes, act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):
        _layers = [nn.MaxPool2d(2)]
        k1 = True
        for in_planes, out_planes in zip(planes[:-1], planes[1:]):
            _layers.extend(conv_sequence(in_planes, out_planes, act_layer, norm_layer, drop_layer, conv_layer,
                                         kernel_size=3 if out_planes > in_planes else 1,
                                         padding=1 if out_planes > in_planes else 0, bias=False))
            k1 = not k1

        return nn.Sequential(*_layers)


class DarknetV1(nn.Sequential):
    def __init__(self, layout, num_classes=10, in_channels=3, stem_channels=64,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):
        super().__init__(OrderedDict([
            ('features', DarknetBodyV1(layout, in_channels, stem_channels,
                                       act_layer, norm_layer, drop_layer, conv_layer)),
            ('global_pool', nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())),
            ('classifier', nn.Linear(layout[2][-1], num_classes))]))

        init_module(self, 'leaky_relu')


class DarknetBodyV2(nn.Sequential):

    def __init__(self, layout, in_channels=3, stem_channels=32, passthrough=False,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):

        if act_layer is None:
            act_layer = nn.LeakyReLU(0.1, inplace=True)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        in_chans = [stem_channels] + [_layout[0] for _layout in layout[:-1]]

        super().__init__(OrderedDict([
            ('stem', nn.Sequential(*conv_sequence(in_channels, stem_channels,
                                                  act_layer, norm_layer, drop_layer, conv_layer,
                                                  kernel_size=3, padding=1, bias=False))),
            ('layers', nn.Sequential(*[self._make_layer(num_blocks, _in_chans, out_chans,
                                                        act_layer, norm_layer, drop_layer, conv_layer)
                                       for _in_chans, (out_chans, num_blocks) in zip(in_chans, layout)]))])
        )

        self.passthrough = passthrough

    @staticmethod
    def _make_layer(num_blocks, in_planes, out_planes,
                    act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):
        layers = [nn.MaxPool2d(2)]
        layers.extend(conv_sequence(in_planes, out_planes, act_layer, norm_layer, drop_layer, conv_layer,
                                    kernel_size=3, padding=1, stride=1, bias=False))
        for _ in range(num_blocks):
            layers.extend(conv_sequence(out_planes, out_planes // 2, act_layer, norm_layer, drop_layer, conv_layer,
                                        kernel_size=1, padding=0, stride=1, bias=False) +
                          conv_sequence(out_planes // 2, out_planes, act_layer, norm_layer, drop_layer, conv_layer,
                                        kernel_size=3, padding=1, stride=1, bias=False))

        return nn.Sequential(*layers)

    def forward(self, x):

        if self.passthrough:
            x = self.stem(x)
            for idx, layer in enumerate(self.layers):
                x = layer(x)
                if idx == len(self.layers) - 2:
                    aux = x.clone()

            return x, aux
        else:
            return super().forward(x)


class DarknetV2(nn.Sequential):
    def __init__(self, layout, num_classes=10, in_channels=3, stem_channels=32,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):

        super().__init__(OrderedDict([
            ('features', DarknetBodyV2(layout, in_channels, stem_channels, False,
                                       act_layer, norm_layer, drop_layer, conv_layer)),
            ('classifier', nn.Conv2d(layout[-1][0], num_classes, 1)),
            ('global_pool', nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()))]))

        init_module(self, 'leaky_relu')


class ResBlock(_ResBlock):

    def __init__(self, planes, mid_planes, act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):
        super().__init__(
            conv_sequence(planes, mid_planes, act_layer, norm_layer, drop_layer, conv_layer,
                          kernel_size=1, bias=False) +
            conv_sequence(mid_planes, planes, act_layer, norm_layer, drop_layer, conv_layer,
                          kernel_size=3, padding=1, bias=False),
            None, act_layer)
        if drop_layer is not None:
            self.dropblock = DropBlock2d(0.1, 7, inplace=True)

        # The backpropagation does not seem to appreciate inplace activation on the residual branch
        if hasattr(self.conv[-1], 'inplace'):
            self.conv[-1].inplace = False

    def forward(self, x):
        out = super().forward(x)
        if hasattr(self, 'dropblock'):
            out = self.dropblock(out)

        return out


class DarknetBodyV3(nn.Sequential):

    def __init__(self, layout, in_channels=3, stem_channels=32,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):

        if act_layer is None:
            act_layer = nn.LeakyReLU(0.1, inplace=True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        in_chans = [stem_channels] + [_layout[0] for _layout in layout[:-1]]

        super().__init__(OrderedDict([
            ('stem', nn.Sequential(*conv_sequence(in_channels, stem_channels,
                                                  act_layer, norm_layer, drop_layer, conv_layer,
                                                  kernel_size=3, padding=1, bias=False))),
            ('layers', nn.Sequential(*[self._make_layer(num_blocks, _in_chans, out_chans,
                                                        act_layer, norm_layer, drop_layer, conv_layer)
                                       for _in_chans, (out_chans, num_blocks) in zip(in_chans, layout)]))])
        )

    @staticmethod
    def _make_layer(num_blocks, in_planes, out_planes,
                    act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):

        layers = conv_sequence(in_planes, out_planes, act_layer, norm_layer, drop_layer, conv_layer,
                               kernel_size=3, padding=1, stride=2, bias=False)
        layers.extend([ResBlock(out_planes, out_planes // 2, act_layer, norm_layer, drop_layer, conv_layer)
                       for _ in range(num_blocks)])

        return nn.Sequential(*layers)


class DarknetV3(nn.Sequential):
    def __init__(self, layout, num_classes=10, in_channels=3, stem_channels=32,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):

        super().__init__(OrderedDict([
            ('features', DarknetBodyV3(layout, in_channels, stem_channels,
                                       act_layer, norm_layer, drop_layer, conv_layer)),
            ('global_pool', nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())),
            ('classifier', nn.Linear(layout[-1][0], num_classes))]))

        init_module(self, 'leaky_relu')


class CSPStage(nn.Module):

    def __init__(self, in_channels, out_channels, num_blocks=1,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):
        super().__init__()
        self.base_layer = nn.Sequential(*conv_sequence(in_channels, out_channels,
                                                       act_layer, norm_layer, drop_layer, conv_layer,
                                                       kernel_size=3, padding=1, stride=2, bias=False))
        compression = 2 if num_blocks > 1 else 1
        self.part1 = nn.Sequential(*conv_sequence(out_channels, out_channels // compression,
                                                  act_layer, norm_layer, drop_layer, conv_layer,
                                                  kernel_size=1, bias=False))
        self.part2 = nn.Sequential(*conv_sequence(out_channels, out_channels // compression,
                                                  act_layer, norm_layer, drop_layer, conv_layer,
                                                  kernel_size=1, bias=False),
                                   *[ResBlock(out_channels // compression,
                                              out_channels // compression if num_blocks > 1 else in_channels,
                                              act_layer, norm_layer, drop_layer, conv_layer)
                                     for _ in range(num_blocks)],
                                   *conv_sequence(out_channels // compression, out_channels // compression,
                                                  act_layer, norm_layer, drop_layer, conv_layer,
                                                  kernel_size=1, bias=False))
        self.transition = nn.Sequential(*conv_sequence(2 * out_channels // compression, out_channels,
                                                       act_layer, norm_layer, drop_layer, conv_layer,
                                                       kernel_size=1, bias=False))

    def forward(self, x):
        x = self.base_layer(x)
        p1 = self.part1(x)
        p2 = self.part2(x)

        return self.transition(torch.cat([p1, p2], dim=1))


class DarknetBodyV4(nn.Sequential):

    def __init__(self, layout, in_channels=3, stem_channels=32, num_features=1,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):

        super().__init__()

        if act_layer is None:
            act_layer = Mish()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if drop_layer is None:
            drop_layer = DropBlock2d

        in_chans = [stem_channels] + [_layout[0] for _layout in layout[:-1]]

        super().__init__(OrderedDict([
            ('stem', nn.Sequential(*conv_sequence(in_channels, stem_channels,
                                                  act_layer, norm_layer, drop_layer, conv_layer,
                                                  kernel_size=3, padding=1, bias=False))),
            ('layers', nn.Sequential(*[CSPStage(_in_chans, out_chans, num_blocks,
                                                act_layer, norm_layer, drop_layer, conv_layer)
                                       for _in_chans, (out_chans, num_blocks) in zip(in_chans, layout)]))])
        )

        self.num_features = num_features

    def forward(self, x):

        if self.num_features == 1:
            return super().forward(x)
        else:
            x = self.stem(x)
            features = []
            for idx, stage in enumerate(self.layers):
                x = stage(x)
                if idx >= (len(self.layers) - self.num_features):
                    features.append(x)

            return features


class DarknetV4(nn.Sequential):
    def __init__(self, layout, num_classes=10, in_channels=3, stem_channels=32, num_features=1,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):

        super().__init__(OrderedDict([
            ('features', DarknetBodyV4(layout, in_channels, stem_channels, num_features,
                                       act_layer, norm_layer, drop_layer, conv_layer)),
            ('global_pool', nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())),
            ('classifier', nn.Linear(layout[-1][0], num_classes))]))

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


def cspdarknet53(pretrained=False, progress=True, **kwargs):
    """CSP-Darknet-53 from
    `"CSPNet: A New Backbone that can Enhance Learning Capability of CNN" <https://arxiv.org/pdf/1911.11929.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _darknet('cspdarknet53', pretrained, progress, **kwargs)
