# -*- coding: utf-8 -*-

"""
Implementation of DarkNet
"""

import sys
import logging
from collections import OrderedDict
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.resnet import conv1x1, conv3x3

from ..nn.init import init_module


__all__ = ['DarknetV1', 'DarknetV2', 'DarknetV3', 'darknet24', 'darknet19', 'darknet53']


default_cfgs = {
    'darknet24': {'arch': 'DarknetV1',
                  'layout': [[128, 256, 256, 512], [*([256, 512] * 4), 512, 1024], [512, 1024, 512, 1024]],
                  'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.0/darknet24_224-054704e0.pth'},
    'darknet19': {'arch': 'DarknetV2',
                  'layout': [(128, 1), (256, 1), (512, 2), (1024, 2)],
                  'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.0/darknet19_224-5aad1493.pth'},
    'darknet53': {'arch': 'DarknetV3',
                  'layout': [(1, 64, 128), (2, 128, 256), (8, 256, 512), (8, 512, 1024), (4, 1024)],
                  'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.0/darknet53_224-42576ca0.pth'},
}


class DarkBlockV1(nn.Sequential):
    def __init__(self, planes):

        layers = []
        k1 = True
        for in_planes, out_planes in zip(planes[:-1], planes[1:]):
            layers.append(conv1x1(in_planes, out_planes) if k1 else conv3x3(in_planes, out_planes))
            layers.append(nn.LeakyReLU(inplace=True))
            k1 = not k1

        super().__init__(*layers)


class DarknetBodyV1(nn.Module):
    def __init__(self, layout):

        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 7, padding=3, stride=2)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = conv3x3(64, 192)

        self.block1 = DarkBlockV1([192] + layout[0])
        self.block2 = DarkBlockV1(layout[0][-1:] + layout[1])
        self.block3 = DarkBlockV1(layout[1][-1:] + layout[2])

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.activation(self.conv2(x))
        x = self.pool(x)

        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)

        return x


class DarknetV1(nn.Sequential):
    def __init__(self, layout, num_classes=10, norm_layer=None):

        super().__init__(OrderedDict([
            ('features', DarknetBodyV1(layout)),
            ('global_pool', nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())),
            ('classifier', nn.Linear(layout[2][-1], num_classes))]))

        init_module(self, 'leaky_relu')


class DarkBlockV2(nn.Sequential):
    def __init__(self, in_planes, out_planes, nb_compressions=0, norm_layer=None):

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers = [conv3x3(in_planes, out_planes),
                  norm_layer(out_planes),
                  nn.LeakyReLU(0.1, inplace=True)]
        for _ in range(nb_compressions):
            layers.extend([conv1x1(out_planes, in_planes),
                           norm_layer(in_planes),
                           nn.LeakyReLU(0.1, inplace=True),
                           conv3x3(in_planes, out_planes),
                           norm_layer(out_planes),
                           nn.LeakyReLU(0.1, inplace=True)])

        super().__init__(*layers)


class DarknetBodyV2(nn.Module):
    def __init__(self, layout, passthrough=False, norm_layer=None):

        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv1x1(3, 32)
        self.bn1 = norm_layer(32)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = conv3x3(32, 64)
        self.bn2 = norm_layer(64)
        self.block1 = DarkBlockV2(64, *layout[0], norm_layer)
        self.block2 = DarkBlockV2(layout[0][0], *layout[1], norm_layer)
        self.block3 = DarkBlockV2(layout[1][0], *layout[2], norm_layer)
        self.block4 = DarkBlockV2(layout[2][0], *layout[3], norm_layer)
        self.passthrough = passthrough

    def forward(self, x):

        x = self.activation(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.block3(x)
        if self.passthrough:
            y = x
        x = self.block4(self.pool(x))

        if self.passthrough:
            return x, y
        else:
            return x


class DarknetV2(nn.Sequential):
    def __init__(self, layout, num_classes=10, norm_layer=None):

        super().__init__(OrderedDict([
            ('features', DarknetBodyV2(layout, norm_layer)),
            ('classifier', conv1x1(layout[-1][0], num_classes)),
            ('global_pool', nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()))]))

        init_module(self, 'leaky_relu')


class DarkBlockV3(nn.Module):
    def __init__(self, planes, mid_planes, norm_layer=None):

        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv1x1(planes, mid_planes)
        self.bn1 = norm_layer(mid_planes)
        self.conv2 = conv3x3(mid_planes, planes)
        self.bn2 = norm_layer(planes)

        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out += identity
        out = self.activation(out)

        return x


class DarknetBodyV3(nn.Module):

    def _make_layer(self, num_blocks, in_planes, out_planes=None):

        layers = [DarkBlockV3(in_planes, in_planes // 2, self._norm_layer) for _ in range(num_blocks)]
        if isinstance(out_planes, int):
            layers.extend([
                conv3x3(in_planes, out_planes, stride=2),
                self._norm_layer(out_planes),
                nn.LeakyReLU(0.1, inplace=True)
            ])

        return nn.Sequential(*layers)

    def __init__(self, layout, norm_layer=None):

        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.conv1 = conv3x3(3, 32)
        self.bn1 = self._norm_layer(32)
        self.conv2 = conv3x3(32, 64, stride=2)
        self.bn2 = self._norm_layer(64)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

        self.block1 = self._make_layer(*layout[0])
        self.block2 = self._make_layer(*layout[1])
        self.block3 = self._make_layer(*layout[2])
        self.block4 = self._make_layer(*layout[3])
        self.block5 = self._make_layer(*layout[4])

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        return x


class DarknetV3(nn.Sequential):
    def __init__(self, layout, num_classes=10, norm_layer=None):

        super().__init__(OrderedDict([
            ('features', DarknetBodyV3(layout, norm_layer)),
            ('global_pool', nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())),
            ('classifier', nn.Linear(layout[4][-1], num_classes))]))

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
