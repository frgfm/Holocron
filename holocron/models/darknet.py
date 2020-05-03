# -*- coding: utf-8 -*-

"""
Implementation of DarkNet
"""

from collections import OrderedDict
import torch.nn as nn

__all__ = ['DarknetV1', 'DarknetV2', 'darknet24', 'darknet19']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DarkBlockV1(nn.Sequential):

    def __init__(self, planes):

        layers = []
        k1 = True
        for in_planes, out_planes in zip(planes[:-1], planes[1:]):
            layers.append(conv1x1(in_planes, out_planes) if k1 else conv3x3(in_planes, out_planes))
            layers.append(nn.LeakyReLU(inplace=True))
            k1 = not k1

        super(DarkBlockV1, self).__init__(*layers)


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


class DarknetV1(nn.Module):

    def __init__(self, layout, num_classes=10):

        super().__init__()

        self.features = DarknetBodyV1(layout)

        # Pooling (7, 7) or global ?
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(layout[2][-1], num_classes))

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)
        return x


class DarkBlockV2(nn.Sequential):

    def __init__(self, in_planes, out_planes, nb_compressions=0):

        layers = [conv3x3(in_planes, out_planes),
                  nn.BatchNorm2d(out_planes),
                  nn.LeakyReLU(0.1, inplace=True)]
        for _ in range(nb_compressions):
            layers.extend([conv1x1(out_planes, in_planes),
                           nn.BatchNorm2d(in_planes),
                           nn.LeakyReLU(0.1, inplace=True),
                           conv3x3(in_planes, out_planes),
                           nn.BatchNorm2d(out_planes),
                           nn.LeakyReLU(0.1, inplace=True)])

        super(DarkBlockV2, self).__init__(*layers)


class DarknetBodyV2(nn.Module):

    def __init__(self, layout, passthrough=False):

        super().__init__()

        self.conv1 = conv1x1(3, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = conv3x3(32, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = DarkBlockV2(64, *layout[0])
        self.block2 = DarkBlockV2(layout[0][0], *layout[1])
        self.block3 = DarkBlockV2(layout[1][0], *layout[2])
        self.block4 = DarkBlockV2(layout[2][0], *layout[3])
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


class DarknetV2(nn.Module):

    def __init__(self, layout, num_classes=10):

        super().__init__()

        self.features = DarknetBodyV2(layout)

        self.classifier = nn.Sequential(
            conv1x1(layout[-1][0], num_classes),
            nn.BatchNorm2d(num_classes),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


def darknet24(pretrained=False, progress=True, **kwargs):
    """Darknet-24 from
    `"You Only Look Once: Unified, Real-Time Object Detection" <https://pjreddie.com/media/files/papers/yolo_1.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return DarknetV1([[128, 256, 256, 512], [*([256, 512] * 4), 512, 1024], [512, 1024, 512, 1024]], **kwargs)


def darknet19(pretrained=False, progress=True):
    """Darknet-19 from
    `"YOLO9000: Better, Faster, Stronger" <https://pjreddie.com/media/files/papers/YOLO9000.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return DarknetV2([(128, 1), (256, 1), (512, 2), (1024, 2)], **kwargs)
