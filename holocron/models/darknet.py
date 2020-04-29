# -*- coding: utf-8 -*-

"""
Implementation of DarkNet
"""

from collections import OrderedDict
import torch.nn as nn

__all__ = ['Darknet', 'darknet19']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DarkBlock(nn.Sequential):

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

        super(DarkBlock, self).__init__(*layers)


class DarknetBody(nn.Module):

    def __init__(self, layout):

        super().__init__()

        self.conv1 = conv1x1(3, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = conv3x3(32, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = DarkBlock(64, *layout[0])
        self.block2 = DarkBlock(layout[0][0], *layout[1])
        self.block3 = DarkBlock(layout[1][0], *layout[2])
        self.block4 = DarkBlock(layout[2][0], *layout[3])

    def forward(self, x):

        x = self.activation(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        x = self.block4(x)

        return x


class Darknet(nn.Module):

    def __init__(self, layout, num_classes):

        super().__init__()

        self.features = DarknetBody(layout)

        self.classifier = nn.Sequential(
            conv1x1(layout[3][0], num_classes),
            nn.BatchNorm2d(num_classes),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


def darknet19(num_classes):
    """Implements Darknet19 as described in https://pjreddie.com/media/files/papers/YOLO9000.pdf

    Args:
        num_classes (int, optional): number of output classes

    Returns:
        torch.nn.Module: classification model
    """

    return Darknet([(128, 1), (256, 1), (512, 2), (1024, 2)], num_classes)
