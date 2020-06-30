# -*- coding: utf-8 -*-

"""
Implementation of Res2Net
"""

import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class TridentBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, branches=3):
        super(TridentBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        if branches < 1:
            raise NotImplementedError("The number of branches needs to be superior or equal to 1")
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        # self.conv2 = TridentConv2d(width, width, stride, dilations=range(1, branches + 1))
        self.conv2 = conv3x3(width, width, stride, groups=groups)
        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.branches = branches

    def forward(self, x):
        if len(x) != self.branches:
            raise NotImplementedError(f"Expected the same number of inputs as branches. Received: {len(x)}")

        out = []
        for idx in range(self.branches):
            identity = x[idx]
            out.append(x[idx])
            # Shared conv1x1
            out[-1] = self.conv1(out[-1])
            out[-1] = self.bn1(out[-1])
            out[-1] = self.relu(out[-1])
            # Shared conv3x3 with different dilation rates
            out[-1] = F.conv2d(out[-1], weight=self.conv2.weight, bias=self.conv2.bias, stride=self.conv2.stride,
                               padding=idx + 1, dilation=idx + 1, groups=self.conv2.groups)
            # out[-1] = self.conv2(out[-1])
            out[-1] = self.bn2(out[-1])
            out[-1] = self.relu(out[-1])
            # Shared conv1x1
            out[-1] = self.conv3(out[-1])
            out[-1] = self.bn3(out[-1])

            if self.downsample is not None:
                identity = self.downsample(x[idx])

            out[-1] = self.relu(out[-1] + identity)

        return out
