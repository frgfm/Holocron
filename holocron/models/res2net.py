# -*- coding: utf-8 -*-

"""
Implementation of Res2Net
based on https://github.com/gasvn/Res2Net
"""

import warnings
import torch
import torch.nn as nn
from torchvision.models.resnet import conv1x1, conv3x3
from torchvision.models.utils import load_state_dict_from_url


RESNET_LAYERS = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3],
                 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}

RES2NEXT_PARAMS = {50: dict(groups=8, width_per_group=4),
                   101: dict(groups=8, width_per_group=8)}

URLS = {
    'res2net50_48w_2s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_48w_2s-afed724a.pth',
    'res2net50_26w_4s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_26w_4s-06e79181.pth',
    'res2net50_14w_8s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_14w_8s-6527dddc.pth',
    'res2net50_26w_6s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_26w_6s-19041792.pth',
    'res2net50_26w_8s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_26w_8s-2c7c9f12.pth',
    'res2net101_26w_4s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net101_26w_4s-02a759a1.pth',
    'res2next50_4w_4s_8c': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2next50_4s-6ef7e7bf.pth'
}


class Res2Block(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=4, dilation=1, scale=4, first_block=False, norm_layer=None):
        """Implements a residual block
        Args:
            inplanes (int): input channel dimensionality
            planes (int): output channel dimensionality
            stride (int): stride used for conv3x3
            downsample (torch.nn.Module): module used for downsampling
            groups: num of convolution groups
            base_width: base width
            dilation (int): dilation rate of conv3x3
            scale (int): scaling ratio for cascade convs
            first_block (bool): whether the block is the first to be placed in the conv layer
            norm_layer (torch.nn.Module): norm layer to be used in blocks
        """
        super(Res2Block, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width * scale)
        self.bn1 = norm_layer(width * scale)

        # If scale == 1, single conv else identity & (scale - 1) convs
        nb_branches = max(scale, 2) - 1
        if first_block:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        self.convs = nn.ModuleList([conv3x3(width, width, stride, groups, dilation)
                                    for _ in range(nb_branches)])
        self.bns = nn.ModuleList([norm_layer(width) for _ in range(nb_branches)])
        self.first_block = first_block
        self.scale = scale

        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Chunk the feature map
        xs = torch.chunk(out, self.scale, dim=1)
        # Initialize output as empty tensor for proper concatenation
        y = 0
        for idx, conv in enumerate(self.convs):
            # Add previous y-value
            if self.first_block:
                y = xs[idx]
            else:
                y += xs[idx]
            y = conv(y)
            y = self.relu(self.bns[idx](y))
            # Concatenate with previously computed values
            out = torch.cat((out, y), 1) if idx > 0 else y
        # Use last chunk as x1
        if self.scale > 1:
            if self.first_block:
                out = torch.cat((out, self.pool(xs[len(self.convs)])), 1)
            else:
                out = torch.cat((out, xs[len(self.convs)]), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):
    """Implements a Res2Net model as described in https://arxiv.org/pdf/1904.01169.pdf

    Args:
        block (torch.nn.Module): class constructor to be used for residual blocks
        layers (list<int>): layout of layers
        num_classes (int): number of output classes
        zero_init_residual (bool): whether the residual connections should be initialized at zero
        groups (int): number of convolution groups
        width_per_group (int): number of channels per group
        scale (int): scaling ratio within blocks
        replace_stride_with_dilation (list<bool>): whether stride should be traded for dilation
        norm_layer (torch.nn.Module): norm layer to be used
    """

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=26, scale=4, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(Res2Net, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.scale = scale
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottle2neck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, self.scale, first_block=True, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                scale=self.scale, first_block=False, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def res2net(depth, num_classes, width_per_group=26, scale=4, pretrained=False, progress=True, **kwargs):
    """Instantiate a Res2Net model

    Args:
        depth (int): depth of the model
        num_classes (int): number of output classes
        scale (int): number of branches for cascade convolutions
        pretrained (bool): whether the model should load pretrained weights (ImageNet training)
        progress (bool): whether a progress bar should be displayed while downloading pretrained weights
        **kwargs: optional arguments of torchvision.models.resnet.ResNet

    Returns:
        model (torch.nn.Module): loaded Pytorch model
    """

    if RESNET_LAYERS.get(depth) is None:
        raise NotImplementedError(f"This specific architecture is not defined for that depth: {depth}")

    block = Res2Block if depth >= 50 else BasicBlock

    model = Res2Net(block, RESNET_LAYERS.get(depth), num_classes=num_classes, scale=scale, **kwargs)

    if pretrained:
        state_dict = None
        try:
            state_dict = load_state_dict_from_url(URLS.get(f"res2net{depth}_{width_per_group}w_{scale}s"),
                                                  map_location=torch.device('cpu'),
                                                  progress=progress)
        except Exception as e:
            warnings.warn(f"While downloading state_dict, received:\n{e}\nSkipping weight loading...")
        if isinstance(state_dict, dict):
            # Remove FC params from dict
            for key in ('fc.weight', 'fc.bias'):
                state_dict.pop(key, None)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            if any(unexpected) or any(not elt.startswith('fc.') for elt in missing):
                raise KeyError(f"Weight loading failed.\nMissing parameters: {missing}\n"
                               f"Unexpected parameters: {unexpected}")

    return model


def res2next(depth, num_classes, width_per_group=4, scale=4, pretrained=False, progress=True, **kwargs):
    """Instantiate a Res2NeXt model

    Args:
        depth (int): depth of the model
        num_classes (int): number of output classes
        scale (int): number of branches for cascade convolutions
        pretrained (bool): whether the model should load pretrained weights (ImageNet training)
        progress (bool): whether a progress bar should be displayed while downloading pretrained weights
        **kwargs: optional arguments of torchvision.models.resnet.ResNet

    Returns:
        model (torch.nn.Module): loaded Pytorch model
    """

    if RESNET_LAYERS.get(depth) is None:
        raise NotImplementedError(f"This specific architecture is not defined for that depth: {depth}")

    block = Res2Block if depth >= 50 else BasicBlock

    kwargs.update(RES2NEXT_PARAMS.get(depth))
    model = Res2Net(block, RESNET_LAYERS.get(depth), num_classes=num_classes, scale=scale, **kwargs)

    if pretrained:
        state_dict = None
        try:
            model_name = f"res2next{depth}_{width_per_group}w_{scale}s_{kwargs['groups']}c"
            state_dict = load_state_dict_from_url(URLS.get(model_name),
                                                  map_location=torch.device('cpu'),
                                                  progress=progress)
        except Exception as e:
            warnings.warn(f"While downloading state_dict, received:\n{e}\nSkipping weight loading...")
        if isinstance(state_dict, dict):
            # Remove FC params from dict
            for key in ('fc.weight', 'fc.bias'):
                state_dict.pop(key, None)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            if any(unexpected) or any(not elt.startswith('fc.') for elt in missing):
                raise KeyError(f"Weight loading failed.\nMissing parameters: {missing}\n"
                               f"Unexpected parameters: {unexpected}")

    return model
