# -*- coding: utf-8 -*-

"""
Personal implementation of UNet models
"""

import sys
import logging
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

from ...nn.init import init_module


__all__ = ['UNet', 'unet', 'UNetp', 'unetp', 'UNetpp', 'unetpp', 'UNet3p', 'unet3p']


default_cfgs = {
    'unet': {'arch': 'UNet',
             'layout': [64, 128, 256, 512, 1024],
             'url': None},
    'unetp': {'arch': 'UNetp',
              'layout': [64, 128, 256, 512, 1024],
              'url': None},
    'unetpp': {'arch': 'UNetpp',
               'layout': [64, 128, 256, 512, 1024],
               'url': None},
    'unet3p': {'arch': 'UNet3p',
               'layout': [64, 128, 256, 512, 1024],
               'url': None}
}


def conv1x1(in_chan, out_chan):
    return nn.Conv2d(in_chan, out_chan, 1)


def conv3x3(in_chan, out_chan, padding=0):
    return nn.Conv2d(in_chan, out_chan, 3, padding=padding)


def conv_bn_act(in_chan, out_chan, kernel_size, padding=0, bn=False, act=True):
    layers = [nn.Conv2d(in_chan, out_chan, kernel_size, padding=padding)]
    if bn:
        layers.append(nn.BatchNorm2d(out_chan))
    if act:
        layers.append(nn.ReLU(inplace=True))

    return layers


class DownPath(nn.Sequential):
    def __init__(self, in_chan, out_chan, downsample=True, padding=0, bn=False):
        layers = [nn.MaxPool2d(2)] if downsample else []
        layers.extend([*conv_bn_act(in_chan, out_chan, 3, padding, bn),
                       *conv_bn_act(out_chan, out_chan, 3, padding, bn)])
        super().__init__(*layers)


class UpPath(nn.Module):
    def __init__(self, in_chan, out_chan, num_skips=1, conv_transpose=False, padding=0, bn=False):
        super().__init__()

        if conv_transpose:
            self.upsample = nn.ConvTranspose2d(in_chan, in_chan // 2, 2, stride=2)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Estimate the number of channels in the upsampled feature map
        up_chan = in_chan // 2 if conv_transpose else in_chan
        self.block = nn.Sequential(*conv_bn_act(num_skips * in_chan // 2 + up_chan, out_chan, 3, padding, bn),
                                   *conv_bn_act(out_chan, out_chan, 3, padding, bn))
        self.num_skips = num_skips

    def forward(self, downfeats, upfeat):

        if not isinstance(downfeats, list):
            downfeats = [downfeats]
        if len(downfeats) != self.num_skips:
            raise ValueError
        # Upsample expansive features
        _upfeat = self.upsample(upfeat)
        # Crop contracting path features
        for idx, downfeat in enumerate(downfeats):
            delta_w = downfeat.shape[-1] - _upfeat.shape[-1]
            delta_h = downfeat.shape[-2] - _upfeat.shape[-2]
            downfeats[idx] = downfeat[..., delta_h // 2:-delta_h // 2, delta_w // 2:-delta_w // 2]
        # Concatenate both feature maps and forward them
        return self.block(torch.cat((*downfeats, _upfeat), dim=1))


class UNet(nn.Module):
    """Implements a U-Net architecture

    Args:
        layout (list<int>): number of channels after each contracting block
        in_channels (int, optional): number of channels in the input tensor
        num_classes (int, optional): number of output classes
    """
    def __init__(self, layout, in_channels=1, num_classes=10):
        super().__init__()

        # Contracting path
        _layout = [in_channels] + layout
        _pool = False
        for num, in_chan, out_chan in zip(range(1, len(_layout)), _layout[:-1], _layout[1:]):
            self.add_module(f"down{num}", DownPath(in_chan, out_chan, _pool))
            _pool = True

        # Expansive path
        _layout = layout[::-1]
        for num, in_chan, out_chan in zip(range(len(layout) - 1, 0, -1), _layout[:-1], _layout[1:]):
            self.add_module(f"up{num}", UpPath(in_chan, out_chan))

        # Classifier
        self.classifier = conv1x1(64, num_classes)

        init_module(self, 'relu')

    def forward(self, x):

        # Contracting path
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.down5(x4)

        # Expansive path
        x = self.up4(x4, x)
        x = self.up3(x3, x)
        x = self.up2(x2, x)
        x = self.up1(x1, x)

        # Classifier
        x = self.classifier(x)
        return x


class UNetp(nn.Module):
    """Implements a UNet+ architecture

    Args:
        layout (list<int>): number of channels after each contracting block
        in_channels (int, optional): number of channels in the input tensor
        num_classes (int, optional): number of output classes
    """
    def __init__(self, layout, in_channels=1, num_classes=10):
        super().__init__()

        # Contracting path
        _layout = [in_channels] + layout
        _pool = False
        for num, in_chan, out_chan in zip(range(1, len(_layout)), _layout[:-1], _layout[1:]):
            self.add_module(f"down{num}", DownPath(in_chan, out_chan, _pool))
            _pool = True

        # Expansive path
        _layout = layout[::-1]
        for row, in_chan, out_chan, cols in zip(range(len(layout) - 1, 0, -1), _layout[:-1], _layout[1:],
                                                range(1, len(layout))):
            for col in range(1, cols + 1):
                self.add_module(f"up{row}{col}", UpPath(in_chan, out_chan))

        # Classifier
        self.classifier = conv1x1(64, num_classes)

        init_module(self, 'relu')

    def forward(self, x):

        # Contracting path
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.down5(x4)

        # Nested Expansive path
        x1 = self.up11(x1, x2)
        x2 = self.up21(x2, x3)
        x3 = self.up31(x3, x4)
        x = self.up41(x4, x)

        x1 = self.up12(x1, x2)
        x2 = self.up22(x2, x3)
        x = self.up32(x3, x)

        x1 = self.up13(x1, x2)
        x = self.up23(x2, x)

        x = self.up14(x1, x)

        # Classifier
        x = self.classifier(x)
        return x


class UNetpp(nn.Module):
    """Implements a UNet++ architecture

    Args:
        layout (list<int>): number of channels after each contracting block
        in_channels (int, optional): number of channels in the input tensor
        num_classes (int, optional): number of output classes
    """
    def __init__(self, layout, in_channels=1, num_classes=10):
        super().__init__()

        # Contracting path
        _layout = [in_channels] + layout
        _pool = False
        for num, in_chan, out_chan in zip(range(1, len(_layout)), _layout[:-1], _layout[1:]):
            self.add_module(f"down{num}", DownPath(in_chan, out_chan, _pool))
            _pool = True

        # Expansive path
        _layout = layout[::-1]
        for row, in_chan, out_chan, cols in zip(range(len(layout) - 1, 0, -1), _layout[:-1], _layout[1:],
                                                range(1, len(layout))):
            for col in range(1, cols + 1):
                self.add_module(f"up{row}{col}", UpPath(in_chan, out_chan, num_skips=col))

        # Classifier
        self.classifier = conv1x1(64, num_classes)

        init_module(self, 'relu')

    def forward(self, x):

        # Contracting path
        x10 = self.down1(x)
        x20 = self.down2(x10)
        x30 = self.down3(x20)
        x40 = self.down4(x30)
        x = self.down5(x40)

        # Nested Expansive path
        x11 = self.up11(x10, x20)
        x21 = self.up21(x20, x30)
        x31 = self.up31(x30, x40)
        x = self.up41(x40, x)

        x12 = self.up12([x10, x11], x21)
        x22 = self.up22([x20, x21], x31)
        x = self.up32([x30, x31], x)

        x13 = self.up13([x10, x11, x12], x22)
        x = self.up23([x20, x21, x22], x)

        x = self.up14([x10, x11, x12, x13], x)

        # Classifier
        x = self.classifier(x)
        return x


class FSAggreg(nn.Module):
    def __init__(self, e_chans, skip_chan, d_chans):
        super().__init__()
        # Check stem conv channels
        base_chan = e_chans[0] if len(e_chans) > 0 else skip_chan
        # Get UNet depth
        depth = len(e_chans) + 1 + len(d_chans)
        # Downsample = max pooling + conv for channel reduction
        self.downsamples = nn.ModuleList([nn.Sequential(nn.MaxPool2d(2 ** (len(e_chans) - idx)),
                                                        conv3x3(e_chan, base_chan, 1))
                                          for idx, e_chan in enumerate(e_chans)])
        self.skip = conv3x3(skip_chan, base_chan, 1) if len(e_chans) > 0 else nn.Identity()
        # Upsample = bilinear interpolation + conv for channel reduction
        self.upsamples = nn.ModuleList([nn.Sequential(nn.Upsample(scale_factor=2 ** (idx + 1),
                                                                  mode='bilinear', align_corners=True),
                                                      conv3x3(d_chan, base_chan, 1))
                                        for idx, d_chan in enumerate(d_chans)])

        self.block = nn.Sequential(*conv_bn_act(depth * base_chan, depth * base_chan, 3, 1, True))

    def forward(self, downfeats, feat, upfeats):

        if len(downfeats) != len(self.downsamples) or len(upfeats) != len(self.upsamples):
            raise ValueError

        # Concatenate full-scale features
        x = torch.cat((*[downsample(downfeat) for downsample, downfeat in zip(self.downsamples, downfeats)],
                       self.skip(feat),
                       *[upsample(upfeat) for upsample, upfeat in zip(self.upsamples, upfeats)]), dim=1)

        return self.block(x)


class UNet3p(nn.Module):
    """Implements a UNet3+ architecture

    Args:
        layout (list<int>): number of channels after each contracting block
        in_channels (int, optional): number of channels in the input tensor
        num_classes (int, optional): number of output classes
    """
    def __init__(self, layout, in_channels=1, num_classes=10):
        super().__init__()

        # Contracting path
        _layout = [in_channels] + layout
        _pool = False
        for num, in_chan, out_chan in zip(range(1, len(_layout)), _layout[:-1], _layout[1:]):
            self.add_module(f"down{num}", DownPath(in_chan, out_chan, _pool, 1, True))
            _pool = True

        # Expansive path
        for row in range(len(layout) - 1, 0, -1):
            self.add_module(f"up{row}", FSAggreg(layout[:row - 1], layout[row - 1], [320] * (4 - row) + layout[-1:]))

        # Classifier
        self.classifier = conv1x1(320, num_classes)

        init_module(self, 'relu')

    def forward(self, x):

        # Contracting path
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        # Full-scale Expansive path
        x4 = self.up4([x1, x2, x3], x4, [x5])
        x3 = self.up3([x1, x2], x3, [x4, x5])
        x2 = self.up2([x1], x2, [x3, x4, x5])
        x1 = self.up1([], x1, [x2, x3, x4, x5])

        # Classifier
        x = self.classifier(x1)
        return x


def _unet(arch, pretrained, progress, **kwargs):
    # Retrieve the correct Darknet layout type
    unet_type = sys.modules[__name__].__dict__[default_cfgs[arch]['arch']]
    # Build the model
    model = unet_type(default_cfgs[arch]['layout'], **kwargs)
    # Load pretrained parameters
    if pretrained:
        if default_cfgs[arch]['url'] is None:
            logging.warning(f"Invalid model URL for {arch}, using default initialization.")
        else:
            state_dict = load_state_dict_from_url(default_cfgs[arch]['url'],
                                                  progress=progress)
            model.load_state_dict(state_dict)

    return model


def unet(pretrained=False, progress=True, **kwargs):
    """U-Net from
    `"U-Net: Convolutional Networks for Biomedical Image Segmentation" <https://arxiv.org/pdf/1505.04597.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: semantic segmentation model
    """

    return _unet('unet', pretrained, progress, **kwargs)


def unetp(pretrained=False, progress=True, **kwargs):
    """UNet+ from
    `"UNet++: A Nested U-Net Architecture for Medical Image Segmentation" <https://arxiv.org/pdf/1807.10165.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: semantic segmentation model
    """

    return _unet('unetp', pretrained, progress, **kwargs)


def unetpp(pretrained=False, progress=True, **kwargs):
    """UNet++ from
    `"UNet++: A Nested U-Net Architecture for Medical Image Segmentation" <https://arxiv.org/pdf/1807.10165.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: semantic segmentation model
    """

    return _unet('unetpp', pretrained, progress, **kwargs)


def unet3p(pretrained=False, progress=True, **kwargs):
    """UNet3+ from
    `"UNet 3+: A Full-Scale Connected UNet For Medical Image Segmentation" <https://arxiv.org/pdf/2004.08790.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: semantic segmentation model
    """

    return _unet('unet3p', pretrained, progress, **kwargs)
