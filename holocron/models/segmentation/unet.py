# -*- coding: utf-8 -*-

"""
Personal implementation of UNet models
"""

import sys
import logging
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url


__all__ = ['UNet', 'unet']


default_cfgs = {
    'unet': {'arch': 'UNet',
             'layout': [64, 128, 256, 512, 1024],
             'url': None}
}


def conv1x1(in_chan, out_chan):

    return nn.Conv2d(in_chan, out_chan, 1)


def conv3x3(in_chan, out_chan, padding=0):

    return nn.Conv2d(in_chan, out_chan, 3, padding=padding)


class DownLayer(nn.Sequential):
    def __init__(self, in_chan, out_chan, pool=True):
        layers = [nn.MaxPool2d(2)] if pool else []
        layers.extend([conv3x3(in_chan, out_chan), nn.ReLU(inplace=True),
                       conv3x3(out_chan, out_chan), nn.ReLU(inplace=True)])
        super().__init__(*layers)


class UpLayer(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(in_chan, out_chan, 2, stride=2)
        self.block = nn.Sequential(conv3x3(in_chan, out_chan), nn.ReLU(inplace=True),
                                   conv3x3(out_chan, out_chan), nn.ReLU(inplace=True))

    def forward(self, upfeat, downfeat):
        # Upsample expansive features
        upfeat = self.upconv(upfeat)
        # Crop contracting path features
        delta_w = downfeat.shape[-1] - upfeat.shape[-1]
        delta_h = downfeat.shape[-2] - upfeat.shape[-2]
        downfeat = downfeat[..., delta_h // 2:-delta_h // 2, delta_w // 2:-delta_w // 2]
        # Concatenate both feature maps and forward them
        upfeat = self.block(torch.cat((downfeat, upfeat), dim=1))

        return upfeat


class UNet(nn.Module):
    def __init__(self, layout, in_channels=1, num_classes=10):
        super().__init__()

        # Contracting path
        _layout = [in_channels] + layout
        _pool = False
        for num, in_chan, out_chan in zip(range(1, len(_layout)), _layout[:-1], _layout[1:]):
            self.add_module(f"down{num}", DownLayer(in_chan, out_chan, _pool))
            _pool = True

        # Expansive path
        _layout = layout[::-1]
        for num, in_chan, out_chan in zip(range(len(layout) - 1, 0, -1), _layout[:-1], _layout[1:]):
            self.add_module(f"up{num}", UpLayer(in_chan, out_chan))

        # Classifier
        self.classifier = conv1x1(64, num_classes)

    def forward(self, x):

        # Contracting path
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.down5(x4)

        # Expansive path
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        # Classifier
        x = self.classifier(x)
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
