# Copyright (C) 2019-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import sys
import torch
from torch import Tensor
import torch.nn as nn
from typing import Dict, Any, Union, Optional, Callable, List

from ...nn.init import init_module
from ..utils import conv_sequence, load_pretrained_params


__all__ = ['UNet', 'unet']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'unet': {'arch': 'UNet',
             'layout': [64, 128, 256, 512, 1024],
             'url': None},
}


def down_path(
    in_chan: int,
    out_chan: int,
    downsample: bool = True,
    padding: int = 0,
    act_layer: Optional[nn.Module] = None,
    norm_layer: Optional[Callable[[int], nn.Module]] = None,
    drop_layer: Optional[Callable[..., nn.Module]] = None,
    conv_layer: Optional[Callable[..., nn.Module]] = None
) -> nn.Sequential:

    layers: List[nn.Module] = [nn.MaxPool2d(2)] if downsample else []
    layers.extend([*conv_sequence(in_chan, out_chan, act_layer, norm_layer, drop_layer, conv_layer,
                                  kernel_size=3, padding=padding),
                   *conv_sequence(out_chan, out_chan, act_layer, norm_layer, drop_layer, conv_layer,
                                  kernel_size=3, padding=padding)])
    return nn.Sequential(*layers)


class UpPath(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        bilinear_upsampling: bool = True,
        padding: int = 0,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()

        self.upsample: nn.Module
        if bilinear_upsampling:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.ConvTranspose2d(in_chan, in_chan // 2, 2, stride=2)

        self.block = nn.Sequential(*conv_sequence(in_chan, out_chan,
                                                  act_layer, norm_layer, drop_layer, conv_layer,
                                                  kernel_size=3, padding=padding),
                                   *conv_sequence(out_chan, out_chan,
                                                  act_layer, norm_layer, drop_layer, conv_layer,
                                                  kernel_size=3, padding=padding))

    def forward(self, downfeats: Union[Tensor, List[Tensor]], upfeat: Tensor) -> Tensor:

        if not isinstance(downfeats, list):
            downfeats = [downfeats]
        # Upsample expansive features
        _upfeat = self.upsample(upfeat)
        # Crop contracting path features
        for idx, downfeat in enumerate(downfeats):
            if downfeat.shape != _upfeat.shape:
                delta_w = downfeat.shape[-1] - _upfeat.shape[-1]
                w_slice = slice(delta_w // 2, -delta_w // 2 if delta_w > 0 else downfeat.shape[-1])
                delta_h = downfeat.shape[-2] - _upfeat.shape[-2]
                h_slice = slice(delta_h // 2, -delta_h // 2 if delta_h > 0 else downfeat.shape[-2])
                downfeats[idx] = downfeat[..., h_slice, w_slice]
        # Concatenate both feature maps and forward them
        return self.block(torch.cat((*downfeats, _upfeat), dim=1))


class UNet(nn.Module):
    """Implements a U-Net architecture

    Args:
        layout: number of channels after each contracting block
        in_channels: number of channels in the input tensor
        num_classes: number of output classes
        act_layer: activation layer
        norm_layer: normalization layer
        drop_layer: dropout layer
        conv_layer: convolutional layer
        same_padding: enforces same padding in convolutions
        bilinear_upsampling: replaces transposed conv by bilinear interpolation for upsampling
    """
    def __init__(
        self,
        layout: List[int],
        in_channels: int = 3,
        num_classes: int = 10,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
        same_padding: bool = True,
        bilinear_upsampling: bool = True,
    ) -> None:
        super().__init__()

        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        # Contracting path
        self.encoders = nn.ModuleList([])
        _layout = [in_channels] + layout
        if bilinear_upsampling:
            _layout[-1] = _layout[-1] // 2
        _pool = False
        for in_chan, out_chan in zip(_layout[:-1], _layout[1:]):
            self.encoders.append(down_path(in_chan, out_chan, _pool, int(same_padding),
                                           act_layer, norm_layer, drop_layer, conv_layer))
            _pool = True

        # Expansive path
        self.decoders = nn.ModuleList([])
        _layout = [chan // 2 if bilinear_upsampling else chan for chan in layout[::-1][1:-1]] + [layout[0]]
        for in_chan, out_chan in zip(layout[::-1][:-1], _layout):
            self.decoders.append(UpPath(in_chan, out_chan, bilinear_upsampling, int(same_padding),
                                        act_layer, norm_layer, drop_layer, conv_layer))

        # Classifier
        self.classifier = nn.Conv2d(layout[0], num_classes, 1)

        init_module(self, 'relu')

    def forward(self, x: Tensor) -> Tensor:

        xs: List[Tensor] = []
        # Contracting path
        for encoder in self.encoders[:-1]:
            xs.append(encoder(xs[-1] if len(xs) > 0 else x))
        x = self.encoders[-1](xs[-1])

        # Expansive path
        for decoder in self.decoders:
            x = decoder(xs.pop(), x)

        # Classifier
        x = self.classifier(x)
        return x


def _unet(arch: str, pretrained: bool, progress: bool, **kwargs: Any) -> nn.Module:
    # Retrieve the correct Darknet layout type
    unet_type = sys.modules[__name__].__dict__[default_cfgs[arch]['arch']]
    # Build the model
    model = unet_type(default_cfgs[arch]['layout'], **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'], progress)

    return model


def unet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNet:
    """U-Net from
    `"U-Net: Convolutional Networks for Biomedical Image Segmentation" <https://arxiv.org/pdf/1505.04597.pdf>`_

    .. image:: https://github.com/frgfm/Holocron/releases/download/v0.1.3/unet.png
        :align: center

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: semantic segmentation model
    """

    return _unet('unet', pretrained, progress, **kwargs)  # type: ignore[return-value]
