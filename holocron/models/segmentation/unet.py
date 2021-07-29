# Copyright (C) 2019-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import sys
from collections import OrderedDict
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, Any, Union, Optional, Callable, List, Tuple
from torchvision.models import vgg11, resnet34
from torchvision.models._utils import IntermediateLayerGetter

from ...nn.init import init_module
from ...nn import SiLU, GlobalAvgPool2d
from ..rexnet import rexnet1_3x
from ..utils import conv_sequence, load_pretrained_params


__all__ = ['UNet', 'unet', 'DynamicUNet', 'unet_vgg11', 'unet_tvresnet34', 'unet_rexnet13', 'unet2']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'unet': {
        'encoder_layout': [64, 128, 256, 512],
        'url': None
    },
    'unet2': {
        'encoder_layout': [64, 128, 256, 512],
        'backbone_layers': ['0', '1', '2', '3'],
        'url': None
    },
    'unet_vgg11': {
        'backbone_layers': ['1', '4', '9', '14', '19'],
        'url': None
    },
    'unet_tvresnet34': {
        'backbone_layers': ['relu', 'layer1', 'layer2', 'layer3', 'layer4'],
        'url': None
    },
    'unet_rexnet13': {
        'backbone_layers': ['3', '5', '7', '13', '18'],
        'url': None,
    },
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
        up_chan: int,
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
            self.upsample = nn.ConvTranspose2d(up_chan, out_chan, 2, stride=2)

        self.block = nn.Sequential(*conv_sequence(up_chan, out_chan,
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
                w_slice = slice(delta_w // 2, -(delta_w // 2) if delta_w > 0 else downfeat.shape[-1])
                delta_h = downfeat.shape[-2] - _upfeat.shape[-2]
                h_slice = slice(delta_h // 2, -(delta_h // 2) if delta_h > 0 else downfeat.shape[-2])
                downfeats[idx] = downfeat[..., h_slice, w_slice]
        # Concatenate both feature maps and forward them
        return self.block(torch.cat((*downfeats, _upfeat), dim=1))


class UNetBackbone(nn.Sequential):
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
    ) -> None:

        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        # Contracting path
        _layers: List[nn.Module] = []
        _layout = [in_channels] + layout
        _pool = False
        for in_chan, out_chan in zip(_layout[:-1], _layout[1:]):
            _layers.append(down_path(in_chan, out_chan, _pool, int(same_padding),
                                     act_layer, norm_layer, drop_layer, conv_layer))
            _pool = True

        super().__init__(OrderedDict([
            ('features', nn.Sequential(*_layers)),
            ('pool', GlobalAvgPool2d(flatten=True)),
            ('head', nn.Linear(layout[-1], num_classes))]))

        init_module(self, 'relu')


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
        self.encoder = nn.ModuleList([])
        _layout = [in_channels] + layout
        _pool = False
        for in_chan, out_chan in zip(_layout[:-1], _layout[1:]):
            self.encoder.append(down_path(in_chan, out_chan, _pool, int(same_padding),
                                           act_layer, norm_layer, drop_layer, conv_layer))
            _pool = True

        self.bridge = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            *conv_sequence(layout[-1], 2 * layout[-1],
                           act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=1),
            *conv_sequence(2 * layout[-1], layout[-1],
                           act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=1),
        )

        # Expansive path
        self.decoders = nn.ModuleList([])
        _layout = [chan // 2 if bilinear_upsampling else chan for chan in layout[::-1][:-1]] + [layout[0]]
        for in_chan, out_chan in zip([2 * layout[-1]] + layout[::-1][:-1], _layout):
            self.decoders.append(UpPath(in_chan, out_chan, bilinear_upsampling, int(same_padding),
                                        act_layer, norm_layer, drop_layer, conv_layer))

        # Classifier
        self.classifier = nn.Conv2d(layout[0], num_classes, 1)

        init_module(self, 'relu')

    def forward(self, x: Tensor) -> Tensor:

        xs: List[Tensor] = []
        # Contracting path
        for encoder in self.encoder:
            xs.append(encoder(xs[-1] if len(xs) > 0 else x))
        x = self.bridge(xs[-1])

        # Expansive path
        for decoder in self.decoders:

            x = decoder(xs.pop(), x)

        # Classifier
        x = self.classifier(x)
        return x


class UBlock(nn.Module):
    def __init__(
        self,
        left_chan: int,
        up_chan: int,
        out_chan: int,
        padding: int = 0,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()

        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        self.upsample = nn.Sequential(
            *conv_sequence(up_chan, up_chan * 2 ** 2, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=1),
            nn.PixelShuffle(upscale_factor=2)
        )

        self.bn = nn.BatchNorm2d(left_chan) if norm_layer is None else norm_layer(in_chan)

        self.block = nn.Sequential(
            act_layer,
            *conv_sequence(left_chan + up_chan, out_chan, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=padding),
            *conv_sequence(out_chan, out_chan, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=padding)
        )

    def forward(self, downfeat: Tensor, upfeat: Tensor) -> Tensor:

        # Upsample expansive features
        _upfeat = self.upsample(upfeat)

        # Crop upsampled features
        if downfeat.shape[-2:] != _upfeat.shape[-2:]:
            _upfeat = F.interpolate(_upfeat, downfeat.shape[-2:], mode='nearest')

        # Concatenate both feature maps and forward them
        return self.block(torch.cat((self.bn(downfeat), _upfeat), dim=1))


class DynamicUNet(nn.Module):
    """Implements a dymanic U-Net architecture

    Args:
        encoder: feature extractor used for encoding
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
        encoder: IntermediateLayerGetter,
        num_classes: int = 10,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
        same_padding: bool = True,
        input_shape: Optional[Tuple[int, int, int]] = None,
        final_upsampling: bool = False,
    ) -> None:
        super().__init__()

        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        self.encoder = encoder
        # Determine all feature map shapes
        training_mode = self.encoder.training
        self.encoder.eval()
        input_shape = (3, 256, 256) if input_shape is None else input_shape
        with torch.no_grad():
            shapes = [v.shape[1:] for v in self.encoder(torch.zeros(1, *input_shape)).values()]
        chans = [s[0] for s in shapes]
        if training_mode:
            self.encoder.train()

        # Middle layers
        self.bridge = nn.Sequential(
            nn.BatchNorm2d(chans[-1]) if norm_layer is None else norm_layer(chans[-1]),
            act_layer,
            *conv_sequence(chans[-1], 2 * chans[-1],
                           act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=1),
            *conv_sequence(2 * chans[-1], chans[-1],
                           act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=1)
        )

        # Expansive path
        self.decoders = nn.ModuleList([])
        _layout = chans[::-1][1:] + [chans[0]]
        for up_chan, out_chan in zip(chans[::-1], _layout):
            self.decoders.append(UBlock(up_chan, up_chan, out_chan, int(same_padding),
                                        act_layer, norm_layer, drop_layer, conv_layer))

        # Final upsampling if sizes don't match
        self.upsample: Optional[nn.Sequential] = None
        if final_upsampling:
            self.upsample = nn.Sequential(
                *conv_sequence(chans[0], chans[0] * 2 ** 2,
                               act_layer, norm_layer, drop_layer, conv_layer, kernel_size=1),
                nn.PixelShuffle(upscale_factor=2)
            )

        # Classifier
        self.classifier = nn.Conv2d(chans[0], num_classes, 1)

        init_module(self, 'relu')

    def forward(self, x: Tensor) -> Tensor:

        # Contracting path
        xs: List[Tensor] = list(self.encoder(x).values())
        x = self.bridge(xs[-1])

        # Expansive path
        for decoder in self.decoders:
            x = decoder(xs.pop(), x)

        if self.upsample is not None:
            x = self.upsample(x)

        # Classifier
        x = self.classifier(x)
        return x


def _unet(arch: str, pretrained: bool, progress: bool, **kwargs: Any) -> UNet:
    # Build the model
    model = UNet(default_cfgs[arch]['encoder_layout'], **kwargs)
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

    return _unet('unet', pretrained, progress, **kwargs)


def _dynamic_unet(arch: str, backbone: nn.Module, pretrained: bool, progress: bool, **kwargs: Any) -> DynamicUNet:
    # Build the encoder
    encoder = IntermediateLayerGetter(
        backbone,
        {name: str(idx) for idx, name in enumerate(default_cfgs[arch]['backbone_layers'])}
    )
    # Build the model
    model = DynamicUNet(encoder, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'], progress)

    return model


def unet2(pretrained: bool = False, progress: bool = True, in_channels: int = 3, **kwargs: Any) -> DynamicUNet:
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

    backbone = UNetBackbone(default_cfgs['unet2']['encoder_layout'], in_channels=in_channels).features

    return _dynamic_unet('unet2', backbone, pretrained, progress, **kwargs)  # type: ignore[arg-type]


def unet_vgg11(
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    progress: bool = True,
    **kwargs: Any
) -> DynamicUNet:

    backbone = vgg11(pretrained=pretrained_backbone and not pretrained).features

    return _dynamic_unet('unet_vgg11', backbone, pretrained, progress, **kwargs)


def unet_tvresnet34(
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    progress: bool = True,
    **kwargs: Any
) -> DynamicUNet:

    backbone = resnet34(pretrained=pretrained_backbone and not pretrained)
    kwargs['final_upsampling'] = kwargs.get('final_upsampling', True)

    return _dynamic_unet('unet_tvresnet34', backbone, pretrained, progress, **kwargs)


def unet_rexnet13(
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    progress: bool = True,
    in_channels: int = 3,
    **kwargs: Any
) -> DynamicUNet:

    backbone = rexnet1_3x(pretrained=pretrained_backbone and not pretrained, in_channels=in_channels).features
    kwargs['final_upsampling'] = kwargs.get('final_upsampling', True)
    kwargs['act_layer'] = kwargs.get('act_layer', SiLU())
    # hotfix of https://github.com/pytorch/vision/issues/3802
    backbone[21] = SiLU()

    return _dynamic_unet('unet_rexnet13', backbone, pretrained, progress, **kwargs)
