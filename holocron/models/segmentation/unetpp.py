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
from .unet import down_path, UpPath


__all__ = ['UNetp', 'unetp', 'UNetpp', 'unetpp']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'unetp': {'arch': 'UNetp',
              'layout': [64, 128, 256, 512, 1024],
              'url': None},
    'unetpp': {'arch': 'UNetpp',
               'layout': [64, 128, 256, 512, 1024],
               'url': None},
}


class UNetp(nn.Module):
    """Implements a UNet+ architecture

    Args:
        layout (list<int>): number of channels after each contracting block
        in_channels (int, optional): number of channels in the input tensor
        num_classes (int, optional): number of output classes
        act_layer: activation layer
        norm_layer: normalization layer
        drop_layer: dropout layer
        conv_layer: convolutional layer
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
    ) -> None:
        super().__init__()

        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        # Contracting path
        self.encoders = nn.ModuleList([])
        _layout = [in_channels] + layout
        _pool = False
        for in_chan, out_chan in zip(_layout[:-1], _layout[1:]):
            self.encoders.append(down_path(in_chan, out_chan, _pool, 1,
                                           act_layer, norm_layer, drop_layer, conv_layer))
            _pool = True

        # Expansive path
        self.decoders = nn.ModuleList([])
        for next_chan, row_chan, num_cells in zip(layout[1:], layout[:-1], range(len(_layout) - 1, 0, -1)):
            self.decoders.append(nn.ModuleList([
                UpPath(next_chan + row_chan, row_chan, 1, True, 1,
                       act_layer, norm_layer, drop_layer, conv_layer)
                for _ in range(num_cells + 1)
            ]))

        # Classifier
        self.classifier = nn.Conv2d(layout[0], num_classes, 1)

        init_module(self, 'relu')

    def forward(self, x: Tensor) -> Tensor:

        xs: List[Tensor] = []
        # Contracting path
        for encoder in self.encoders:
            xs.append(encoder(xs[-1] if len(xs) > 0 else x))

        # Nested expansive path
        for j in range(len(self.decoders)):
            for i in range(len(self.decoders) - j):
                xs[i] = self.decoders[i][j](xs[i], xs[i + 1] if (i + 1) < (len(self.decoders) - j) else xs.pop())

        return self.classifier(xs.pop())


class UNetpp(nn.Module):
    """Implements a UNet++ architecture

    Args:
        layout (list<int>): number of channels after each contracting block
        in_channels (int, optional): number of channels in the input tensor
        num_classes (int, optional): number of output classes
        act_layer: activation layer
        norm_layer: normalization layer
        drop_layer: dropout layer
        conv_layer: convolutional layer
    """
    def __init__(
        self,
        layout: List[int],
        in_channels: int = 3,
        num_classes: int = 10,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()

        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        # Contracting path
        self.encoders = nn.ModuleList([])
        _layout = [in_channels] + layout
        _pool = False
        for in_chan, out_chan in zip(_layout[:-1], _layout[1:]):
            self.encoders.append(down_path(in_chan, out_chan, _pool, 1, act_layer, norm_layer, drop_layer, conv_layer))
            _pool = True

        # Expansive path
        self.decoders = nn.ModuleList([])
        for next_chan, row_chan, num_cells in zip(layout[1:], layout[:-1], range(len(_layout) - 1, 0, -1)):
            self.decoders.append(nn.ModuleList([
                UpPath(next_chan + num_skips * row_chan, row_chan, num_skips, True, 1,
                       act_layer, norm_layer, drop_layer, conv_layer)
                for num_skips in range(1, num_cells + 2)
            ]))

        # Classifier
        self.classifier = nn.Conv2d(layout[0], num_classes, 1)

        init_module(self, 'relu')

    def forward(self, x: Tensor) -> Tensor:

        xs: List[List[Tensor]] = []
        # Contracting path
        for encoder in self.encoders:
            xs.append([encoder(xs[-1][0] if len(xs) > 0 else x)])

        # Nested expansive path
        for j in range(len(self.decoders)):
            for i in range(len(self.decoders) - j):
                xs[i].append(self.decoders[i][j](
                    xs[i][:j + 1],
                    xs[i + 1][j] if (i + 1) < (len(self.decoders) - j) else xs.pop()[-1]
                ))

        # Classifier
        x = self.classifier(xs.pop()[-1])
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


def unetp(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNetp:
    """UNet+ from `"UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation"
    <https://arxiv.org/pdf/1912.05074.pdf>`_

    .. image:: https://github.com/frgfm/Holocron/releases/download/v0.1.3/unetp.png
        :align: center

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: semantic segmentation model
    """

    return _unet('unetp', pretrained, progress, **kwargs)  # type: ignore[return-value]


def unetpp(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNetpp:
    """UNet++ from `"UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation"
    <https://arxiv.org/pdf/1912.05074.pdf>`_

    .. image:: https://github.com/frgfm/Holocron/releases/download/v0.1.3/unetpp.png
        :align: center

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: semantic segmentation model
    """

    return _unet('unetpp', pretrained, progress, **kwargs)  # type: ignore[return-value]
