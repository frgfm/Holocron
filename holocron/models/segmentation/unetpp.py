# Copyright (C) 2020-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import sys
from typing import Any, Callable, Dict, List, Optional

import torch.nn as nn
from torch import Tensor

from ...nn.init import init_module
from ..utils import conv_sequence, load_pretrained_params
from .unet import UpPath, down_path

__all__ = ["UNetp", "unetp", "UNetpp", "unetpp"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "unetp": {"arch": "UNetp", "layout": [64, 128, 256, 512], "url": None},
    "unetpp": {"arch": "UNetpp", "layout": [64, 128, 256, 512], "url": None},
}


class UNetp(nn.Module):
    """Implements a UNet+ architecture

    Args:
        layout: number of channels after each contracting block
        in_channels: number of channels in the input tensor
        num_classes: number of output classes
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
        self.encoder = nn.ModuleList([])
        _layout = [in_channels] + layout
        _pool = False
        for in_chan, out_chan in zip(_layout[:-1], _layout[1:]):
            self.encoder.append(down_path(in_chan, out_chan, _pool, 1, act_layer, norm_layer, drop_layer, conv_layer))
            _pool = True

        self.bridge = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            *conv_sequence(
                layout[-1], 2 * layout[-1], act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=1
            ),
            *conv_sequence(
                2 * layout[-1], layout[-1], act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=1
            ),
        )

        # Expansive path
        self.decoder = nn.ModuleList([])
        _layout = [layout[-1]] + layout[1:][::-1]
        for left_chan, up_chan, num_cells in zip(layout[::-1], _layout, range(1, len(layout) + 1)):
            self.decoder.append(
                nn.ModuleList(
                    [
                        UpPath(left_chan + up_chan, left_chan, True, 1, act_layer, norm_layer, drop_layer, conv_layer)
                        for _ in range(num_cells)
                    ]
                )
            )

        # Classifier
        self.classifier = nn.Conv2d(layout[0], num_classes, 1)

        init_module(self, "relu")

    def forward(self, x: Tensor) -> Tensor:

        xs: List[Tensor] = []
        # Contracting path
        for encoder in self.encoder:
            xs.append(encoder(xs[-1] if len(xs) > 0 else x))

        xs.append(self.bridge(xs[-1]))

        # Nested expansive path
        for j in range(len(self.decoder)):
            for i in range(len(xs) - 1):
                up_feat = xs[i + 1] if (i + 2) < len(xs) else xs.pop()
                xs[i] = self.decoder[-1 - i][j](xs[i], up_feat)

        return self.classifier(xs.pop())


class UNetpp(nn.Module):
    """Implements a UNet++ architecture

    Args:
        layout: number of channels after each contracting block
        in_channels: number of channels in the input tensor
        num_classes: number of output classes
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
        self.encoder = nn.ModuleList([])
        _layout = [in_channels] + layout
        _pool = False
        for in_chan, out_chan in zip(_layout[:-1], _layout[1:]):
            self.encoder.append(down_path(in_chan, out_chan, _pool, 1, act_layer, norm_layer, drop_layer, conv_layer))
            _pool = True

        self.bridge = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            *conv_sequence(
                layout[-1], 2 * layout[-1], act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=1
            ),
            *conv_sequence(
                2 * layout[-1], layout[-1], act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=1
            ),
        )

        # Expansive path
        self.decoder = nn.ModuleList([])
        _layout = [layout[-1]] + layout[1:][::-1]
        for left_chan, up_chan, num_cells in zip(layout[::-1], _layout, range(1, len(layout) + 1)):
            self.decoder.append(
                nn.ModuleList(
                    [
                        UpPath(
                            up_chan + (idx + 1) * left_chan,
                            left_chan,
                            True,
                            1,
                            act_layer,
                            norm_layer,
                            drop_layer,
                            conv_layer,
                        )
                        for idx in range(num_cells)
                    ]
                )
            )

        # Classifier
        self.classifier = nn.Conv2d(layout[0], num_classes, 1)

        init_module(self, "relu")

    def forward(self, x: Tensor) -> Tensor:

        xs: List[List[Tensor]] = []
        # Contracting path
        for encoder in self.encoder:
            xs.append([encoder(xs[-1][0] if len(xs) > 0 else x)])

        xs.append([self.bridge(xs[-1][-1])])

        # Nested expansive path
        for j in range(len(self.decoder)):
            for i in range(len(xs) - 1):
                up_feat = xs[i + 1][j] if (i + 2) < len(xs) else xs.pop()[-1]
                xs[i].append(self.decoder[-1 - i][j](xs[i], up_feat))

        # Classifier
        return self.classifier(xs.pop()[-1])


def _unet(arch: str, pretrained: bool, progress: bool, **kwargs: Any) -> nn.Module:
    # Retrieve the correct Darknet layout type
    unet_type = sys.modules[__name__].__dict__[default_cfgs[arch]["arch"]]
    # Build the model
    model = unet_type(default_cfgs[arch]["layout"], **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"], progress)

    return model


def unetp(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNetp:
    """UNet+ from `"UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation"
    <https://arxiv.org/pdf/1912.05074.pdf>`_

    .. image:: https://github.com/frgfm/Holocron/releases/download/v0.1.3/unetp.png
        :align: center

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        progress: If True, displays a progress bar of the download to stderr

    Returns:
        semantic segmentation model
    """

    return _unet("unetp", pretrained, progress, **kwargs)  # type: ignore[return-value]


def unetpp(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNetpp:
    """UNet++ from `"UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation"
    <https://arxiv.org/pdf/1912.05074.pdf>`_

    .. image:: https://github.com/frgfm/Holocron/releases/download/v0.1.3/unetpp.png
        :align: center

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        progress: If True, displays a progress bar of the download to stderr

    Returns:
        semantic segmentation model
    """

    return _unet("unetpp", pretrained, progress, **kwargs)  # type: ignore[return-value]
