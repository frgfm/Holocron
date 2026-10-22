# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import sys
from collections.abc import Callable
from itertools import pairwise
from typing import Any, cast

from torch import Tensor, nn

from ...nn.init import init_module
from ..utils import conv_sequence, load_pretrained_params
from .unet import UpPath, down_path

__all__ = ["UNetp", "UNetpp", "unetp", "unetpp"]


default_cfgs: dict[str, dict[str, Any]] = {
    "unetp": {"arch": "UNetp", "layout": [64, 128, 256, 512], "url": None},
    "unetpp": {"arch": "UNetpp", "layout": [64, 128, 256, 512], "url": None},
}


def _encoder_and_bridge(
    layout: list[int],
    in_channels: int,
    act_layer: nn.Module,
    norm_layer: Callable[..., nn.Module] | None,
    drop_layer: Callable[..., nn.Module] | None,
    conv_layer: Callable[..., nn.Module] | None,
) -> tuple[nn.ModuleList, nn.Sequential]:
    """Build the shared contracting path of UNet+ and UNet++.

    Returns:
        Encoder blocks and bridge, in their original construction order.
    """
    layers = (act_layer, norm_layer, drop_layer, conv_layer)
    encoder = nn.ModuleList(
        down_path(in_chan, out_chan, idx > 0, 1, *layers)
        for idx, (in_chan, out_chan) in enumerate(pairwise([in_channels, *layout]))
    )
    bridge = nn.Sequential(
        nn.MaxPool2d((2, 2)),
        *conv_sequence(layout[-1], 2 * layout[-1], *layers, kernel_size=3, padding=1),
        *conv_sequence(2 * layout[-1], layout[-1], *layers, kernel_size=3, padding=1),
    )
    return encoder, bridge


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
        layout: list[int],
        in_channels: int = 3,
        num_classes: int = 10,
        act_layer: nn.Module | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
        drop_layer: Callable[..., nn.Module] | None = None,
        conv_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()

        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        # Contracting path
        self.encoder, self.bridge = _encoder_and_bridge(
            layout, in_channels, act_layer, norm_layer, drop_layer, conv_layer
        )

        # Expansive path
        self.decoder = nn.ModuleList([])
        layout_ = [layout[-1], *layout[1:][::-1]]
        for left_chan, up_chan, num_cells in zip(layout[::-1], layout_, range(1, len(layout) + 1), strict=True):
            self.decoder.append(
                nn.ModuleList([
                    UpPath(left_chan + up_chan, left_chan, True, 1, act_layer, norm_layer, drop_layer, conv_layer)
                    for _ in range(num_cells)
                ])
            )

        # Classifier
        self.classifier = nn.Conv2d(layout[0], num_classes, 1)

        init_module(self, "relu")

    def forward(self, x: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        xs: list[Tensor] = []
        # Contracting path
        for encoder in self.encoder:
            xs.append(encoder(xs[-1] if len(xs) > 0 else x))

        xs.append(self.bridge(xs[-1]))

        # Nested expansive path
        for j in range(len(self.decoder)):
            for i in range(len(xs) - 1):
                up_feat = xs[i + 1] if (i + 2) < len(xs) else xs.pop()
                xs[i] = cast(nn.ModuleList, self.decoder[-1 - i])[j](xs[i], up_feat)

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
        layout: list[int],
        in_channels: int = 3,
        num_classes: int = 10,
        act_layer: nn.Module | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
        drop_layer: Callable[..., nn.Module] | None = None,
        conv_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()

        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        # Contracting path
        self.encoder, self.bridge = _encoder_and_bridge(
            layout, in_channels, act_layer, norm_layer, drop_layer, conv_layer
        )

        # Expansive path
        self.decoder = nn.ModuleList([])
        layout_ = [layout[-1], *layout[1:][::-1]]
        for left_chan, up_chan, num_cells in zip(layout[::-1], layout_, range(1, len(layout) + 1), strict=True):
            self.decoder.append(
                nn.ModuleList([
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
                ])
            )

        # Classifier
        self.classifier = nn.Conv2d(layout[0], num_classes, 1)

        init_module(self, "relu")

    def forward(self, x: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        xs: list[list[Tensor]] = []
        # Contracting path
        for encoder in self.encoder:
            xs.append([encoder(xs[-1][0] if len(xs) > 0 else x)])

        xs.append([self.bridge(xs[-1][-1])])

        # Nested expansive path
        for j in range(len(self.decoder)):
            for i in range(len(xs) - 1):
                up_feat = xs[i + 1][j] if (i + 2) < len(xs) else xs.pop()[-1]
                xs[i].append(cast(nn.ModuleList, self.decoder[-1 - i])[j](xs[i], up_feat))

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
    """UNet+ from ["UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation"](https://arxiv.org/pdf/1912.05074.pdf)

    ![UNet+ architecture](https://github.com/frgfm/Holocron/releases/download/v0.1.3/unetp.png)

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of [`UNetp`][holocron.models.segmentation.unetpp.UNetp]

    Returns:
        semantic segmentation model
    """
    return _unet("unetp", pretrained, progress, **kwargs)  # type: ignore[return-value]


def unetpp(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNetpp:
    """UNet++ from ["UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation"](https://arxiv.org/pdf/1912.05074.pdf)

    ![UNet++ architecture](https://github.com/frgfm/Holocron/releases/download/v0.1.3/unetpp.png)

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of [`UNetpp`][holocron.models.segmentation.unetpp.UNetpp]

    Returns:
        semantic segmentation model
    """
    return _unet("unetpp", pretrained, progress, **kwargs)  # type: ignore[return-value]
