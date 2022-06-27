# Copyright (C) 2020-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ...nn.init import init_module
from ..utils import conv_sequence, load_pretrained_params
from .unet import down_path

__all__ = ["UNet3p", "unet3p"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "unet3p": {"arch": "UNet3p", "layout": [64, 128, 256, 512, 1024], "url": None}
}


class FSAggreg(nn.Module):
    def __init__(
        self,
        e_chans: List[int],
        skip_chan: int,
        d_chans: List[int],
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:

        super().__init__()

        # Check stem conv channels
        base_chan = e_chans[0] if len(e_chans) > 0 else skip_chan
        # Get UNet depth
        depth = len(e_chans) + 1 + len(d_chans)
        # Downsample = max pooling + conv for channel reduction
        self.downsamples = nn.ModuleList(
            [
                nn.Sequential(nn.MaxPool2d(2 ** (len(e_chans) - idx)), nn.Conv2d(e_chan, base_chan, 3, padding=1))
                for idx, e_chan in enumerate(e_chans)
            ]
        )
        self.skip = nn.Conv2d(skip_chan, base_chan, 3, padding=1) if len(e_chans) > 0 else nn.Identity()
        # Upsample = bilinear interpolation + conv for channel reduction
        self.upsamples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=2 ** (idx + 1), mode="bilinear", align_corners=True),
                    nn.Conv2d(d_chan, base_chan, 3, padding=1),
                )
                for idx, d_chan in enumerate(d_chans)
            ]
        )

        self.block = nn.Sequential(
            *conv_sequence(
                depth * base_chan,
                depth * base_chan,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, downfeats: List[Tensor], feat: Tensor, upfeats: List[Tensor]):

        if len(downfeats) != len(self.downsamples) or len(upfeats) != len(self.upsamples):
            raise ValueError(
                f"Expected {len(self.downsamples)} encoding & {len(self.upsamples)} decoding features, "
                f"received: {len(downfeats)} & {len(upfeats)}"
            )

        # Concatenate full-scale features
        x = torch.cat(
            (
                *[downsample(downfeat) for downsample, downfeat in zip(self.downsamples, downfeats)],
                self.skip(feat),
                *[upsample(upfeat) for upsample, upfeat in zip(self.upsamples, upfeats)],
            ),
            dim=1,
        )

        return self.block(x)


class UNet3p(nn.Module):
    """Implements a UNet3+ architecture

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
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Contracting path
        self.encoder = nn.ModuleList([])
        _layout = [in_channels] + layout
        _pool = False
        for in_chan, out_chan in zip(_layout[:-1], _layout[1:]):
            self.encoder.append(down_path(in_chan, out_chan, _pool, 1, act_layer, norm_layer, drop_layer, conv_layer))
            _pool = True

        # Expansive path
        self.decoder = nn.ModuleList([])
        for row in range(len(layout) - 1):
            self.decoder.append(
                FSAggreg(
                    layout[:row],
                    layout[row],
                    [len(layout) * layout[0]] * (len(layout) - 2 - row) + layout[-1:],
                    act_layer,
                    norm_layer,
                    drop_layer,
                    conv_layer,
                )
            )

        # Classifier
        self.classifier = nn.Conv2d(len(layout) * layout[0], num_classes, 1)

        init_module(self, "relu")

    def forward(self, x: Tensor) -> Tensor:

        xs: List[Tensor] = []
        # Contracting path
        for encoder in self.encoder:
            xs.append(encoder(xs[-1] if len(xs) > 0 else x))

        # Full-scale expansive path
        for idx in range(len(self.decoder) - 1, -1, -1):
            xs[idx] = self.decoder[idx](xs[:idx], xs[idx], xs[idx + 1 :])

        # Classifier
        x = self.classifier(xs[0])
        return x


def _unet(arch: str, pretrained: bool, progress: bool, **kwargs: Any) -> nn.Module:
    # Build the model
    model = UNet3p(default_cfgs[arch]["layout"], **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"], progress)

    return model


def unet3p(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNet3p:
    """UNet3+ from
    `"UNet 3+: A Full-Scale Connected UNet For Medical Image Segmentation" <https://arxiv.org/pdf/2004.08790.pdf>`_

    .. image:: https://github.com/frgfm/Holocron/releases/download/v0.1.3/unet3p.png
        :align: center

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        progress: If True, displays a progress bar of the download to stderr

    Returns:
        semantic segmentation model
    """

    return _unet("unet3p", pretrained, progress, **kwargs)  # type: ignore[return-value]
