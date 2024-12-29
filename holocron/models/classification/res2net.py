# Copyright (C) 2019-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
Implementation of Res2Net
based on https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/res2net.py
"""

import math
from enum import Enum
from typing import Any, Callable, List, Optional, Union

import torch
import torch.nn as nn

from ..checkpoints import Checkpoint, _handle_legacy_pretrained
from ..utils import _checkpoint, _configure_model, conv_sequence
from .resnet import ResNet, _ResBlock

__all__ = ["Bottle2neck", "Res2Net50_26w_4s_Checkpoint", "res2net50_26w_4s"]


class ScaleConv2d(nn.Module):
    def __init__(
        self,
        scale: int,
        planes: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        downsample: bool = False,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        self.scale = scale
        self.width = planes // scale
        self.conv = nn.ModuleList([
            nn.Sequential(
                *conv_sequence(
                    self.width,
                    self.width,
                    act_layer,
                    norm_layer,
                    drop_layer,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=groups,
                    bias=(norm_layer is None),
                )
            )
            for _ in range(max(1, scale - 1))
        ])

        if downsample:
            self.downsample = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        else:
            self.downsample = None  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split the channel dimension into groups of self.width channels
        split_x = torch.split(x, self.width, 1)
        out: List[torch.Tensor] = []
        for idx, layer in enumerate(self.conv):
            # If downsampled, don't add previous branch
            res = split_x[idx] if idx == 0 or self.downsample is not None else out[-1] + split_x[idx]
            out.append(layer(res))
        # Use the last chunk as shortcut connection
        if self.scale > 1:
            # If the convs were strided, the shortcut needs to be downsampled
            if self.downsample is not None:
                out.append(self.downsample(split_x[-1]))
            else:
                out.append(split_x[-1])

        return torch.cat(out, 1)


class Bottle2neck(_ResBlock):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 26,
        dilation: int = 1,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        scale: int = 4,
    ) -> None:
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        # Check if ScaleConv2d needs to downsample the identity branch
        downsample_ = stride > 1 or downsample is not None

        width = math.floor(planes * (base_width / 64.0)) * groups
        super().__init__(
            [
                *conv_sequence(
                    inplanes,
                    width * scale,
                    act_layer,
                    norm_layer,
                    drop_layer,
                    kernel_size=1,
                    stride=1,
                    bias=(norm_layer is None),
                ),
                ScaleConv2d(scale, width * scale, 3, stride, groups, downsample_, act_layer, norm_layer, drop_layer),
                *conv_sequence(
                    width * scale,
                    planes * self.expansion,
                    None,
                    norm_layer,
                    drop_layer,
                    kernel_size=1,
                    stride=1,
                    bias=(norm_layer is None),
                ),
            ],
            downsample,
            act_layer,
        )


def _res2net(
    checkpoint: Union[Checkpoint, None],
    progress: bool,
    num_blocks: List[int],
    out_chans: List[int],
    width_per_group: int,
    scale: int,
    **kwargs: Any,
) -> ResNet:
    # Build the model
    model = model = ResNet(
        Bottle2neck,  # type: ignore[arg-type]
        num_blocks,
        out_chans,
        width_per_group=width_per_group,
        block_args={"scale": scale},
        **kwargs,
    )
    return _configure_model(model, checkpoint, progress=progress)


class Res2Net50_26w_4s_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="res2net50_26w_4s",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/res2net50_26w_4s_224-345170e8.pth",
        acc1=0.9394,
        acc5=0.9941,
        sha256="345170e8ff75d10330af55674090b0d9aa751e14b6f3b4a95bb8ea6cdd65be4b",
        size=95020747,
        num_params=23670610,
        commit="6e32c5b578711a2ef3731a8f8c61760ed9f03e58",
        train_args=(
            "./imagenette2-320/ --arch res2net50_26w_4s --batch-size 64 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENETTE


def res2net50_26w_4s(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """Res2Net-50 26wx4s from
    `"Res2Net: A New Multi-scale Backbone Architecture" <https://arxiv.org/pdf/1904.01169.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _res2net

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.Res2Net50_26w_4s_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        Res2Net50_26w_4s_Checkpoint.DEFAULT.value,
    )
    return _res2net(checkpoint, progress, [3, 4, 6, 3], [64, 128, 256, 512], 26, 4, **kwargs)
