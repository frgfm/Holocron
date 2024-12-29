# Copyright (C) 2020-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections import OrderedDict
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from holocron.nn import DropBlock2d, GlobalAvgPool2d
from holocron.nn.init import init_module

from ..checkpoints import Checkpoint, _handle_legacy_pretrained
from ..utils import _checkpoint, _configure_model, conv_sequence
from .resnet import _ResBlock

__all__ = ["Darknet53_Checkpoint", "DarknetV3", "darknet53"]


class ResBlock(_ResBlock):
    def __init__(
        self,
        planes: int,
        mid_planes: int,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
            conv_sequence(
                planes,
                mid_planes,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=1,
                bias=(norm_layer is None),
            )
            + conv_sequence(
                mid_planes,
                planes,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=1,
                bias=(norm_layer is None),
            ),
            None,
            None,
        )
        if drop_layer is not None:
            self.dropblock = DropBlock2d(0.1, 7, inplace=True)

        # The backpropagation does not seem to appreciate inplace activation on the residual branch
        if hasattr(self.conv[-1], "inplace"):
            self.conv[-1].inplace = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        if hasattr(self, "dropblock"):
            out = self.dropblock(out)

        return out


class DarknetBodyV3(nn.Sequential):
    def __init__(
        self,
        layout: List[Tuple[int, int]],
        in_channels: int = 3,
        stem_channels: int = 32,
        num_features: int = 1,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if act_layer is None:
            act_layer = nn.LeakyReLU(0.1, inplace=True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        in_chans = [stem_channels] + [_layout[0] for _layout in layout[:-1]]

        super().__init__(
            OrderedDict([
                (
                    "stem",
                    nn.Sequential(
                        *conv_sequence(
                            in_channels,
                            stem_channels,
                            act_layer,
                            norm_layer,
                            drop_layer,
                            conv_layer,
                            kernel_size=3,
                            padding=1,
                            bias=(norm_layer is None),
                        )
                    ),
                ),
                (
                    "layers",
                    nn.Sequential(*[
                        self._make_layer(
                            num_blocks, _in_chans, out_chans, act_layer, norm_layer, drop_layer, conv_layer
                        )
                        for _in_chans, (out_chans, num_blocks) in zip(in_chans, layout, strict=False)
                    ]),
                ),
            ])
        )
        self.num_features = num_features

    @staticmethod
    def _make_layer(
        num_blocks: int,
        in_planes: int,
        out_planes: int,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> nn.Sequential:
        layers = conv_sequence(
            in_planes,
            out_planes,
            act_layer,
            norm_layer,
            drop_layer,
            conv_layer,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=(norm_layer is None),
        )
        layers.extend([
            ResBlock(out_planes, out_planes // 2, act_layer, norm_layer, drop_layer, conv_layer)
            for _ in range(num_blocks)
        ])

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self.num_features == 1:
            return super().forward(x)

        self.stem: nn.Sequential
        self.layers: nn.Sequential
        x = self.stem(x)
        features = []
        for idx, stage in enumerate(self.layers):
            x = stage(x)
            if idx >= (len(self.layers) - self.num_features):
                features.append(x)

        return features


class DarknetV3(nn.Sequential):
    def __init__(
        self,
        layout: List[Tuple[int, int]],
        num_classes: int = 10,
        in_channels: int = 3,
        stem_channels: int = 32,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
            OrderedDict([
                (
                    "features",
                    DarknetBodyV3(layout, in_channels, stem_channels, 1, act_layer, norm_layer, drop_layer, conv_layer),
                ),
                ("pool", GlobalAvgPool2d(flatten=True)),
                ("classifier", nn.Linear(layout[-1][0], num_classes)),
            ])
        )

        init_module(self, "leaky_relu")


def _darknet(
    checkpoint: Union[Checkpoint, None],
    progress: bool,
    layout: List[Tuple[int, int]],
    **kwargs: Any,
) -> DarknetV3:
    # Build the model
    model = DarknetV3(layout, **kwargs)
    return _configure_model(model, checkpoint, progress=progress)


class Darknet53_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="darknet53",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/darknet53_224-5015f3fd.pth",
        acc1=0.9417,
        acc5=0.9957,
        sha256="5015f3fdf0963342e0c54790127350375ba269d871feed48f8328b2e43cf7819",
        size=162584273,
        num_params=40595178,
        commit="6e32c5b578711a2ef3731a8f8c61760ed9f03e58",
        train_args=(
            "./imagenette2-320/ --arch darknet53 --batch-size 64 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENETTE


def darknet53(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> DarknetV3:
    """Darknet-53 from
    `"YOLOv3: An Incremental Improvement" <https://pjreddie.com/media/files/papers/YOLOv3.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _darknet

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.Darknet53_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        Darknet53_Checkpoint.DEFAULT.value,
    )
    return _darknet(checkpoint, progress, [(64, 1), (128, 2), (256, 8), (512, 8), (1024, 4)], **kwargs)
