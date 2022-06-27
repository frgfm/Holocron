# Copyright (C) 2020-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from holocron.nn import DropBlock2d, GlobalAvgPool2d
from holocron.nn.init import init_module

from ..presets import IMAGENETTE
from ..utils import conv_sequence, load_pretrained_params
from .resnet import _ResBlock

__all__ = ["DarknetV3", "darknet53"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "darknet53": {
        **IMAGENETTE,
        "input_shape": (3, 256, 256),
        "url": "https://github.com/frgfm/Holocron/releases/download/v0.1.2/darknet53_256-f57b8429.pth",
    },
}


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
            OrderedDict(
                [
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
                        nn.Sequential(
                            *[
                                self._make_layer(
                                    num_blocks, _in_chans, out_chans, act_layer, norm_layer, drop_layer, conv_layer
                                )
                                for _in_chans, (out_chans, num_blocks) in zip(in_chans, layout)
                            ]
                        ),
                    ),
                ]
            )
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
        layers.extend(
            [
                ResBlock(out_planes, out_planes // 2, act_layer, norm_layer, drop_layer, conv_layer)
                for _ in range(num_blocks)
            ]
        )

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
            OrderedDict(
                [
                    (
                        "features",
                        DarknetBodyV3(
                            layout, in_channels, stem_channels, 1, act_layer, norm_layer, drop_layer, conv_layer
                        ),
                    ),
                    ("pool", GlobalAvgPool2d(flatten=True)),
                    ("classifier", nn.Linear(layout[-1][0], num_classes)),
                ]
            )
        )

        init_module(self, "leaky_relu")


def _darknet(arch: str, pretrained: bool, progress: bool, layout: List[Tuple[int, int]], **kwargs: Any) -> DarknetV3:
    # Build the model
    model = DarknetV3(layout, **kwargs)
    model.default_cfg = default_cfgs[arch]  # type: ignore[assignment]
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"], progress)

    return model


def darknet53(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DarknetV3:
    """Darknet-53 from
    `"YOLOv3: An Incremental Improvement" <https://pjreddie.com/media/files/papers/YOLOv3.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _darknet("darknet53", pretrained, progress, [(64, 1), (128, 2), (256, 8), (512, 8), (1024, 4)], **kwargs)
