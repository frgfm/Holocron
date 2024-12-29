# Copyright (C) 2020-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections import OrderedDict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from holocron.nn import GlobalAvgPool2d
from holocron.nn.init import init_module

from ..checkpoints import Checkpoint, _handle_legacy_pretrained
from ..presets import IMAGENETTE
from ..utils import _checkpoint, _configure_model, conv_sequence

__all__ = ["Darknet19_Checkpoint", "DarknetV2", "darknet19"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "darknet19": {
        **IMAGENETTE.__dict__,
        "input_shape": (3, 224, 224),
        "url": "https://github.com/frgfm/Holocron/releases/download/v0.1.3/darknet19_224-b1ce16a5.pt",
    },
}


class DarknetBodyV2(nn.Sequential):
    def __init__(
        self,
        layout: List[Tuple[int, int]],
        in_channels: int = 3,
        stem_channels: int = 32,
        passthrough: bool = False,
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

        self.passthrough = passthrough

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
        layers: List[nn.Module] = [nn.MaxPool2d(2)]
        layers.extend(
            conv_sequence(
                in_planes,
                out_planes,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=(norm_layer is None),
            )
        )
        for _ in range(num_blocks):
            layers.extend(
                conv_sequence(
                    out_planes,
                    out_planes // 2,
                    act_layer,
                    norm_layer,
                    drop_layer,
                    conv_layer,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    bias=(norm_layer is None),
                )
                + conv_sequence(
                    out_planes // 2,
                    out_planes,
                    act_layer,
                    norm_layer,
                    drop_layer,
                    conv_layer,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias=(norm_layer is None),
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.passthrough:
            self.stem: nn.Sequential
            self.layers: nn.Sequential
            x = self.stem(x)
            for idx, layer in enumerate(self.layers):
                x = layer(x)
                if idx == len(self.layers) - 2:
                    aux = x.clone()

            return x, aux
        return super().forward(x)


class DarknetV2(nn.Sequential):
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
                    DarknetBodyV2(
                        layout, in_channels, stem_channels, False, act_layer, norm_layer, drop_layer, conv_layer
                    ),
                ),
                ("classifier", nn.Conv2d(layout[-1][0], num_classes, 1)),
                ("pool", GlobalAvgPool2d(flatten=True)),
            ])
        )

        init_module(self, "leaky_relu")


def _darknet(
    checkpoint: Union[Checkpoint, None],
    progress: bool,
    layout: List[Tuple[int, int]],
    **kwargs: Any,
) -> DarknetV2:
    # Build the model
    model = DarknetV2(layout, **kwargs)
    return _configure_model(model, checkpoint, progress=progress)


class Darknet19_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="darknet19",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/darknet19_224-32fd3f97.pth",
        acc1=0.9386,
        acc5=0.9936,
        sha256="32fd3f979586556554652d650c44a59747c7762d81140cadbcd795179a3877ec",
        size=79387724,
        num_params=19827626,
        commit="6e32c5b578711a2ef3731a8f8c61760ed9f03e58",
        train_args=(
            "./imagenette2-320/ --arch darknet19 --batch-size 64 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENETTE


def darknet19(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> DarknetV2:
    """Darknet-19 from
    `"YOLO9000: Better, Faster, Stronger" <https://pjreddie.com/media/files/papers/YOLO9000.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _darknet

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.Darknet19_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        Darknet19_Checkpoint.DEFAULT.value,
    )
    return _darknet(checkpoint, progress, [(64, 0), (128, 1), (256, 1), (512, 2), (1024, 2)], **kwargs)
