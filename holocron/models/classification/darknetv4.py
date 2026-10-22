# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections import OrderedDict
from collections.abc import Callable
from enum import Enum
from typing import Any

import torch
from torch import nn

from holocron.nn import DropBlock2d, GlobalAvgPool2d
from holocron.nn.init import init_module

from ..checkpoints import Checkpoint, _handle_legacy_pretrained
from ..presets import IMAGENETTE
from ..utils import _checkpoint, _configure_model, _imagenette_recipe, conv_sequence
from .darknetv3 import ResBlock

__all__ = ["CSPDarknet53_Checkpoint", "CSPDarknet53_Mish_Checkpoint", "DarknetV4", "cspdarknet53", "cspdarknet53_mish"]


default_cfgs: dict[str, dict[str, Any]] = {
    "cspdarknet53": {
        **IMAGENETTE.__dict__,
        "input_shape": (3, 224, 224),
        "url": "https://github.com/frgfm/Holocron/releases/download/v0.1.3/cspdarknet53_224-d2a17b18.pt",
    },
    "cspdarknet53_mish": {
        **IMAGENETTE.__dict__,
        "input_shape": (3, 224, 224),
        "url": "https://github.com/frgfm/Holocron/releases/download/v0.1.3/cspdarknet53_mish_256-32d8ec68.pt",
    },
}


class CSPStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        act_layer: nn.Module | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
        drop_layer: Callable[..., nn.Module] | None = None,
        conv_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        compression = 2 if num_blocks > 1 else 1
        self.base_layer = nn.Sequential(
            *conv_sequence(
                in_channels,
                out_channels,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=1,
                stride=2,
                bias=(norm_layer is None),
            ),
            # Share the conv
            *conv_sequence(
                out_channels,
                2 * out_channels // compression,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=1,
                bias=(norm_layer is None),
            ),
        )
        self.main = nn.Sequential(
            *[
                ResBlock(
                    out_channels // compression,
                    out_channels // compression if num_blocks > 1 else in_channels,
                    act_layer,
                    norm_layer,
                    drop_layer,
                    conv_layer,
                )
                for _ in range(num_blocks)
            ],
            *conv_sequence(
                out_channels // compression,
                out_channels // compression,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=1,
                bias=(norm_layer is None),
            ),
        )
        self.transition = nn.Sequential(
            *conv_sequence(
                2 * out_channels // compression,
                out_channels,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=1,
                bias=(norm_layer is None),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        x = self.base_layer(x)
        x1, x2 = x.chunk(2, dim=1)
        return self.transition(torch.cat([x1, self.main(x2)], dim=1))


class DarknetBodyV4(nn.Sequential):
    def __init__(
        self,
        layout: list[tuple[int, int]],
        in_channels: int = 3,
        stem_channels: int = 32,
        num_features: int = 1,
        act_layer: nn.Module | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
        drop_layer: Callable[..., nn.Module] | None = None,
        conv_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()

        if act_layer is None:
            act_layer = nn.LeakyReLU(inplace=True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        in_chans = [stem_channels] + [layout_[0] for layout_ in layout[:-1]]

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
                    "stages",
                    nn.Sequential(*[
                        CSPStage(in_chans_, out_chans, num_blocks, act_layer, norm_layer, drop_layer, conv_layer)
                        for in_chans_, (out_chans, num_blocks) in zip(in_chans, layout, strict=True)
                    ]),
                ),
            ])
        )

        self.num_features: int = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        if self.num_features == 1:
            return super().forward(x)

        self.stem: nn.Sequential
        self.stages: nn.Sequential
        x = self.stem(x)
        features = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx >= (len(self.stages) - self.num_features):
                features.append(x)

        return features


class DarknetV4(nn.Sequential):
    def __init__(
        self,
        layout: list[tuple[int, int]],
        num_classes: int = 10,
        in_channels: int = 3,
        stem_channels: int = 32,
        num_features: int = 1,
        act_layer: nn.Module | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
        drop_layer: Callable[..., nn.Module] | None = None,
        conv_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__(
            OrderedDict([
                (
                    "features",
                    DarknetBodyV4(
                        layout,
                        in_channels,
                        stem_channels,
                        num_features,
                        act_layer,
                        norm_layer,
                        drop_layer,
                        conv_layer,
                    ),
                ),
                ("pool", GlobalAvgPool2d(flatten=True)),
                ("classifier", nn.Linear(layout[-1][0], num_classes)),
            ])
        )

        init_module(self, "leaky_relu")


def _darknet(
    checkpoint: Checkpoint | None,
    progress: bool,
    layout: list[tuple[int, int]],
    **kwargs: Any,
) -> DarknetV4:
    # Build the model
    model = DarknetV4(layout, **kwargs)
    return _configure_model(model, checkpoint, progress=progress)


class CSPDarknet53_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="cspdarknet53",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/cspdarknet53_224-7a69463a.pth",
        acc1=0.9450,
        acc5=0.9964,
        sha256="7a69463a4bd445beb6691dfd6ef7378efcf941f75d07d60034106ebedfcb82f8",
        size=106732575,
        num_params=26627434,
        commit="6e32c5b578711a2ef3731a8f8c61760ed9f03e58",
        train_args=_imagenette_recipe("cspdarknet53"),
    )
    DEFAULT = IMAGENETTE


def cspdarknet53(
    pretrained: bool = False,
    checkpoint: Checkpoint | None = None,
    progress: bool = True,
    **kwargs: Any,
) -> DarknetV4:
    """CSP-Darknet-53 from
    ["CSPNet: A New Backbone that can Enhance Learning Capability of CNN"](https://arxiv.org/pdf/1911.11929.pdf)

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of [`DarknetV4`][holocron.models.classification.darknetv4.DarknetV4]

    Returns:
        classification model

    ::: holocron.models.CSPDarknet53_Checkpoint
        options:
            heading_level: 4
            show_if_no_docstring: true
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        CSPDarknet53_Checkpoint.DEFAULT.value,
    )
    return _darknet(checkpoint, progress, [(64, 1), (128, 2), (256, 8), (512, 8), (1024, 4)], **kwargs)


class CSPDarknet53_Mish_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="cspdarknet53_mish",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/cspdarknet53_mish_224-1b660b3c.pth",
        acc1=0.9465,
        acc5=0.9969,
        sha256="1b660b3cb144195100c99ee3b9b863c37a5b5a59619c8de8c588b3d2af954b15",
        size=106737530,
        num_params=26627434,
        commit="6e32c5b578711a2ef3731a8f8c61760ed9f03e58",
        train_args=_imagenette_recipe("cspdarknet53_mish", batch_size=32),
    )
    DEFAULT = IMAGENETTE


def cspdarknet53_mish(
    pretrained: bool = False,
    checkpoint: Checkpoint | None = None,
    progress: bool = True,
    **kwargs: Any,
) -> DarknetV4:
    """Modified version of CSP-Darknet-53 from
    ["CSPNet: A New Backbone that can Enhance Learning Capability of CNN"](https://arxiv.org/pdf/1911.11929.pdf)
    with Mish as activation layer and DropBlock as regularization layer.

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of [`DarknetV4`][holocron.models.classification.darknetv4.DarknetV4]

    Returns:
        classification model

    ::: holocron.models.CSPDarknet53_Mish_Checkpoint
        options:
            heading_level: 4
            show_if_no_docstring: true
    """
    kwargs["act_layer"] = nn.Mish(inplace=True)
    kwargs["drop_layer"] = DropBlock2d

    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        CSPDarknet53_Mish_Checkpoint.DEFAULT.value,
    )
    return _darknet(checkpoint, progress, [(64, 1), (128, 2), (256, 8), (512, 8), (1024, 4)], **kwargs)
