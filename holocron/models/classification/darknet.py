# Copyright (C) 2020-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

import torch.nn as nn

from holocron.nn import GlobalAvgPool2d
from holocron.nn.init import init_module

from ..presets import IMAGENETTE
from ..utils import conv_sequence, load_pretrained_params

__all__ = ["DarknetV1", "darknet24"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "darknet24": {
        **IMAGENETTE,
        "input_shape": (3, 224, 224),
        "url": "https://github.com/frgfm/Holocron/releases/download/v0.1.3/darknet24_224-816d72cb.pt",
    },
}


class DarknetBodyV1(nn.Sequential):
    def __init__(
        self,
        layout: List[List[int]],
        in_channels: int = 3,
        stem_channels: int = 64,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:

        if act_layer is None:
            act_layer = nn.LeakyReLU(0.1, inplace=True)

        in_chans = [stem_channels] + [_layout[-1] for _layout in layout[:-1]]

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
                                kernel_size=7,
                                padding=3,
                                stride=2,
                                bias=(norm_layer is None),
                            )
                        ),
                    ),
                    (
                        "layers",
                        nn.Sequential(
                            *[
                                self._make_layer([_in_chans] + planes, act_layer, norm_layer, drop_layer, conv_layer)
                                for _in_chans, planes in zip(in_chans, layout)
                            ]
                        ),
                    ),
                ]
            )
        )

    @staticmethod
    def _make_layer(
        planes: List[int],
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> nn.Sequential:
        _layers: List[nn.Module] = [nn.MaxPool2d(2)]
        k1 = True
        for in_planes, out_planes in zip(planes[:-1], planes[1:]):
            _layers.extend(
                conv_sequence(
                    in_planes,
                    out_planes,
                    act_layer,
                    norm_layer,
                    drop_layer,
                    conv_layer,
                    kernel_size=3 if out_planes > in_planes else 1,
                    padding=1 if out_planes > in_planes else 0,
                    bias=(norm_layer is None),
                )
            )
            k1 = not k1

        return nn.Sequential(*_layers)


class DarknetV1(nn.Sequential):
    def __init__(
        self,
        layout: List[List[int]],
        num_classes: int = 10,
        in_channels: int = 3,
        stem_channels: int = 64,
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
                        DarknetBodyV1(
                            layout, in_channels, stem_channels, act_layer, norm_layer, drop_layer, conv_layer
                        ),
                    ),
                    ("pool", GlobalAvgPool2d(flatten=True)),
                    ("classifier", nn.Linear(layout[2][-1], num_classes)),
                ]
            )
        )

        init_module(self, "leaky_relu")


def _darknet(arch: str, pretrained: bool, progress: bool, layout: List[List[int]], **kwargs: Any) -> DarknetV1:
    # Build the model
    model = DarknetV1(layout, **kwargs)
    model.default_cfg = default_cfgs[arch]  # type: ignore[assignment]
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"], progress)

    return model


def darknet24(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DarknetV1:
    """Darknet-24 from
    `"You Only Look Once: Unified, Real-Time Object Detection" <https://pjreddie.com/media/files/papers/yolo_1.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _darknet(
        "darknet24",
        pretrained,
        progress,
        [[192], [128, 256, 256, 512], [*([256, 512] * 4), 512, 1024], [512, 1024] * 2],
        **kwargs,
    )
