# Copyright (C) 2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops.stochastic_depth import StochasticDepth

from holocron.nn import GlobalAvgPool2d

from ..presets import IMAGENETTE
from ..utils import conv_sequence, load_pretrained_params
from .resnet import _ResBlock

__all__ = ["convnext_micro", "convnext_tiny", "convnext_small", "convnext_base", "convnext_large", "convnext_xl"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "convnext_micro": {
        **IMAGENETTE,
        "input_shape": (3, 224, 224),
        "url": None,
    },
    "convnext_tiny": {
        **IMAGENETTE,
        "input_shape": (3, 224, 224),
        "url": None,
    },
    "convnext_small": {
        **IMAGENETTE,
        "input_shape": (3, 224, 224),
        "url": None,
    },
    "convnext_base": {
        **IMAGENETTE,
        "input_shape": (3, 224, 224),
        "url": None,
    },
    "convnext_large": {
        **IMAGENETTE,
        "input_shape": (3, 224, 224),
        "url": None,
    },
    "convnext_xl": {
        **IMAGENETTE,
        "input_shape": (3, 224, 224),
        "url": None,
    },
}


class LayerNorm2d(nn.LayerNorm):
    """Compatibility wrapper of LayerNorm on 2D tensors"""

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class LayerScale(nn.Module):
    """Learnable channel-wise scaling"""

    def __init__(self, chans: int, scale: float = 1e-6) -> None:
        super().__init__()
        self.register_parameter("weight", nn.Parameter(scale * torch.ones(chans)))  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight.reshape(1, -1, *((1,) * (x.ndim - 2)))  # type: ignore[operator]


class Bottlenext(_ResBlock):
    def __init__(
        self,
        inplanes: int,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        chan_expansion: int = 4,
        stochastic_depth_prob: float = 0.1,
        layer_scale: float = 1e-6,
    ) -> None:
        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)
        if act_layer is None:
            act_layer = nn.GELU()

        super().__init__(
            [
                # Depth-conv (groups = in_channels): spatial awareness
                *conv_sequence(
                    inplanes,
                    inplanes,
                    None,
                    norm_layer,
                    drop_layer,
                    kernel_size=7,
                    padding=3,
                    stride=1,
                    bias=True,
                    groups=inplanes,
                ),
                # 1x1 conv: channel awareness
                *conv_sequence(
                    inplanes,
                    inplanes * chan_expansion,
                    act_layer,
                    None,
                    drop_layer,
                    kernel_size=1,
                    stride=1,
                    bias=True,
                ),
                # 1x1 conv: channel mapping
                *conv_sequence(
                    inplanes * chan_expansion,
                    inplanes,
                    None,
                    None,
                    drop_layer,
                    kernel_size=1,
                    stride=1,
                    bias=True,
                ),
                LayerScale(inplanes, layer_scale),
                StochasticDepth(stochastic_depth_prob, "row"),
            ],
            None,
            None,
        )


class ConvNeXt(nn.Sequential):
    def __init__(
        self,
        num_blocks: List[int],
        planes: List[int],
        num_classes: int = 10,
        in_channels: int = 3,
        stem_channels: int = 96,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        stochastic_depth_prob: float = 0.0,
    ) -> None:

        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)
        if act_layer is None:
            act_layer = nn.GELU()
        self.dilation = 1

        in_planes = stem_channels
        # Patchify-like stem
        _layers = conv_sequence(
            in_channels,
            in_planes,
            None,
            norm_layer,
            drop_layer,
            conv_layer,
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
        )

        stage_idx, block_idx = 0, 0
        tot_blocks = sum(num_blocks)
        for _num_blocks, _planes in zip(num_blocks, planes):

            # adjust stochastic depth probability based on the depth of the stage block
            sd_probs = [stochastic_depth_prob * (block_idx + _idx) / (tot_blocks - 1.0) for _idx in range(_num_blocks)]
            _stage: List[nn.Module] = [
                Bottlenext(in_planes, act_layer, norm_layer, drop_layer, stochastic_depth_prob=sd_prob)
                for _idx, sd_prob in zip(range(_num_blocks), sd_probs)
            ]
            if stage_idx < len(num_blocks) - 1:
                _stage.append(
                    nn.Sequential(
                        LayerNorm2d(in_planes),
                        nn.Conv2d(in_planes, _planes, kernel_size=2, stride=2),
                    )
                )
                in_planes = _planes
            _layers.append(nn.Sequential(*_stage))
            stage_idx += 1
            block_idx += _num_blocks

        super().__init__(
            OrderedDict(
                [
                    ("features", nn.Sequential(*_layers)),
                    ("pool", GlobalAvgPool2d(flatten=True)),
                    (
                        "head",
                        nn.Sequential(
                            nn.LayerNorm(in_planes, eps=1e-6),
                            nn.Linear(in_planes, num_classes),
                        ),
                    ),
                ]
            )
        )

        # Init all layers
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def _convnext(
    arch: str,
    pretrained: bool,
    progress: bool,
    num_blocks: List[int],
    out_chans: List[int],
    **kwargs: Any,
) -> ConvNeXt:
    # Build the model
    model = ConvNeXt(
        num_blocks,
        out_chans,
        **kwargs,
    )

    model.default_cfg = default_cfgs[arch]  # type: ignore[assignment]
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"], progress)

    return model


def convnext_micro(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    """Smaller variation of ConvNeXt from
    `"A ConvNet for the 2020s" <https://arxiv.org/pdf/2201.03545.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _convnext(
        "convnext_micro",
        pretrained,
        progress,
        [3, 4, 6, 3],
        [64, 128, 256, 512],
        stem_channels=64,
        **kwargs,
    )


def convnext_tiny(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    """ConvNeXt-T from
    `"A ConvNet for the 2020s" <https://arxiv.org/pdf/2201.03545.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _convnext("convnext_tiny", pretrained, progress, [3, 3, 9, 3], [192, 384, 768, 768], **kwargs)


def convnext_small(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    """ConvNeXt-S from
    `"A ConvNet for the 2020s" <https://arxiv.org/pdf/2201.03545.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _convnext("convnext_tiny", pretrained, progress, [3, 3, 27, 3], [96, 192, 384, 768], **kwargs)


def convnext_base(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    """ConvNeXt-B from
    `"A ConvNet for the 2020s" <https://arxiv.org/pdf/2201.03545.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _convnext("convnext_base", pretrained, progress, [3, 3, 27, 3], [128, 256, 512, 1024], **kwargs)


def convnext_large(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    """ConvNeXt-L from
    `"A ConvNet for the 2020s" <https://arxiv.org/pdf/2201.03545.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _convnext("convnext_large", pretrained, progress, [3, 3, 27, 3], [192, 384, 768, 1536], **kwargs)


def convnext_xl(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    """ConvNeXt-XL from
    `"A ConvNet for the 2020s" <https://arxiv.org/pdf/2201.03545.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _convnext("convnext_xl", pretrained, progress, [3, 3, 27, 3], [256, 512, 1024, 2048], **kwargs)
