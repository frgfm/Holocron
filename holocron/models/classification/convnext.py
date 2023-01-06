# Copyright (C) 2022-2023, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections import OrderedDict
from enum import Enum
from functools import partial
from typing import Any, Callable, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops.stochastic_depth import StochasticDepth

from holocron.nn import GlobalAvgPool2d

from ..checkpoints import (
    Checkpoint,
    Dataset,
    EvalResult,
    Evaluation,
    LoadingMeta,
    Metric,
    PreProcessing,
    TrainingRecipe,
    _handle_legacy_pretrained,
)
from ..presets import IMAGENETTE
from ..utils import conv_sequence, load_pretrained_params
from .resnet import _ResBlock

__all__ = [
    "ConvNeXt",
    "ConvNeXt_Atto_Checkpoint",
    "convnext_atto",
    "convnext_femto",
    "convnext_pico",
    "convnext_nano",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "convnext_xl",
]

_COMMON_SCRIPT = "references/classification/train.py"
_COMMON_PREPROCESSING = PreProcessing(input_shape=(3, 224, 224), mean=IMAGENETTE.mean, std=IMAGENETTE.std)


def _checkpoint(
    arch: str, url: str, acc1: float, acc5: float, sha256: str, size: int, num_params: int, commit: str, train_args: str
) -> Checkpoint:
    return Checkpoint(
        evaluation=Evaluation(
            dataset=Dataset.IMAGENETTE,
            results=[EvalResult(metric=Metric.TOP1_ACC, val=acc1), EvalResult(metric=Metric.TOP5_ACC, val=acc5)],
        ),
        meta=LoadingMeta(
            url=url, sha256=sha256, size=size, num_params=num_params, arch=arch, categories=IMAGENETTE.classes
        ),
        pre_processing=_COMMON_PREPROCESSING,
        recipe=TrainingRecipe(commit=commit, script=_COMMON_SCRIPT, args=train_args),
    )


class LayerNorm2d(nn.LayerNorm):
    """Compatibility wrapper of LayerNorm on 2D tensors"""

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class LayerScale(nn.Module):
    """Learnable channel-wise scaling"""

    def __init__(self, chans: int, scale: float = 1e-6) -> None:
        super().__init__()
        self.register_parameter("weight", nn.Parameter(scale * torch.ones(chans)))

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

        # Patchify-like stem
        _layers = conv_sequence(
            in_channels,
            planes[0],
            None,
            norm_layer,
            drop_layer,
            conv_layer,
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
        )

        block_idx = 0
        tot_blocks = sum(num_blocks)
        for _num_blocks, _planes, _oplanes in zip(num_blocks, planes, planes[1:] + [planes[-1]]):

            # adjust stochastic depth probability based on the depth of the stage block
            sd_probs = [stochastic_depth_prob * (block_idx + _idx) / (tot_blocks - 1.0) for _idx in range(_num_blocks)]
            _stage: List[nn.Module] = [
                Bottlenext(_planes, act_layer, norm_layer, drop_layer, stochastic_depth_prob=sd_prob)
                for _idx, sd_prob in zip(range(_num_blocks), sd_probs)
            ]
            if _planes != _oplanes:
                _stage.append(
                    nn.Sequential(
                        LayerNorm2d(_planes),
                        nn.Conv2d(_planes, _oplanes, kernel_size=2, stride=2),
                    )
                )
            _layers.append(nn.Sequential(*_stage))
            block_idx += _num_blocks

        super().__init__(
            OrderedDict(
                [
                    ("features", nn.Sequential(*_layers)),
                    ("pool", GlobalAvgPool2d(flatten=True)),
                    (
                        "head",
                        nn.Sequential(
                            nn.LayerNorm(planes[-1], eps=1e-6),
                            nn.Linear(planes[-1], num_classes),
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
    checkpoint: Union[Checkpoint, None],
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

    model.default_cfg = checkpoint  # type: ignore[assignment]
    # Load pretrained parameters
    if isinstance(checkpoint, Checkpoint):
        load_pretrained_params(model, checkpoint.meta.url, progress)

    return model


class ConvNeXt_Atto_Checkpoint(Enum):

    IMAGENETTE = _checkpoint(
        arch="convnext_atto",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/convnext_atto_224-4e92d4b1.pth",
        acc1=0.8161,
        acc5=0.9839,
        sha256="4e92d4b1e8389af36a12c5edc9af70601ddfae8603a9b349363aebbcb926024d",
        size=13535258,
        num_params=3377730,
        commit="695fbbb9e5449bc109c71058f756373ca3407c50",
        train_args=(
            "./imagenette2-320/ --arch convnext_atto --batch-size 64 --mixup-alpha 0.2 --amp --device 0"
            " --epochs 100 --lr 1e-3 --wb --label-smoothing 0.1"
        ),
    )
    DEFAULT = IMAGENETTE


def convnext_atto(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ConvNeXt:
    """ConvNeXt-Atto variant of Ross Wightman inspired by
    `"A ConvNet for the 2020s" <https://arxiv.org/pdf/2201.03545.pdf>`_

    Args:
        pretrained: If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress: If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.ConvNeXt_Atto_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        ConvNeXt_Atto_Checkpoint.DEFAULT,  # type: ignore[arg-type]
    )
    return _convnext(checkpoint, progress, [2, 2, 6, 2], [40, 80, 160, 320], **kwargs)


def convnext_femto(
    pretrained: bool = False, checkpoint: Union[Checkpoint, None] = None, progress: bool = True, **kwargs: Any
) -> ConvNeXt:
    """ConvNeXt-Femto variant of Ross Wightman inspired by
    `"A ConvNet for the 2020s" <https://arxiv.org/pdf/2201.03545.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """
    checkpoint = _handle_legacy_pretrained(pretrained, checkpoint, None)
    return _convnext(checkpoint, progress, [2, 2, 6, 2], [48, 96, 192, 384], **kwargs)


def convnext_pico(
    pretrained: bool = False, checkpoint: Union[Checkpoint, None] = None, progress: bool = True, **kwargs: Any
) -> ConvNeXt:
    """ConvNeXt-Pico variant of Ross Wightman inspired by
    `"A ConvNet for the 2020s" <https://arxiv.org/pdf/2201.03545.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """
    checkpoint = _handle_legacy_pretrained(pretrained, checkpoint, None)
    return _convnext(checkpoint, progress, [2, 2, 6, 2], [64, 128, 256, 512], **kwargs)


def convnext_nano(
    pretrained: bool = False, checkpoint: Union[Checkpoint, None] = None, progress: bool = True, **kwargs: Any
) -> ConvNeXt:
    """ConvNeXt-Nano variant of Ross Wightman inspired by
    `"A ConvNet for the 2020s" <https://arxiv.org/pdf/2201.03545.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """
    checkpoint = _handle_legacy_pretrained(pretrained, checkpoint, None)
    return _convnext(checkpoint, progress, [2, 2, 8, 2], [80, 160, 320, 640], **kwargs)


def convnext_tiny(
    pretrained: bool = False, checkpoint: Union[Checkpoint, None] = None, progress: bool = True, **kwargs: Any
) -> ConvNeXt:
    """ConvNeXt-T from
    `"A ConvNet for the 2020s" <https://arxiv.org/pdf/2201.03545.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """
    checkpoint = _handle_legacy_pretrained(pretrained, checkpoint, None)
    return _convnext(checkpoint, progress, [3, 3, 9, 3], [96, 192, 384, 768], **kwargs)


def convnext_small(
    pretrained: bool = False, checkpoint: Union[Checkpoint, None] = None, progress: bool = True, **kwargs: Any
) -> ConvNeXt:
    """ConvNeXt-S from
    `"A ConvNet for the 2020s" <https://arxiv.org/pdf/2201.03545.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """
    checkpoint = _handle_legacy_pretrained(pretrained, checkpoint, None)
    return _convnext(checkpoint, progress, [3, 3, 27, 3], [96, 192, 384, 768], **kwargs)


def convnext_base(
    pretrained: bool = False, checkpoint: Union[Checkpoint, None] = None, progress: bool = True, **kwargs: Any
) -> ConvNeXt:
    """ConvNeXt-B from
    `"A ConvNet for the 2020s" <https://arxiv.org/pdf/2201.03545.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """
    checkpoint = _handle_legacy_pretrained(pretrained, checkpoint, None)
    return _convnext(checkpoint, progress, [3, 3, 27, 3], [128, 256, 512, 1024], **kwargs)


def convnext_large(
    pretrained: bool = False, checkpoint: Union[Checkpoint, None] = None, progress: bool = True, **kwargs: Any
) -> ConvNeXt:
    """ConvNeXt-L from
    `"A ConvNet for the 2020s" <https://arxiv.org/pdf/2201.03545.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """
    checkpoint = _handle_legacy_pretrained(pretrained, checkpoint, None)
    return _convnext(checkpoint, progress, [3, 3, 27, 3], [192, 384, 768, 1536], **kwargs)


def convnext_xl(
    pretrained: bool = False, checkpoint: Union[Checkpoint, None] = None, progress: bool = True, **kwargs: Any
) -> ConvNeXt:
    """ConvNeXt-XL from
    `"A ConvNet for the 2020s" <https://arxiv.org/pdf/2201.03545.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """
    checkpoint = _handle_legacy_pretrained(pretrained, checkpoint, None)
    return _convnext(checkpoint, progress, [3, 3, 27, 3], [256, 512, 1024, 2048], **kwargs)
