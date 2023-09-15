# Copyright (C) 2019-2023, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections import OrderedDict
from enum import Enum
from math import ceil
from typing import Any, Callable, Optional, Union

import torch.nn as nn

from holocron.nn import GlobalAvgPool2d, init

from ..checkpoints import (
    Checkpoint,
    Dataset,
    Evaluation,
    LoadingMeta,
    Metric,
    PreProcessing,
    TrainingRecipe,
    _handle_legacy_pretrained,
)
from ..presets import IMAGENET, IMAGENETTE
from ..utils import _configure_model, conv_sequence

__all__ = [
    "SEBlock",
    "ReXBlock",
    "ReXNet",
    "ReXNet1_0x_Checkpoint",
    "rexnet1_0x",
    "ReXNet1_3x_Checkpoint",
    "rexnet1_3x",
    "ReXNet1_5x_Checkpoint",
    "rexnet1_5x",
    "ReXNet2_0x_Checkpoint",
    "rexnet2_0x",
    "ReXNet2_2x_Checkpoint",
    "rexnet2_2x",
]


class SEBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        se_ratio: int = 12,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.pool = GlobalAvgPool2d(flatten=False)
        self.conv = nn.Sequential(
            *conv_sequence(
                channels,
                channels // se_ratio,
                act_layer,
                norm_layer,
                drop_layer,
                kernel_size=1,
                stride=1,
                bias=(norm_layer is None),
            ),
            *conv_sequence(channels // se_ratio, channels, nn.Sigmoid(), None, drop_layer, kernel_size=1, stride=1),
        )

    def forward(self, x):

        y = self.pool(x)
        y = self.conv(y)
        return x * y


class ReXBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        t: int,
        stride: int,
        use_se: bool = True,
        se_ratio: int = 12,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if act_layer is None:
            act_layer = nn.ReLU6(inplace=True)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels
        self.out_channels = channels

        _layers = []
        if t != 1:
            dw_channels = in_channels * t
            _layers.extend(
                conv_sequence(
                    in_channels,
                    dw_channels,
                    nn.SiLU(inplace=True),
                    norm_layer,
                    drop_layer,
                    kernel_size=1,
                    stride=1,
                    bias=(norm_layer is None),
                )
            )
        else:
            dw_channels = in_channels

        _layers.extend(
            conv_sequence(
                dw_channels,
                dw_channels,
                None,
                norm_layer,
                drop_layer,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=(norm_layer is None),
                groups=dw_channels,
            )
        )

        if use_se:
            _layers.append(SEBlock(dw_channels, se_ratio, act_layer, norm_layer, drop_layer))

        _layers.append(act_layer)
        _layers.extend(
            conv_sequence(
                dw_channels, channels, None, norm_layer, drop_layer, kernel_size=1, stride=1, bias=(norm_layer is None)
            )
        )
        self.conv = nn.Sequential(*_layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_shortcut:
            out[:, : self.in_channels] += x

        return out


class ReXNet(nn.Sequential):
    def __init__(
        self,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        num_classes: int = 1000,
        in_channels: int = 3,
        in_planes: int = 16,
        final_planes: int = 180,
        use_se: bool = True,
        se_ratio: int = 12,
        dropout_ratio: float = 0.2,
        bn_momentum: float = 0.9,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """Mostly adapted from https://github.com/clovaai/rexnet/blob/master/rexnetv1.py"""
        super().__init__()

        if act_layer is None:
            act_layer = nn.SiLU(inplace=True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        num_blocks = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]
        num_blocks = [ceil(element * depth_mult) for element in num_blocks]
        strides = sum([[element] + [1] * (num_blocks[idx] - 1) for idx, element in enumerate(strides)], [])
        depth = sum(num_blocks)

        stem_channel = 32 / width_mult if width_mult < 1.0 else 32
        inplanes = in_planes / width_mult if width_mult < 1.0 else in_planes

        # The following channel configuration is a simple instance to make each layer become an expand layer
        chans = [int(round(width_mult * stem_channel))]
        chans.extend([int(round(width_mult * (inplanes + idx * final_planes / depth))) for idx in range(depth)])

        ses = [False] * (num_blocks[0] + num_blocks[1]) + [use_se] * sum(num_blocks[2:])

        _layers = conv_sequence(
            in_channels,
            chans[0],
            act_layer,
            norm_layer,
            drop_layer,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=(norm_layer is None),
        )

        t = 1
        for in_c, c, s, se in zip(chans[:-1], chans[1:], strides, ses):
            _layers.append(ReXBlock(in_channels=in_c, channels=c, t=t, stride=s, use_se=se, se_ratio=se_ratio))
            t = 6

        pen_channels = int(width_mult * 1280)
        _layers.extend(
            conv_sequence(
                chans[-1],
                pen_channels,
                act_layer,
                norm_layer,
                drop_layer,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=(norm_layer is None),
            )
        )

        super().__init__(
            OrderedDict(
                [
                    ("features", nn.Sequential(*_layers)),
                    ("pool", GlobalAvgPool2d(flatten=True)),
                    ("head", nn.Sequential(nn.Dropout(dropout_ratio), nn.Linear(pen_channels, num_classes))),
                ]
            )
        )

        # Init all layers
        init.init_module(self, nonlinearity="relu")


def _rexnet(
    checkpoint: Union[Checkpoint, None],
    progress: bool,
    width_mult: float,
    depth_mult: float,
    **kwargs: Any,
) -> ReXNet:
    # Build the model
    model = ReXNet(width_mult, depth_mult, **kwargs)
    return _configure_model(model, checkpoint, progress=progress)


def _checkpoint(
    arch: str,
    url: str,
    acc1: float,
    acc5: float,
    sha256: str,
    size: int,
    num_params: int,
    commit: Union[str, None] = None,
    train_args: Union[str, None] = None,
    dataset: Dataset = Dataset.IMAGENETTE,
) -> Checkpoint:
    preset = IMAGENETTE if dataset == Dataset.IMAGENETTE else IMAGENET
    return Checkpoint(
        evaluation=Evaluation(
            dataset=dataset,
            results={Metric.TOP1_ACC: acc1, Metric.TOP5_ACC: acc5},
        ),
        meta=LoadingMeta(
            url=url, sha256=sha256, size=size, num_params=num_params, arch=arch, categories=preset.classes
        ),
        pre_processing=PreProcessing(input_shape=(3, 224, 224), mean=preset.mean, std=preset.std),
        recipe=TrainingRecipe(commit=commit, script="references/classification/train.py", args=train_args),
    )


class ReXNet1_0x_Checkpoint(Enum):
    # Porting of Ross Wightman's weights
    IMAGENET1K = _checkpoint(
        arch="rexnet1_0x",
        url="https://github.com/frgfm/Holocron/releases/download/v0.1.2/rexnet1_0x_224-ab7b9733.pth",
        dataset=Dataset.IMAGENET1K,
        acc1=0.7786,
        acc5=0.93870,
        sha256="ab7b973341a59832099f6ee2a41eb51121b287ad4adaae8b2cd8dd92ef058f01",
        size=14351299,
        num_params=4796186,
    )

    IMAGENETTE = _checkpoint(
        arch="rexnet1_0x",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/rexnet1_0x_224-7c19fd53.pth",
        acc1=0.9439,
        acc5=0.9962,
        sha256="7c19fd53a5433927e9b4b22fa9cb0833eb1e4c3254b4079b6818fce650a77943",
        size=14351299,
        num_params=3527996,
        commit="d4a59999179b42fc0d3058ac6b76cc41f49dd56e",
        train_args=(
            "./imagenette2-320/ --arch rexnet1_0x --batch-size 64 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENET1K


def rexnet1_0x(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ReXNet:
    """ReXNet-1.0x from
    `"ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"
    <https://arxiv.org/pdf/2007.00992.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.ReXNet1_0x_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        ReXNet1_0x_Checkpoint.DEFAULT.value,
    )
    return _rexnet(checkpoint, progress, 1, 1, **kwargs)


class ReXNet1_3x_Checkpoint(Enum):
    # Porting of Ross Wightman's weights
    IMAGENET1K = _checkpoint(
        arch="rexnet1_3x",
        url="https://github.com/frgfm/Holocron/releases/download/v0.1.2/rexnet1_3x_224-95479104.pth",
        dataset=Dataset.IMAGENET1K,
        acc1=0.7950,
        acc5=0.9468,
        sha256="95479104024ce294abbdd528df62bd1a23e67a9db2956e1d6cdb9a9759dc1c69",
        size=14351299,
        num_params=7556198,
    )

    IMAGENETTE = _checkpoint(
        arch="rexnet1_3x",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/rexnet1_3x_224-cf85ae91.pth",
        acc1=0.9488,
        acc5=0.9939,
        sha256="cf85ae919cbc9484f9fa150106451f68d2e84c73f1927a1b80aeeaa243ccd65b",
        size=23920480,
        num_params=5907848,
        commit="d4a59999179b42fc0d3058ac6b76cc41f49dd56e",
        train_args=(
            "./imagenette2-320/ --arch rexnet1_3x --batch-size 64 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENET1K


def rexnet1_3x(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ReXNet:
    """ReXNet-1.3x from
    `"ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"
    <https://arxiv.org/pdf/2007.00992.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.ReXNet1_3x_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        ReXNet1_3x_Checkpoint.DEFAULT.value,
    )
    return _rexnet(checkpoint, progress, 1.3, 1, **kwargs)


class ReXNet1_5x_Checkpoint(Enum):
    # Porting of Ross Wightman's weights
    IMAGENET1K = _checkpoint(
        arch="rexnet1_5x",
        url="https://github.com/frgfm/Holocron/releases/download/v0.1.2/rexnet1_5x_224-c42a16ac.pth",
        dataset=Dataset.IMAGENET1K,
        acc1=0.8031,
        acc5=0.9517,
        sha256="c42a16ac73470d64852b8317ba9e875c833595a90a086b90490a696db9bb6a96",
        size=14351299,
        num_params=9727562,
    )

    IMAGENETTE = _checkpoint(
        arch="rexnet1_5x",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/rexnet1_5x_224-4b9d7a59.pth",
        acc1=0.9447,
        acc5=0.9962,
        sha256="4b9d7a5901da6c2b9386987a6120bc86089d84df7727e43b78a4dfe2fc1c719a",
        size=31625286,
        num_params=7825772,
        commit="d4a59999179b42fc0d3058ac6b76cc41f49dd56e",
        train_args=(
            "./imagenette2-320/ --arch rexnet1_5x --batch-size 64 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENET1K


def rexnet1_5x(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ReXNet:
    """ReXNet-1.5x from
    `"ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"
    <https://arxiv.org/pdf/2007.00992.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.ReXNet1_5x_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        ReXNet1_5x_Checkpoint.DEFAULT.value,
    )
    return _rexnet(checkpoint, progress, 1.5, 1, **kwargs)


class ReXNet2_0x_Checkpoint(Enum):
    # Porting of Ross Wightman's weights
    IMAGENET1K = _checkpoint(
        arch="rexnet2_0x",
        url="https://github.com/frgfm/Holocron/releases/download/v0.1.2/rexnet2_0x_224-c8802402.pth",
        dataset=Dataset.IMAGENET1K,
        acc1=0.8031,
        acc5=0.9517,
        sha256="c8802402442551c77fe3874f84d4d7eb1bd67cce274375db11a869ed074a1089",
        size=14351299,
        num_params=16365244,
    )

    IMAGENETTE = _checkpoint(
        arch="rexnet2_0x",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/rexnet2_0x_224-3f00641e.pth",
        acc1=0.9524,
        acc5=0.9957,
        sha256="3f00641e48a6d1d3c9794534eb372467e0730700498933c9e79e60c838671d13",
        size=55724412,
        num_params=13829854,
        commit="d4a59999179b42fc0d3058ac6b76cc41f49dd56e",
        train_args=(
            "./imagenette2-320/ --arch rexnet2_0x --batch-size 32 --grad-acc 2 --mixup-alpha 0.2 --amp --device 0"
            " --epochs 100 --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176"
            " --val-resize-size 232 --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENET1K


def rexnet2_0x(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ReXNet:
    """ReXNet-2.0x from
    `"ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"
    <https://arxiv.org/pdf/2007.00992.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.ReXNet2_0x_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        ReXNet2_0x_Checkpoint.DEFAULT.value,
    )
    return _rexnet(checkpoint, progress, 2, 1, **kwargs)


class ReXNet2_2x_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="rexnet2_2x",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/rexnet2_2x_224-b23b2847.pth",
        acc1=0.9544,
        acc5=0.9946,
        sha256="b23b28475329e413bfb491503460db8f47a838ec8dcdc5d13ade6f40ee5841a6",
        size=67217933,
        num_params=16694966,
        commit="d4a59999179b42fc0d3058ac6b76cc41f49dd56e",
        train_args=(
            "./imagenette2-320/ --arch rexnet2_2x --batch-size 32 --grad-acc 2 --mixup-alpha 0.2 --amp --device 0"
            " --epochs 100 --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176"
            " --val-resize-size 232 --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENETTE


def rexnet2_2x(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ReXNet:
    """ReXNet-2.2x from
    `"ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"
    <https://arxiv.org/pdf/2007.00992.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.ReXNet2_2x_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        ReXNet2_2x_Checkpoint.DEFAULT.value,
    )
    return _rexnet(checkpoint, progress, 2.2, 1, **kwargs)
