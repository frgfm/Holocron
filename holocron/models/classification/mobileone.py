# Copyright (C) 2022-2023, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections import OrderedDict
from enum import Enum
from typing import Any, Callable, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from holocron.nn import GlobalAvgPool2d, init

from ..checkpoints import Checkpoint, _handle_legacy_pretrained
from ..utils import _checkpoint, _configure_model, conv_sequence, fuse_conv_bn

__all__ = [
    "MobileOne_S0_Checkpoint",
    "mobileone_s0",
    "MobileOne_S1_Checkpoint",
    "mobileone_s1",
    "MobileOne_S2_Checkpoint",
    "mobileone_s2",
    "MobileOne_S3_Checkpoint",
    "mobileone_s3",
]


class DepthConvBlock(nn.ModuleList):
    """Implements a reparametrizeable depth-wise convolutional block"""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        stride: int = 1,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        _layers = [norm_layer(channels)] if stride == 1 else []
        _layers.append(
            nn.Sequential(
                *conv_sequence(channels, channels, kernel_size=1, stride=stride, norm_layer=norm_layer, groups=channels)
            ),
        )
        _layers.extend(
            [
                nn.Sequential(
                    *conv_sequence(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=1,
                        stride=stride,
                        norm_layer=norm_layer,
                        groups=channels,
                    )
                )
                for _ in range(num_blocks)
            ]
        )
        super().__init__(_layers)

    def forward(self, x: Tensor) -> Tensor:
        return sum(mod(x) for mod in self)

    def reparametrize(self) -> nn.Conv2d:
        _chans = self[1][0].in_channels
        # Fuse the conv & BN
        _conv = nn.Conv2d(_chans, _chans, 3, padding=1, bias=True, stride=self[1][0].stride, groups=_chans).to(
            self[1][0].weight.data.device
        )
        _conv.weight.data.zero_()
        _conv.bias.data.zero_()  # type: ignore[union-attr]
        bn_idx, conv1_idx, branch_idx = (None, 0, 1) if isinstance(self[0], nn.Sequential) else (0, 1, 2)
        # BN branch
        if isinstance(bn_idx, int):
            bn = self[bn_idx]
            scale_factor = bn.weight.data / torch.sqrt(bn.running_var + bn.eps)
            _conv.bias.data += bn.bias.data - scale_factor * bn.running_mean  # type: ignore[union-attr]
            _conv.weight.data[..., 1, 1] += scale_factor.unsqueeze(1)

        # Conv 1x1 branch
        _k, _b = fuse_conv_bn(self[conv1_idx][0], self[conv1_idx][1])
        _conv.bias.data += _b  # type: ignore[union-attr]
        _conv.weight.data[..., 1:2, 1:2] += _k

        # Conv 3x3 branches
        for mod_idx in range(branch_idx, len(self)):
            _k, _b = fuse_conv_bn(self[mod_idx][0], self[mod_idx][1])
            _conv.bias.data += _b  # type: ignore[union-attr]
            _conv.weight.data += _k

        return _conv


class PointConvBlock(nn.ModuleList):
    """Implements a reparametrizeable point-wise convolutional block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        _layers = [norm_layer(out_channels)] if out_channels == in_channels else []
        _layers.extend(
            [
                nn.Sequential(*conv_sequence(in_channels, out_channels, kernel_size=1, norm_layer=norm_layer))
                for _ in range(num_blocks)
            ]
        )
        super().__init__(_layers)

    def forward(self, x: Tensor) -> Tensor:
        return sum(mod(x) for mod in self)

    def reparametrize(self) -> nn.Conv2d:
        seq_idx = 1 if not isinstance(self[0], nn.Sequential) else 0
        in_chans, out_chans = self[seq_idx][0].in_channels, self[seq_idx][0].out_channels
        # Fuse the conv & BN
        _conv = nn.Conv2d(in_chans, out_chans, 1, bias=True).to(self[seq_idx][0].weight.data.device)
        _conv.weight.data.zero_()
        _conv.bias.data.zero_()  # type: ignore[union-attr]
        bn_idx, branch_idx = (None, 0) if isinstance(self[0], nn.Sequential) else (0, 1)
        # BN branch
        if isinstance(bn_idx, int):
            bn = self[bn_idx]
            scale_factor = bn.weight.data / torch.sqrt(bn.running_var + bn.eps)
            _conv.bias.data += bn.bias.data - scale_factor * bn.running_mean  # type: ignore[union-attr]
            for chan_idx in range(_conv.weight.data.shape[0]):
                _conv.weight.data[chan_idx, chan_idx] += scale_factor[chan_idx]

        # Conv branches
        for mod_idx in range(branch_idx, len(self)):
            _k, _b = fuse_conv_bn(self[mod_idx][0], self[mod_idx][1])
            _conv.bias.data += _b  # type: ignore[union-attr]
            _conv.weight.data += _k

        return _conv


class MobileOneBlock(nn.Sequential):
    """Implements the bottleneck block of MobileOne"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        overparam_factor: int = 1,
        stride: int = 1,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        super().__init__(
            DepthConvBlock(in_channels, overparam_factor, stride, norm_layer),
            act_layer,
            PointConvBlock(in_channels, out_channels, overparam_factor, norm_layer),
            act_layer,
        )

    def reparametrize(self) -> None:
        """Reparametrize the depth-wise & point-wise blocks"""
        self[0] = self[0].reparametrize()
        self[2] = self[2].reparametrize()


class MobileOne(nn.Sequential):
    def __init__(
        self,
        num_blocks: List[int],
        width_multipliers: List[float],
        overparam_factor: int = 1,
        num_classes: int = 10,
        in_channels: int = 3,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        base_planes = [64, 128, 256, 512]
        planes = [int(round(mult * chans)) for mult, chans in zip(width_multipliers, base_planes)]

        in_planes = min(64, planes[0])
        # Stem
        _layers: List[nn.Module] = [MobileOneBlock(in_channels, in_planes, overparam_factor, 2, act_layer, norm_layer)]

        # Consecutive convolutional blocks
        for _num_blocks, _planes in zip(num_blocks, planes):
            # Stride & channel changes
            _stage = [MobileOneBlock(in_planes, _planes, overparam_factor, 2, act_layer, norm_layer)]
            # Depth
            _stage.extend(
                [
                    MobileOneBlock(_planes, _planes, overparam_factor, 1, act_layer, norm_layer)
                    for _ in range(_num_blocks - 1)
                ]
            )
            in_planes = _planes

            _layers.append(nn.Sequential(*_stage))

        super().__init__(
            OrderedDict(
                [
                    ("features", nn.Sequential(*_layers)),
                    ("pool", GlobalAvgPool2d(flatten=True)),
                    ("head", nn.Linear(in_planes, num_classes)),
                ]
            )
        )

        # Init all layers
        init.init_module(self, nonlinearity="relu")

    def reparametrize(self) -> None:
        """Reparametrize the block by fusing convolutions and BN in each branch, then fusing all branches"""
        self.features: nn.Sequential
        # Stem
        self.features[0].reparametrize()
        for stage in self.features[1:]:
            for block in stage:
                block.reparametrize()


def _mobileone(
    checkpoint: Union[Checkpoint, None],
    progress: bool,
    width_multipliers: List[float],
    overparam_factor: int,
    **kwargs: Any,
) -> MobileOne:
    # Build the model
    model = MobileOne([2, 8, 10, 1], width_multipliers, overparam_factor, **kwargs)
    return _configure_model(model, checkpoint, progress=progress)


class MobileOne_S0_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="mobileone_s0",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/mobileone_s0_224-9ddd1fe9.pth",
        acc1=0.8808,
        acc5=0.9883,
        sha256="9ddd1fe9d6c0a73d3c4d51d3c967a8a27ff5e545705afc557b4d4ac0f34395cb",
        size=17708169,
        num_params=4277991,
        commit="d4a59999179b42fc0d3058ac6b76cc41f49dd56e",
        train_args=(
            "./imagenette2-320/ --arch mobileone_s0 --batch-size 64 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENETTE


def mobileone_s0(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> MobileOne:
    """MobileOne-S0 from
    `"An Improved One millisecond Mobile Backbone" <https://arxiv.org/pdf/2206.04040.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _mobileone

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.MobileOne_S0_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        MobileOne_S0_Checkpoint.DEFAULT,  # type: ignore[arg-type]
    )
    return _mobileone(checkpoint, progress, [0.75, 1.0, 1.0, 2.0], 4, **kwargs)


class MobileOne_S1_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="mobileone_s1",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/mobileone_s1_224-d4ec5433.pth",
        acc1=0.9126,
        acc5=0.9918,
        sha256="d4ec5433cff3d55d562b7a35fc0c95568ff8f4591bf822dd3e699535bdff90eb",
        size=14594817,
        num_params=3555188,
        commit="d4a59999179b42fc0d3058ac6b76cc41f49dd56e",
        train_args=(
            "./imagenette2-320/ --arch mobileone_s1 --batch-size 64 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENETTE


def mobileone_s1(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> MobileOne:
    """MobileOne-S1 from
    `"An Improved One millisecond Mobile Backbone" <https://arxiv.org/pdf/2206.04040.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _mobileone

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.MobileOne_S1_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        MobileOne_S1_Checkpoint.DEFAULT,  # type: ignore[arg-type]
    )
    return _mobileone(checkpoint, progress, [1.5, 1.5, 2.0, 2.5], 1, **kwargs)


class MobileOne_S2_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="mobileone_s2",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/mobileone_s2_224-b748859c.pth",
        acc1=0.9131,
        acc5=0.9921,
        sha256="b748859c45a636ea22f0f68a3b7e75e5fb6ffb31178a5a3137931a21b4c41697",
        size=23866479,
        num_params=5854324,
        commit="d4a59999179b42fc0d3058ac6b76cc41f49dd56e",
        train_args=(
            "./imagenette2-320/ --arch mobileone_s2 --batch-size 64 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENETTE


def mobileone_s2(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> MobileOne:
    """MobileOne-S2 from
    `"An Improved One millisecond Mobile Backbone" <https://arxiv.org/pdf/2206.04040.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _mobileone

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.MobileOne_S2_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        MobileOne_S2_Checkpoint.DEFAULT,  # type: ignore[arg-type]
    )
    return _mobileone(checkpoint, progress, [1.5, 2.0, 2.5, 4.0], 1, **kwargs)


class MobileOne_S3_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="mobileone_s3",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/mobileone_s3_224-7f357baf.pth",
        acc1=0.9106,
        acc5=0.9931,
        sha256="7f357baf0754136b4a02e7aec4129874db93ee462f43588b77def730db0b2bca",
        size=33080943,
        num_params=8140276,
        commit="d4a59999179b42fc0d3058ac6b76cc41f49dd56e",
        train_args=(
            "./imagenette2-320/ --arch mobileone_s3 --batch-size 64 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENETTE


def mobileone_s3(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> MobileOne:
    """MobileOne-S3 from
    `"An Improved One millisecond Mobile Backbone" <https://arxiv.org/pdf/2206.04040.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _mobileone

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.MobileOne_S3_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        MobileOne_S3_Checkpoint.DEFAULT,  # type: ignore[arg-type]
    )
    return _mobileone(checkpoint, progress, [2.0, 2.5, 3.0, 4.0], 1, **kwargs)
