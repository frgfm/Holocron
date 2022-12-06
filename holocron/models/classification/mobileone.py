# Copyright (C) 2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from holocron.nn import GlobalAvgPool2d, init

from ..presets import IMAGENETTE
from ..utils import conv_sequence, fuse_conv_bn, load_pretrained_params

__all__ = ["mobileone_s0", "mobileone_s1", "mobileone_s2", "mobileone_s3"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "mobileone_s0": {
        **IMAGENETTE,
        "input_shape": (3, 224, 224),
        "url": None,
    },
    "mobileone_s1": {
        **IMAGENETTE,
        "input_shape": (3, 224, 224),
        "url": None,
    },
    "mobileone_s2": {
        **IMAGENETTE,
        "input_shape": (3, 224, 224),
        "url": None,
    },
    "mobileone_s3": {
        **IMAGENETTE,
        "input_shape": (3, 224, 224),
        "url": None,
    },
}


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
        in_chans, out_chans = self[1][0].in_channels, self[1][0].out_channels
        # Fuse the conv & BN
        _conv = nn.Conv2d(in_chans, out_chans, 1, bias=True).to(self[1][0].weight.data.device)
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
    arch: str,
    pretrained: bool,
    progress: bool,
    width_multipliers: List[float],
    overparam_factor: int,
    **kwargs: Any,
) -> MobileOne:
    # Build the model
    model = MobileOne(
        [2, 8, 10, 1],
        width_multipliers,
        overparam_factor,
        **kwargs,
    )

    model.default_cfg = default_cfgs[arch]  # type: ignore[assignment]
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"], progress)

    return model


def mobileone_s0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileOne:
    """MobileOne-S0 from
    `"An Improved One millisecond Mobile Backbone" <https://arxiv.org/pdf/2206.04040.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _mobileone("mobileone_s0", pretrained, progress, [0.75, 1.0, 1.0, 2.0], 4, **kwargs)


def mobileone_s1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileOne:
    """MobileOne-S1 from
    `"An Improved One millisecond Mobile Backbone" <https://arxiv.org/pdf/2206.04040.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _mobileone("mobileone_s1", pretrained, progress, [1.5, 1.5, 2.0, 2.5], 1, **kwargs)


def mobileone_s2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileOne:
    """MobileOne-S2 from
    `"An Improved One millisecond Mobile Backbone" <https://arxiv.org/pdf/2206.04040.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _mobileone("mobileone_s2", pretrained, progress, [1.5, 2.0, 2.5, 4.0], 1, **kwargs)


def mobileone_s3(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileOne:
    """MobileOne-S3 from
    `"An Improved One millisecond Mobile Backbone" <https://arxiv.org/pdf/2206.04040.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _mobileone("mobileone_s3", pretrained, progress, [2.0, 2.5, 3.0, 4.0], 1, **kwargs)
