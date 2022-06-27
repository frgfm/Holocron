# Copyright (C) 2020-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..presets import IMAGENETTE
from ..utils import conv_sequence, load_pretrained_params
from .resnet import ResNet, _ResBlock

__all__ = ["Tridentneck", "tridentnet50"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "tridentnet50": {
        **IMAGENETTE,
        "input_shape": (3, 224, 224),
        "url": "https://github.com/frgfm/Holocron/releases/download/v0.1.2/tridentnet50_224-98b4ce9c.pth",
    },
}


class TridentConv2d(nn.Conv2d):

    num_branches: int = 3

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if self.dilation[0] != 1 and self.dilation[0] != self.num_branches:
            raise ValueError(f"expected dilation to either be 1 or {self.num_branches}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % self.num_branches != 0:
            raise ValueError("expected number of channels of input tensor to be a multiple of `num_branches`.")

        # Dilation for each chunk
        if self.dilation[0] == 1:
            dilations = [1] * self.num_branches
        else:
            dilations = [1 + idx for idx in range(self.num_branches)]

        # Use shared weight to apply the convolution
        out = torch.cat(
            [
                F.conv2d(
                    _x,
                    self.weight,
                    self.bias,
                    self.stride,
                    tuple(dilation * p for p in self.padding),  # type: ignore[misc]
                    (dilation,) * len(self.dilation),
                    self.groups,
                )
                for _x, dilation in zip(torch.chunk(x, self.num_branches, 1), dilations)
            ],
            1,
        )

        return out


class Tridentneck(_ResBlock):

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 3,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        width = int(planes * (base_width / 64.0)) * groups
        # Concatenate along the channel axis and enlarge BN to leverage parallelization
        super().__init__(
            [
                *conv_sequence(
                    inplanes,
                    width,
                    act_layer,
                    norm_layer,
                    drop_layer,
                    TridentConv2d,
                    bn_channels=3 * width,
                    kernel_size=1,
                    stride=1,
                    bias=(norm_layer is None),
                    dilation=1,
                ),
                *conv_sequence(
                    width,
                    width,
                    act_layer,
                    norm_layer,
                    drop_layer,
                    TridentConv2d,
                    bn_channels=3 * width,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=groups,
                    bias=(norm_layer is None),
                    dilation=3,
                ),
                *conv_sequence(
                    width,
                    planes * self.expansion,
                    None,
                    norm_layer,
                    drop_layer,
                    TridentConv2d,
                    bn_channels=3 * planes * self.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=(norm_layer is None),
                    dilation=1,
                ),
            ],
            downsample,
            act_layer,
        )


def _tridentnet(
    arch: str,
    pretrained: bool,
    progress: bool,
    num_blocks: List[int],
    out_chans: List[int],
    **kwargs: Any,
) -> ResNet:
    # Build the model
    model = ResNet(Tridentneck, num_blocks, out_chans, num_repeats=3, **kwargs)  # type: ignore[arg-type]
    model.default_cfg = default_cfgs[arch]  # type: ignore[assignment]
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"], progress)

    return model


def tridentnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """TridentNet-50 from
    `"Scale-Aware Trident Networks for Object Detection" <https://arxiv.org/pdf/1901.01892.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _tridentnet("tridentnet50", pretrained, progress, [3, 4, 6, 3], [64, 128, 256, 512], **kwargs)
