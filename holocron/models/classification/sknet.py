# Copyright (C) 2020-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

from holocron.nn import GlobalAvgPool2d

from ..checkpoints import Checkpoint, _handle_legacy_pretrained
from ..presets import IMAGENETTE
from ..utils import _checkpoint, _configure_model, conv_sequence
from .resnet import ResNet, _ResBlock

__all__ = ["SKBottleneck", "SKConv2d", "SKNet50_Checkpoint", "SoftAttentionLayer", "sknet50", "sknet101", "sknet152"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "sknet50": {
        **IMAGENETTE.__dict__,
        "input_shape": (3, 224, 224),
        "url": "https://github.com/frgfm/Holocron/releases/download/v0.1.3/sknet50_224-5d2160f2.pth",
    },
    "sknet101": {
        **IMAGENETTE.__dict__,
        "input_shape": (3, 224, 224),
        "url": None,
    },
    "sknet152": {
        **IMAGENETTE.__dict__,
        "input_shape": (3, 224, 224),
        "url": None,
    },
}


class SoftAttentionLayer(nn.Sequential):
    def __init__(
        self,
        channels: int,
        sa_ratio: int = 16,
        out_multiplier: int = 1,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
            GlobalAvgPool2d(flatten=False),
            *conv_sequence(
                channels,
                max(channels // sa_ratio, 32),
                act_layer,
                norm_layer,
                drop_layer,
                kernel_size=1,
                stride=1,
                bias=(norm_layer is None),
            ),
            *conv_sequence(
                max(channels // sa_ratio, 32),
                channels * out_multiplier,
                nn.Sigmoid(),
                None,
                drop_layer,
                kernel_size=1,
                stride=1,
            ),
        )


class SKConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        m: int = 2,
        sa_ratio: int = 16,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.path_convs = nn.ModuleList([
            nn.Sequential(
                *conv_sequence(
                    in_channels,
                    out_channels,
                    act_layer,
                    norm_layer,
                    drop_layer,
                    kernel_size=3,
                    bias=(norm_layer is None),
                    dilation=idx + 1,
                    padding=idx + 1,
                    **kwargs,
                )
            )
            for idx in range(m)
        ])
        self.sa = SoftAttentionLayer(out_channels, sa_ratio, m, act_layer, norm_layer, drop_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        paths = torch.stack([path_conv(x) for path_conv in self.path_convs], dim=1)

        b, m, c = paths.shape[:3]
        z = self.sa(paths.sum(dim=1)).view(b, m, c, 1, 1)
        attention_factors = torch.softmax(z, dim=1)
        return (attention_factors * paths).sum(dim=1)


class SKBottleneck(_ResBlock):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 32,
        base_width: int = 64,
        dilation: int = 1,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        width = int(planes * (base_width / 64.0)) * groups
        super().__init__(
            [
                *conv_sequence(
                    inplanes,
                    width,
                    act_layer,
                    norm_layer,
                    drop_layer,
                    conv_layer,
                    kernel_size=1,
                    stride=1,
                    bias=(norm_layer is None),
                    **kwargs,
                ),
                SKConv2d(width, width, 2, 16, act_layer, norm_layer, drop_layer, groups=groups, stride=stride),
                *conv_sequence(
                    width,
                    planes * self.expansion,
                    None,
                    norm_layer,
                    drop_layer,
                    conv_layer,
                    kernel_size=1,
                    stride=1,
                    bias=(norm_layer is None),
                    **kwargs,
                ),
            ],
            downsample,
            act_layer,
        )


def _sknet(
    checkpoint: Union[Checkpoint, None],
    progress: bool,
    num_blocks: List[int],
    out_chans: List[int],
    **kwargs: Any,
) -> ResNet:
    # Build the model
    model = ResNet(SKBottleneck, num_blocks, out_chans, **kwargs)  # type: ignore[arg-type]
    return _configure_model(model, checkpoint, progress=progress)


class SKNet50_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="sknet50",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/sknet50_224-e2349031.pth",
        acc1=0.9437,
        acc5=0.9954,
        sha256="e2349031c838a4661cd729dbc7825605c9e0c966bd89bbcc9b39f0e324894d1f",
        size=141253623,
        num_params=35224394,
        commit="6e32c5b578711a2ef3731a8f8c61760ed9f03e58",
        train_args=(
            "./imagenette2-320/ --arch sknet50 --batch-size 64 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENETTE


def sknet50(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """SKNet-50 from
    `"Selective Kernel Networks" <https://arxiv.org/pdf/1903.06586.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _sknet

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.SKNet50_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        SKNet50_Checkpoint.DEFAULT.value,
    )
    return _sknet(checkpoint, progress, [3, 4, 6, 3], [64, 128, 256, 512], **kwargs)


def sknet101(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """SKNet-101 from
    `"Selective Kernel Networks" <https://arxiv.org/pdf/1903.06586.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _sknet

    Returns:
        torch.nn.Module: classification model
    """
    return _sknet(checkpoint if pretrained else None, progress, [3, 4, 23, 3], [64, 128, 256, 512], **kwargs)


def sknet152(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """SKNet-152 from
    `"Selective Kernel Networks" <https://arxiv.org/pdf/1903.06586.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _sknet

    Returns:
        torch.nn.Module: classification model
    """
    return _sknet(checkpoint if pretrained else None, progress, [3, 8, 86, 3], [64, 128, 256, 512], **kwargs)
