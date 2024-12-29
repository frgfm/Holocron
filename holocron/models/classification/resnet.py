# Copyright (C) 2020-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections import OrderedDict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch.nn as nn
from torch import Tensor

from holocron.nn import GlobalAvgPool2d, init

from ..checkpoints import Checkpoint, _handle_legacy_pretrained
from ..presets import IMAGENET, IMAGENETTE
from ..utils import _checkpoint, _configure_model, conv_sequence

__all__ = [
    "BasicBlock",
    "Bottleneck",
    "ResNeXt50_32x4d_Checkpoint",
    "ResNet",
    "ResNet18_Checkpoint",
    "ResNet34_Checkpoint",
    "ResNet50D_Checkpoint",
    "ResNet50_Checkpoint",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet50d",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "resnet18": {**IMAGENET.__dict__, "input_shape": (3, 224, 224), "url": None},
    "resnet34": {**IMAGENET.__dict__, "input_shape": (3, 224, 224), "url": None},
    "resnet50": {
        **IMAGENETTE.__dict__,
        "input_shape": (3, 256, 256),
        "url": "https://github.com/frgfm/Holocron/releases/download/v0.1.2/resnet50_256-5e6206e0.pth",
    },
    "resnet101": {**IMAGENET.__dict__, "input_shape": (3, 224, 224), "url": None},
    "resnet152": {**IMAGENET.__dict__, "input_shape": (3, 224, 224), "url": None},
    "resnext50_32x4d": {**IMAGENET.__dict__, "input_shape": (3, 224, 224), "url": None},
    "resnext101_32x8d": {**IMAGENET.__dict__, "input_shape": (3, 224, 224), "url": None},
    "resnet50d": {
        **IMAGENETTE.__dict__,
        "input_shape": (3, 224, 224),
        "url": "https://github.com/frgfm/Holocron/releases/download/v0.1.3/resnet50d_224-e315ba9d.pt",
    },
}


class _ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self, convs: List[nn.Module], downsample: Optional[nn.Module] = None, act_layer: Optional[nn.Module] = None
    ) -> None:
        super().__init__()

        # Main branch
        self.conv = nn.Sequential(*convs)
        # Shortcut connection
        self.downsample = downsample

        if isinstance(act_layer, nn.Module):
            self.activation = act_layer

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if hasattr(self, "activation"):
            out = self.activation(out)

        return out


class BasicBlock(_ResBlock):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            [
                *conv_sequence(
                    inplanes,
                    planes,
                    act_layer,
                    norm_layer,
                    drop_layer,
                    conv_layer,
                    kernel_size=3,
                    stride=stride,
                    padding=dilation,
                    groups=groups,
                    bias=(norm_layer is None),
                    dilation=dilation,
                    **kwargs,
                ),
                *conv_sequence(
                    planes,
                    planes,
                    None,
                    norm_layer,
                    drop_layer,
                    conv_layer,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    groups=groups,
                    bias=(norm_layer is None),
                    dilation=dilation,
                    **kwargs,
                ),
            ],
            downsample,
            act_layer,
        )


class Bottleneck(_ResBlock):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
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
                *conv_sequence(
                    width,
                    width,
                    act_layer,
                    norm_layer,
                    drop_layer,
                    conv_layer,
                    kernel_size=3,
                    stride=stride,
                    padding=dilation,
                    groups=groups,
                    bias=(norm_layer is None),
                    dilation=dilation,
                    **kwargs,
                ),
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


class ChannelRepeat(nn.Module):
    def __init__(self, chan_repeats: int = 1) -> None:
        super().__init__()
        self.chan_repeats = chan_repeats

    def forward(self, x: Tensor) -> Tensor:
        repeats = [1] * x.ndim
        # Repeat the tensor along the channel dimension
        repeats[1] = self.chan_repeats
        return x.repeat(*repeats)


class ResNet(nn.Sequential):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        num_blocks: List[int],
        planes: List[int],
        num_classes: int = 10,
        in_channels: int = 3,
        zero_init_residual: bool = False,
        width_per_group: int = 64,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        deep_stem: bool = False,
        stem_pool: bool = True,
        avg_downsample: bool = False,
        num_repeats: int = 1,
        block_args: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> None:
        if conv_layer is None:
            conv_layer = nn.Conv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)
        self.dilation = 1

        in_planes = 64
        # Deep stem from ResNet-C
        if deep_stem:
            layers = [
                *conv_sequence(
                    in_channels,
                    in_planes // 2,
                    act_layer,
                    norm_layer,
                    drop_layer,
                    conv_layer,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=(norm_layer is None),
                ),
                *conv_sequence(
                    in_planes // 2,
                    in_planes // 2,
                    act_layer,
                    norm_layer,
                    drop_layer,
                    conv_layer,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=(norm_layer is None),
                ),
                *conv_sequence(
                    in_planes // 2,
                    in_planes,
                    act_layer,
                    norm_layer,
                    drop_layer,
                    conv_layer,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=(norm_layer is None),
                ),
            ]
        else:
            layers = conv_sequence(
                in_channels,
                in_planes,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=(norm_layer is None),
            )
        if stem_pool:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Optional tensor repetitions along channel axis (mainly for TridentNet)
        if num_repeats > 1:
            layers.append(ChannelRepeat(num_repeats))

        # Consecutive convolutional blocks
        stride = 1
        # Block args
        if block_args is None:
            block_args = {"groups": 1}
        if not isinstance(block_args, list):
            block_args = [block_args] * len(num_blocks)
        for _num_blocks, _planes, _block_args in zip(num_blocks, planes, block_args):
            layers.append(
                self._make_layer(
                    block,
                    _num_blocks,
                    in_planes,
                    _planes,
                    stride,
                    width_per_group,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop_layer=drop_layer,
                    avg_downsample=avg_downsample,
                    num_repeats=num_repeats,
                    block_args=_block_args,
                )
            )
            in_planes = block.expansion * _planes
            stride = 2

        super().__init__(
            OrderedDict([
                ("features", nn.Sequential(*layers)),
                ("pool", GlobalAvgPool2d(flatten=True)),
                ("head", nn.Linear(num_repeats * in_planes, num_classes)),
            ])
        )

        # Init all layers
        init.init_module(self, nonlinearity="relu")

        # Init shortcut
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    m.convs[2][1].weight.data.zero_()
                elif isinstance(m, BasicBlock):
                    m.convs[1][1].weight.data.zero_()

    @staticmethod
    def _make_layer(
        block: Type[Union[BasicBlock, Bottleneck]],
        num_blocks: int,
        in_planes: int,
        planes: int,
        stride: int = 1,
        width_per_group: int = 64,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
        avg_downsample: bool = False,
        num_repeats: int = 1,
        block_args: Optional[Dict[str, Any]] = None,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_planes != planes * block.expansion:
            # Downsampling from ResNet-D
            if avg_downsample:
                downsample = nn.Sequential(
                    nn.AvgPool2d(stride, ceil_mode=True, count_include_pad=False),
                    *conv_sequence(
                        num_repeats * in_planes,
                        num_repeats * planes * block.expansion,
                        None,
                        norm_layer,
                        drop_layer,
                        conv_layer,
                        kernel_size=1,
                        stride=1,
                        bias=(norm_layer is None),
                    ),
                )
            else:
                downsample = nn.Sequential(
                    *conv_sequence(
                        num_repeats * in_planes,
                        num_repeats * planes * block.expansion,
                        None,
                        norm_layer,
                        drop_layer,
                        conv_layer,
                        kernel_size=1,
                        stride=stride,
                        bias=(norm_layer is None),
                    )
                )
        if block_args is None:
            block_args = {}
        layers = [
            block(
                in_planes,
                planes,
                stride,
                downsample,
                base_width=width_per_group,
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_layer=drop_layer,
                **block_args,
            )
        ]
        layers.extend([
            block(
                block.expansion * planes,
                planes,
                1,
                None,
                base_width=width_per_group,
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_layer=drop_layer,
                **block_args,
            )
            for _ in range(num_blocks - 1)
        ])

        return nn.Sequential(*layers)


def _resnet(
    checkpoint: Union[Checkpoint, None],
    progress: bool,
    block: Type[Union[BasicBlock, Bottleneck]],
    num_blocks: List[int],
    out_chans: List[int],
    **kwargs: Any,
) -> ResNet:
    # Build the model
    model = ResNet(block, num_blocks, out_chans, **kwargs)
    return _configure_model(model, checkpoint, progress=progress)


class ResNet18_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="resnet18",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/resnet18_224-fc07006c.pth",
        acc1=0.9361,
        acc5=0.9946,
        sha256="fc07006c894cac8cf380fed699bc5a68463698753c954632f52bb8595040f781",
        size=44787043,
        num_params=11181642,
        commit="6e32c5b578711a2ef3731a8f8c61760ed9f03e58",
        train_args=(
            "./imagenette2-320/ --arch resnet18 --batch-size 64 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENETTE


def resnet18(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """ResNet-18 from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, loads that checkpoint
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _resnet

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.ResNet18_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        ResNet18_Checkpoint.DEFAULT.value,
    )
    return _resnet(checkpoint, progress, BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512], **kwargs)


class ResNet34_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="resnet34",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/resnet34_224-412b0792.pth",
        acc1=0.9381,
        acc5=0.9949,
        sha256="412b07927cc1938ee3add8d0f6bb18b42786646182f674d75f1433d086914485",
        size=85267035,
        num_params=21289802,
        commit="6e32c5b578711a2ef3731a8f8c61760ed9f03e58",
        train_args=(
            "./imagenette2-320/ --arch resnet34 --batch-size 64 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENETTE


def resnet34(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """ResNet-34 from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, load that checkpoint on the model
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _resnet

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.ResNet34_Checkpoint
        :members:
    """
    return _resnet(checkpoint, progress, BasicBlock, [3, 4, 6, 3], [64, 128, 256, 512], **kwargs)


class ResNet50_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="resnet50",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/resnet50_224-5b913f0b.pth",
        acc1=0.9378,
        acc5=0.9954,
        sha256="5b913f0b8148b483ba15541ab600cf354ca42b326e4896c4c3dbc51eb1e80e70",
        size=94384682,
        num_params=23528522,
        commit="6e32c5b578711a2ef3731a8f8c61760ed9f03e58",
        train_args=(
            "./imagenette2-320/ --arch resnet50 --batch-size 64 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENETTE


def resnet50(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """ResNet-50 from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, load that checkpoint on the model
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _resnet

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.ResNet50_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        ResNet50_Checkpoint.DEFAULT.value,
    )
    return _resnet(checkpoint, progress, Bottleneck, [3, 4, 6, 3], [64, 128, 256, 512], **kwargs)


class ResNet50D_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="resnet50d",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/resnet50d_224-6218d936.pth",
        acc1=0.9465,
        acc5=0.9952,
        sha256="6218d936fa67c0047f1ec65564213db538aa826d84f2df1d4fa3224531376e6c",
        size=94464810,
        num_params=23547754,
        commit="6e32c5b578711a2ef3731a8f8c61760ed9f03e58",
        train_args=(
            "./imagenette2-320/ --arch resnet50d --batch-size 64 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENETTE


def resnet50d(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """ResNet-50-D from
    `"Bag of Tricks for Image Classification with Convolutional Neural Networks"
    <https://arxiv.org/pdf/1812.01187.pdf>`_

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, load that checkpoint on the model
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _resnet

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.ResNet50D_Checkpoint
        :members:
    """
    return _resnet(
        checkpoint,
        progress,
        Bottleneck,
        [3, 4, 6, 3],
        [64, 128, 256, 512],
        deep_stem=True,
        avg_downsample=True,
        **kwargs,
    )


def resnet101(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """ResNet-101 from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, load that checkpoint on the model
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _resnet

    Returns:
        torch.nn.Module: classification model
    """
    return _resnet(checkpoint, progress, Bottleneck, [3, 4, 23, 3], [64, 128, 256, 512], **kwargs)


def resnet152(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """ResNet-152 from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, load that checkpoint on the model
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _resnet

    Returns:
        torch.nn.Module: classification model
    """
    return _resnet(checkpoint, progress, Bottleneck, [3, 8, 86, 3], [64, 128, 256, 512], **kwargs)


class ResNeXt50_32x4d_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="resnext50_32x4d",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/resnext50_32x4d_224-5832c4ce.pth",
        acc1=0.9455,
        acc5=0.9949,
        sha256="5832c4ce33522a9eb7a8b5abe31cf30621721a92d4f99b4b332a007d81d071fe",
        size=92332638,
        num_params=23000394,
        commit="6e32c5b578711a2ef3731a8f8c61760ed9f03e58",
        train_args=(
            "./imagenette2-320/ --arch resnext50_32x4d --batch-size 64 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENETTE


def resnext50_32x4d(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """ResNeXt-50 from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, load that checkpoint on the model
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _resnet

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.ResNeXt50_32x4d_Checkpoint
        :members:
    """
    kwargs["width_per_group"] = 4
    block_args = {"groups": 32}
    return _resnet(
        checkpoint,
        progress,
        Bottleneck,
        [3, 4, 6, 3],
        [64, 128, 256, 512],
        block_args=block_args,
        **kwargs,
    )


def resnext101_32x8d(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """ResNeXt-101 from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        checkpoint: If specified, load that checkpoint on the model
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _resnet

    Returns:
        torch.nn.Module: classification model
    """
    kwargs["width_per_group"] = 8
    block_args = {"groups": 32}
    return _resnet(
        checkpoint,
        progress,
        Bottleneck,
        [3, 4, 23, 3],
        [64, 128, 256, 512],
        block_args=block_args,
        **kwargs,
    )
