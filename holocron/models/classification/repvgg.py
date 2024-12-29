# Copyright (C) 2021-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections import OrderedDict
from enum import Enum
from typing import Any, Callable, List, Optional, Union

import torch
import torch.nn as nn

from holocron.nn import GlobalAvgPool2d, init

from ..checkpoints import Checkpoint, _handle_legacy_pretrained
from ..utils import _checkpoint, _configure_model, conv_sequence, fuse_conv_bn

__all__ = [
    "RepBlock",
    "RepVGG",
    "RepVGG",
    "RepVGG_A0_Checkpoint",
    "RepVGG_A1_Checkpoint",
    "RepVGG_A2_Checkpoint",
    "RepVGG_B0_Checkpoint",
    "RepVGG_B1_Checkpoint",
    "RepVGG_B2_Checkpoint",
    "repvgg_a0",
    "repvgg_a1",
    "repvgg_a2",
    "repvgg_b0",
    "repvgg_b1",
    "repvgg_b2",
    "repvgg_b3",
]


class RepBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        identity: bool = True,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        self.branches: Union[nn.Conv2d, nn.ModuleList] = nn.ModuleList([
            nn.Sequential(
                *conv_sequence(inplanes, planes, None, norm_layer, kernel_size=3, padding=1, stride=stride),
            ),
            nn.Sequential(
                *conv_sequence(inplanes, planes, None, norm_layer, kernel_size=1, padding=0, stride=stride),
            ),
        ])

        self.activation = act_layer

        if identity:
            if inplanes != planes:
                raise ValueError("The number of input and output channels must be identical if identity is used")
            self.branches.append(norm_layer(planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.branches(x) if isinstance(self.branches, nn.Conv2d) else sum(branch(x) for branch in self.branches)
        return self.activation(out)

    def reparametrize(self) -> None:
        """Reparametrize the block by fusing convolutions and BN in each branch, then fusing all branches"""
        if not isinstance(self.branches, nn.ModuleList):
            raise AssertionError
        inplanes = self.branches[0][0].weight.data.shape[1]
        planes = self.branches[0][0].weight.data.shape[0]
        # Instantiate the equivalent Conv 3x3
        rep = nn.Conv2d(inplanes, planes, 3, padding=1, bias=True, stride=self.branches[0][0].stride)

        # Fuse convolutions with their BN
        fused_k3, fused_b3 = fuse_conv_bn(*self.branches[0])
        fused_k1, fused_b1 = fuse_conv_bn(*self.branches[1])

        # Conv 3x3
        rep.weight.data = fused_k3
        rep.bias.data = fused_b3  # type: ignore[union-attr]

        # Conv 1x1
        rep.weight.data[..., 1:2, 1:2] += fused_k1
        rep.bias.data += fused_b1  # type: ignore[union-attr]

        # Identity
        if len(self.branches) == 3:
            scale_factor = self.branches[2].weight.data / (self.branches[2].running_var + self.branches[2].eps).sqrt()
            # Identity is mapped as a diagonal matrix relatively to the out/in channel dimensions
            rep.weight.data[range(planes), range(inplanes), 1, 1] += scale_factor
            rep.bias.data += self.branches[2].bias.data  # type: ignore[union-attr]
            rep.bias.data -= scale_factor * self.branches[2].running_mean  # type: ignore[union-attr]

        # Update main branch & delete the others
        self.branches = rep


class RepVGG(nn.Sequential):
    """Implements a reparametrized version of VGG as described in
    `"RepVGG: Making VGG-style ConvNets Great Again" <https://arxiv.org/pdf/2101.03697.pdf>`_

    Args:
        num_blocks: list of number of blocks per stage
        planes: list of output channels of each stage
        width_multiplier: multiplier for the output channels of all stages apart from the last
        final_width_multiplier: multiplier for the output channels of the last stage
        num_classes: number of output classes
        in_channels: number of input channels
        act_layer: the activation layer to use
        norm_layer: the normalization layer to use
    """

    def __init__(
        self,
        num_blocks: List[int],
        planes: List[int],
        width_multiplier: float,
        final_width_multiplier: float,
        num_classes: int = 10,
        in_channels: int = 3,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        if len(num_blocks) != len(planes):
            raise AssertionError("the length of `num_blocks` and `planes` are expected to be the same")

        stages: List[nn.Sequential] = []
        # Assign the width multipliers
        chans = [in_channels, int(min(1, width_multiplier) * planes[0])]
        chans.extend([int(width_multiplier * chan) for chan in planes[1:-1]])
        chans.append(int(final_width_multiplier * planes[-1]))

        # Build the layers
        for nb_blocks, in_chan, out_chan in zip(num_blocks, chans[:-1], chans[1:], strict=False):
            layers = [RepBlock(in_chan, out_chan, 2, False, act_layer, norm_layer)]
            layers.extend([RepBlock(out_chan, out_chan, 1, True, act_layer, norm_layer) for _ in range(nb_blocks)])
            stages.append(nn.Sequential(*layers))

        super().__init__(
            OrderedDict([
                ("features", nn.Sequential(*stages)),
                ("pool", GlobalAvgPool2d(flatten=True)),
                ("head", nn.Linear(chans[-1], num_classes)),
            ])
        )
        # Init all layers
        init.init_module(self, nonlinearity="relu")

    def reparametrize(self) -> None:
        """Reparametrize the block by fusing convolutions and BN in each branch, then fusing all branches"""
        self.features: nn.Sequential
        for stage in self.features:
            for block in stage:
                block.reparametrize()


def _repvgg(
    checkpoint: Union[Checkpoint, None],
    progress: bool,
    num_blocks: List[int],
    a: float,
    b: float,
    **kwargs: Any,
) -> RepVGG:
    # Build the model
    model = RepVGG(num_blocks, [64, 64, 128, 256, 512], a, b, **kwargs)
    return _configure_model(model, checkpoint, progress=progress)


class RepVGG_A0_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="repvgg_a0",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/repvgg_a0_224-d3f54b28.pth",
        acc1=0.9292,
        acc5=0.9946,
        sha256="d3f54b28567fcd7e3e32ffbcffb5bb5c64fd97b7139cba0bfe9ad0bd7765cdaa",
        size=99183419,
        num_params=24741642,
        commit="d4a59999179b42fc0d3058ac6b76cc41f49dd56e",
        train_args=(
            "./imagenette2-320/ --arch repvgg_a0 --batch-size 64 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENETTE


def repvgg_a0(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> RepVGG:
    """RepVGG-A0 from
    `"RepVGG: Making VGG-style ConvNets Great Again" <https://arxiv.org/pdf/2101.03697.pdf>`_

    Args:
        pretrained: If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _repvgg

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.RepVGG_A0_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        RepVGG_A0_Checkpoint.DEFAULT.value,
    )
    return _repvgg(checkpoint, progress, [1, 2, 4, 14, 1], 0.75, 2.5, **kwargs)


class RepVGG_A1_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="repvgg_a1",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/repvgg_a1_224-8d3269fb.pth",
        acc1=0.9378,
        acc5=0.9918,
        sha256="8d3269fb5181c0fe75ef617872238135f3002f41e82e5ef7492d62a402ffae50",
        size=120724868,
        num_params=30119946,
        commit="d4a59999179b42fc0d3058ac6b76cc41f49dd56e",
        train_args=(
            "./imagenette2-320/ --arch repvgg_a1 --batch-size 64 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENETTE


def repvgg_a1(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> RepVGG:
    """RepVGG-A1 from
    `"RepVGG: Making VGG-style ConvNets Great Again" <https://arxiv.org/pdf/2101.03697.pdf>`_

    Args:
        pretrained: If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _repvgg

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.RepVGG_A1_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        RepVGG_A1_Checkpoint.DEFAULT.value,
    )
    return _repvgg(checkpoint, progress, [1, 2, 4, 14, 1], 1, 2.5, **kwargs)


class RepVGG_A2_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="repvgg_a2",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/repvgg_a2_224-cb442207.pth",
        acc1=0.9363,
        acc5=0.9939,
        sha256="cb442207d0c4627e3a16d7a8b4bf5342a182fd924cf4a044ac3a832014e7d4cf",
        size=194822538,
        num_params=48629514,
        commit="d4a59999179b42fc0d3058ac6b76cc41f49dd56e",
        train_args=(
            "./imagenette2-320/ --arch repvgg_a2 --batch-size 64 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENETTE


def repvgg_a2(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> RepVGG:
    """RepVGG-A2 from
    `"RepVGG: Making VGG-style ConvNets Great Again" <https://arxiv.org/pdf/2101.03697.pdf>`_

    Args:
        pretrained: If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _repvgg

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.RepVGG_A2_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        RepVGG_A2_Checkpoint.DEFAULT.value,
    )
    return _repvgg(checkpoint, progress, [1, 2, 4, 14, 1], 1.5, 2.75, **kwargs)


class RepVGG_B0_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="repvgg_b0",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/repvgg_b0_224-fdcdd2b7.pth",
        acc1=0.9269,
        acc5=0.9921,
        sha256="fdcdd2b739f19b47572be5a98ec407c08935d02adf1ab0bf90d7bc92c710fe2d",
        size=127668600,
        num_params=31845642,
        commit="d4a59999179b42fc0d3058ac6b76cc41f49dd56e",
        train_args=(
            "./imagenette2-320/ --arch repvgg_b0 --batch-size 64 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENETTE


def repvgg_b0(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> RepVGG:
    """RepVGG-B0 from
    `"RepVGG: Making VGG-style ConvNets Great Again" <https://arxiv.org/pdf/2101.03697.pdf>`_

    Args:
        pretrained: If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _repvgg

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.RepVGG_B0_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        RepVGG_B0_Checkpoint.DEFAULT.value,
    )
    return _repvgg(checkpoint, progress, [1, 4, 6, 16, 1], 1, 2.5, **kwargs)


class RepVGG_B1_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="repvgg_b1",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/repvgg_b1_224-3e5b28d7.pth",
        acc1=0.9396,
        acc5=0.9939,
        sha256="3e5b28d7803965546efadeb20abb84d8fef765dd08170677467a9c06294224c4",
        size=403763795,
        num_params=100829194,
        commit="d4a59999179b42fc0d3058ac6b76cc41f49dd56e",
        train_args=(
            "./imagenette2-320/ --arch repvgg_b1 --batch-size 64 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENETTE


def repvgg_b1(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> RepVGG:
    """RepVGG-B1 from
    `"RepVGG: Making VGG-style ConvNets Great Again" <https://arxiv.org/pdf/2101.03697.pdf>`_

    Args:
        pretrained: If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _repvgg

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.RepVGG_B1_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        RepVGG_B1_Checkpoint.DEFAULT.value,
    )
    return _repvgg(checkpoint, progress, [1, 4, 6, 16, 1], 2, 4, **kwargs)


class RepVGG_B2_Checkpoint(Enum):
    IMAGENETTE = _checkpoint(
        arch="repvgg_b2",
        url="https://github.com/frgfm/Holocron/releases/download/v0.2.1/repvgg_b2_224-dc810d88.pth",
        acc1=0.9414,
        acc5=0.9957,
        sha256="dc810d889e8533f3ab24d75d8bf4cec84380abfb3b10ee01009997eab6a35d4b",
        size=630382163,
        num_params=157462410,
        commit="d4a59999179b42fc0d3058ac6b76cc41f49dd56e",
        train_args=(
            "./imagenette2-320/ --arch repvgg_b2 --batch-size 32 --grad-acc 2 --mixup-alpha 0.2 --amp --device 0"
            " --epochs 100 --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176"
            " --val-resize-size 232 --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENETTE


def repvgg_b2(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> RepVGG:
    """RepVGG-B2 from
    `"RepVGG: Making VGG-style ConvNets Great Again" <https://arxiv.org/pdf/2101.03697.pdf>`_

    Args:
        pretrained: If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _repvgg

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.RepVGG_B2_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        RepVGG_B2_Checkpoint.DEFAULT.value,
    )
    return _repvgg(checkpoint, progress, [1, 4, 6, 16, 1], 2.5, 5, **kwargs)


def repvgg_b3(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> RepVGG:
    """RepVGG-B3 from
    `"RepVGG: Making VGG-style ConvNets Great Again" <https://arxiv.org/pdf/2101.03697.pdf>`_

    Args:
        pretrained: If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of _repvgg

    Returns:
        torch.nn.Module: classification model
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        None,
    )
    return _repvgg(checkpoint, progress, [1, 4, 6, 16, 1], 3, 5, **kwargs)
