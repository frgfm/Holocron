# Copyright (C) 2021-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

from holocron.nn import GlobalAvgPool2d

from ..presets import IMAGENETTE
from ..utils import fuse_conv_bn, load_pretrained_params

__all__ = [
    "RepVGG",
    "RepBlock",
    "RepVGG",
    "repvgg_a0",
    "repvgg_a1",
    "repvgg_a2",
    "repvgg_b0",
    "repvgg_b1",
    "repvgg_b2",
    "repvgg_b3",
]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "repvgg_a0": {
        **IMAGENETTE,
        "input_shape": (3, 224, 224),
        "url": "https://github.com/frgfm/Holocron/releases/download/v0.1.3/repvgg_a0_224-150f4b9d.pt",
    },
    "repvgg_a1": {
        **IMAGENETTE,
        "input_shape": (3, 224, 224),
        "url": "https://github.com/frgfm/Holocron/releases/download/v0.1.3/repvgg_a1_224-870b9e4b.pt",
    },
    "repvgg_a2": {
        **IMAGENETTE,
        "input_shape": (3, 224, 224),
        "url": "https://github.com/frgfm/Holocron/releases/download/v0.1.3/repvgg_a2_224-7051289a.pt",
    },
    "repvgg_b0": {
        **IMAGENETTE,
        "input_shape": (3, 224, 224),
        "url": "https://github.com/frgfm/Holocron/releases/download/v0.1.3/repvgg_b0_224-7e9c3fc7.pth",
    },
    "repvgg_b1": {
        **IMAGENETTE,
        "input_shape": (3, 224, 224),
        "url": None,
    },
    "repvgg_b2": {
        **IMAGENETTE,
        "input_shape": (3, 224, 224),
        "url": None,
    },
    "repvgg_b3": {
        **IMAGENETTE,
        "input_shape": (3, 224, 224),
        "url": None,
    },
}


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

        self.branches: Union[nn.Conv2d, nn.ModuleList] = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(inplanes, planes, 3, padding=1, bias=(norm_layer is None), stride=stride),
                    norm_layer(planes),
                ),
                nn.Sequential(
                    nn.Conv2d(inplanes, planes, 1, padding=0, bias=(norm_layer is None), stride=stride),
                    norm_layer(planes),
                ),
            ]
        )

        self.activation = act_layer

        if identity:
            if inplanes != planes:
                raise ValueError("The number of input and output channels must be identical if identity is used")
            self.branches.append(nn.BatchNorm2d(planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if isinstance(self.branches, nn.Conv2d):
            out = self.branches(x)
        else:
            out = sum(branch(x) for branch in self.branches)

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

        _stages: List[nn.Sequential] = []
        # Assign the width multipliers
        chans = [in_channels, int(min(1, width_multiplier) * planes[0])]
        chans.extend([int(width_multiplier * chan) for chan in planes[1:-1]])
        chans.append(int(final_width_multiplier * planes[-1]))

        # Build the layers
        for nb_blocks, in_chan, out_chan in zip(num_blocks, chans[:-1], chans[1:]):
            _layers = [RepBlock(in_chan, out_chan, 2, False, act_layer, norm_layer)]
            _layers.extend([RepBlock(out_chan, out_chan, 1, True, act_layer, norm_layer) for _ in range(nb_blocks)])
            _stages.append(nn.Sequential(*_layers))

        super().__init__(
            OrderedDict(
                [
                    ("features", nn.Sequential(*_stages)),
                    ("pool", GlobalAvgPool2d(flatten=True)),
                    ("head", nn.Linear(chans[-1], num_classes)),
                ]
            )
        )

    def reparametrize(self) -> None:
        """Reparametrize the block by fusing convolutions and BN in each branch, then fusing all branches"""
        self.features: nn.Sequential
        for stage in self.features:
            for block in stage:
                block.reparametrize()


def _repvgg(
    arch: str,
    pretrained: bool,
    progress: bool,
    num_blocks: List[int],
    out_chans: List[int],
    a: float,
    b: float,
    **kwargs: Any,
) -> RepVGG:

    # Build the model
    model = RepVGG(num_blocks, [64, 64, 128, 256, 512], a, b, **kwargs)
    model.default_cfg = default_cfgs[arch]  # type: ignore[assignment]
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"], progress)

    return model


def repvgg_a0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RepVGG:
    """RepVGG-A0 from
    `"RepVGG: Making VGG-style ConvNets Great Again" <https://arxiv.org/pdf/2101.03697.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _repvgg("repvgg_a0", pretrained, progress, [1, 2, 4, 14, 1], [64, 64, 128, 256, 512], 0.75, 2.5, **kwargs)


def repvgg_a1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RepVGG:
    """RepVGG-A1 from
    `"RepVGG: Making VGG-style ConvNets Great Again" <https://arxiv.org/pdf/2101.03697.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _repvgg("repvgg_a1", pretrained, progress, [1, 2, 4, 14, 1], [64, 64, 128, 256, 512], 1, 2.5, **kwargs)


def repvgg_a2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RepVGG:
    """RepVGG-A2 from
    `"RepVGG: Making VGG-style ConvNets Great Again" <https://arxiv.org/pdf/2101.03697.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _repvgg("repvgg_a2", pretrained, progress, [1, 2, 4, 14, 1], [64, 64, 128, 256, 512], 1.5, 2.75, **kwargs)


def repvgg_b0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RepVGG:
    """RepVGG-B0 from
    `"RepVGG: Making VGG-style ConvNets Great Again" <https://arxiv.org/pdf/2101.03697.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _repvgg("repvgg_b0", pretrained, progress, [1, 4, 6, 16, 1], [64, 64, 128, 256, 512], 1, 2.5, **kwargs)


def repvgg_b1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RepVGG:
    """RepVGG-B1 from
    `"RepVGG: Making VGG-style ConvNets Great Again" <https://arxiv.org/pdf/2101.03697.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _repvgg("repvgg_b1", pretrained, progress, [1, 4, 6, 16, 1], [64, 64, 128, 256, 512], 2, 4, **kwargs)


def repvgg_b2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RepVGG:
    """RepVGG-B2 from
    `"RepVGG: Making VGG-style ConvNets Great Again" <https://arxiv.org/pdf/2101.03697.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _repvgg("repvgg_b2", pretrained, progress, [1, 4, 6, 16, 1], [64, 64, 128, 256, 512], 2.5, 5, **kwargs)


def repvgg_b3(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RepVGG:
    """RepVGG-B3 from
    `"RepVGG: Making VGG-style ConvNets Great Again" <https://arxiv.org/pdf/2101.03697.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _repvgg("repvgg_b3", pretrained, progress, [1, 4, 6, 16, 1], [64, 64, 128, 256, 512], 3, 5, **kwargs)
