# Copyright (C) 2020-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import Any, Callable, Dict, List, Optional, Type, Union

from torch.nn import Module

from holocron.nn import PyConv2d

from ..presets import IMAGENETTE
from ..utils import conv_sequence, load_pretrained_params
from .resnet import ResNet, _ResBlock

__all__ = ["PyBottleneck", "pyconv_resnet50", "pyconvhg_resnet50"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "pyconv_resnet50": {
        **IMAGENETTE,
        "input_shape": (3, 224, 224),
        "url": None,
    },
    "pyconvhg_resnet50": {
        **IMAGENETTE,
        "input_shape": (3, 224, 224),
        "url": None,
    },
}


class PyBottleneck(_ResBlock):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[Module] = None,
        groups: List[int] = [1],
        base_width: int = 64,
        dilation: int = 1,
        act_layer: Optional[Module] = None,
        norm_layer: Optional[Callable[[int], Module]] = None,
        drop_layer: Optional[Callable[..., Module]] = None,
        num_levels: int = 2,
        **kwargs: Any,
    ) -> None:

        width = int(planes * (base_width / 64.0)) * min(groups)

        super().__init__(
            [
                *conv_sequence(
                    inplanes,
                    width,
                    act_layer,
                    norm_layer,
                    drop_layer,
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
                    conv_layer=PyConv2d,
                    kernel_size=3,
                    stride=stride,
                    padding=dilation,
                    groups=groups,
                    bias=(norm_layer is None),
                    dilation=dilation,
                    num_levels=num_levels,
                    **kwargs,
                ),
                *conv_sequence(
                    width,
                    planes * self.expansion,
                    None,
                    norm_layer,
                    drop_layer,
                    kernel_size=1,
                    stride=1,
                    bias=(norm_layer is None),
                    **kwargs,
                ),
            ],
            downsample,
            act_layer,
        )


class PyHGBottleneck(PyBottleneck):
    expansion: int = 2


def _pyconvresnet(
    arch: str,
    pretrained: bool,
    progress: bool,
    block: Type[Union[PyBottleneck, PyHGBottleneck]],
    num_blocks: List[int],
    out_chans: List[int],
    width_per_group: int,
    groups: List[List[int]],
    **kwargs: Any,
) -> ResNet:

    # Build the model
    model = ResNet(
        block,  # type: ignore[arg-type]
        num_blocks,
        out_chans,
        stem_pool=False,
        width_per_group=width_per_group,
        block_args=[dict(num_levels=len(group), groups=group) for group in groups],
        **kwargs,
    )
    model.default_cfg = default_cfgs[arch]  # type: ignore[assignment]
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"], progress)

    return model


def pyconv_resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """PyConvResNet-50 from `"Pyramidal Convolution: Rethinking Convolutional Neural Networks
    for Visual Recognition" <https://arxiv.org/pdf/2006.11538.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _pyconvresnet(
        "pyconv_resnet50",
        pretrained,
        progress,
        PyBottleneck,
        [3, 4, 6, 3],
        [64, 128, 256, 512],
        64,
        [[1, 4, 8, 16], [1, 4, 8], [1, 4], [1]],
        **kwargs,
    )


def pyconvhg_resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """PyConvHGResNet-50 from `"Pyramidal Convolution: Rethinking Convolutional Neural Networks
    for Visual Recognition" <https://arxiv.org/pdf/2006.11538.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _pyconvresnet(
        "pyconvhg_resnet50",
        pretrained,
        progress,
        PyHGBottleneck,
        [3, 4, 6, 3],
        [128, 256, 512, 1024],
        2,
        [[32, 32, 32, 32], [32, 64, 64], [32, 64], [32]],
        **kwargs,
    )
