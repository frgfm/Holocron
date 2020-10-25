"""
Implementation of Res2Net
based on https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/res2net.py
"""

import math
import torch
import torch.nn as nn
from .resnet import _ResBlock, ResNet
from .utils import conv_sequence, load_pretrained_params
from typing import Optional, Callable, Any, Dict


__all__ = ['Bottle2neck', 'res2net50_26w_4s']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'res2net50_26w_4s': {
        'num_blocks': [3, 4, 6, 3], 'width_per_group': 26, 'scale': 4,
        'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/res2net50_26w_4s_224-97cfc954.pth'},
}


class ScaleConv2d(nn.Module):
    def __init__(
        self,
        scale: int,
        planes: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        downsample: bool = False,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()

        self.scale = scale
        self.width = planes // scale
        self.conv = nn.ModuleList([nn.Sequential(*conv_sequence(self.width, self.width,
                                                                act_layer, norm_layer, drop_layer,
                                                                kernel_size=3, stride=stride, padding=1,
                                                                groups=groups, bias=False))
                                   for _ in range(max(1, scale - 1))])

        if downsample:
            self.downsample = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        else:
            self.downsample = None  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Split the channel dimension into groups of self.width channels
        split_x = torch.split(x, self.width, 1)
        out = []
        for idx, layer in enumerate(self.conv):
            # If downsampled, don't add previous branch
            if idx == 0 or self.downsample is not None:
                _res = split_x[idx]
            else:
                _res = out[-1] + split_x[idx]
            out.append(layer(_res))
        # Use the last chunk as shortcut connection
        if self.scale > 1:
            # If the convs were strided, the shortcut needs to be downsampled
            if self.downsample is not None:
                out.append(self.downsample(split_x[-1]))
            else:
                out.append(split_x[-1])

        return torch.cat(out, 1)


class Bottle2neck(_ResBlock):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 26,
        dilation: int = 1,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        scale: int = 4
    ) -> None:
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        # Check if ScaleConv2d needs to downsample the identity branch
        _downsample = stride > 1 or downsample is not None

        width = int(math.floor(planes * (base_width / 64.))) * groups
        super().__init__(
            [*conv_sequence(inplanes, width * scale, act_layer, norm_layer, drop_layer, kernel_size=1,
                            stride=1, bias=False),
             ScaleConv2d(scale, width * scale, 3, stride, groups, _downsample, act_layer, norm_layer, drop_layer),
             *conv_sequence(width * scale, planes * self.expansion, None, norm_layer, drop_layer, kernel_size=1,
                            stride=1, bias=False)],
            downsample, act_layer)


def _res2net(arch: str, pretrained: bool, progress: bool, **kwargs: Any) -> ResNet:
    # Build the model
    model = ResNet(Bottle2neck, default_cfgs[arch]['num_blocks'], [64, 128, 256, 512],  # type: ignore[arg-type]
                   width_per_group=default_cfgs[arch]['width_per_group'],
                   block_args=dict(scale=default_cfgs[arch]['scale']),
                   **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'], progress)

    return model


def res2net50_26w_4s(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """Res2Net-50 26wx4s from
    `"Res2Net: A New Multi-scale Backbone Architecture" <https://arxiv.org/pdf/1904.01169.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _res2net('res2net50_26w_4s', pretrained, progress, **kwargs)
