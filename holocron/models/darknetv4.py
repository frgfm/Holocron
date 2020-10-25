from collections import OrderedDict
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, List, Tuple, Union

from ..nn.init import init_module
from .utils import conv_sequence, load_pretrained_params
from holocron.nn import Mish, DropBlock2d, GlobalAvgPool2d
from .darknetv3 import ResBlock


__all__ = ['DarknetV4', 'cspdarknet53', 'cspdarknet53_mish']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'cspdarknet53': {'arch': 'DarknetV4',
                     'layout': [(64, 1), (128, 2), (256, 8), (512, 8), (1024, 4)],
                     'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/cspdarknet53_256-3ef96818.pth'},
    'cspdarknet53_mish': {'arch': 'DarknetV4',
                          'layout': [(64, 1), (128, 2), (256, 8), (512, 8), (1024, 4)],
                          'url': None},
}


class CSPStage(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        compression = 2 if num_blocks > 1 else 1
        self.base_layer = nn.Sequential(*conv_sequence(in_channels, out_channels,
                                                       act_layer, norm_layer, drop_layer, conv_layer,
                                                       kernel_size=3, padding=1, stride=2, bias=False),
                                        # Share the conv
                                        *conv_sequence(out_channels, 2 * out_channels // compression,
                                                       act_layer, norm_layer, drop_layer, conv_layer,
                                                       kernel_size=1, bias=False))
        self.main = nn.Sequential(*[ResBlock(out_channels // compression,
                                             out_channels // compression if num_blocks > 1 else in_channels,
                                             act_layer, norm_layer, drop_layer, conv_layer)
                                    for _ in range(num_blocks)],
                                  *conv_sequence(out_channels // compression, out_channels // compression,
                                                 act_layer, norm_layer, drop_layer, conv_layer,
                                                 kernel_size=1, bias=False))
        self.transition = nn.Sequential(*conv_sequence(2 * out_channels // compression, out_channels,
                                                       act_layer, norm_layer, drop_layer, conv_layer,
                                                       kernel_size=1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_layer(x)
        x1, x2 = x.chunk(2, dim=1)
        return self.transition(torch.cat([x1, self.main(x2)], dim=1))


class DarknetBodyV4(nn.Sequential):

    def __init__(
        self,
        layout: List[Tuple[int, int]],
        in_channels: int = 3,
        stem_channels: int = 32,
        num_features: int = 1,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:

        super().__init__()

        if act_layer is None:
            act_layer = nn.LeakyReLU(inplace=True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        in_chans = [stem_channels] + [_layout[0] for _layout in layout[:-1]]

        super().__init__(OrderedDict([
            ('stem', nn.Sequential(*conv_sequence(in_channels, stem_channels,
                                                  act_layer, norm_layer, drop_layer, conv_layer,
                                                  kernel_size=3, padding=1, bias=False))),
            ('stages', nn.Sequential(*[CSPStage(_in_chans, out_chans, num_blocks,
                                                act_layer, norm_layer, drop_layer, conv_layer)
                                       for _in_chans, (out_chans, num_blocks) in zip(in_chans, layout)]))])
        )

        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:

        if self.num_features == 1:
            return super().forward(x)
        else:
            self.stem: nn.Sequential
            self.stages: nn.Sequential
            x = self.stem(x)
            features = []
            for idx, stage in enumerate(self.stages):
                x = stage(x)
                if idx >= (len(self.stages) - self.num_features):
                    features.append(x)

            return features


class DarknetV4(nn.Sequential):
    def __init__(
        self,
        layout: List[Tuple[int, int]],
        num_classes: int = 10,
        in_channels: int = 3,
        stem_channels: int = 32,
        num_features: int = 1,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:

        super().__init__(OrderedDict([
            ('features', DarknetBodyV4(layout, in_channels, stem_channels, num_features,
                                       act_layer, norm_layer, drop_layer, conv_layer)),
            ('pool', GlobalAvgPool2d(flatten=True)),
            ('classifier', nn.Linear(layout[-1][0], num_classes))]))

        init_module(self, 'leaky_relu')


def _darknet(arch: str, pretrained: bool, progress: bool, **kwargs: Any) -> DarknetV4:
    # Build the model
    model = DarknetV4(default_cfgs[arch]['layout'], **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'], progress)

    return model


def cspdarknet53(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DarknetV4:
    """CSP-Darknet-53 from
    `"CSPNet: A New Backbone that can Enhance Learning Capability of CNN" <https://arxiv.org/pdf/1911.11929.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _darknet('cspdarknet53', pretrained, progress, **kwargs)  # type: ignore[return-value]


def cspdarknet53_mish(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DarknetV4:
    """Modified version of CSP-Darknet-53 from
    `"CSPNet: A New Backbone that can Enhance Learning Capability of CNN" <https://arxiv.org/pdf/1911.11929.pdf>`_
    with Mish as activation layer and DropBlock as regularization layer.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    kwargs['act_layer'] = Mish()
    kwargs['drop_layer'] = DropBlock2d

    return _darknet('cspdarknet53_mish', pretrained, progress, **kwargs)  # type: ignore[return-value]
