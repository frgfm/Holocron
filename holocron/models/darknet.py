import sys
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

from ..nn.init import init_module
from .utils import conv_sequence
from holocron.nn import GlobalAvgPool2d
from typing import Dict, Any, Optional, Callable, List, Tuple


__all__ = ['DarknetV1', 'darknet24']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'darknet24': {'arch': 'DarknetV1',
                  'layout': [[192], [128, 256, 256, 512], [*([256, 512] * 4), 512, 1024], [512, 1024] * 2],
                  'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/darknet24_224-55729a5c.pth'},
}


class DarknetBodyV1(nn.Sequential):
    def __init__(
        self,
        layout: List[List[int]],
        in_channels: int = 3,
        stem_channels: int = 64,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:

        if act_layer is None:
            act_layer = nn.LeakyReLU(0.1, inplace=True)

        in_chans = [stem_channels] + [_layout[-1] for _layout in layout[:-1]]

        super().__init__(OrderedDict([
            ('stem', nn.Sequential(*conv_sequence(in_channels, stem_channels,
                                                  act_layer, norm_layer, drop_layer, conv_layer,
                                                  kernel_size=7, padding=3, stride=2, bias=False))),
            ('layers', nn.Sequential(*[self._make_layer([_in_chans] + planes,
                                                        act_layer, norm_layer, drop_layer, conv_layer)
                                       for _in_chans, planes in zip(in_chans, layout)]))])
        )

    @staticmethod
    def _make_layer(
        planes: List[int],
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None
    ) -> nn.Sequential:
        _layers: List[nn.Module] = [nn.MaxPool2d(2)]
        k1 = True
        for in_planes, out_planes in zip(planes[:-1], planes[1:]):
            _layers.extend(conv_sequence(in_planes, out_planes, act_layer, norm_layer, drop_layer, conv_layer,
                                         kernel_size=3 if out_planes > in_planes else 1,
                                         padding=1 if out_planes > in_planes else 0, bias=False))
            k1 = not k1

        return nn.Sequential(*_layers)


class DarknetV1(nn.Sequential):
    def __init__(
        self,
        layout: List[List[int]],
        num_classes: int = 10,
        in_channels: int = 3,
        stem_channels: int = 64,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__(OrderedDict([
            ('features', DarknetBodyV1(layout, in_channels, stem_channels,
                                       act_layer, norm_layer, drop_layer, conv_layer)),
            ('pool', GlobalAvgPool2d(flatten=True)),
            ('classifier', nn.Linear(layout[2][-1], num_classes))]))

        init_module(self, 'leaky_relu')


def _darknet(arch: str, pretrained: bool, progress: bool, **kwargs: Any) -> nn.Sequential:

    # Retrieve the correct Darknet layout type
    darknet_type = sys.modules[__name__].__dict__[default_cfgs[arch]['arch']]
    # Build the model
    model = darknet_type(default_cfgs[arch]['layout'], **kwargs)
    # Load pretrained parameters
    if pretrained:
        if default_cfgs[arch]['url'] is None:
            logging.warning(f"Invalid model URL for {arch}, using default initialization.")
        else:
            state_dict = load_state_dict_from_url(default_cfgs[arch]['url'],
                                                  progress=progress)
            model.load_state_dict(state_dict)

    return model


def darknet24(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DarknetV1:
    """Darknet-24 from
    `"You Only Look Once: Unified, Real-Time Object Detection" <https://pjreddie.com/media/files/papers/yolo_1.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _darknet('darknet24', pretrained, progress, **kwargs)  # type: ignore[return-value]
