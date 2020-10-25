from collections import OrderedDict
import torch.nn as nn

from ..nn.init import init_module
from .utils import conv_sequence, load_pretrained_params
from holocron.nn import GlobalAvgPool2d
from typing import Dict, Any, Optional, Callable, List


__all__ = ['DarknetV1', 'darknet24']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'darknet24': {'layout': [[192], [128, 256, 256, 512], [*([256, 512] * 4), 512, 1024], [512, 1024] * 2],
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


def _darknet(arch: str, pretrained: bool, progress: bool, **kwargs: Any) -> DarknetV1:
    # Build the model
    model = DarknetV1(default_cfgs[arch]['layout'], **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'], progress)

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

    return _darknet('darknet24', pretrained, progress, **kwargs)
