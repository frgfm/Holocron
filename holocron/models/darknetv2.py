from collections import OrderedDict
import torch
import torch.nn as nn

from ..nn.init import init_module
from .utils import conv_sequence, load_pretrained_params
from holocron.nn import GlobalAvgPool2d
from typing import Dict, Any, Optional, Callable, List, Tuple, Union


__all__ = ['DarknetV2', 'darknet19']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'darknet19': {'arch': 'DarknetV2',
                  'layout': [(64, 0), (128, 1), (256, 1), (512, 2), (1024, 2)],
                  'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/darknet19_224-a48304cd.pth'},
}


class DarknetBodyV2(nn.Sequential):

    def __init__(
        self,
        layout: List[Tuple[int, int]],
        in_channels: int = 3,
        stem_channels: int = 32,
        passthrough: bool = False,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:

        if act_layer is None:
            act_layer = nn.LeakyReLU(0.1, inplace=True)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        in_chans = [stem_channels] + [_layout[0] for _layout in layout[:-1]]

        super().__init__(OrderedDict([
            ('stem', nn.Sequential(*conv_sequence(in_channels, stem_channels,
                                                  act_layer, norm_layer, drop_layer, conv_layer,
                                                  kernel_size=3, padding=1, bias=False))),
            ('layers', nn.Sequential(*[self._make_layer(num_blocks, _in_chans, out_chans,
                                                        act_layer, norm_layer, drop_layer, conv_layer)
                                       for _in_chans, (out_chans, num_blocks) in zip(in_chans, layout)]))])
        )

        self.passthrough = passthrough

    @staticmethod
    def _make_layer(
        num_blocks: int,
        in_planes: int,
        out_planes: int,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None
    ) -> nn.Sequential:
        layers: List[nn.Module] = [nn.MaxPool2d(2)]
        layers.extend(conv_sequence(in_planes, out_planes, act_layer, norm_layer, drop_layer, conv_layer,
                                    kernel_size=3, padding=1, stride=1, bias=False))
        for _ in range(num_blocks):
            layers.extend(conv_sequence(out_planes, out_planes // 2, act_layer, norm_layer, drop_layer, conv_layer,
                                        kernel_size=1, padding=0, stride=1, bias=False) +
                          conv_sequence(out_planes // 2, out_planes, act_layer, norm_layer, drop_layer, conv_layer,
                                        kernel_size=3, padding=1, stride=1, bias=False))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if self.passthrough:
            self.stem: nn.Sequential
            self.layers: nn.Sequential
            x = self.stem(x)
            for idx, layer in enumerate(self.layers):
                x = layer(x)
                if idx == len(self.layers) - 2:
                    aux = x.clone()

            return x, aux
        else:
            return super().forward(x)


class DarknetV2(nn.Sequential):
    def __init__(
        self,
        layout: List[Tuple[int, int]],
        num_classes: int = 10,
        in_channels: int = 3,
        stem_channels: int = 32,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:

        super().__init__(OrderedDict([
            ('features', DarknetBodyV2(layout, in_channels, stem_channels, False,
                                       act_layer, norm_layer, drop_layer, conv_layer)),
            ('classifier', nn.Conv2d(layout[-1][0], num_classes, 1)),
            ('pool', GlobalAvgPool2d(flatten=True))]))

        init_module(self, 'leaky_relu')


def _darknet(arch: str, pretrained: bool, progress: bool, **kwargs: Any) -> DarknetV2:
    # Build the model
    model = DarknetV2(default_cfgs[arch]['layout'], **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'], progress)

    return model


def darknet19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DarknetV2:
    """Darknet-19 from
    `"YOLO9000: Better, Faster, Stronger" <https://pjreddie.com/media/files/papers/YOLO9000.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _darknet('darknet19', pretrained, progress, **kwargs)  # type: ignore[return-value]
