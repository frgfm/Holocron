import torch
import torch.nn as nn
from holocron.nn import GlobalAvgPool2d
from .resnet import ResNet, _ResBlock
from .utils import conv_sequence, load_pretrained_params
from typing import Optional, Callable, Any, Dict


__all__ = ['SoftAttentionLayer', 'SKConv2d', 'SKBottleneck', 'sknet50', 'sknet101', 'sknet152']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'sknet50': {'block': 'SKBottleneck', 'num_blocks': [3, 4, 6, 3],
                'url': None},
    'sknet101': {'block': 'SKBottleneck', 'num_blocks': [3, 4, 23, 3],
                 'url': None},
    'sknet152': {'block': 'SKBottleneck', 'num_blocks': [3, 8, 86, 3],
                 'url': None},
}


class SoftAttentionLayer(nn.Sequential):

    def __init__(
        self,
        channels: int,
        sa_ratio: int = 16,
        out_multiplier: int = 1,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__(GlobalAvgPool2d(flatten=False),
                         *conv_sequence(channels, max(channels // sa_ratio, 32), act_layer, norm_layer, drop_layer,
                                        kernel_size=1, stride=1, bias=False),
                         *conv_sequence(max(channels // sa_ratio, 32), channels * out_multiplier,
                                        nn.Sigmoid(), None, drop_layer, kernel_size=1, stride=1))


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
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.path_convs = nn.ModuleList([nn.Sequential(*conv_sequence(in_channels, out_channels,
                                                                      act_layer, norm_layer, drop_layer,
                                                                      kernel_size=3, bias=False, dilation=idx + 1,
                                                                      padding=idx + 1, **kwargs))
                                         for idx in range(m)])
        self.sa = SoftAttentionLayer(out_channels, sa_ratio, m, act_layer, norm_layer, drop_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        paths = torch.stack([path_conv(x) for path_conv in self.path_convs], dim=1)

        b, m, c = paths.shape[:3]
        z = self.sa(paths.sum(dim=1)).view(b, m, c, 1, 1)
        attention_factors = torch.softmax(z, dim=1)
        out = (attention_factors * paths).sum(dim=1)

        return out


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
        **kwargs: Any
    ) -> None:

        width = int(planes * (base_width / 64.)) * groups
        super().__init__(
            [*conv_sequence(inplanes, width, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=1,
                            stride=1, bias=False, **kwargs),
             SKConv2d(width, width, 2, 16, act_layer, norm_layer, drop_layer, groups=groups, stride=stride),
             *conv_sequence(width, planes * self.expansion, None, norm_layer, drop_layer, conv_layer, kernel_size=1,
                            stride=1, bias=False, **kwargs)],
            downsample, act_layer)


def _sknet(arch: str, pretrained: bool, progress: bool, **kwargs: Any) -> ResNet:

    # Build the model
    model = ResNet(SKBottleneck, default_cfgs[arch]['num_blocks'],  # type: ignore[arg-type]
                   [64, 128, 256, 512], **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'], progress)

    return model


def sknet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """SKNet-50 from
    `"Selective Kernel Networks" <https://arxiv.org/pdf/1903.06586.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _sknet('sknet50', pretrained, progress, **kwargs)


def sknet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """SKNet-101 from
    `"Selective Kernel Networks" <https://arxiv.org/pdf/1903.06586.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _sknet('sknet50', pretrained, progress, **kwargs)


def sknet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """SKNet-152 from
    `"Selective Kernel Networks" <https://arxiv.org/pdf/1903.06586.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _sknet('sknet50', pretrained, progress, **kwargs)
