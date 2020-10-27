import sys
from collections import OrderedDict
import torch.nn as nn
from torch import Tensor
from holocron.nn import init, GlobalAvgPool2d
from .utils import conv_sequence, load_pretrained_params
from typing import Dict, Any, List, Optional, Callable, Union, Type


__all__ = ['BasicBlock', 'Bottleneck', 'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnext50_32x4d', 'resnext101_32x8d', 'resnet50d']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'resnet18': {'block': 'BasicBlock', 'num_blocks': [2, 2, 2, 2],
                 'url': None},
    'resnet34': {'block': 'BasicBlock', 'num_blocks': [3, 4, 6, 3],
                 'url': None},
    'resnet50': {'block': 'Bottleneck', 'num_blocks': [3, 4, 6, 3],
                 'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/resnet50_256-5e6206e0.pth'},
    'resnet101': {'block': 'Bottleneck', 'num_blocks': [3, 4, 23, 3],
                  'url': None},
    'resnet152': {'block': 'Bottleneck', 'num_blocks': [3, 8, 86, 3],
                  'url': None},
    'resnext50_32x4d': {'block': 'Bottleneck', 'num_blocks': [3, 4, 6, 3],
                        'url': None},
    'resnext101_32x8d': {'block': 'Bottleneck', 'num_blocks': [3, 4, 23, 3],
                         'url': None},
    'resnet50d': {'block': 'Bottleneck', 'num_blocks': [3, 4, 6, 3],
                  'url': 'https://github.com/frgfm/Holocron/releases/download/v0.1.2/resnet50d_224-499c0b54.pth'},
}


class _ResBlock(nn.Module):

    expansion: int = 1

    def __init__(
        self,
        convs: List[nn.Module],
        downsample: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = None
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
        if hasattr(self, 'activation'):
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
        **kwargs: Any
    ) -> None:
        super().__init__(
            [*conv_sequence(inplanes, planes, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3,
                            stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation, **kwargs),
             *conv_sequence(planes, planes, None, norm_layer, drop_layer, conv_layer, kernel_size=3,
                            stride=1, padding=dilation, groups=groups, bias=False, dilation=dilation, **kwargs)],
            downsample, act_layer)


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
        **kwargs: Any
    ) -> None:

        width = int(planes * (base_width / 64.)) * groups
        super().__init__(
            [*conv_sequence(inplanes, width, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=1,
                            stride=1, bias=False, **kwargs),
             *conv_sequence(width, width, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3,
                            stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation, **kwargs),
             *conv_sequence(width, planes * self.expansion, None, norm_layer, drop_layer, conv_layer, kernel_size=1,
                            stride=1, bias=False, **kwargs)],
            downsample, act_layer)


class ChannelRepeat(nn.Module):
    def __init__(self, chan_repeats: int = 1) -> None:
        super().__init__()
        self.chan_repeats = chan_repeats

    def forward(self, x: Tensor) -> Tensor:
        repeats = [1] * x.ndim  # type: ignore[attr-defined]
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
        block_args: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
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
            _layers = [*conv_sequence(in_channels, in_planes // 2, act_layer, norm_layer, drop_layer, conv_layer,
                                      kernel_size=3, stride=2, padding=1, bias=False),
                       *conv_sequence(in_planes // 2, in_planes // 2, act_layer, norm_layer, drop_layer, conv_layer,
                                      kernel_size=3, stride=1, padding=1, bias=False),
                       *conv_sequence(in_planes // 2, in_planes, act_layer, norm_layer, drop_layer, conv_layer,
                                      kernel_size=3, stride=1, padding=1, bias=False)]
        else:
            _layers = conv_sequence(in_channels, in_planes, act_layer, norm_layer, drop_layer, conv_layer,
                                    kernel_size=7, stride=2, padding=3, bias=False)
        if stem_pool:
            _layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Optional tensor repetitions along channel axis (mainly for TridentNet)
        if num_repeats > 1:
            _layers.append(ChannelRepeat(num_repeats))

        # Consecutive convolutional blocks
        stride = 1
        # Block args
        if block_args is None:
            block_args = dict(groups=1)
        if not isinstance(block_args, list):
            block_args = [block_args] * len(num_blocks)
        for _num_blocks, _planes, _block_args in zip(num_blocks, planes, block_args):
            _layers.append(self._make_layer(block, _num_blocks, in_planes, _planes, stride, width_per_group,
                                            act_layer=act_layer, norm_layer=norm_layer, drop_layer=drop_layer,
                                            avg_downsample=avg_downsample, num_repeats=num_repeats,
                                            block_args=_block_args))
            in_planes = block.expansion * _planes
            stride = 2

        super().__init__(OrderedDict([
            ('features', nn.Sequential(*_layers)),
            ('pool', GlobalAvgPool2d(flatten=True)),
            ('head', nn.Linear(num_repeats * in_planes, num_classes))]))

        # Init all layers
        init.init_module(self, nonlinearity='relu')

        # Init shortcut
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    m.convs[2][1].weight.data.zero_()  # type: ignore[index, union-attr]
                elif isinstance(m, BasicBlock):
                    m.convs[1][1].weight.data.zero_()  # type: ignore[index, union-attr]

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
        block_args: Optional[Dict[str, Any]] = None
    ) -> nn.Sequential:

        downsample = None
        if stride != 1 or in_planes != planes * block.expansion:
            # Downsampling from ResNet-D
            if avg_downsample:
                downsample = nn.Sequential(nn.AvgPool2d(stride, ceil_mode=True, count_include_pad=False),
                                           *conv_sequence(num_repeats * in_planes,
                                                          num_repeats * planes * block.expansion,
                                                          None, norm_layer, drop_layer, conv_layer,
                                                          kernel_size=1, stride=1, bias=False))
            else:
                downsample = nn.Sequential(*conv_sequence(num_repeats * in_planes,
                                                          num_repeats * planes * block.expansion,
                                                          None, norm_layer, drop_layer, conv_layer,
                                                          kernel_size=1, stride=stride, bias=False))
        if block_args is None:
            block_args = {}
        layers = [block(in_planes, planes, stride, downsample, base_width=width_per_group,
                        act_layer=act_layer, norm_layer=norm_layer, drop_layer=drop_layer, **block_args)]

        for _ in range(num_blocks - 1):
            layers.append(block(block.expansion * planes, planes, 1, None, base_width=width_per_group,
                                act_layer=act_layer, norm_layer=norm_layer, drop_layer=drop_layer, **block_args))

        return nn.Sequential(*layers)


def _resnet(arch: str, pretrained: bool, progress: bool, **kwargs: Any) -> ResNet:

    # Retrieve the correct block type
    block = sys.modules[__name__].__dict__[default_cfgs[arch]['block']]

    # Build the model
    model = ResNet(block, default_cfgs[arch]['num_blocks'], [64, 128, 256, 512], **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'], progress)

    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _resnet('resnet18', pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-34 from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _resnet('resnet34', pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-50 from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _resnet('resnet50', pretrained, progress, **kwargs)


def resnet50d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-50-D from
    `"Bag of Tricks for Image Classification with Convolutional Neural Networks"
    <https://arxiv.org/pdf/1812.01187.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _resnet('resnet50d', pretrained, progress, deep_stem=True, avg_downsample=True, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-101 from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _resnet('resnet101', pretrained, progress, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-152 from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _resnet('resnet152', pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNeXt-50 from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    kwargs['width_per_group'] = 4
    block_args = dict(groups=32)
    return _resnet('resnext50_32x4d', pretrained, progress, block_args=block_args, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNeXt-101 from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    kwargs['width_per_group'] = 8
    block_args = dict(groups=32)
    return _resnet('resnext101_32x8d', pretrained, progress, block_args=block_args, **kwargs)
