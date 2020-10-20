import logging
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from holocron.nn import GlobalAvgPool2d
from .resnet import ResNet, _ResBlock
from .utils import conv_sequence


__all__ = ['SoftAttentionLayer', 'SKConv2d', 'SKBottleneck', 'sknet50', 'sknet101', 'sknet152']


default_cfgs = {
    'sknet50': {'block': 'SKBottleneck', 'num_blocks': [3, 4, 6, 3],
                'url': None},
    'sknet101': {'block': 'SKBottleneck', 'num_blocks': [3, 4, 23, 3],
                 'url': None},
    'sknet152': {'block': 'SKBottleneck', 'num_blocks': [3, 8, 86, 3],
                 'url': None},
}


class SoftAttentionLayer(nn.Sequential):

    def __init__(self, channels, sa_ratio=16, out_multiplier=1, act_layer=None, norm_layer=None, drop_layer=None):
        super().__init__(GlobalAvgPool2d(flatten=False),
                         *conv_sequence(channels, max(channels // sa_ratio, 32), act_layer, norm_layer, drop_layer,
                                        kernel_size=1, stride=1, bias=False),
                         *conv_sequence(max(channels // sa_ratio, 32), channels * out_multiplier,
                                        nn.Sigmoid(), None, drop_layer, kernel_size=1, stride=1))


class SKConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, m=2, sa_ratio=16,
                 act_layer=None, norm_layer=None, drop_layer=None, **kwargs):
        super().__init__()
        self.path_convs = nn.ModuleList([nn.Sequential(*conv_sequence(in_channels, out_channels,
                                                                      act_layer, norm_layer, drop_layer,
                                                                      kernel_size=3, bias=False, dilation=idx + 1,
                                                                      padding=idx + 1, **kwargs))
                                         for idx in range(m)])
        self.sa = SoftAttentionLayer(out_channels, sa_ratio, m, act_layer, norm_layer, drop_layer)

    def forward(self, x):

        paths = torch.stack([path_conv(x) for path_conv in self.path_convs], dim=1)

        b, m, c = paths.shape[:3]
        z = self.sa(paths.sum(dim=1)).view(b, m, c, 1, 1)
        attention_factors = torch.softmax(z, dim=1)
        out = (attention_factors * paths).sum(dim=1)

        return out


class SKBottleneck(_ResBlock):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=32, base_width=64, dilation=1,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None, **kwargs):

        width = int(planes * (base_width / 64.)) * groups
        super().__init__(
            [*conv_sequence(inplanes, width, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=1,
                            stride=1, bias=False, **kwargs),
             SKConv2d(width, width, 2, 16, act_layer, norm_layer, drop_layer, groups=groups, stride=stride),
             *conv_sequence(width, planes * self.expansion, None, norm_layer, drop_layer, conv_layer, kernel_size=1,
                            stride=1, bias=False, **kwargs)],
            downsample, act_layer)


def _sknet(arch, pretrained, progress, **kwargs):

    # Build the model
    model = ResNet(SKBottleneck, default_cfgs[arch]['num_blocks'], [64, 128, 256, 512], **kwargs)
    # Load pretrained parameters
    if pretrained:
        if default_cfgs[arch]['url'] is None:
            logging.warning(f"Invalid model URL for {arch}, using default initialization.")
        else:
            state_dict = load_state_dict_from_url(default_cfgs[arch]['url'],
                                                  progress=progress)
            model.load_state_dict(state_dict)

    return model


def sknet50(pretrained=False, progress=True, **kwargs):
    """SKNet-50 from
    `"Selective Kernel Networks" <https://arxiv.org/pdf/1903.06586.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _sknet('sknet50', pretrained, progress, **kwargs)


def sknet101(pretrained=False, progress=True, **kwargs):
    """SKNet-101 from
    `"Selective Kernel Networks" <https://arxiv.org/pdf/1903.06586.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _sknet('sknet50', pretrained, progress, **kwargs)


def sknet152(pretrained=False, progress=True, **kwargs):
    """SKNet-152 from
    `"Selective Kernel Networks" <https://arxiv.org/pdf/1903.06586.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """

    return _sknet('sknet50', pretrained, progress, **kwargs)
