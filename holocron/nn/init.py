import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from typing import Optional


def init_module(module: nn.Module, nonlinearity: Optional[str] = None) -> None:
    """Initializes pytorch modules

    Args:
        module: module to initialize
        nonlinearity: linearity to initialize convolutions for
    """

    for m in module.modules():
        if isinstance(m, _ConvNd):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity=nonlinearity)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
