# Copyright (C) 2019-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import torch.nn as nn
from torch.nn.modules.conv import _ConvNd


def init_module(module: nn.Module, nonlinearity: str = "relu") -> None:
    """Initializes pytorch modules

    Args:
        module: module to initialize
        nonlinearity: linearity to initialize convolutions for
    """

    for m in module.modules():
        if isinstance(m, _ConvNd):
            nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity=nonlinearity)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
