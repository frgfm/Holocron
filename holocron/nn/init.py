# -*- coding: utf-8 -*-

"""
Parameter initialization
"""

import torch.nn as nn


def init_module(module, nonlinearity=None):
    """Initializes pytorch modules

    Args:
        module (torch.nn.Module): module to initialize
        nonlinearity (str, optional): linearity to initialize convolutions for
    """

    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity=nonlinearity)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
