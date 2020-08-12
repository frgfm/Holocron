# -*- coding: utf-8 -*-

"""
Utilities for models
"""

import torch.nn as nn


__all__ = ['conv_sequence']


def conv_sequence(in_channels, out_channels, act_layer=None, norm_layer=None, drop_layer=None,
                  conv_layer=None, bn_channels=None, **kwargs):

    if conv_layer is None:
        conv_layer = nn.Conv2d
    if bn_channels is None:
        bn_channels = out_channels

    conv_seq = [conv_layer(in_channels, out_channels, **kwargs)]

    if callable(norm_layer):
        conv_seq.append(norm_layer(bn_channels))
    if callable(act_layer):
        conv_seq.append(act_layer)
    if callable(drop_layer):
        conv_seq.append(drop_layer(inplace=True))

    return conv_seq
