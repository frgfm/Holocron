#!usr/bin/python
# -*- coding: utf-8 -*-

'''
Functional interface
'''

import torch
import torch.nn.functional as F


def mish(x, inplace=False):
    """Implements the Mish activation function

    Args:
        x (torch.Tensor): input tensor
    """
    if inplace:
    	raise NotImplementedError()

    return x * torch.tanh(F.softplus(x))
