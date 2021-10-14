# Copyright (C) 2019-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import pytest
import torch
from holocron import nn


def _test_conv2d(mod, input_shape, output_shape):

    x = torch.rand(*input_shape)

    out = mod(x)
    assert out.shape == output_shape
    # Check that backprop works
    out.sum().backward()


def test_normconv2d():
    _test_conv2d(nn.NormConv2d(8, 16, 3, padding=1), (2, 8, 16, 16), (2, 16, 16, 16))
    _test_conv2d(nn.NormConv2d(8, 16, 3, padding=1, padding_mode='reflect'), (2, 8, 16, 16), (2, 16, 16, 16))


def test_add2d():
    _test_conv2d(nn.Add2d(8, 16, 3, padding=1), (2, 8, 16, 16), (2, 16, 16, 16))
    _test_conv2d(nn.Add2d(8, 16, 3, padding=1, padding_mode='reflect'), (2, 8, 16, 16), (2, 16, 16, 16))


def test_slimconv2d():
    _test_conv2d(nn.SlimConv2d(8, 3, padding=1, r=32, L=2), (2, 8, 16, 16), (2, 6, 16, 16))


def test_pyconv2d():
    for num_levels in range(1, 5):
        _test_conv2d(nn.PyConv2d(8, 16, 3, num_levels, padding=1), (2, 8, 16, 16), (2, 16, 16, 16))


def test_lambdalayer():

    with pytest.raises(AssertionError):
        nn.LambdaLayer(3, 31, 16)
    with pytest.raises(AssertionError):
        nn.LambdaLayer(3, 32, 16, r=2)
    with pytest.raises(AssertionError):
        nn.LambdaLayer(3, 32, 16, r=None, n=None)

    _test_conv2d(nn.LambdaLayer(8, 32, 16, r=13), (2, 8, 32, 32), (2, 32, 32, 32))


def test_involution2d():
    _test_conv2d(nn.Involution2d(8, 3, 1, reduction_ratio=2), (2, 8, 16, 16), (2, 8, 16, 16))
    _test_conv2d(nn.Involution2d(8, 3, 1, 2, reduction_ratio=2), (2, 8, 16, 16), (2, 8, 8, 8))
