# Copyright (C) 2019-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import inspect

import torch

from holocron.nn import functional as F
from holocron.nn.modules import activation


def _test_activation_function(fn, input_shape):

    # Optional testing
    fn_args = inspect.signature(fn).parameters.keys()
    cfg = {}
    if 'inplace' in fn_args:
        cfg['inplace'] = [False, True]

    # Generate inputs
    x = torch.rand(input_shape)

    # Optional argument testing
    kwargs = {}
    for inplace in cfg.get('inplace', [None]):
        if isinstance(inplace, bool):
            kwargs['inplace'] = inplace
        out = fn(x, **kwargs)
        assert out.shape == x.shape
        if kwargs.get('inplace', False):
            assert x.data_ptr() == out.data_ptr()


def test_silu():
    _test_activation_function(F.silu, (4, 3, 32, 32))
    assert repr(activation.SiLU()) == "SiLU()"


def test_mish():
    _test_activation_function(F.mish, (4, 3, 32, 32))
    assert repr(activation.Mish()) == "Mish()"


def test_hard_mish():
    _test_activation_function(F.hard_mish, (4, 3, 32, 32))
    assert repr(activation.HardMish()) == "HardMish()"


def test_nl_relu():
    _test_activation_function(F.nl_relu, (4, 3, 32, 32))
    assert repr(activation.NLReLU()) == "NLReLU()"


def test_frelu():
    mod = activation.FReLU(8).eval()
    with torch.no_grad():
        _test_activation_function(mod.forward, (4, 8, 32, 32))
    assert len(repr(mod).split('\n')) == 4
