# Copyright (C) 2019-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import torch
from holocron import nn


def _test_attention_mod(mod):

    x = torch.rand(2, 4, 8, 8)
    # Check that attention preserves shape
    mod = mod.eval()
    with torch.no_grad():
        out = mod(x)
    assert x.shape == out.shape
    # Check that it doesn't break backprop
    mod = mod.train()
    out = mod(x)
    out.sum().backward()
    assert isinstance(next(mod.parameters()).grad, torch.Tensor)


def test_sam():
    _test_attention_mod(nn.SAM(4))


def test_triplet_attention():
    _test_attention_mod(nn.TripletAttention())
