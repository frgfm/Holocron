# Copyright (C) 2019-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import pytest
import torch
from holocron.nn.modules import upsample
from holocron.nn import functional as F


def test_stackupsample2d():

    num_batches = 2
    num_chan = 4
    x = torch.arange(num_batches * num_chan * 4 ** 2).view(num_batches, num_chan, 4, 4)

    # Test functional API
    with pytest.raises(AssertionError):
        F.stack_upsample2d(x, 3)

    # Check that it's the inverse of concat_downsample2d
    x = torch.rand((num_batches, num_chan, 32, 32))
    down = F.concat_downsample2d(x, scale_factor=2)
    up = F.stack_upsample2d(down, scale_factor=2)
    assert torch.equal(up, x)

    # module interface
    mod = upsample.StackUpsample2d(scale_factor=2)
    assert torch.equal(mod(down), up)
    assert repr(mod) == "StackUpsample2d(scale_factor=2)"
