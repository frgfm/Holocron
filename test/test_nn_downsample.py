# Copyright (C) 2019-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import pytest
import torch
from torch import nn

from holocron.nn import functional as F
from holocron.nn.modules import downsample


def test_concatdownsample2d():

    num_batches = 2
    num_chan = 4
    scale_factor = 2
    x = torch.arange(num_batches * num_chan * 4 ** 2).view(num_batches, num_chan, 4, 4)

    # Test functional API
    with pytest.raises(AssertionError):
        F.concat_downsample2d(x, 3)
    out = F.concat_downsample2d(x, scale_factor)
    assert out.shape == (num_batches, num_chan * scale_factor ** 2,
                         x.shape[2] // scale_factor, x.shape[3] // scale_factor)

    # Check first and last values
    assert torch.equal(out[0][0], torch.tensor([[0, 2], [8, 10]]))
    assert torch.equal(out[0][-num_chan], torch.tensor([[5, 7], [13, 15]]))
    # Test module
    mod = downsample.ConcatDownsample2d(scale_factor)
    assert torch.equal(mod(x), out)
    # Test JIT module
    mod = downsample.ConcatDownsample2dJit(scale_factor)
    assert torch.equal(mod(x), out)


def test_globalavgpool2d():

    x = torch.rand(2, 4, 16, 16)

    # Check that ops are doing the same thing
    ref = nn.AdaptiveAvgPool2d(1)
    mod = downsample.GlobalAvgPool2d(flatten=False)
    out = mod(x)
    assert torch.equal(out, ref(x))
    assert out.data_ptr != x.data_ptr

    # Check that flatten works
    x = torch.rand(2, 4, 16, 16)
    mod = downsample.GlobalAvgPool2d(flatten=True)
    assert torch.equal(mod(x), ref(x).view(*x.shape[:2]))


def test_blurpool2d():

    with pytest.raises(AssertionError):
        downsample.BlurPool2d(1, 0)

    # Generate inputs
    num_batches = 2
    num_chan = 8
    x = torch.rand((num_batches, num_chan, 5, 5))
    mod = downsample.BlurPool2d(num_chan, stride=2)

    # Optional argument testing
    with torch.no_grad():
        out = mod(x)
    assert out.shape == (num_batches, num_chan, 3, 3)

    k = torch.tensor([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
    assert torch.allclose(out[..., 1, 1], (x[..., 1:-1, 1:-1] * k[None, None, ...]).sum(dim=(2, 3)), atol=1e-7)


def test_zpool():

    num_batches = 2
    num_chan = 4
    x = torch.rand((num_batches, num_chan, 32, 32))

    # Test functional API
    out = F.z_pool(x, 1)
    assert out.shape == (num_batches, 2, 32, 32)
    assert out[0, 0, 0, 0].item() == x[0, :, 0, 0].max().item()
    assert out[0, 1, 0, 0].item() == x[0, :, 0, 0].mean().item()

    # Test module
    mod = downsample.ZPool(1)
    assert torch.equal(mod(x), out)
