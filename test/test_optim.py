# Copyright (C) 2019-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import torch
from torch.nn import functional as F
from torchvision.models import resnet18

from holocron import optim


def _test_optimizer(name: str) -> None:

    lr = 1e-4
    input_shape = (3, 224, 224)
    num_batches = 4
    # Get model and optimizer
    model = resnet18(num_classes=10)
    for n, m in model.named_children():
        if n != 'fc':
            for p in m.parameters():
                p.requires_grad_(False)
    optimizer = optim.__dict__[name](model.fc.parameters(), lr=lr)

    # Save param value
    _p = model.fc.weight
    p_val = _p.data.clone()

    # Random inputs
    input_t = torch.rand((num_batches, *input_shape), dtype=torch.float32)
    target = torch.zeros(num_batches, dtype=torch.long)

    # Update
    optimizer.zero_grad()
    output = model(input_t)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()

    # Test
    assert _p.grad is not None
    assert not torch.equal(_p.data, p_val)


def test_lars():
    _test_optimizer('Lars')


def test_lamb():
    _test_optimizer('Lamb')


def test_ralars():
    _test_optimizer('RaLars')


def test_tadam():
    _test_optimizer('TAdam')


def test_adabelief():
    _test_optimizer('AdaBelief')


def test_adamp():
    _test_optimizer('AdamP')
