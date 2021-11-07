# Copyright (C) 2019-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Any

import torch
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small

from holocron import optim


def _test_optimizer(name: str, **kwargs: Any) -> None:

    lr = 1e-4
    input_shape = (3, 224, 224)
    num_batches = 4
    # Get model and optimizer
    model = mobilenet_v3_small(num_classes=10)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.classifier[3].parameters():
        p.requires_grad_(True)
    optimizer = optim.__dict__[name](model.classifier[3].parameters(), lr=lr, **kwargs)

    # Save param value
    _p = model.classifier[3].weight
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
    _test_optimizer('Lars', momentum=0.9, weight_decay=2e-5)


def test_lamb():
    _test_optimizer('Lamb', weight_decay=2e-5)


def test_ralars():
    _test_optimizer('RaLars', weight_decay=2e-5)


def test_tadam():
    _test_optimizer('TAdam')


def test_adabelief():
    _test_optimizer('AdaBelief')


def test_adamp():
    _test_optimizer('AdamP')
