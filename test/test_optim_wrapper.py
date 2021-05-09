# Copyright (C) 2020-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import torch
from torch.nn import functional as F
from torch.optim import SGD
from torchvision.models import resnet18

from holocron.optim import wrapper


def _test_wrapper(name: str) -> None:

    lr = 1e-4
    input_shape = (3, 224, 224)
    num_batches = 4
    # Get model, optimizer and criterion
    model = resnet18(num_classes=10)
    for n, m in model.named_children():
        if n != 'fc':
            for p in m.parameters():
                p.requires_grad_(False)
    # Pick an optimizer whose update is easy to verify
    optimizer = SGD(model.fc.parameters(), lr=lr)

    # Wrap the optimizer
    opt_wrapper = wrapper.__dict__[name](optimizer)

    # Check gradient reset
    opt_wrapper.zero_grad()
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                assert torch.all(p.grad == 0.)

    # Check update step
    _p = model.fc.weight
    p_val = _p.data.clone()

    # Random inputs
    input_t = torch.rand((num_batches, *input_shape), dtype=torch.float32)
    target = torch.zeros(num_batches, dtype=torch.long)

    # Update
    output = model(input_t)
    loss = F.cross_entropy(output, target)
    loss.backward()
    opt_wrapper.step()

    # Check update rule
    assert not torch.equal(_p.data, p_val - lr * _p.grad)


def test_lookahead():
    _test_wrapper('Lookahead')


def test_scout():
    _test_wrapper('Scout')