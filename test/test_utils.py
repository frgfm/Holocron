# Copyright (C) 2019-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import torch
from torch.utils.data import DataLoader, Dataset

from holocron import utils


class MockDataset(Dataset):
    """Mock dataset generating a random sample and a fixed zero target"""
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __getitem__(self, idx):
        return torch.rand(32), 0

    def __len__(self):
        return self.n


def test_mixup():

    num_it = 10
    batch_size = 2
    # Generate all dependencies
    loader = DataLoader(MockDataset(num_it * batch_size), batch_size=batch_size, collate_fn=utils.data.mixup_collate)

    inputs, targets_a, targets_b, lam = next(iter(loader))
    assert inputs.shape == (batch_size, 32)
    assert targets_a.shape == targets_b.shape


def _train_one_batch(model, x, target, optimizer, criterion, device):
    """Mock batch training function"""

    x, target = x.to(device), target.to(device)
    output = model(x)
    batch_loss = criterion(output, target)

    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    return batch_loss.item()


def test_lr_finder():
    num_it = 10
    batch_size = 2
    start_lr, end_lr = 1e-7, 10
    # Generate all dependencies
    model = torch.nn.Linear(32, 5)
    train_loader = DataLoader(MockDataset(num_it * batch_size), batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    # Perform the iterations
    lrs, losses = utils.misc.lr_finder(_train_one_batch, model, train_loader, optimizer, criterion,
                                       num_it=num_it, start_lr=start_lr, end_lr=end_lr, stop_div=False)

    # Check integrity of results
    assert isinstance(lrs, list) and isinstance(losses, list)
    assert len(lrs) == len(losses) == num_it
    assert lrs[0] == start_lr and abs(lrs[-1] - end_lr) / lrs[-1] < 1e-7
