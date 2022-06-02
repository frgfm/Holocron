import pytest
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

    batch_size = 8
    num_classes = 10
    shape = (3, 32, 32)
    with pytest.raises(ValueError):
        utils.data.Mixup(num_classes, alpha=-1.0)
    # Generate all dependencies
    mix = utils.data.Mixup(num_classes, alpha=0.2)
    img, target = torch.rand((batch_size, *shape)), torch.arange(num_classes)[:batch_size]
    mix_img, mix_target = mix(img.clone(), target.clone())
    assert img.shape == (batch_size, *shape)
    assert not torch.equal(img, mix_img)
    assert mix_target.dtype == torch.float32 and mix_target.shape == (batch_size, num_classes)
    assert torch.all(mix_target.sum(dim=1) == 1.0)
    count = (mix_target > 0).sum(dim=1)
    assert torch.all((count == 2.0) | (count == 1.0))

    # Alpha = 0 case
    mix = utils.data.Mixup(num_classes, alpha=0.0)
    mix_img, mix_target = mix(img.clone(), target.clone())
    assert torch.equal(img, mix_img)
    assert mix_target.dtype == torch.float32 and mix_target.shape == (batch_size, num_classes)
    assert torch.all(mix_target.sum(dim=1) == 1.0)
    assert torch.all((mix_target > 0).sum(dim=1) == 1.0)


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
    # Generate all dependencies
    model = torch.nn.Linear(32, 5)
    train_loader = DataLoader(MockDataset(num_it * batch_size), batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    # Perform the iterations
    lrs, losses = utils.misc.lr_finder(
        _train_one_batch,
        model,
        train_loader,
        optimizer,
        criterion,
        num_it=num_it,
        start_lr=start_lr,
        end_lr=end_lr,
        stop_div=False,
    )

    # Check integrity of results
    assert isinstance(lrs, list) and isinstance(losses, list)
    assert len(lrs) == len(losses) == num_it
    assert lrs[0] == start_lr and abs(lrs[-1] - end_lr) / lrs[-1] < 1e-7
