import unittest
import torch
from holocron import utils


class MockDataset(torch.utils.data.Dataset):
    """Mock dataset generating a random sample and a fixed zero target"""
    def __init__(self, n):
        super(MockDataset, self).__init__()
        self.n = n

    def __getitem__(self, idx):
        return torch.rand(32), torch.zeros(1, dtype=torch.long)

    def __len__(self):
        return self.n


def train_one_batch(model, x, target, optimizer, criterion, device):
    """Mock batch training function"""

    x, target = x.to(device), target.to(device)
    output = model(x)
    batch_loss = criterion(output, target.view(-1))

    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    return batch_loss.item()


class MiscTester(unittest.TestCase):
    def test_lr_finder(self):

        num_it = 10
        batch_size = 2
        start_lr, end_lr = 1e-7, 10
        # Generate all dependencies
        model = torch.nn.Linear(32, 5)
        train_loader = torch.utils.data.DataLoader(MockDataset(num_it * batch_size), batch_size=batch_size)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        # Perform the iterations
        lrs, losses = utils.misc.lr_finder(train_one_batch, model, train_loader, optimizer, criterion,
                                           num_it=num_it, start_lr=start_lr, end_lr=end_lr, stop_div=False)

        # Check integrity of results
        self.assertIsInstance(lrs, list)
        self.assertIsInstance(losses, list)
        self.assertEqual(len(lrs), num_it)
        self.assertEqual(len(losses), num_it)
        self.assertEqual(lrs[0], start_lr)
        self.assertAlmostEqual(lrs[-1], end_lr)


class DataTester(unittest.TestCase):
    def test_mixup(self):

        num_it = 10
        batch_size = 2
        # Generate all dependencies
        loader = torch.utils.data.DataLoader(MockDataset(num_it * batch_size), batch_size=batch_size,
                                             collate_fn=utils.data.mixup_collate)

        inputs, targets_a, targets_b, lam = next(iter(loader))
        self.assertEqual(inputs.shape, (batch_size, 32))
        self.assertEqual(targets_a.shape, targets_b.shape)


if __name__ == '__main__':
    unittest.main()
