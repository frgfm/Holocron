import unittest
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision.models import resnet18
from holocron import optim


class Tester(unittest.TestCase):

    def _get_model(self, num_classes=50):
        return resnet18(num_classes=num_classes)

    def _test_lr_scheduler(self, name, lr=1e-4, ratio_preserved=True, **kwargs):

        # Get model and optimizer
        model = self._get_model()
        # Create param groups
        bias_params, weight_params = [], []
        for n, p in model.named_parameters():
            if n.endswith('.bias'):
                bias_params.append(p)
            else:
                weight_params.append(p)
        optimizer = Adam([dict(params=weight_params, lr=2 * lr),
                          dict(params=bias_params, lr=lr)])

        scheduler = optim.lr_scheduler.__dict__[name](optimizer, **kwargs)

        # Check that LR is different after a scheduler step
        scheduler.step()
        self.assertNotEqual(optimizer.param_groups[1]['lr'], lr)

        # Check that LR ratio is preserved
        if ratio_preserved:
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'] / optimizer.param_groups[1]['lr'], 2.)

    def test_onecycle(self, steps=500):

        self._test_lr_scheduler('OneCycleScheduler', total_size=steps, cycle_momentum=False)

    def _test_optimizer(self, name, input_shape=(3, 224, 224), nb_batches=4):

        # Get model and optimizer
        model = self._get_model()
        optimizer = optim.__dict__[name](model.parameters(), lr=1e-4)
        criterion = CrossEntropyLoss()

        # Save param value
        _p = list(model.parameters())[0]
        p_val = _p.data.clone()

        # Random inputs
        input_t = torch.rand((nb_batches, 3, 224, 224), dtype=torch.float32)
        target = torch.zeros(nb_batches, dtype=torch.long)

        # Update
        optimizer.zero_grad()
        output = model(input_t)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Test
        self.assertIsNotNone(_p.grad)
        self.assertFalse(torch.equal(_p.data, p_val))


tested_opt = ['Lamb', 'Lars', 'RAdam', 'RaLars']

for opt_name in tested_opt:
    def do_test(self, fn_name=opt_name):
        input_shape = (3, 224, 224)
        nb_batches = 4
        self._test_optimizer(opt_name, input_shape, nb_batches)

    setattr(Tester, "test_" + opt_name, do_test)


if __name__ == '__main__':
    unittest.main()
