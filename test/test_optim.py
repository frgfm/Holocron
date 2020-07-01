import unittest
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision.models import resnet18
from holocron import optim


def _get_model(num_classes=50):
    return resnet18(num_classes=num_classes)


def _get_learnable_param(model):

    for p in model.parameters():
        if p.requires_grad:
            return p
    raise AssertionError("No learnable parameter found")


class Tester(unittest.TestCase):
    def _test_lr_scheduler(self, name, lr=1e-4, ratio_preserved=True, **kwargs):

        # Get model and optimizer
        model = _get_model()
        # Create param groups
        bias_params, weight_params = [], []
        for n, p in model.named_parameters():
            if n.endswith('.bias'):
                bias_params.append(p)
            else:
                weight_params.append(p)
        optimizer = SGD([dict(params=weight_params, lr=2 * lr),
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

    def _test_optimizer(self, name):

        lr = 1e-4
        input_shape = (3, 224, 224)
        nb_batches = 4
        # Get model and optimizer
        model = _get_model()
        optimizer = optim.__dict__[name](model.parameters(), lr=lr)
        criterion = CrossEntropyLoss()

        # Save param value
        _p = _get_learnable_param(model)
        p_val = _p.data.clone()

        # Random inputs
        input_t = torch.rand((nb_batches, *input_shape), dtype=torch.float32)
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

    def _test_wrapper(self, name):

        lr = 1e-4
        input_shape = (3, 224, 224)
        nb_batches = 4
        # Get model, optimizer and criterion
        model = _get_model()
        # Pick an optimizer whose update is easy to verify
        optimizer = SGD(model.parameters(), lr=lr)
        criterion = CrossEntropyLoss()

        # Wrap the optimizer
        wrapper = optim.wrapper.__dict__[name](optimizer)

        # Check gradient reset
        wrapper.zero_grad()
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    self.assertTrue(torch.all(p.grad == 0.))

        # Check update step
        _p = _get_learnable_param(model)
        p_val = _p.data.clone()

        # Random inputs
        input_t = torch.rand((nb_batches, *input_shape), dtype=torch.float32)
        target = torch.zeros(nb_batches, dtype=torch.long)

        # Update
        output = model(input_t)
        loss = criterion(output, target)
        loss.backward()
        wrapper.step()

        # Check update rule
        self.assertFalse(torch.equal(_p.data, p_val - lr * _p.grad))


for opt_name in ['Lars', 'Lamb', 'RAdam', 'RaLars', 'TAdam']:
    def opt_test(self, opt_name=opt_name):
        self._test_optimizer(opt_name)

    setattr(Tester, "test_" + opt_name, opt_test)


for wrapper_name in ['Lookahead', 'Scout']:
    def wrap_test(self, wrapper_name=wrapper_name):
        self._test_wrapper(wrapper_name)

    setattr(Tester, "test_" + wrapper_name, wrap_test)


if __name__ == '__main__':
    unittest.main()
