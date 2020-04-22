import unittest
import inspect
import torch
from holocron.nn import functional as F
from holocron.nn.modules import activation, loss


class Tester(unittest.TestCase):

    def _test_activation_function(self, name, input_shape):
        fn = F.__dict__[name]

        # Optional testing
        fn_args = inspect.signature(fn).parameters.keys()
        cfg = {}
        if 'inplace' in fn_args:
            cfg['inplace'] = [False, True]

        # Generate inputs
        x = torch.rand(input_shape)

        # Optional argument testing
        kwargs = {}
        for inplace in cfg.get('inplace', [None]):
            if isinstance(inplace, bool):
                kwargs['inplace'] = inplace
            out = fn(x, **kwargs)
            self.assertEqual(out.size(), x.size())
            if kwargs.get('inplace', False):
                self.assertEqual(x.data_ptr(), out.data_ptr())

    def _test_loss_function(self, name):

        num_batches = 2
        num_classes = 4
        # 4 classes
        x = torch.ones(num_batches, num_classes, 20, 20)
        x[:, 0, ...] = 10

        # Identical target
        target = torch.zeros((num_batches, 20, 20), dtype=torch.long)
        loss_fn = F.__dict__[name]
        self.assertAlmostEqual(loss_fn(x, target).item(), 0)
        self.assertTrue(torch.allclose(loss_fn(x, target, reduction='none'),
                                       torch.zeros((num_batches, 20, 20), dtype=x.dtype)))

        # Check that class rescaling works
        x = torch.rand(num_batches, num_classes, 20, 20)
        target = (num_classes * torch.rand(num_batches, 20, 20)).to(torch.long)
        weights = torch.ones(num_classes)
        self.assertEqual(loss_fn(x, target).item(), loss_fn(x, target, weight=weights).item())

        # Check that ignore_index works
        self.assertEqual(loss_fn(x, target).item(), loss_fn(x, target, ignore_index=num_classes).item())
        # Ignore an index we are certain to be in the target
        self.assertNotEqual(loss_fn(x, target).item(),
                            loss_fn(x, target, ignore_index=torch.unique(target)[0].item()).item())

        # Test reduction
        self.assertEqual(loss_fn(x, target, reduction='sum').item(), loss_fn(x, target, reduction='none').sum().item())
        self.assertEqual(loss_fn(x, target).item(),
                         (loss_fn(x, target, reduction='sum') / target.view(-1).shape[0]).item())

    def test_focal_loss(self):

        # Common verification
        self._test_loss_function('focal_loss')

        num_batches = 2
        num_classes = 4
        x = torch.rand(num_batches, num_classes, 20, 20)
        target = (num_classes * torch.rand(num_batches, 20, 20)).to(torch.long)

        # Value check
        self.assertAlmostEqual(F.focal_loss(x, target, gamma=0).item(),
                               torch.nn.functional.cross_entropy(x, target).item(), places=5)
        # Equal probabilities
        x = torch.ones(num_batches, num_classes, 20, 20)
        self.assertAlmostEqual((1 - 1 / num_classes) * F.focal_loss(x, target, gamma=0).item(),
                               F.focal_loss(x, target, gamma=1).item(), places=5)

    def _test_activation_module(self, name, input_shape):
        module = activation.__dict__[name]

        # Optional testing
        fn_args = inspect.signature(module).parameters.keys()
        cfg = {}
        if 'inplace' in fn_args:
            cfg['inplace'] = [False, True]

        # Generate inputs
        x = torch.rand(input_shape)

        # Optional argument testing
        kwargs = {}
        for inplace in cfg.get('inplace', [None]):
            if isinstance(inplace, bool):
                kwargs['inplace'] = inplace
            out = module(**kwargs)(x)
            self.assertEqual(out.size(), x.size())
            if kwargs.get('inplace', False):
                self.assertEqual(x.data_ptr(), out.data_ptr())

    def _test_loss_module(self, name):

        num_batches = 2
        num_classes = 4
        # 4 classes
        x = torch.ones(num_batches, num_classes, 20, 20)
        x[:, 0, ...] = 10

        # Identical target
        target = torch.zeros((num_batches, 20, 20), dtype=torch.long)
        criterion = loss.__dict__[name]()
        self.assertAlmostEqual(criterion(x, target).item(), 0)
        criterion = loss.__dict__[name](reduction='none')
        self.assertTrue(torch.allclose(criterion(x, target),
                                       torch.zeros((num_batches, 20, 20), dtype=x.dtype)))


act_fns = ['mish', 'nl_relu']

for fn_name in act_fns:
    def do_test(self, fn_name=fn_name):
        input_shape = (32, 3, 224, 224)
        self._test_activation_function(fn_name, input_shape)

    setattr(Tester, "test_" + fn_name, do_test)

act_modules = ['Mish', 'NLReLU']

for mod_name in act_modules:
    def do_test(self, mod_name=mod_name):
        input_shape = (32, 3, 224, 224)
        self._test_activation_module(mod_name, input_shape)

    setattr(Tester, "test_" + mod_name, do_test)


loss_modules = ['FocalLoss']

for mod_name in loss_modules:
    def do_test(self, mod_name=mod_name):
        self._test_loss_module(mod_name)

    setattr(Tester, "test_" + mod_name, do_test)


if __name__ == '__main__':
    unittest.main()
