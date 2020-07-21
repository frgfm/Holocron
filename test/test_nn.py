import unittest
import inspect
import torch
import torch.nn as nn
from holocron.nn import functional as F
from holocron.nn.init import init_module
from holocron.nn.modules import activation, conv, loss, downsample, dropblock


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

    def _test_loss_function(self, name, same_loss=0, multi_label=False):

        num_batches = 2
        num_classes = 4
        # 4 classes
        x = torch.ones(num_batches, num_classes)
        x[:, 0, ...] = 10

        # Identical target
        if multi_label:
            target = torch.zeros_like(x)
            target[:, 0] = 1.
        else:
            target = torch.zeros(num_batches, dtype=torch.long)
        loss_fn = F.__dict__[name]
        self.assertAlmostEqual(loss_fn(x, target).item(), same_loss, places=3)
        self.assertTrue(torch.allclose(loss_fn(x, target, reduction='none'),
                                       same_loss * torch.ones(num_batches, dtype=x.dtype),
                                       atol=1e-3))

        # Check that class rescaling works
        x = torch.rand(num_batches, num_classes)
        if multi_label:
            target = torch.rand(x.shape)
        else:
            target = (num_classes * torch.rand(num_batches)).to(torch.long)
        weights = torch.ones(num_classes)
        self.assertEqual(loss_fn(x, target).item(), loss_fn(x, target, weight=weights).item())

        # Check that ignore_index works
        self.assertEqual(loss_fn(x, target).item(), loss_fn(x, target, ignore_index=num_classes).item())
        # Ignore an index we are certain to be in the target
        if multi_label:
            ignore_index = torch.unique(target.argmax(dim=1))[0].item()
        else:
            ignore_index = torch.unique(target)[0].item()
        self.assertNotEqual(loss_fn(x, target).item(),
                            loss_fn(x, target, ignore_index=ignore_index).item())

        # Test reduction
        self.assertAlmostEqual(loss_fn(x, target, reduction='sum').item(),
                               loss_fn(x, target, reduction='none').sum().item(), places=6)
        self.assertAlmostEqual(loss_fn(x, target, reduction='mean').item(),
                               (loss_fn(x, target, reduction='sum') / target.shape[0]).item(), places=6)

    def test_focal_loss(self):

        # Common verification
        self._test_loss_function('focal_loss')

        num_batches = 2
        num_classes = 4
        x = torch.rand(num_batches, num_classes, 20, 20)
        target = (num_classes * torch.rand(num_batches, 20, 20)).to(torch.long)

        # Value check
        self.assertAlmostEqual(F.focal_loss(x, target, gamma=0).item(),
                               nn.functional.cross_entropy(x, target).item(), places=5)
        # Equal probabilities
        x = torch.ones(num_batches, num_classes, 20, 20)
        self.assertAlmostEqual((1 - 1 / num_classes) * F.focal_loss(x, target, gamma=0).item(),
                               F.focal_loss(x, target, gamma=1).item(), places=5)

    def test_ls_celoss(self):

        num_batches = 2
        num_classes = 4

        # Common verification
        self._test_loss_function('ls_cross_entropy', 0.1 / num_classes * (num_classes - 1) * 9)

        x = torch.rand(num_batches, num_classes, 20, 20)
        target = (num_classes * torch.rand(num_batches, 20, 20)).to(torch.long)

        # Value check
        self.assertAlmostEqual(F.ls_cross_entropy(x, target, eps=0).item(),
                               nn.functional.cross_entropy(x, target).item(), places=5)
        self.assertAlmostEqual(F.ls_cross_entropy(x, target, eps=1).item(),
                               -1 / num_classes * nn.functional.log_softmax(x, dim=1).sum(dim=1).mean().item(),
                               places=5)

    def test_multilabel_cross_entropy(self):

        num_batches = 2
        num_classes = 4

        # Common verification
        self._test_loss_function('multilabel_cross_entropy', multi_label=True)

        x = torch.rand(num_batches, num_classes, 20, 20)
        target = torch.zeros_like(x)
        target[:, 0] = 1.

        # Value check
        self.assertAlmostEqual(F.multilabel_cross_entropy(x, target).item(),
                               nn.functional.cross_entropy(x, target.argmax(dim=1)).item(), places=5)

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

    def _test_loss_module(self, name, fn_name, multi_label=False):

        num_batches = 2
        num_classes = 4
        x = torch.rand(num_batches, num_classes, 20, 20)

        # Identical target
        if multi_label:
            target = torch.rand(x.shape)
        else:
            target = (num_classes * torch.rand(num_batches, 20, 20)).to(torch.long)
        criterion = loss.__dict__[name]()
        self.assertEqual(criterion(x, target).item(),
                         F.__dict__[fn_name](x, target).item())
        criterion = loss.__dict__[name](reduction='none')
        self.assertTrue(torch.equal(criterion(x, target),
                                    F.__dict__[fn_name](x, target, reduction='none')))

    def test_concatdownsample2d(self):

        num_batches = 2
        num_chan = 4
        scale_factor = 2
        x = torch.arange(num_batches * num_chan * 4 ** 2).view(num_batches, num_chan, 4, 4)

        # Test functional API
        self.assertRaises(AssertionError, F.concat_downsample2d, x, 3)
        out = F.concat_downsample2d(x, scale_factor)
        self.assertEqual(out.shape, (num_batches, num_chan * scale_factor ** 2,
                                     x.shape[2] // scale_factor, x.shape[3] // scale_factor))

        # Check first and last values
        self.assertTrue(torch.equal(out[0][0], torch.tensor([[0, 2], [8, 10]])))
        self.assertTrue(torch.equal(out[0][-num_chan], torch.tensor([[5, 7], [13, 15]])))
        # Test module
        mod = downsample.ConcatDownsample2d(scale_factor)
        self.assertTrue(torch.equal(mod(x), out))

    def test_init(self):

        module = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))

        # Check that each layer was initialized correctly
        init_module(module, 'leaky_relu')
        self.assertTrue(torch.all(module[0].bias.data == 0))
        self.assertTrue(torch.all(module[1].weight.data == 1))
        self.assertTrue(torch.all(module[1].bias.data == 0))

    def test_mixuploss(self):

        num_batches = 8
        num_classes = 10
        # Generate inputs
        x = torch.rand((num_batches, num_classes, 20, 20))
        target_a = torch.rand((num_batches, num_classes, 20, 20))
        target_b = torch.rand((num_batches, num_classes, 20, 20))
        lam = 0.9

        # Take a criterion compatible with one-hot encoded targets
        criterion = loss.MultiLabelCrossEntropy()
        mixup_criterion = loss.MixupLoss(criterion)

        # Check the repr
        self.assertEqual(mixup_criterion.__repr__(), f"Mixup_{criterion.__repr__()}")

        # Check the forward
        out = mixup_criterion(x, target_a, target_b, lam)
        self.assertEqual(out.item(), (lam * criterion(x, target_a) + (1 - lam) * criterion(x, target_b)).item())
        self.assertEqual(mixup_criterion(x, target_a, target_b, 1).item(), criterion(x, target_a).item())
        self.assertEqual(mixup_criterion(x, target_a, target_b, 0).item(), criterion(x, target_b).item())

    def _test_xcorr2d(self, name):

        x = torch.rand(2, 8, 19, 19)

        # Normalized Conv
        for padding_mode in ['zeros', 'reflect']:
            mod = conv.__dict__[name](8, 16, 3, padding=1, padding_mode=padding_mode)

            with torch.no_grad():
                out = mod(x)
            self.assertEqual(out.shape, (2, 16, 19, 19))

    def test_slimconv2d(self):

        x = torch.rand(2, 8, 19, 19)

        mod = conv.SlimConv2d(8, 3, padding=1, r=32, L=2)

        with torch.no_grad():
            out = mod(x)
        self.assertEqual(out.shape, (2, 6, 19, 19))

    def test_dropblock2d(self):

        x = torch.rand(2, 8, 19, 19)

        # Drop probability of 1
        mod = dropblock.DropBlock2d(1., 1, inplace=False)

        with torch.no_grad():
            out = mod(x)
        self.assertTrue(torch.equal(out, torch.zeros_like(x)))

        # Drop probability of 0
        mod = dropblock.DropBlock2d(0., 3, inplace=False)

        with torch.no_grad():
            out = mod(x)
        self.assertTrue(torch.equal(out, x))
        self.assertNotEqual(out.data_ptr, x.data_ptr)

        # Check inplace
        mod = dropblock.DropBlock2d(1., 3, inplace=True)

        with torch.no_grad():
            out = mod(x)
        self.assertEqual(out.data_ptr, x.data_ptr)


act_fns = ['silu', 'mish', 'nl_relu']

for fn_name in act_fns:
    def do_test(self, fn_name=fn_name):
        input_shape = (32, 3, 224, 224)
        self._test_activation_function(fn_name, input_shape)

    setattr(Tester, "test_" + fn_name, do_test)

act_modules = ['SiLU', 'Mish', 'NLReLU']

for mod_name in act_modules:
    def do_test(self, mod_name=mod_name):
        input_shape = (32, 3, 224, 224)
        self._test_activation_module(mod_name, input_shape)

    setattr(Tester, "test_" + mod_name, do_test)


loss_modules = [('FocalLoss', 'focal_loss'),
                ('LabelSmoothingCrossEntropy', 'ls_cross_entropy')]

for (mod_name, fn_name) in loss_modules:
    def do_test(self, mod_name=mod_name, fn_name=fn_name):
        self._test_loss_module(mod_name, fn_name, multi_label=False)

    setattr(Tester, "test_" + mod_name, do_test)


loss_modules = [('MultiLabelCrossEntropy', 'multilabel_cross_entropy')]

for (mod_name, fn_name) in loss_modules:
    def do_test(self, mod_name=mod_name, fn_name=fn_name):
        self._test_loss_module(mod_name, fn_name, multi_label=True)

    setattr(Tester, "test_" + mod_name, do_test)


xcorr_modules = ['NormConv2d', 'Add2d']

for mod_name in xcorr_modules:
    def do_test(self, mod_name=mod_name):
        self._test_xcorr2d(mod_name)

    setattr(Tester, "test_" + mod_name, do_test)


if __name__ == '__main__':
    unittest.main()
