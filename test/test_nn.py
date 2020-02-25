import unittest
import inspect
import torch
from holocron.nn import functional as F
from holocron.nn.modules import activation


def get_functionals():
    # Get all activation functions
    return [k for k, v in F.__dict__.items() if callable(v)]


def get_activation_modules():
    # Get all activation modules
    return [k for k, v in activation.__dict__.items() if callable(v)]


class Tester(unittest.TestCase):

    def _test_activation_functions(self, name, input_shape):
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

    def _test_activation_modules(self, name, input_shape):
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


for fn_name in get_functionals():
    def do_test(self, fn_name=fn_name):
        input_shape = (32, 3, 224, 224)
        self._test_activation_functions(fn_name, input_shape)

    setattr(Tester, "test_" + fn_name, do_test)

for activation_name in get_activation_modules():
    def do_test(self, activation_name=activation_name):
        input_shape = (32, 3, 224, 224)
        self._test_activation_modules(activation_name, input_shape)

    setattr(Tester, "test_" + activation_name, do_test)


if __name__ == '__main__':
    unittest.main()
