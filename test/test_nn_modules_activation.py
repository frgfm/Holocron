import unittest
import torch
from holocron.nn.modules import activation


def get_activation_modules():
    # Get all activation functions
    return [k for k, v in activation.__dict__.items() if callable(v)]


class Tester(unittest.TestCase):

    def _test_activation_modules(self, name, input_shape):
        module = activation.__dict__[name]()
        x = torch.rand(input_shape)
        out = module(x)
        self.assertEqual(out.size(), x.size())


for activation_name in get_activation_modules():
    def do_test(self, activation_name=activation_name):
        input_shape = (32, 3, 224, 224)
        self._test_activation_modules(activation_name, input_shape)

    setattr(Tester, "test_" + activation_name, do_test)


if __name__ == '__main__':
    unittest.main()
