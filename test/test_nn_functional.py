import unittest
import torch
from holocron.nn import functional as F 


def get_activation_functions():
    # Get all activation functions
    return [k for k, v in F.__dict__.items() if callable(v)]


class Tester(unittest.TestCase):

    def _test_activation_functions(self, name, input_shape):
        fn = F.__dict__[name]
        x = torch.rand(input_shape)
        out = fn(x)
        self.assertEqual(out.size(), x.size())


for fn_name in get_activation_functions():
    def do_test(self, fn_name=fn_name):
        input_shape = (32, 3, 224, 224)
        self._test_activation_functions(fn_name, input_shape)

    setattr(Tester, "test_" + fn_name, do_test)


if __name__ == '__main__':
    unittest.main()
