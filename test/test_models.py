import unittest
import torch
from holocron import models


class Tester(unittest.TestCase):

    def _test_res2nets(self, name, input_shape):
        # passing num_class equal to a number other than 1000 helps in making the test
        # more enforcing in nature
        model = models.__dict__[name](depth=50, num_classes=50, pretrained=True)
        model.eval()
        x = torch.rand(input_shape)
        out = model(x)
        self.assertEqual(out.shape[-1], 50)


for model_name in ['res2net', 'res2next']:
    def do_test(self, model_name=model_name):
        input_shape = (4, 3, 224, 224)
        self._test_res2nets(model_name, input_shape)

    setattr(Tester, "test_" + model_name, do_test)


if __name__ == '__main__':
    unittest.main()
