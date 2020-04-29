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

    def _test_classification_model(self, name):

        num_classes = 10
        x = torch.rand((2, 3, 224, 224))
        model = models.__dict__[name](num_classes=num_classes).eval()
        with torch.no_grad():
            out = model(x)

        self.assertEqual(out.shape[0], x.shape[0])
        self.assertEqual(out.shape[-1], num_classes)


for model_name in ['res2net', 'res2next']:
    def do_test(self, model_name=model_name):
        input_shape = (4, 3, 224, 224)
        self._test_res2nets(model_name, input_shape)

    setattr(Tester, "test_" + model_name, do_test)

for model_name in ['darknet19']:
    def do_test(self, model_name=model_name):
        self._test_classification_model(model_name)

    setattr(Tester, "test_" + model_name, do_test)


if __name__ == '__main__':
    unittest.main()
