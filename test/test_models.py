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
        model = models.__dict__[name](pretrained=True, num_classes=num_classes).eval()
        with torch.no_grad():
            out = model(x)

        self.assertEqual(out.shape[0], x.shape[0])
        self.assertEqual(out.shape[-1], num_classes)

    def _test_detection_model(self, name, size):

        num_classes = 10
        num_batches = 2
        x = torch.rand((num_batches, 3, size, size))
        model = models.__dict__[name](pretrained=True, num_classes=num_classes).eval()
        with torch.no_grad():
            out = model(x)

        self.assertIsInstance(out, list)
        self.assertEqual(len(out), x.shape[0])
        if len(out) > 0:
            self.assertIsInstance(out[0].get('boxes'), torch.Tensor)
            self.assertIsInstance(out[0].get('scores'), torch.Tensor)
            self.assertIsInstance(out[0].get('labels'), torch.Tensor)

        # Training mode without target
        model = model.train()
        self.assertRaises(ValueError, model, x)
        # Generate targets
        num_boxes = [2, 3]
        gt_boxes = []
        for num in num_boxes:
            _boxes = torch.rand((num, 4), dtype=torch.float)
            # Ensure format xmin, ymin, xmax, ymax
            _boxes[:, :2] *= _boxes[:, 2:]
            # Ensure some anchors will be assigned
            _boxes[0, :2] = 0
            _boxes[0, 2:] = 1
        gt_boxes = [torch.rand((num, 4)) for num in num_boxes]
        gt_labels = [(num_classes * torch.rand(num)).to(dtype=torch.long) for num in num_boxes]

        # Loss computation
        loss = model(x, gt_boxes, gt_labels)
        self.assertIsInstance(loss, dict)
        for subloss in loss.values():
            self.assertIsInstance(subloss, torch.Tensor)


for model_name in ['res2net', 'res2next']:
    def do_test(self, model_name=model_name):
        input_shape = (4, 3, 224, 224)
        self._test_res2nets(model_name, input_shape)

    setattr(Tester, "test_" + model_name, do_test)

for model_name in ['darknet24', 'darknet19', 'darknet53']:
    def do_test(self, model_name=model_name):
        self._test_classification_model(model_name)

    setattr(Tester, "test_" + model_name, do_test)


for model_name, size in [('yolov1', 448), ('yolov2', 416)]:
    def do_test(self, model_name=model_name, size=size):
        self._test_detection_model(model_name, size)

    setattr(Tester, "test_" + model_name, do_test)


if __name__ == '__main__':
    unittest.main()
