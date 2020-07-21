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
        # Check backbone pretrained
        model = models.__dict__[name](pretrained_backbone=True, num_classes=num_classes).eval()
        with torch.no_grad():
            out = model(x)

        self.assertIsInstance(out, list)
        self.assertEqual(len(out), x.shape[0])
        if len(out) > 0:
            self.assertIsInstance(out[0].get('boxes'), torch.Tensor)
            self.assertIsInstance(out[0].get('scores'), torch.Tensor)
            self.assertIsInstance(out[0].get('labels'), torch.Tensor)

        # Check that list of Tensors does not change output
        x_list = [torch.rand(3, size, size) for _ in range(num_batches)]
        with torch.no_grad():
            out_list = model(x_list)
            self.assertEqual(len(out_list), len(out))

        # Training mode without target
        model = model.train()
        self.assertRaises(ValueError, model, x)
        # Generate targets
        num_boxes = [3, 4]
        gt_boxes = []
        for num in num_boxes:
            _boxes = torch.rand((num, 4), dtype=torch.float)
            # Ensure format xmin, ymin, xmax, ymax
            _boxes[:, :2] *= _boxes[:, 2:]
            # Ensure some anchors will be assigned
            _boxes[0, :2] = 0
            _boxes[0, 2:] = 1
            # Check cases where cell can get two assignments
            _boxes[1, :2] = 0.2
            _boxes[1, 2:] = 0.8
            gt_boxes.append(_boxes)
        gt_labels = [(num_classes * torch.rand(num)).to(dtype=torch.long) for num in num_boxes]

        # Loss computation
        loss = model(x, gt_boxes, gt_labels)
        self.assertIsInstance(loss, dict)
        for subloss in loss.values():
            self.assertIsInstance(subloss, torch.Tensor)
            self.assertFalse(torch.isnan(subloss))

        # Loss computation with no GT
        gt_boxes = [torch.zeros((0, 4)) for _ in num_boxes]
        gt_labels = [torch.zeros(0, dtype=torch.long) for _ in num_boxes]
        loss = model(x, gt_boxes, gt_labels)

    def _test_segmentation_model(self, name, size, out_size):

        num_classes = 10
        num_batches = 2
        num_channels = 1
        x = torch.rand((num_batches, num_channels, size, size))
        model = models.__dict__[name](pretrained=True, num_classes=num_classes).eval()
        with torch.no_grad():
            out = model(x)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (num_batches, num_classes, out_size, out_size))


for model_name in ['res2net', 'res2next']:
    def do_test(self, model_name=model_name):
        input_shape = (4, 3, 224, 224)
        self._test_res2nets(model_name, input_shape)

    setattr(Tester, "test_" + model_name, do_test)

for model_name in ['darknet24', 'darknet19', 'darknet53',
                   'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                   'resnext50_32x4d', 'resnext101_32x8d',
                   'rexnet1_0x', 'rexnet1_3x', 'rexnet1_5x', 'rexnet2_0x', 'rexnet2_2x']:
    def do_test(self, model_name=model_name):
        self._test_classification_model(model_name)

    setattr(Tester, "test_" + model_name, do_test)


for model_name, size in [('yolov1', 448), ('yolov2', 416)]:
    def do_test(self, model_name=model_name, size=size):
        self._test_detection_model(model_name, size)

    setattr(Tester, "test_" + model_name, do_test)


# Lower input size to avoid OOM with CI
for model_name, size, out_size in [('unet', 572, 388), ('unetp', 256, 256), ('unetpp', 256, 256), ('unet3p', 320, 320)]:
    def do_test(self, model_name=model_name, size=size, out_size=out_size):
        self._test_segmentation_model(model_name, size, out_size)

    setattr(Tester, "test_" + model_name, do_test)


if __name__ == '__main__':
    unittest.main()
