import unittest
import torch
from holocron.nn import DropBlock2d, BlurPool2d, SAM
from holocron import models


class ModelTester(unittest.TestCase):

    def _test_classification_model(self, name, num_classes=10):

        batch_size = 2
        x = torch.rand((batch_size, 3, 224, 224))
        model = models.__dict__[name](pretrained=True, num_classes=num_classes).eval()
        with torch.no_grad():
            out = model(x)

        self.assertEqual(out.shape[0], x.shape[0])
        self.assertEqual(out.shape[-1], num_classes)

        # Check backprop is OK
        target = torch.zeros(batch_size, dtype=torch.long)
        model.train()
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, target)
        loss.backward()

    def _test_detection_model(self, name, size):

        num_classes = 10
        batch_size = 2
        x = torch.rand((batch_size, 3, size, size))
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
        x_list = [torch.rand(3, size, size) for _ in range(batch_size)]
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
        loss = model(x, [dict(boxes=boxes, labels=labels) for boxes, labels in zip(gt_boxes, gt_labels)])
        self.assertIsInstance(loss, dict)
        for subloss in loss.values():
            self.assertIsInstance(subloss, torch.Tensor)
            self.assertTrue(subloss.requires_grad)
            self.assertFalse(torch.isnan(subloss))

        # Loss computation with no GT
        gt_boxes = [torch.zeros((0, 4)) for _ in num_boxes]
        gt_labels = [torch.zeros(0, dtype=torch.long) for _ in num_boxes]
        loss = model(x, [dict(boxes=boxes, labels=labels) for boxes, labels in zip(gt_boxes, gt_labels)])

    def _test_segmentation_model(self, name, size, out_size):

        num_classes = 10
        batch_size = 2
        num_channels = 1
        x = torch.rand((batch_size, num_channels, size, size))
        model = models.__dict__[name](pretrained=True, num_classes=num_classes).eval()
        with torch.no_grad():
            out = model(x)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (batch_size, num_classes, out_size, out_size))

    def _test_conv_seq(self, conv_seq, expected_classes, expected_channels):

        self.assertEqual(len(conv_seq), len(expected_classes))
        for _layer, mod_class in zip(conv_seq, expected_classes):
            self.assertIsInstance(_layer, mod_class)

        input_t = torch.rand(1, conv_seq[0].in_channels, 224, 224)
        out = torch.nn.Sequential(*conv_seq)(input_t)
        self.assertEqual(out.shape[:2], (1, expected_channels))
        out.sum().backward()

    def test_conv_sequence(self):

        mod = models.utils.conv_sequence(3, 32, kernel_size=3, act_layer=torch.nn.ReLU(inplace=True),
                                         norm_layer=torch.nn.BatchNorm2d, drop_layer=DropBlock2d, attention_layer=SAM)

        self._test_conv_seq(mod, [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU, SAM, DropBlock2d], 32)
        self.assertEqual(mod[0].kernel_size, (3, 3))

        mod = models.utils.conv_sequence(3, 32, kernel_size=3, stride=2, act_layer=torch.nn.ReLU(inplace=True),
                                         norm_layer=torch.nn.BatchNorm2d, drop_layer=DropBlock2d, blurpool=True)
        self._test_conv_seq(mod, [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU, BlurPool2d, DropBlock2d], 32)
        self.assertEqual(mod[0].kernel_size, (3, 3))
        self.assertEqual(mod[0].stride, (1, 1))
        self.assertEqual(mod[3].stride, 2)


for model_name in ['darknet24', 'darknet19', 'darknet53', 'cspdarknet53', 'cspdarknet53_mish',
                   'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                   'resnext50_32x4d', 'resnext101_32x8d',
                   'resnet50d',
                   'res2net50_26w_4s',
                   'tridentnet50',
                   'pyconv_resnet50', 'pyconvhg_resnet50',
                   'rexnet1_0x', 'rexnet1_3x', 'rexnet1_5x', 'rexnet2_0x', 'rexnet2_2x',
                   'sknet50', 'sknet101', 'sknet152']:
    num_classes = 1000 if model_name in ['rexnet1_0x', 'rexnet1_3x', 'rexnet1_5x', 'rexnet2_0x'] else 10

    def do_test(self, model_name=model_name, num_classes=num_classes):
        self._test_classification_model(model_name, num_classes)

    setattr(ModelTester, "test_" + model_name, do_test)


for model_name, size in [('yolov1', 448), ('yolov2', 416), ('yolov4', 608)]:
    def do_test(self, model_name=model_name, size=size):
        self._test_detection_model(model_name, size)

    setattr(ModelTester, "test_" + model_name, do_test)


# Lower input size to avoid OOM with CI
for model_name, size, out_size in [('unet', 572, 388), ('unetp', 256, 256), ('unetpp', 256, 256), ('unet3p', 320, 320)]:
    def do_test(self, model_name=model_name, size=size, out_size=out_size):
        self._test_segmentation_model(model_name, size, out_size)

    setattr(ModelTester, "test_" + model_name, do_test)


if __name__ == '__main__':
    unittest.main()
