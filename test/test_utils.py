import unittest
import requests
from io import BytesIO
from PIL import Image
import torch
from torchvision.models import mobilenet_v2
from torchvision.transforms import transforms

from holocron import utils


class Tester(unittest.TestCase):

    def test_gradcam(self):

        # Get a pretrained model
        model = mobilenet_v2(pretrained=True)
        conv_layer = 'features'

        # Hook the corresponding layer in the model
        gradcam = utils.ActivationMapper(model, conv_layer)

        # Get a dog image
        URL = 'https://www.woopets.fr/assets/races/000/066/big-portrait/border-collie.jpg'
        response = requests.get(URL)

        # Forward an image
        pil_img = Image.open(BytesIO(response.content), mode='r').convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = preprocess(pil_img)
        out = model(img_tensor.unsqueeze(0))

        # Border collie index in ImageNet
        class_idx = 232

        # Use the hooked data to compute activation map
        activation_map = gradcam.get_activation_maps(out, class_idx)

        self.assertIsInstance(activation_map, torch.Tensor)
        self.assertEqual(activation_map.shape, (1, 7, 7))

    def test_get_module_names(self):

        # Get a model
        model = mobilenet_v2().eval()

        layer_names = utils.get_module_names(model)

        self.assertIsInstance(layer_names, list)
        self.assertEqual(len(layer_names), 141)
        self.assertEqual(layer_names[42], 'features.6.conv.0.2')

    def test_module_summary(self):

        # Get a model
        model = mobilenet_v2().eval()

        exec_sum = utils.module_summary(model, input_shape=(3, 224, 224))

        self.assertIsInstance(exec_sum, list)
        self.assertEqual(len(exec_sum), 141)
        self.assertEqual(exec_sum[42]['output_shape'], (None, 192, 28, 28))
