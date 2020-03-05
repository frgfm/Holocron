import unittest
import requests
from io import BytesIO
from PIL import Image
import torch
from torchvision.models import resnet18
from torchvision.transforms import transforms

from holocron import utils


class Tester(unittest.TestCase):

    def test_gradcam(self):

        # Get a pretrained model
        model = resnet18(pretrained=True)
        conv_layer = 'layer4'

        # Hook the corresponding layer in the model
        gradcam = utils.ActivationMapper(model, conv_layer)

        # Get a dog image
        URL = 'https://www.woopets.fr/assets/races/000/066/big-portrait/border-collie.jpg'
        response = requests.get(URL)

        # Forward an image
        pil_img = Image.open(BytesIO(response.content), mode='r').convert('RGB')
        preprocess = transforms.Compose([
           transforms.Resize((224,224)),
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
