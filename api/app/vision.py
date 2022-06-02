# Copyright (C) 2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import io

import torch
from PIL import Image
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, PILToTensor, Resize
from torchvision.transforms.functional import InterpolationMode

from holocron.models.classification import rexnet1_5x

__all__ = ["classification_model", "classification_preprocessor", "decode_image"]

classification_model = rexnet1_5x(pretrained=True).eval()
classification_preprocessor = Compose(
    [
        Resize(classification_model.default_cfg["input_shape"][1:], interpolation=InterpolationMode.BILINEAR),
        PILToTensor(),
        ConvertImageDtype(torch.float32),
        Normalize(classification_model.default_cfg["mean"], classification_model.default_cfg["std"]),
    ]
)


def decode_image(img_data: bytes) -> torch.Tensor:
    return Image.open(io.BytesIO(img_data))
