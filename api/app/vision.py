# Copyright (C) 2022-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import io
import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from PIL import Image

from app.config import settings

__all__ = ["classify_image", "decode_image"]

logger = logging.getLogger("uvicorn.warning")

# Download model config & checkpoint
with Path(hf_hub_download(settings.CLF_HUB_REPO, filename="config.json")).open("rb") as f:
    CLF_CFG = json.load(f)

CLF_ORT = onnxruntime.InferenceSession(hf_hub_download(settings.CLF_HUB_REPO, filename="model.onnx"))

logger.info(f"Model loading completed: {settings.CLF_HUB_REPO}")


def decode_image(img_data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(img_data))


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Preprocess an image for inference

    Args:
        pil_img: a valid pillow image

    Returns:
        the resized and normalized image of shape (1, C, H, W)
    """
    # Resizing (PIL takes (W, H) order for resizing)
    img = pil_img.resize(CLF_CFG["input_shape"][-2:][::-1], Image.BILINEAR)
    # (H, W, C) --> (C, H, W)
    img = np.asarray(img).transpose((2, 0, 1)).astype(np.float32) / 255
    # Normalization
    img -= np.array(CLF_CFG["mean"])[:, None, None]
    img /= np.array(CLF_CFG["std"])[:, None, None]

    return img[None, ...]


def classify_image(pil_img: Image.Image) -> np.ndarray:
    np_img = preprocess_image(pil_img)
    ort_input = {CLF_ORT.get_inputs()[0].name: np_img}

    # Inference
    ort_out = CLF_ORT.run(None, ort_input)
    # sigmoid
    return 1 / (1 + np.exp(-ort_out[0][0]))
