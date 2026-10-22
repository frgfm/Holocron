# Copyright (C) 2022-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import io
import json
import logging
import math
from pathlib import Path

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from PIL import Image

from app.config import settings

__all__ = ["classify_image", "decode_image", "load_model", "validate_model_config"]

logger = logging.getLogger("uvicorn.warning")


def validate_model_config(config: dict) -> None:
    """Validate the preprocessing fields consumed by the API and demo.

    Raises:
        TypeError: If config is not an object.
        ValueError: If required classification fields are invalid.
    """
    if not isinstance(config, dict):
        raise TypeError("Hub config.json must be a JSON object")
    classes, shape, mean, std = (config.get(key) for key in ("classes", "input_shape", "mean", "std"))
    valid_classes = (
        isinstance(classes, list) and bool(classes) and all(isinstance(label, str) and label for label in classes)
    )
    valid_shape = (
        isinstance(shape, list)
        and len(shape) == 3
        and shape[0] == 3
        and all(type(value) is int and 0 < value <= 8192 for value in shape)
    )
    valid_stats = (
        isinstance(mean, list)
        and isinstance(std, list)
        and len(mean) == len(std) == 3
        and all(type(value) in {int, float} and math.isfinite(value) for value in mean + std)
        and all(value > 0 for value in std)
    )
    if not (valid_classes and valid_shape and valid_stats):
        raise ValueError("Hub config.json has invalid classification settings")


def load_model(repo_id: str, revision: str) -> tuple[dict, onnxruntime.InferenceSession]:
    """Load config first, then the matching model from one immutable Hub commit."""
    config_path = hf_hub_download(repo_id, filename="config.json", revision=revision)
    with Path(config_path).open("rb") as f:
        config = json.load(f)
    validate_model_config(config)
    model_path = hf_hub_download(repo_id, filename="model.onnx", revision=revision)
    return config, onnxruntime.InferenceSession(model_path)


CLF_CFG, CLF_ORT = load_model(settings.CLF_HUB_REPO, settings.CLF_HUB_REVISION)

logger.info(f"Model loading completed: {settings.CLF_HUB_REPO}")


def decode_image(img_data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(img_data), formats=("JPEG", "PNG")) as image:
            if image.width > 8192 or image.height > 8192 or image.width * image.height > 25_000_000:
                raise ValueError("Image dimensions exceed the supported limit")
            image.load()
            return image.convert("RGB")
    except Image.DecompressionBombError as exc:
        raise ValueError("Image dimensions exceed the supported limit") from exc


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
