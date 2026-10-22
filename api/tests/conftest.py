import asyncio
import json
import tempfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image


class FakeSession:
    @staticmethod
    def get_inputs():
        """Return the input name expected by the API."""
        return [SimpleNamespace(name="input")]

    @staticmethod
    def run(_outputs, _inputs):
        """Return a fixed fake ONNX output."""
        return [np.array([[2.0]], dtype=np.float32)]


with tempfile.TemporaryDirectory(prefix="holocron-api-test-") as model_dir:
    _model_config = Path(model_dir) / "config.json"
    _model_config.write_text(
        json.dumps({
            "classes": ["French horn"],
            "input_shape": [3, 224, 224],
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        })
    )
    with (
        patch("huggingface_hub.hf_hub_download", return_value=str(_model_config)),
        patch("onnxruntime.InferenceSession", return_value=FakeSession()),
    ):
        from app import main
        from app.main import app


@pytest.fixture(autouse=True)
def reset_classification_slots(monkeypatch):
    monkeypatch.setattr(main, "CLASSIFICATION_SLOTS", asyncio.Semaphore(2))


@pytest.fixture
def mock_classification_image(image_format="JPEG"):
    image = Image.new("RGB", (32, 32), "red")
    output = BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


@pytest_asyncio.fixture
async def test_app_asyncio():
    # for httpx>=20, follow_redirects=True (cf. https://github.com/encode/httpx/releases/tag/0.20.0)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", follow_redirects=True) as ac:
        yield ac  # testing happens here
