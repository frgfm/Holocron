# Copyright (C) 2022-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import io
import json
import sys
from types import SimpleNamespace

import httpx
import huggingface_hub
import numpy as np
import onnxruntime
import pytest
import uvicorn
from PIL import Image

from demo import app as demo


@pytest.fixture
def isolated_api_modules(monkeypatch):
    """Import the API afresh so the demo's selected model is loaded."""
    modules = {name: module for name, module in sys.modules.items() if name == "app" or name.startswith("app.")}
    for name in modules:
        del sys.modules[name]
    monkeypatch.syspath_prepend(str(demo.API_DIR))
    monkeypatch.setenv("CLF_HUB_REPO", demo.DEFAULT_REPO)
    monkeypatch.setenv("CLF_HUB_REVISION", demo.DEFAULT_REVISION)
    yield
    for name in tuple(sys.modules):
        if name == "app" or name.startswith("app."):
            del sys.modules[name]
    sys.modules.update(modules)


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_api_modules")
async def test_default_model_pin_ui_and_top_three(monkeypatch, tmp_path):
    config = {
        "classes": ["first", "second", "third", "fourth"],
        "input_shape": [3, 2, 2],
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    downloads = []

    def download(repo, filename, revision):
        downloads.append((repo, filename, revision))
        return str(config_path)

    class Session:
        @staticmethod
        def get_inputs():
            return [SimpleNamespace(name="input")]

        @staticmethod
        def run(_outputs, _inputs):
            return [np.array([[1.0, 4.0, 3.0, 2.0]], dtype=np.float32)]

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda _path: Session())

    captured = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: captured.update(app=app, **kwargs))
    monkeypatch.setattr(sys, "argv", ["demo/app.py", "--port", "3000"])
    demo.main()
    assert captured["port"] == 3000
    assert downloads == [
        (demo.DEFAULT_REPO, "config.json", demo.DEFAULT_REVISION),
        (demo.DEFAULT_REPO, "model.onnx", demo.DEFAULT_REVISION),
    ]

    output = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(output, format="PNG")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=captured["app"]), base_url="http://test") as client:
        page = await client.get("/")
        response = await client.post("/predictions", files={"file": ("image.png", output.getvalue())})
    assert page.status_code == 200
    assert "Choose a JPEG or PNG" in page.text
    assert response.status_code == 200, response.text
    predictions = response.json()
    assert [item["name"] for item in predictions] == ["second", "third", "fourth"]
    expected = np.exp([4.0, 3.0, 2.0]) / np.exp([1.0, 4.0, 3.0, 2.0]).sum()
    assert [item["confidence"] for item in predictions] == pytest.approx(expected.tolist())


def test_custom_model_requires_revision():
    with pytest.raises(ValueError, match="--revision is required"):
        demo.build_app("owner/model", None)
