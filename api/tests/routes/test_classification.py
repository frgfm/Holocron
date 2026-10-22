import asyncio
import json
import threading
from io import BytesIO

import httpx
import numpy as np
import pytest
from PIL import GifImagePlugin, Image, ImageFile
from starlette.requests import Request

from app import main, vision
from app.config import Settings
from app.vision import CLF_CFG, decode_image, preprocess_image, validate_model_config


@pytest.mark.asyncio
async def test_health(test_app_asyncio):
    response = await test_app_asyncio.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert float(response.headers["x-process-time"]) >= 0


@pytest.mark.asyncio
async def test_classification(test_app_asyncio, mock_classification_image):
    response = await test_app_asyncio.post("/classification", files={"file": mock_classification_image})
    assert response.status_code == 200
    json_response = response.json()

    # Check that IoU with GT if reasonable
    assert isinstance(json_response, dict)
    assert json_response["value"] == "French horn"
    assert json_response["confidence"] >= 0.8


@pytest.mark.asyncio
async def test_unsupported_gif_is_a_client_error_before_inference(test_app_asyncio, monkeypatch):
    image = Image.new("RGB", (8, 8), "red")
    output = BytesIO()
    image.save(output, format="GIF")
    monkeypatch.setattr(GifImagePlugin.GifImageFile, "_open", lambda _image: pytest.fail("GIF parser must not run"))
    monkeypatch.setattr(main, "classify_image", lambda _image: pytest.fail("inference must not run"))

    response = await test_app_asyncio.post("/classification", files={"file": ("x.gif", output.getvalue())})
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_valid_png_is_accepted(test_app_asyncio):
    image = Image.new("L", (8, 8), 128)
    output = BytesIO()
    image.save(output, format="PNG")
    response = await test_app_asyncio.post("/classification", files={"file": ("x.png", output.getvalue())})
    assert response.status_code == 200


def test_model_config_is_validated_before_model_download():
    with pytest.raises(ValueError, match="classification settings"):
        validate_model_config({"classes": ["a"], "input_shape": [1, 224, 224], "mean": [0, 0, 0], "std": [1, 1, 1]})


def test_invalid_config_stops_before_model_download(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"classes": [], "input_shape": [3, 224, 224]}))
    downloads = []

    def fake_download(_repo, filename, revision):
        assert revision == "a" * 40
        downloads.append(filename)
        return config_path

    monkeypatch.setattr(vision, "hf_hub_download", fake_download)
    monkeypatch.setattr(vision.onnxruntime, "InferenceSession", lambda _path: pytest.fail("model must not load"))

    with pytest.raises(ValueError, match="classification settings"):
        vision.load_model("owner/model", "a" * 40)
    assert downloads == ["config.json"]


def test_config_and_onnx_downloads_share_the_pinned_commit(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"classes": ["class"], "input_shape": [3, 224, 224], "mean": [0, 0, 0], "std": [1, 1, 1]})
    )
    downloads = []
    revision = "a" * 40

    def fake_download(repo, filename, revision):
        downloads.append((repo, filename, revision))
        return config_path

    monkeypatch.setattr(vision, "hf_hub_download", fake_download)
    monkeypatch.setattr(vision.onnxruntime, "InferenceSession", lambda _path: object())
    vision.load_model("owner/model", revision)

    assert downloads == [("owner/model", "config.json", revision), ("owner/model", "model.onnx", revision)]


def test_image_dimension_limit_precedes_pixel_decode(monkeypatch):
    output = BytesIO()
    Image.new("RGB", (5001, 5000), "red").save(output, format="PNG")
    monkeypatch.setattr(ImageFile.ImageFile, "load", lambda _image: pytest.fail("oversized image must not decode"))

    with pytest.raises(ValueError, match="dimensions"):
        decode_image(output.getvalue())


@pytest.mark.parametrize("image_format", ["JPEG", "PNG"])
def test_rgb_supported_inputs_keep_the_original_preprocessing(image_format):
    image = Image.new("RGB", (32, 32), "red")
    output = BytesIO()
    image.save(output, format=image_format)
    data = output.getvalue()

    before_gates = Image.open(BytesIO(data))
    expected = before_gates.resize(CLF_CFG["input_shape"][-2:][::-1], Image.Resampling.BILINEAR)
    expected = np.asarray(expected).transpose((2, 0, 1)).astype(np.float32) / 255
    expected -= np.array(CLF_CFG["mean"])[:, None, None]
    expected /= np.array(CLF_CFG["std"])[:, None, None]

    assert np.array_equal(preprocess_image(decode_image(data)), expected[None, ...])


def test_invalid_image_bytes_are_rejected():
    with pytest.raises((OSError, ValueError)):
        decode_image(b"not an image")


def test_debug_defaults_off_and_invalid_model_settings_fail_early():
    assert Settings(DEBUG=False).DEBUG is False
    with pytest.raises(ValueError, match="commit SHA"):
        Settings(CLF_HUB_REVISION="main")
    with pytest.raises(ValueError, match="repository ID"):
        Settings(CLF_HUB_REPO="https://huggingface.co/frgfm/rexnet1_5x")


def test_debug_defaults_off_when_unset(monkeypatch):
    monkeypatch.delenv("DEBUG", raising=False)
    assert Settings().DEBUG is False
    assert main.app.debug is False


@pytest.mark.asyncio
async def test_malformed_upload_is_a_controlled_client_error(test_app_asyncio):
    response = await test_app_asyncio.post("/classification", files={"file": ("x.jpg", b"truncated image")})
    assert response.status_code == 400
    assert "Traceback" not in response.text
    await main.CLASSIFICATION_SLOTS.acquire()
    await main.CLASSIFICATION_SLOTS.acquire()
    main.CLASSIFICATION_SLOTS.release()
    main.CLASSIFICATION_SLOTS.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("image_format", ["JPEG", "PNG"])
async def test_truncated_upload_is_a_client_error(test_app_asyncio, image_format):
    output = BytesIO()
    Image.new("RGB", (32, 32), "red").save(output, format=image_format)
    data = output.getvalue()
    response = await test_app_asyncio.post("/classification", files={"file": ("image", data[: len(data) // 2])})
    assert response.status_code == 400
    assert "Traceback" not in response.text


@pytest.mark.asyncio
async def test_full_request_capacity_fails_fast_with_503(test_app_asyncio):
    await main.CLASSIFICATION_SLOTS.acquire()
    await main.CLASSIFICATION_SLOTS.acquire()
    response = await test_app_asyncio.post("/classification", content=b"ignored")
    assert response.status_code == 503
    main.CLASSIFICATION_SLOTS.release()
    main.CLASSIFICATION_SLOTS.release()


class OversizedStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"a" * (main.MAX_REQUEST_BYTES // 2 + 1)
        yield b"b" * (main.MAX_REQUEST_BYTES // 2 + 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("content_length", [None, "1"])
async def test_chunked_body_limit_is_enforced_before_multipart_and_slots_are_released(monkeypatch, content_length):
    async def no_multipart_parser(_request):
        await asyncio.sleep(0)
        pytest.fail("oversized bytes must be rejected before multipart parsing")

    monkeypatch.setattr(Request, "form", no_multipart_parser)
    headers = {"content-type": "multipart/form-data; boundary=x"}
    if content_length is None:
        headers["transfer-encoding"] = "chunked"
    else:
        headers["content-length"] = content_length
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        response = await client.post(
            "/classification",
            content=OversizedStream(),
            headers=headers,
        )
    assert response.status_code == 413
    await main.CLASSIFICATION_SLOTS.acquire()
    await main.CLASSIFICATION_SLOTS.acquire()
    main.CLASSIFICATION_SLOTS.release()
    main.CLASSIFICATION_SLOTS.release()


@pytest.mark.asyncio
async def test_classification_slot_is_held_until_cancelled_inference_finishes(
    test_app_asyncio, mock_classification_image, monkeypatch
):
    started, finish = threading.Event(), threading.Event()

    def slow_inference(_image):
        started.set()
        finish.wait(timeout=3)
        return np.array([1.0])

    monkeypatch.setattr(main, "classify_image", slow_inference)
    request = asyncio.create_task(
        test_app_asyncio.post("/classification", files={"file": ("x.jpg", mock_classification_image)})
    )
    assert await asyncio.to_thread(started.wait, 2)
    request.cancel()
    await asyncio.sleep(0)
    await main.CLASSIFICATION_SLOTS.acquire()
    request.cancel()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(main.CLASSIFICATION_SLOTS.acquire(), timeout=0.01)
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await request
    await asyncio.wait_for(main.CLASSIFICATION_SLOTS.acquire(), timeout=0.1)
    main.CLASSIFICATION_SLOTS.release()
    main.CLASSIFICATION_SLOTS.release()


@pytest.mark.asyncio
async def test_body_read_timeout_returns_408_and_releases_slot(monkeypatch):
    sent = []

    async def receive():
        await asyncio.sleep(1)

    async def send(message):
        await asyncio.sleep(0)
        sent.append(message)

    async def unused_app(_scope, _receive, _send):
        await asyncio.sleep(0)
        pytest.fail("timed-out request must not reach the app")

    async def invoke(scope, receive, send):
        await main.BoundedRequestMiddleware(unused_app)(scope, receive, send)

    monkeypatch.setattr(main, "BODY_READ_TIMEOUT_SECONDS", 0.01)
    scope = {"type": "http", "method": "POST", "headers": [], "path": "/classification"}
    await invoke(scope, receive, send)
    assert sent[0]["status"] == 408
    await main.CLASSIFICATION_SLOTS.acquire()
    await main.CLASSIFICATION_SLOTS.acquire()
    main.CLASSIFICATION_SLOTS.release()
    main.CLASSIFICATION_SLOTS.release()
