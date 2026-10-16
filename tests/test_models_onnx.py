import argparse
import inspect

import onnx
import onnxruntime
import pytest
import torch
from torch import nn

from holocron import models
from holocron.models.detection.yolo import _post_process  # noqa: PLC2701
from scripts.export_to_onnx import _match_detections, _outputs, export_model, main  # noqa: PLC2701


@torch.no_grad()
def _calibrate(model, side, momentum=0.99):
    # Fresh running stats can make deep networks almost constant; emulate a training batch.
    norms = [module for module in model.modules() if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))]
    if norms:
        for module in norms:
            module.train()
            module.momentum = momentum  # Retain initial variance to avoid enormous normalization gains.
        images = torch.rand(2, 3, side, side)
        images[0] = 0
        model(images)
    model.eval()


@torch.inference_mode()
def _responds_to_input(model, images):
    prediction, blank = model(images), model(torch.zeros_like(images))
    if isinstance(prediction, torch.Tensor):
        return not torch.allclose(prediction, blank, rtol=1e-3, atol=3e-5)
    try:
        _match_detections(_outputs(blank), _outputs(prediction))
    except AssertionError:
        return True
    return False


@pytest.mark.parametrize("arch", models.list_models())
def test_model_onnx_inference(arch, tmp_path):
    torch.manual_seed(42)
    task = models.get_model_info(arch).task
    factory = getattr(getattr(models, task), arch)
    kwargs = {"num_classes": 3}
    if "pretrained_backbone" in inspect.signature(factory).parameters:
        kwargs["pretrained_backbone"] = False
    side = 448 if arch == "yolov1" else 64
    model = models.get_model(arch, **kwargs).eval()
    # Detector factories can initialize constant heads; exercise input-dependent predictions.
    if task == "detection":
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and not torch.count_nonzero(module.weight):
                # Keep box logits moderate: exponential decoding amplifies round-off.
                nn.init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    # Give class logits a margin instead of rounding to the same confidence.
                    nn.init.normal_(module.bias, std=0.1)
        model.box_score_thresh = 0
    if arch == "yolo26n":
        # Keep synthetic class scores sensitive to inputs instead of near zero.
        for branch in model.one_to_one.classes:
            nn.init.zeros_(branch[-1].bias)
        _calibrate(model, side, momentum=0.95)
    path = tmp_path / "model.onnx"
    images = torch.rand(1, 3, side, side)
    responds = _responds_to_input(model, images)
    for _ in range(2):
        if responds:
            break
        # Retain extra variance for stable normalization folding in the deep YOLO head.
        _calibrate(model, side, momentum=0.95 if arch == "yolo26n" else 0.99)
        responds = _responds_to_input(model, images)
    assert responds
    if task == "detection":
        with torch.inference_mode():
            assert len(model(images)[0]["boxes"]) > 0
            if arch == "yolo26n":
                assert not torch.allclose(
                    model(images)[0]["scores"],
                    model(torch.zeros_like(images))[0]["scores"],
                    rtol=1e-3,
                    atol=3e-5,
                )
    export_model(model, images, path)
    graph = onnx.load(path, load_external_data=False)
    assert graph.opset_import[0].version == 20
    assert [dim.dim_value for dim in graph.graph.input[0].type.tensor_type.shape.dim] == [1, 3, side, side]
    path.unlink()


@pytest.mark.parametrize("nms", [False, True])
def test_yolo26_onnx_batch(nms, tmp_path):
    torch.manual_seed(42)
    model = models.get_model("yolo26n", num_classes=3, box_score_thresh=0, nms=nms)
    export_model(model, torch.rand(2, 3, 64, 96), tmp_path / "model.onnx")


def test_detection_onnx_empty_and_nonempty(tmp_path):
    class PostProcess(nn.Module):
        def forward(self, images):  # noqa: PLR6301
            boxes = images.new_tensor([[[0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5], [0.5, 0.5, 1, 1]]])
            boxes = boxes.expand(images.shape[0], -1, -1)
            scores = images.new_tensor([[[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]]]).expand(images.shape[0], -1, -1)
            return _post_process(boxes, images[:, 0, 0, :3], scores)

    images = torch.ones(2, 3, 4, 4)
    images[1] = 0
    model = PostProcess()
    assert [len(prediction["boxes"]) for prediction in model(images)] == [2, 0]
    export_model(model, images, tmp_path / "model.onnx")


def test_detection_matching_reassigns_ambiguous_records():
    expected = {
        "boxes_0": torch.tensor([[0.5, 0, 1, 1], [0.5009, 0, 1, 1]]),
        "scores_0": torch.tensor([0.5, 0.5]),
        "labels_0": torch.tensor([0, 0]),
    }
    actual = {name: value.clone() for name, value in expected.items()}
    actual["boxes_0"][:, 0] = torch.tensor([0.5004, 0.5])
    _match_detections(actual, expected)
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=1e-3, atol=3e-5)
    actual["labels_0"][0] = 1
    with pytest.raises(AssertionError, match="No matching"):
        _match_detections(actual, expected)


@pytest.mark.parametrize("existing_file", [False, True])
def test_onnx_verification_catches_traced_input_branch(existing_file, tmp_path):
    class InputBranch(nn.Module):
        def forward(self, images):  # noqa: PLR6301
            return images + (1 if images.sum() > 0 else 0)

    path = tmp_path / "model.onnx"
    if existing_file:
        export_model(nn.Identity(), torch.ones(1, 3, 4, 4), path)
        original = path.read_bytes()
    with pytest.raises(AssertionError, match="not close"):
        export_model(InputBranch(), torch.ones(1, 3, 4, 4), path)
    assert list(tmp_path.iterdir()) == ([path] if existing_file else [])
    if existing_file:
        assert path.read_bytes() == original


@pytest.mark.parametrize("arch", ["rexnet1_0x", "repvit_m0_9"])
def test_onnx_verification_catches_frozen_image(arch, tmp_path):
    class FrozenImage(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, images):
            return self.model(images * 0 if torch.onnx.is_in_onnx_export() else images)

    torch.manual_seed(42)
    model = models.get_model(arch, num_classes=3).eval()
    _calibrate(model, 64)
    with pytest.raises(AssertionError, match="not close"):
        export_model(FrozenImage(model), torch.rand(1, 3, 64, 64), tmp_path / "model.onnx")


@pytest.mark.parametrize("arch", ["repvgg_a0", "mobileone_s0"])
def test_onnx_repeated_export(arch, tmp_path):
    model = models.get_model(arch, num_classes=3).eval()
    export_model(model, torch.rand(1, 3, 64, 64), tmp_path / "model.onnx")
    model.reparametrize()
    export_model(model, torch.rand(1, 3, 64, 96), tmp_path / "model.onnx")
    export_model(model, torch.rand(1, 3, 64, 64), tmp_path / "model.onnx")


@pytest.mark.parametrize("trainer_checkpoint", [False, True])
def test_onnx_checkpoint_cli(trainer_checkpoint, tmp_path):
    model = models.get_model("resnet18", num_classes=3, in_channels=1)
    checkpoint = tmp_path / "model.pth"
    state = model.state_dict()
    torch.save({"model": state, "epoch": 1} if trainer_checkpoint else state, checkpoint)
    main(
        argparse.Namespace(
            arch="resnet18",
            checkpoint=str(checkpoint),
            pretrained=False,
            num_classes=3,
            in_channels=1,
            batch_size=1,
            height=64,
            width=96,
            path=str(tmp_path / "model.onnx"),
        )
    )
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = torch.get_num_threads()
    session = onnxruntime.InferenceSession(
        str(tmp_path / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
    )
    images = torch.rand(1, 1, 64, 96)
    with torch.inference_mode():
        expected = model.eval()(images)
    torch.testing.assert_close(
        torch.from_numpy(session.run(None, {"images": images.numpy()})[0]), expected, rtol=1e-3, atol=1e-5
    )
