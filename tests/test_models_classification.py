import os

import pytest
import torch
from torch import nn

from holocron.models import classification


def _test_classification_model(name, num_classes, pretrained):
    batch_size = 2
    x = torch.rand((batch_size, 3, 224, 224))
    model = classification.__dict__[name](pretrained=pretrained, num_classes=num_classes).eval()
    with torch.no_grad():
        out = model(x)

    assert out.shape[0] == x.shape[0]
    assert out.shape[-1] == num_classes

    # Check backprop is OK
    target = torch.zeros(batch_size, dtype=torch.long)
    model.train()
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, target)
    loss.backward()


def test_repvgg_reparametrize():
    num_classes = 10
    batch_size = 2
    x = torch.rand((batch_size, 3, 224, 224))
    model = classification.repvgg_a0(pretrained=False, num_classes=num_classes).eval()
    with torch.no_grad():
        out = model(x)

    # Reparametrize
    model.reparametrize()
    # Check that there is no longer any Conv1x1 or BatchNorm
    for mod in model.modules():
        assert not isinstance(mod, nn.BatchNorm2d)
        if isinstance(mod, nn.Conv2d):
            assert mod.weight.data.shape[2:] == (3, 3)
    # Check that values are still matching
    with torch.no_grad():
        assert torch.allclose(out, model(x), atol=1e-5)


def test_mobileone_reparametrize():
    num_classes = 10
    batch_size = 2
    x = torch.rand((batch_size, 3, 224, 224))
    model = classification.mobileone_s0(pretrained=False, num_classes=num_classes).eval()
    with torch.no_grad():
        out = model(x)

    # Reparametrize
    model.reparametrize()
    # Check that there is no longer any Conv1x1 or BatchNorm
    for mod in model.modules():
        assert not isinstance(mod, nn.BatchNorm2d)
    # Check that values are still matching
    with torch.no_grad():
        assert torch.allclose(out, model(x), atol=1e-3)


@pytest.mark.parametrize(
    "arch, pretrained",
    [
        ["darknet24", True],
        ["darknet19", True],
        ["darknet53", True],
        ["cspdarknet53", True],
        ["cspdarknet53_mish", True],
        ["resnet18", True],
        ["resnet34", True],
        ["resnet50", True],
        ["resnet101", True],
        ["resnet152", True],
        ["resnext50_32x4d", True],
        ["resnext101_32x8d", True],
        ["resnet50d", True],
        ["res2net50_26w_4s", True],
        ["tridentnet50", True],
        ["pyconv_resnet50", True],
        ["pyconvhg_resnet50", True],
        ["rexnet1_0x", True],
        ["rexnet1_3x", False],
        ["rexnet1_5x", False],
        ["rexnet2_0x", False],
        ["rexnet2_2x", False],
        ["sknet50", True],
        ["sknet101", True],
        ["sknet152", True],
        ["repvgg_a0", True],
        ["repvgg_b0", False],
        ["convnext_atto", True],
        ["convnext_femto", False],
        ["convnext_pico", False],
        ["convnext_nano", False],
        ["convnext_tiny", False],
        ["convnext_small", False],
        ["convnext_base", False],
        ["convnext_large", False],
        ["convnext_xl", False],
        ["mobileone_s0", True],
        ["mobileone_s1", False],
        ["mobileone_s2", False],
        ["mobileone_s3", False],
    ],
)
def test_classification_model(arch, pretrained):
    num_classes = 1000 if arch.startswith("rexnet") else 10
    _test_classification_model(arch, num_classes, pretrained)


@pytest.mark.parametrize(
    "arch",
    [
        "darknet24",
        "darknet19",
        "darknet53",
        "cspdarknet53",
        "resnet18",
        "res2net50_26w_4s",
        "tridentnet50",
        "pyconv_resnet50",
        "rexnet1_0x",
        "sknet50",
        "repvgg_a0",
        "convnext_atto",
        "mobileone_s0",
    ],
)
def test_classification_onnx_export(arch, tmpdir_factory):
    model = classification.__dict__[arch](pretrained=False, num_classes=10).eval()
    tmp_path = os.path.join(str(tmpdir_factory.mktemp("onnx")), f"{arch}.onnx")
    img_tensor = torch.rand((1, 3, 224, 224))
    with torch.no_grad():
        torch.onnx.export(model, img_tensor, tmp_path, export_params=True, opset_version=14)
