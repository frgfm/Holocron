import pytest
import torch
from torch import nn

from holocron.models import classification


def _test_classification_model(name, num_classes=10):

    batch_size = 2
    x = torch.rand((batch_size, 3, 224, 224))
    model = classification.__dict__[name](pretrained=True, num_classes=num_classes).eval()
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
        assert torch.allclose(out, model(x), atol=1e-4)


@pytest.mark.parametrize(
    "arch",
    [
        "darknet24",
        "darknet19",
        "darknet53",
        "cspdarknet53",
        "cspdarknet53_mish",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "resnet50d",
        "res2net50_26w_4s",
        "tridentnet50",
        "pyconv_resnet50",
        "pyconvhg_resnet50",
        "rexnet1_0x",
        "rexnet1_3x",
        "rexnet1_5x",
        "rexnet2_0x",
        "rexnet2_2x",
        "sknet50",
        "sknet101",
        "sknet152",
        "repvgg_a0",
        "repvgg_b0",
        "convnext_micro",
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "convnext_large",
        "convnext_xl",
        "mobileone_s0",
        "mobileone_s1",
        "mobileone_s2",
        "mobileone_s3",
    ],
)
def test_classification_model(arch):
    num_classes = 1000 if arch in ["rexnet1_0x", "rexnet1_3x", "rexnet1_5x", "rexnet2_0x"] else 10
    _test_classification_model(arch, num_classes)
