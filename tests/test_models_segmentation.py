import os

import pytest
import torch

from holocron.models import segmentation


def _test_segmentation_model(name, input_shape):

    num_classes = 10
    batch_size = 2
    num_channels = 3
    x = torch.rand((batch_size, num_channels, *input_shape))
    # Check pretrained version
    model = segmentation.__dict__[name](pretrained=True).eval()
    # Check custom number of output classes
    model = segmentation.__dict__[name](pretrained=False, num_classes=num_classes).eval()
    with torch.no_grad():
        out = model(x)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (batch_size, num_classes, *input_shape)


@pytest.mark.parametrize(
    "arch, input_shape",
    [
        ["unet", (256, 256)],
        ["unet2", (256, 256)],
        ["unet_rexnet13", (256, 256)],
        ["unet_tvvgg11", (256, 256)],
        ["unet_tvresnet34", (256, 256)],
        ["unetp", (256, 256)],
        ["unetpp", (256, 256)],
        ["unet3p", (320, 320)],
    ],
)
def test_segmentation_model(arch, input_shape):
    _test_segmentation_model(arch, input_shape)


@pytest.mark.parametrize(
    "arch, input_shape",
    [
        ["unet", (256, 256)],
        ["unet2", (256, 256)],
        ["unetp", (256, 256)],
        ["unetpp", (256, 256)],
        ["unet3p", (320, 320)],
    ],
)
def test_segmentation_onnx_export(arch, input_shape, tmpdir_factory):
    model = segmentation.__dict__[arch](pretrained=False, num_classes=10).eval()
    tmp_path = os.path.join(str(tmpdir_factory.mktemp("onnx")), f"{arch}.onnx")
    img_tensor = torch.rand((1, 3, *input_shape))
    with torch.no_grad():
        torch.onnx.export(model, img_tensor, tmp_path, export_params=True, opset_version=14)
