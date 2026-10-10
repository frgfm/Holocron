from functools import partial

import pytest
import torch
from torchvision.models._utils import IntermediateLayerGetter  # noqa: PLC2701

from holocron.models import segmentation
from holocron.models.segmentation.unet import DynamicUNet, UpPath


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
    ("arch", "input_shape"),
    [
        ("unet", (256, 256)),
        ("unet2", (256, 256)),
        ("unet_rexnet13", (256, 256)),
        ("unet_tvvgg11", (256, 256)),
        ("unet_tvresnet34", (256, 256)),
        ("unetp", (256, 256)),
        ("unetpp", (256, 256)),
        ("unet3p", (320, 320)),
    ],
)
def test_segmentation_model(arch, input_shape):
    _test_segmentation_model(arch, input_shape)


@pytest.mark.parametrize(
    ("model_type", "layout", "kwargs"),
    [
        (segmentation.UNet, [5, 9, 13], {}),
        (segmentation.UNet, [5, 9, 13], {"bilinear_upsampling": False}),
        (segmentation.UNetp, [5, 9, 13], {}),
        (segmentation.UNetpp, [5, 9, 13], {}),
        (segmentation.UNet3p, [5, 9, 13, 17], {}),
    ],
)
@pytest.mark.parametrize("shape", [(32, 48), (35, 39)])
def test_segmentation_backward_arbitrary_shapes(model_type, layout, kwargs, shape):
    model = model_type(layout, in_channels=1, num_classes=3, **kwargs).train()
    output = model(torch.rand(2, 1, *shape))
    assert output.shape == (2, 3, *shape)
    target = torch.randint(3, (2, *shape))
    torch.nn.functional.cross_entropy(output, target).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("bilinear", [True, False])
def test_unet_valid_convolutions(bilinear):
    model = segmentation.UNet([4, 8], num_classes=3, same_padding=False, bilinear_upsampling=bilinear)
    output = model(torch.rand(2, 3, 128, 130))
    assert output.shape[-2] < 128
    assert output.shape[-1] < 130
    output.mean().backward()


def test_up_path_does_not_mutate_dense_skips():
    skips = [torch.rand(2, 4, 13, 15), torch.rand(2, 4, 13, 15)]
    originals = list(skips)
    output = UpPath(12, 4)(skips, torch.rand(2, 4, 6, 7))
    assert output.shape == (2, 4, 8, 10)
    assert all(actual is original for actual, original in zip(skips, originals, strict=True))


@pytest.mark.parametrize("shape", [(32, 48), (35, 39)])
def test_dynamic_unet_preserves_encoder_weights_and_stats(shape):
    backbone = torch.nn.Sequential(
        torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, stride=2, padding=1), torch.nn.BatchNorm2d(4)),
        torch.nn.Conv2d(4, 8, 3, stride=2, padding=1),
    )
    encoder = IntermediateLayerGetter(backbone, {"0": "0", "1": "1"})
    original = {name: value.clone() for name, value in encoder.state_dict().items()}
    model = DynamicUNet(encoder, num_classes=3, input_shape=(3, *shape), final_upsampling=True)
    assert encoder.training
    assert all(torch.equal(value, original[name]) for name, value in encoder.state_dict().items())
    output = model(torch.rand(2, 3, *shape))
    assert output.shape == (2, 3, *shape)
    output.mean().backward()


@pytest.mark.parametrize("factory", [False, True])
def test_unet3p_uses_custom_convolution_for_projections(factory):
    class CustomConv(torch.nn.Conv2d):
        pass

    conv_layer = partial(CustomConv, kernel_size=3) if factory else CustomConv
    model = segmentation.UNet3p([4, 8, 16], conv_layer=conv_layer)
    convolutions = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    assert all(isinstance(module, CustomConv) for module in convolutions if module is not model.classifier)


@pytest.mark.parametrize("arch", ["unet2", "unet_rexnet13"], ids=["plain", "rexnet"])
def test_dynamic_unet_grayscale_factory(arch):
    kwargs = {"pretrained_backbone": False} if arch == "unet_rexnet13" else {}
    model = segmentation.__dict__[arch](
        pretrained=False, in_channels=1, num_classes=3, input_shape=None, **kwargs
    ).eval()
    with torch.no_grad():
        output = model(torch.rand(2, 1, 35, 39))
    assert output.shape == (2, 3, 35, 39)


@pytest.mark.parametrize("num_classes", [2, 21])
def test_unet3p_initial_classifier_logit_scale(num_classes):
    torch.manual_seed(42)
    model = segmentation.UNet3p([64, 128, 256], num_classes=num_classes)
    features = torch.randn(8, 192, 8, 8)
    with torch.no_grad():
        logits = model.classifier(features)
    assert 0.1 < logits.std().item() < 2
