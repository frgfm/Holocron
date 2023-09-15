import pytest
import torch
from torch import nn

from holocron.models import utils
from holocron.models.classification.repvgg import RepVGG
from holocron.nn import SAM, BlurPool2d, DropBlock2d


def _test_conv_seq(conv_seq, expected_classes, expected_channels):
    assert len(conv_seq) == len(expected_classes)
    for _layer, mod_class in zip(conv_seq, expected_classes):
        assert isinstance(_layer, mod_class)

    input_t = torch.rand(1, conv_seq[0].in_channels, 224, 224)
    out = torch.nn.Sequential(*conv_seq)(input_t)
    assert out.shape[:2] == (1, expected_channels)
    out.sum().backward()


def test_conv_sequence():
    mod = utils.conv_sequence(
        3,
        32,
        kernel_size=3,
        act_layer=nn.ReLU(inplace=True),
        norm_layer=nn.BatchNorm2d,
        drop_layer=DropBlock2d,
        attention_layer=SAM,
    )

    _test_conv_seq(mod, [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, SAM, DropBlock2d], 32)
    assert mod[0].kernel_size == (3, 3)

    mod = utils.conv_sequence(
        3,
        32,
        kernel_size=3,
        stride=2,
        act_layer=nn.ReLU(inplace=True),
        norm_layer=nn.BatchNorm2d,
        drop_layer=DropBlock2d,
        blurpool=True,
    )
    _test_conv_seq(mod, [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, BlurPool2d, DropBlock2d], 32)
    assert mod[0].kernel_size == (3, 3)
    assert mod[0].stride == (1, 1)
    assert mod[3].stride == 2
    assert mod[0].bias is None
    # Ensures that bias is added when there is no BN
    mod = utils.conv_sequence(3, 32, kernel_size=3, stride=2, act_layer=nn.ReLU(inplace=True))
    assert isinstance(getattr(mod[0], "bias"), nn.Parameter)


def test_fuse_conv_bn():
    # Check the channel verification
    with pytest.raises(AssertionError):
        utils.fuse_conv_bn(nn.Conv2d(3, 5, 3), nn.BatchNorm2d(3))

    # Prepare candidate modules
    conv = nn.Conv2d(3, 8, 3, padding=1, bias=False).eval()
    bn = nn.BatchNorm2d(8).eval()
    bn.weight.data = torch.rand(8)

    # Create the fused version
    fused_conv = nn.Conv2d(3, 8, 3, padding=1, bias=True).eval()
    k, b = utils.fuse_conv_bn(conv, bn)
    fused_conv.weight.data = k
    fused_conv.bias.data = b

    # Check values
    batch_size = 2
    x = torch.rand((batch_size, 3, 32, 32))
    with torch.no_grad():
        assert torch.allclose(bn(conv(x)), fused_conv(x), atol=1e-6)

    # Check the warning when there is already a bias
    conv = nn.Conv2d(3, 8, 3, padding=1, bias=True).eval()
    k, b = utils.fuse_conv_bn(conv, bn)
    fused_conv.weight.data = k
    fused_conv.bias.data = b
    with torch.no_grad():
        assert torch.allclose(bn(conv(x)), fused_conv(x), atol=1e-6)


def test_model_from_hf_hub():
    model = utils.model_from_hf_hub("frgfm/repvgg_a0")
    # Check model type
    assert isinstance(model, RepVGG)

    # Check num of params
    assert sum(p.data.numel() for p in model.parameters()) == 24741642
