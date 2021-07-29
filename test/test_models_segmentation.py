# Copyright (C) 2019-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import pytest
import torch
from torch import nn
from holocron.models import segmentation


def _test_segmentation_model(name, input_shape):

    num_classes = 10
    batch_size = 2
    num_channels = 3
    x = torch.rand((batch_size, num_channels, *input_shape))
    model = segmentation.__dict__[name](pretrained=True, num_classes=num_classes).eval()
    with torch.no_grad():
        out = model(x)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (batch_size, num_classes, *input_shape)


@pytest.mark.parametrize(
    "arch, input_shape",
    [
        ['unet', (256, 256)],
        ['unet2', (256, 256)],
        ['unet_rexnet13', (256, 256)],
        ['unet_vgg11', (256, 256)],
        ['unet_tvresnet34', (256, 256)],
        ['unetp', (256, 256)],
        ['unetpp', (256, 256)],
        ['unet3p', (320, 320)],
    ],
)
def test_segmentation_model(arch, input_shape):
    _test_segmentation_model(arch, input_shape)
