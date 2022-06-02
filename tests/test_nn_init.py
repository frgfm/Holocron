import torch
from torch import nn

from holocron.nn import init


def test_init():

    module = nn.Sequential(nn.Conv2d(3, 32, 3), nn.BatchNorm2d(32), nn.LeakyReLU(inplace=True))

    # Check that each layer was initialized correctly
    init.init_module(module, "leaky_relu")
    assert torch.all(module[0].bias.data == 0)
    assert torch.all(module[1].weight.data == 1)
    assert torch.all(module[1].bias.data == 0)
