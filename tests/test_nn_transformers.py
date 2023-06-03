import pytest
import torch

from holocron import nn


def test_transformer_encoder_block():

    x = torch.rand(1, 10, 128)

    mod = nn.TransformerEncoderBlock(num_layers=1, num_heads=1, d_model=128, dff=128, dropout=0.1)

    with torch.no_grad():
        out = mod(x)
    assert out.shape == x.shape

    # Check inference mode
    mod.eval()

    with torch.no_grad():
        out = mod(x)
    assert out.shape == x.shape

    # Check d_model divisible by num_heads
    with pytest.raises(AssertionError):
        nn.TransformerEncoderBlock(num_layers=1, num_heads=18, d_model=12, dff=12, dropout=0.0)
