import pytest
import torch
from torch import nn

from holocron import trainer


def test_freeze_bn():

    # Simple module with BN
    mod = nn.Sequential(nn.Conv2d(3, 32, 3), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
    nb = mod[1].num_batches_tracked.clone()
    rm = mod[1].running_mean.clone()
    rv = mod[1].running_var.clone()
    # Freeze & forward
    for p in mod.parameters():
        p.requires_grad_(False)
    mod = trainer.freeze_bn(mod)
    for _ in range(10):
        _ = mod(torch.rand((1, 3, 32, 32)))
    # Check that stats were not updated
    assert torch.equal(mod[1].num_batches_tracked, nb)
    assert torch.equal(mod[1].running_mean, rm)
    assert torch.equal(mod[1].running_var, rv)


def test_freeze_model():

    # Simple model
    mod = nn.Sequential(nn.Conv2d(3, 32, 3), nn.ReLU(inplace=True), nn.Conv2d(32, 64, 3), nn.ReLU(inplace=True))
    mod = trainer.freeze_model(mod, "0")
    # Check that the correct layers were frozen
    assert not any(p.requires_grad for p in mod[0].parameters())
    assert all(p.requires_grad for p in mod[2].parameters())
    with pytest.raises(ValueError):
        trainer.freeze_model(mod, "wrong_layer")

    # Freeze last layer
    for p in mod[-1].parameters():
        p.requires_grad_(False)
    mod = trainer.freeze_model(mod, "0")
    # Ensure the last layer is now unfrozen
    assert all(p.requires_grad for p in mod[-1].parameters())
