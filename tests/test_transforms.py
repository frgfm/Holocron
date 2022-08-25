import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from holocron import transforms as T
from holocron.transforms.interpolation import ResizeMethod


def test_resize():

    # Arg check
    with pytest.raises(AssertionError):
        T.Resize(16)

    with pytest.raises(AssertionError):
        T.Resize((16, 16), mode="stretch")

    with pytest.raises(AssertionError):
        T.Resize((16, 16), mode="pad")

    img1 = np.full((16, 32, 3), 255, dtype=np.uint8)
    img2 = np.full((32, 16, 3), 255, dtype=np.uint8)
    tf = T.Resize((32, 32), mode=ResizeMethod.PAD)
    assert isinstance(tf, nn.Module)

    # PIL Image
    out = tf(Image.fromarray(img1))
    assert isinstance(out, Image.Image)
    assert out.size == (32, 32)
    np_out = np.asarray(out)
    assert np.all(np_out[8:-8] == 255) and np.all(np_out[:8] == 0) and np.all(np_out[-8:]) == 0
    out = tf(Image.fromarray(img2))
    assert isinstance(out, Image.Image)
    assert out.size == (32, 32)
    np_out = np.asarray(out)
    assert np.all(np_out[:, 8:-8] == 255) and np.all(np_out[:, :8] == 0) and np.all(np_out[:, -8:]) == 0
    # Squish
    out = T.Resize((32, 32), mode=ResizeMethod.SQUISH)(Image.fromarray(img1))
    assert np.all(np.asarray(out) == 255)

    # Tensor
    out = tf(torch.from_numpy(img1).to(dtype=torch.float32).permute(2, 0, 1) / 255)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 32, 32)
    np_out = out.numpy()
    assert np.all(np_out[:, 8:-8] == 1) and np.all(np_out[:, :8] == 0) and np.all(np_out[:, -8:]) == 0
    out = tf(torch.from_numpy(img2).to(dtype=torch.float32).permute(2, 0, 1) / 255)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 32, 32)
    np_out = out.numpy()
    assert np.all(np_out[:, :, 8:-8] == 1) and np.all(np_out[:, :, :8] == 0) and np.all(np_out[:, :, -8:]) == 0


def test_randomzoomout():

    # Arg check
    with pytest.raises(AssertionError):
        T.RandomZoomOut(224)

    with pytest.raises(AssertionError):
        T.Resize((16, 16), (1, 0.5))

    pil_img = Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8))
    torch_img = torch.ones((3, 64, 64), dtype=torch.float32)
    tf = T.RandomZoomOut((32, 32), scale=(0.5, 0.99))
    assert isinstance(tf, nn.Module)

    # PIL Image
    out = tf(pil_img)
    assert isinstance(out, Image.Image)
    assert out.size == (32, 32)
    np_out = np.asarray(out)
    assert np.all(np_out[16, 16] == 255) and np_out.mean() < 255

    # Tensor
    out = tf(torch_img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 32, 32)
    np_out = np.asarray(out)
    assert np.all(np_out[:, 16, 16] == 1) and np_out.mean() < 1
