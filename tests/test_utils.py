import numpy as np
import pytest
import torch
from PIL import Image

from holocron import utils


def test_mixup():
    batch_size = 8
    num_classes = 10
    shape = (3, 32, 32)
    with pytest.raises(ValueError):
        utils.data.Mixup(num_classes, alpha=-1.0)
    # Generate all dependencies
    mix = utils.data.Mixup(num_classes, alpha=0.2)
    img, target = torch.rand((batch_size, *shape)), torch.arange(num_classes)[:batch_size]
    mix_img, mix_target = mix(img.clone(), target.clone())
    assert img.shape == (batch_size, *shape)
    assert not torch.equal(img, mix_img)
    assert mix_target.dtype == torch.float32
    assert mix_target.shape == (batch_size, num_classes)
    assert torch.all(mix_target.sum(dim=1) == 1.0)
    count = (mix_target > 0).sum(dim=1)
    assert torch.all((count == 2.0) | (count == 1.0))

    # Alpha = 0 case
    mix = utils.data.Mixup(num_classes, alpha=0.0)
    mix_img, mix_target = mix(img.clone(), target.clone())
    assert torch.equal(img, mix_img)
    assert mix_target.dtype == torch.float32
    assert mix_target.shape == (batch_size, num_classes)
    assert torch.all(mix_target.sum(dim=1) == 1.0)
    assert torch.all((mix_target > 0).sum(dim=1) == 1.0)

    # Binary target
    mix = utils.data.Mixup(1, alpha=0.5)
    img = torch.rand((batch_size, *shape))
    target = torch.concat((torch.zeros(batch_size // 2), torch.ones(batch_size - batch_size // 2)))
    mix_img, mix_target = mix(img.clone(), target.clone())
    assert img.shape == (batch_size, *shape)
    assert not torch.equal(img, mix_img)
    assert mix_target.dtype == torch.float32
    assert mix_target.shape == (batch_size, 1)

    # Already in one-hot
    mix = utils.data.Mixup(num_classes, alpha=0.2)
    img, target = torch.rand((batch_size, *shape)), torch.rand((batch_size, num_classes))
    mix_img, mix_target = mix(img.clone(), target.clone())
    assert img.shape == (batch_size, *shape)
    assert not torch.equal(img, mix_img)
    assert mix_target.dtype == torch.float32
    assert mix_target.shape == (batch_size, num_classes)


@pytest.mark.parametrize(
    ("arr", "fn", "expected", "progress", "num_threads"),
    [
        ([1, 2, 3], lambda x: x**2, [1, 4, 9], False, 3),
        ([1, 2, 3], lambda x: x**2, [1, 4, 9], True, 1),
        ("hello", lambda x: x.upper(), list("HELLO"), True, None),
        ("hello", lambda x: x.upper(), list("HELLO"), False, None),
    ],
)
def test_parallel(arr, fn, expected, progress, num_threads):
    assert utils.parallel(fn, arr, progress=progress, num_threads=num_threads) == expected


def test_find_image_size():
    ds = [(Image.fromarray(np.full((16, 16, 3), 255, dtype=np.uint8)), 0) for _ in range(100)]
    utils.find_image_size(ds, block=False)
