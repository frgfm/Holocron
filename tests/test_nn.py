import torch

from holocron.nn.modules import dropblock


def test_dropblock2d():
    x = torch.rand(2, 4, 16, 16)

    # Drop probability of 1
    mod = dropblock.DropBlock2d(1.0, 1, inplace=False)

    with torch.no_grad():
        out = mod(x)
    assert torch.equal(out, torch.zeros_like(x))

    # Drop probability of 0
    mod = dropblock.DropBlock2d(0.0, 3, inplace=False)

    with torch.no_grad():
        out = mod(x)
    assert torch.equal(out, x)
    assert out.data_ptr == x.data_ptr

    # Check inference mode
    mod = dropblock.DropBlock2d(1.0, 3, inplace=False).eval()

    with torch.no_grad():
        out = mod(x)
    assert torch.equal(out, x)

    # Check inplace
    mod = dropblock.DropBlock2d(1.0, 3, inplace=True)

    with torch.no_grad():
        out = mod(x)
    assert out.data_ptr == x.data_ptr
