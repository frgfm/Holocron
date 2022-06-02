import torch
from torch.nn import CrossEntropyLoss, Linear
from torch.nn.functional import cross_entropy

from holocron import nn
from holocron.nn import functional as F


def _test_loss_function(loss_fn, same_loss=0.0, multi_label=False):

    num_batches = 2
    num_classes = 4
    # 4 classes
    x = torch.ones(num_batches, num_classes)
    x[:, 0, ...] = 100
    x.requires_grad_(True)

    # Identical target
    if multi_label:
        target = torch.zeros_like(x)
        target[:, 0] = 1.0
    else:
        target = torch.zeros(num_batches, dtype=torch.long)
    assert abs(loss_fn(x, target).item() - same_loss) < 1e-3
    assert torch.allclose(
        loss_fn(x, target, reduction="none"), same_loss * torch.ones(num_batches, dtype=x.dtype), atol=1e-3
    )

    # Check that class rescaling works
    x = torch.rand(num_batches, num_classes, requires_grad=True)
    if multi_label:
        target = torch.rand(x.shape)
    else:
        target = (num_classes * torch.rand(num_batches)).to(torch.long)
    weights = torch.ones(num_classes)
    assert loss_fn(x, target).item() == loss_fn(x, target, weight=weights).item()

    # Check that ignore_index works
    assert loss_fn(x, target).item() == loss_fn(x, target, ignore_index=num_classes).item()
    # Ignore an index we are certain to be in the target
    if multi_label:
        ignore_index = torch.unique(target.argmax(dim=1))[0].item()
    else:
        ignore_index = torch.unique(target)[0].item()
    assert loss_fn(x, target).item() != loss_fn(x, target, ignore_index=ignore_index)
    # Check backprop
    loss = loss_fn(x, target, ignore_index=0)
    loss.backward()

    # Test reduction
    assert torch.allclose(loss_fn(x, target, reduction="sum"), loss_fn(x, target, reduction="none").sum(), atol=1e-6)
    assert torch.allclose(
        loss_fn(x, target, reduction="mean"), loss_fn(x, target, reduction="sum") / target.shape[0], atol=1e-6
    )


def test_focal_loss():

    # Common verification
    _test_loss_function(F.focal_loss)

    num_batches = 2
    num_classes = 4
    x = torch.rand(num_batches, num_classes, 20, 20)
    target = (num_classes * torch.rand(num_batches, 20, 20)).to(torch.long)

    # Value check
    assert torch.allclose(F.focal_loss(x, target, gamma=0), cross_entropy(x, target), atol=1e-5)
    # Equal probabilities
    x = torch.ones(num_batches, num_classes, 20, 20)
    assert torch.allclose(
        (1 - 1 / num_classes) * F.focal_loss(x, target, gamma=0), F.focal_loss(x, target, gamma=1), atol=1e-5
    )

    assert repr(nn.FocalLoss()) == "FocalLoss(gamma=2.0, reduction='mean')"


def test_multilabel_cross_entropy():

    num_batches = 2
    num_classes = 4

    # Common verification
    _test_loss_function(F.multilabel_cross_entropy, multi_label=True)

    x = torch.rand(num_batches, num_classes, 20, 20)
    target = torch.zeros_like(x)
    target[:, 0] = 1.0

    # Value check
    assert torch.allclose(F.multilabel_cross_entropy(x, target), cross_entropy(x, target.argmax(dim=1)), atol=1e-5)

    assert repr(nn.MultiLabelCrossEntropy()) == "MultiLabelCrossEntropy(reduction='mean')"


def test_complement_cross_entropy():

    num_batches = 2
    num_classes = 4

    x = torch.rand((num_batches, num_classes, 20, 20), requires_grad=True)
    target = (num_classes * torch.rand(num_batches, 20, 20)).to(torch.long)

    # Backprop
    out = F.complement_cross_entropy(x, target, ignore_index=0)
    out.backward()

    assert repr(nn.ComplementCrossEntropy()) == "ComplementCrossEntropy(gamma=-1, reduction='mean')"


def test_mc_loss():

    num_batches = 2
    num_classes = 4
    xi = 2
    # 4 classes
    x = torch.ones(num_batches, xi * num_classes)
    x[:, 0, ...] = 10
    target = torch.zeros(num_batches, dtype=torch.long)

    mod = Linear(xi * num_classes, xi * num_classes)

    # Check backprop
    for reduction in ["mean", "sum", "none"]:
        for p in mod.parameters():
            p.grad = None
        train_loss = F.mutual_channel_loss(mod(x), target, ignore_index=0, reduction=reduction)
        if reduction == "none":
            assert train_loss.shape == (num_batches,)
            train_loss = train_loss.sum()
        train_loss.backward()
        assert isinstance(mod.weight.grad, torch.Tensor)

    # Check type casting of weights
    for p in mod.parameters():
        p.grad = None
    class_weights = torch.ones(num_classes, dtype=torch.float16)
    ignore_index = 0

    criterion = nn.MutualChannelLoss(weight=class_weights, ignore_index=ignore_index, xi=xi)
    train_loss = criterion(mod(x), target)
    train_loss.backward()
    assert isinstance(mod.weight.grad, torch.Tensor)
    assert repr(criterion) == f"MutualChannelLoss(reduction='mean', xi={xi}, alpha=1)"


def test_cb_loss():

    num_batches = 2
    num_classes = 4
    x = torch.rand(num_batches, num_classes, 20, 20)
    beta = 0.99
    num_samples = 10 * torch.ones(num_classes, dtype=torch.long)

    # Identical target
    target = (num_classes * torch.rand(num_batches, 20, 20)).to(torch.long)
    base_criterion = CrossEntropyLoss()
    base_loss = base_criterion(x, target)
    criterion = nn.ClassBalancedWrapper(base_criterion, num_samples, beta=beta)

    assert isinstance(criterion.criterion, CrossEntropyLoss)
    assert criterion.criterion.weight is not None

    # Value tests
    loss_value = criterion(x, target)
    assert loss_value.shape == base_loss.shape
    # assert torch.allclose(criterion(x, target), (1 - beta) / (1 - beta ** num_samples[0]) * base_loss, atol=1e-5)
    # With pre-existing weights
    base_criterion = CrossEntropyLoss(weight=torch.ones(num_classes, dtype=torch.float32))
    base_weights = base_criterion.weight.clone()
    criterion = nn.ClassBalancedWrapper(base_criterion, num_samples, beta=beta)
    assert not torch.equal(base_weights, criterion.criterion.weight)
    # assert torch.allclose(criterion(x, target), (1 - beta) / (1 - beta ** num_samples[0]) * base_loss, atol=1e-5)

    assert repr(criterion) == "ClassBalancedWrapper(CrossEntropyLoss(), beta=0.99)"


def test_dice_loss():

    num_batches = 2
    num_classes = 4

    x = torch.rand((num_batches, num_classes, 20, 20), requires_grad=True)
    target = torch.rand(num_batches, num_classes, 20, 20)

    # Backprop
    out = F.dice_loss(x, target)
    out.backward()

    # Weighted loss
    class_weights = torch.ones(num_classes)
    class_weights[0] = 2
    out = F.dice_loss(x, target, weight=class_weights)
    out.backward()

    assert repr(nn.DiceLoss()) == "DiceLoss(reduction='mean', gamma=1.0, eps=1e-08)"


def test_poly_loss():

    _test_loss_function(F.poly_loss)

    num_batches = 2
    num_classes = 4

    x = torch.rand((num_batches, num_classes, 20, 20), requires_grad=True)
    target = torch.rand(num_batches, num_classes, 20, 20)
    target = (num_classes * torch.rand(num_batches, 20, 20)).to(torch.long)

    # Backprop
    out = F.poly_loss(x, target)
    out.backward()

    # Weighted loss
    class_weights = torch.ones(num_classes)
    class_weights[0] = 2
    out = F.poly_loss(x, target, weight=class_weights)
    out.backward()

    assert repr(nn.PolyLoss()) == "PolyLoss(eps=2.0, reduction='mean')"
