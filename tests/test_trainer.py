import math
import sys
import warnings
from argparse import Namespace
from io import StringIO

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.models import get_model, get_model_weights
from tqdm.auto import tqdm

from holocron import trainer
from holocron.nn import GlobalAvgPool2d
from holocron.trainer.detection import assign_iou
from references._common import run_training  # noqa: PLC2701


@pytest.mark.parametrize("failure", [False, True])
def test_reference_tracking_reports_training_status(monkeypatch, failure):
    exit_codes = []

    run = Namespace(finish=lambda exit_code=0: exit_codes.append(exit_code))
    monkeypatch.setitem(sys.modules, "wandb", Namespace(init=lambda **_kwargs: run, log=lambda _metrics: None))

    def fit(*_args, **_kwargs):
        if failure:
            raise RuntimeError("training failed")

    learner = Namespace(fit_n_epochs=fit)
    args = Namespace(
        test_only=False,
        find_lr=False,
        check_setup=False,
        wb=True,
        name="test",
        arch="linear",
        epochs=1,
        lr=0.01,
        freeze_until=None,
        sched="cosine",
        norm_wd=None,
    )
    if failure:
        with pytest.raises(RuntimeError, match="training failed"):
            run_training(learner, args, project="test", config={})
    else:
        run_training(learner, args, project="test", config={})
    assert exit_codes == [int(failure)]


class MockClassificationDataset(Dataset):
    """Mock dataset generating a random sample and a fixed zero target"""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def __getitem__(self, idx):
        return torch.rand((3, 32, 32)), 0

    def __len__(self):
        return self.n


class MockBinaryClassificationDataset(Dataset):
    """Mock dataset generating a random sample and a fixed zero probability"""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def __getitem__(self, idx):
        return torch.rand((3, 32, 32)), torch.zeros((1,))

    def __len__(self):
        return self.n


class MockBinaryClassificationDatasetBis(MockBinaryClassificationDataset):
    """Mock dataset generating a random sample and a fixed zero probability"""

    def __getitem__(self, idx):
        return torch.rand((3, 32, 32)), 0


class MockSegDataset(Dataset):
    """Mock dataset generating a random sample and a fixed zero target"""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def __getitem__(self, idx):
        return torch.rand((3, 32, 32)), torch.zeros((32, 32), dtype=torch.long)

    def __len__(self):
        return self.n


class MockDetDataset(Dataset):
    """Mock dataset generating a random sample and a fixed zero target"""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def __getitem__(self, idx):
        boxes = torch.tensor([[0, 0, 1, 1], [0.25, 0.25, 0.75, 0.75]], dtype=torch.float32)
        return torch.rand((3, 320, 320)), {"boxes": boxes, "labels": torch.ones(2, dtype=torch.long)}

    def __len__(self):
        return self.n


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_assign_iou_device(device):
    gt_boxes = torch.tensor([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=torch.float32, device=device)
    pred_boxes = gt_boxes.clone()

    gt_indices, pred_indices = assign_iou(gt_boxes, pred_boxes)

    assert gt_indices.device == gt_boxes.device
    assert pred_indices.device == gt_boxes.device
    torch.testing.assert_close(gt_indices, torch.tensor([0, 1], device=device))
    torch.testing.assert_close(pred_indices, torch.tensor([0, 1], device=device))


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_assign_iou_duplicate_matches(device):
    gt_boxes = torch.tensor(
        [[4, 4, 5, 5], [0, 0, 1, 1], [2, 2, 2.8, 2.8], [2, 2, 3, 3]], dtype=torch.float32, device=device
    )
    pred_boxes = gt_boxes[[1, 3]]

    gt_indices, pred_indices = assign_iou(gt_boxes, pred_boxes)

    assert gt_indices == [1, 3]
    assert pred_indices == [0, 1]


def test_detection_evaluation_duplicate_matches(monkeypatch):
    boxes = torch.tensor([[0, 0, 1, 1], [2, 2, 2.8, 2.8], [2, 2, 3, 3]], dtype=torch.float32)
    target = {"boxes": boxes, "labels": torch.tensor([0, 0, 1])}
    detections = {"boxes": boxes[[0, 2]], "labels": torch.tensor([0, 1])}
    loader = DataLoader([(torch.zeros(3, 8, 8), target)], collate_fn=collate_fn)
    model = nn.Linear(1, 1)
    monkeypatch.setattr(model, "forward", lambda _x: [detections])
    learner = trainer.DetectionTrainer(model, loader, loader, None, torch.optim.SGD(model.parameters(), lr=0.1))

    metrics = learner.evaluate()

    assert metrics == pytest.approx({"loc_err": 0.2, "clf_err": 0.0, "det_err": 0.2, "val_loss": 0.2})


def collate_fn(batch):
    imgs, target = zip(*batch, strict=False)
    return imgs, target


class _CountingSGD(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.used_lrs = []

    def step(self, *args, **kwargs):
        self.used_lrs.append(float(self.param_groups[0]["lr"]))
        return super().step(*args, **kwargs)


class _CountingScheduler(torch.optim.lr_scheduler.MultiplicativeLR):
    instance = None

    def __init__(self, *args, **kwargs):
        self.steps = -1
        super().__init__(*args, **kwargs)
        type(self).instance = self

    def step(self, *args, **kwargs):
        self.steps += 1
        return super().step(*args, **kwargs)


def _linear_trainer(x, target, gradient_acc, optimizer_cls=torch.optim.SGD):
    dataset = TensorDataset(x, target)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = nn.Linear(1, 1, bias=False)
    optimizer = optimizer_cls(model.parameters(), lr=0.1)
    return trainer.Trainer(model, loader, loader, nn.MSELoss(), optimizer, gradient_acc=gradient_acc)


def test_gradient_accumulation_matches_large_batch():
    x = torch.tensor([[1.0], [3.0]])
    target = torch.tensor([[2.0], [0.0]])
    accumulated = _linear_trainer(x, target, gradient_acc=2)
    large_batch = _linear_trainer(x, target, gradient_acc=1)
    large_batch.model.load_state_dict(accumulated.model.state_dict())

    for batch in accumulated.train_loader:
        accumulated._backprop_step(accumulated._get_loss(*batch))
    large_batch._backprop_step(large_batch._get_loss(x, target))

    torch.testing.assert_close(accumulated.model.weight, large_batch.model.weight)


def test_gradient_accumulation_flushes_partial_batch():
    x = torch.arange(1, 6, dtype=torch.float32).unsqueeze(1)
    target = torch.zeros_like(x)
    learner = _linear_trainer(x, target, gradient_acc=2, optimizer_cls=_CountingSGD)
    learner.model.weight.data.fill_(1)
    initial_weight = learner.model.weight.detach().clone()

    learner._reset_scheduler(0.1, 2, sched_type="cosine")
    assert learner.scheduler.T_max == 2 * math.ceil(len(learner.train_loader) / learner.gradient_acc)

    class Scheduler:
        steps = 0

        def step(self):
            self.steps += 1

    learner.scheduler = Scheduler()
    learner._fit_epoch()

    assert (
        len(learner.optimizer.used_lrs)
        == learner.scheduler.steps
        == math.ceil(len(learner.train_loader) / learner.gradient_acc)
    )
    assert learner._grad_count == 0
    assert not torch.equal(learner.model.weight, initial_weight)


@pytest.mark.parametrize(
    ("gradient_acc", "num_it", "expected_steps"),
    [(1, 5, 5), (2, 5, 3), (8, 100, 13), (8, 3, 1)],
)
def test_find_lr_gradient_accumulation(monkeypatch, gradient_acc, num_it, expected_steps):
    x = torch.arange(1, num_it + 1, dtype=torch.float32).unsqueeze(1)
    learner = _linear_trainer(x, torch.zeros_like(x), gradient_acc, optimizer_cls=_CountingSGD)
    losses = list(range(num_it, 0, -1))
    loss_iter = iter(losses)
    learner._get_loss = lambda *_args: learner.model.weight.sum() * 0 + next(loss_iter)
    monkeypatch.setattr("holocron.trainer.core.MultiplicativeLR", _CountingScheduler)
    progress_totals = []
    monkeypatch.setattr(
        "holocron.trainer.core.tqdm", lambda data, **kwargs: progress_totals.append(kwargs["total"]) or data
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        learner.find_lr(start_lr=1e-3, end_lr=1e-1, num_it=num_it)

    expected_losses = []
    for idx in range(0, num_it, gradient_acc):
        window = losses[idx : idx + gradient_acc]
        expected_losses.append(sum(window) / len(window))
    assert len(learner.optimizer.used_lrs) == _CountingScheduler.instance.steps == expected_steps
    assert learner.lr_recorder == pytest.approx(learner.optimizer.used_lrs)
    assert learner.loss_recorder == pytest.approx(expected_losses)
    assert learner.lr_recorder[0] == pytest.approx(1e-3)
    assert learner.lr_recorder[-1] == pytest.approx(1e-1 if expected_steps > 1 else 1e-3)
    assert progress_totals == [expected_steps]
    assert learner._grad_count == 0
    assert not any("lr_scheduler.step() before optimizer.step()" in str(warning.message) for warning in caught)
    if expected_steps > 1:
        plotted_lrs = []
        monkeypatch.setattr("matplotlib.pyplot.plot", lambda lrs, _losses: plotted_lrs.extend(lrs))
        learner.plot_recorder(block=False)
        assert plotted_lrs == pytest.approx(learner.lr_recorder)


def test_find_lr_ignores_amp_overflow(monkeypatch):
    class SkipFirstGradScaler:
        def __init__(self, *_args, **_kwargs):
            self.current_scale = 2.0
            self.skip = True

        @staticmethod
        def scale(loss):
            return loss

        def unscale_(self, _optimizer):
            pass

        def step(self, optimizer):
            if not self.skip:
                optimizer.step()

        def update(self):
            if self.skip:
                self.current_scale /= 2
                self.skip = False

        def get_scale(self):
            return self.current_scale

    x = torch.tensor([[1.0], [2.0], [3.0]])
    learner = _linear_trainer(x, torch.zeros_like(x), gradient_acc=1, optimizer_cls=_CountingSGD)
    losses = iter([1.0, 2.0, 3.0])
    learner._get_loss = lambda *_args: learner.model.weight.sum() * 0 + next(losses)
    learner.amp = True
    monkeypatch.setattr("holocron.trainer.core.GradScaler", SkipFirstGradScaler)
    monkeypatch.setattr("holocron.trainer.core.MultiplicativeLR", _CountingScheduler)
    monkeypatch.setattr("holocron.trainer.core.tqdm", lambda data, **_kwargs: data)

    learner.find_lr(start_lr=1e-3, end_lr=1e-1, num_it=2)

    assert learner.optimizer.used_lrs == pytest.approx([1e-3, 1e-1])
    assert _CountingScheduler.instance.steps == 2
    assert learner.lr_recorder == pytest.approx([1e-3, 1e-1])
    assert learner.loss_recorder == pytest.approx([2.0, 3.0])
    assert learner._grad_count == 0


def _test_trainer(
    learner: trainer.Trainer, num_it: int, ref_param: str, freeze_until: str | None = None, lr: float = 1e-3
) -> None:
    trainer.utils.freeze_model(learner.model.train(), freeze_until)
    learner._reset_opt(lr)
    # Update param groups & LR
    learner.save(learner.output_file)
    checkpoint = torch.load(learner.output_file, map_location="cpu")
    model_w = learner.model.state_dict()[ref_param].clone()
    # Check setup
    learner.check_setup(freeze_until, num_it=num_it, block=False)

    # LR Find
    learner.load(checkpoint)

    with pytest.raises(AssertionError):
        learner.plot_recorder(block=False)

    with pytest.raises(ValueError):
        learner.find_lr(freeze_until, num_it=num_it + 1)

    # All params are frozen
    for p in learner.model.parameters():
        p.requires_grad_(False)
    with pytest.raises(AssertionError):
        learner._set_params()

    # Test norm weight decay
    learner.find_lr(freeze_until, norm_weight_decay=5e-4, num_it=num_it)
    assert len(learner.lr_recorder) == len(learner.loss_recorder)
    learner.plot_recorder(block=False)

    # Training
    # Perform the iterations
    learner.load(checkpoint)
    with pytest.raises(ValueError):
        learner.fit_n_epochs(1, 1e-3, freeze_until, sched_type="my_scheduler")
    learner.fit_n_epochs(1, 1e-3, freeze_until)
    # Check that params were updated
    assert not torch.equal(learner.model.state_dict()[ref_param], model_w)
    learner.load(checkpoint)
    learner.fit_n_epochs(1, 1e-3, freeze_until, sched_type="cosine")
    # Check that params were updated
    assert not torch.equal(learner.model.state_dict()[ref_param], model_w)

    # Gradient accumulation
    learner.load(checkpoint)
    assert torch.equal(learner.model.state_dict()[ref_param], model_w)
    learner.model.train()
    learner.gradient_acc = 2
    learner._reset_opt(lr)
    train_iter = iter(learner.train_loader)
    assert all(torch.all(p.grad == 0) for p in learner.model.parameters() if p.requires_grad and p.grad is not None)
    x, target = next(train_iter)
    x, target = learner.to_cuda(x, target)
    loss = learner._get_loss(x, target)
    learner._backprop_step(loss)
    assert torch.equal(learner.model.state_dict()[ref_param], model_w)
    assert all(torch.any(p.grad != 0) for p in learner.model.parameters() if p.requires_grad and p.grad is not None)
    # With accumulation of 2, the update step is performed every 2 batches
    x, target = next(train_iter)
    x, target = learner.to_cuda(x, target)
    loss = learner._get_loss(x, target)
    learner._backprop_step(loss)
    assert not torch.equal(learner.model.state_dict()[ref_param], model_w)
    assert all(torch.all(p.grad == 0) for p in learner.model.parameters() if p.requires_grad and p.grad is not None)


def test_classification_trainer(tmpdir_factory):
    folder = tmpdir_factory.mktemp("checkpoints")
    file_path = str(folder.join("nested", "tmp.pt"))

    num_it = 100
    batch_size = 8
    # Generate all dependencies
    model = nn.Sequential(nn.Conv2d(3, 32, 3), nn.ReLU(inplace=True), GlobalAvgPool2d(flatten=True), nn.Linear(32, 5))
    train_loader = DataLoader(MockClassificationDataset(num_it * batch_size), batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    with pytest.raises(ValueError if torch.cuda.is_available() else AssertionError):
        trainer.ClassificationTrainer(model, train_loader, train_loader, criterion, optimizer, gpu=7)

    learner = trainer.ClassificationTrainer(
        model,
        train_loader,
        train_loader,
        criterion,
        optimizer,
        output_file=file_path,
        gpu=0 if torch.cuda.is_available() else None,
    )

    _test_trainer(learner, num_it, "3.weight", None)
    # AMP
    learner = trainer.ClassificationTrainer(
        model,
        train_loader,
        train_loader,
        criterion,
        optimizer,
        output_file=file_path,
        gpu=0 if torch.cuda.is_available() else None,
        amp=torch.cuda.is_available(),
    )
    # Top losses
    learner.plot_top_losses((0, 0, 0), (1, 1, 1), [str(idx) for idx in range(5)], block=False)
    _test_trainer(learner, num_it, "3.weight", None)


def test_classification_trainer_few_classes():
    num_it = 10
    batch_size = 8
    # Generate all dependencies
    model = nn.Sequential(nn.Conv2d(3, 32, 3), nn.ReLU(inplace=True), GlobalAvgPool2d(flatten=True), nn.Linear(32, 3))
    train_loader = DataLoader(MockClassificationDataset(num_it * batch_size), batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    learner = trainer.ClassificationTrainer(model, train_loader, train_loader, criterion, optimizer)
    # Fewer than 5 classes
    assert learner.evaluate()["acc5"] == 0


def test_binary_classification_trainer():
    num_it = 10
    batch_size = 8
    # Generate all dependencies
    model = nn.Sequential(nn.Conv2d(3, 32, 3), nn.ReLU(inplace=True), GlobalAvgPool2d(flatten=True), nn.Linear(32, 1))
    train_loader = DataLoader(MockBinaryClassificationDataset(num_it * batch_size), batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    learner = trainer.BinaryClassificationTrainer(model, train_loader, train_loader, criterion, optimizer)

    res = learner.evaluate()
    assert 0 <= res["acc"] <= 1

    # Check that it works also for incorrect shaped / data-formatted targets
    train_loader = DataLoader(MockBinaryClassificationDatasetBis(num_it * batch_size), batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    gpu = 0 if torch.cuda.is_available() else None
    learner = trainer.BinaryClassificationTrainer(
        model, train_loader, train_loader, criterion, optimizer, gpu=gpu, amp=gpu is not None
    )

    res = learner.evaluate()
    assert 0 <= res["acc"] <= 1

    # Top losses
    learner.plot_top_losses((0, 0, 0), (1, 1, 1), block=False)


def test_segmentation_trainer(tmpdir_factory):
    folder = tmpdir_factory.mktemp("checkpoints")
    file_path = str(folder.join("tmp.pt"))

    num_it = 100
    batch_size = 8
    # Generate all dependencies
    model = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 5, 3, padding=1))
    train_loader = DataLoader(MockSegDataset(num_it * batch_size), batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    learner = trainer.SegmentationTrainer(
        model,
        train_loader,
        train_loader,
        criterion,
        optimizer,
        num_classes=5,
        output_file=file_path,
        gpu=0 if torch.cuda.is_available() else None,
    )

    _test_trainer(learner, num_it, "2.weight", None)
    # AMP
    learner = trainer.SegmentationTrainer(
        model,
        train_loader,
        train_loader,
        criterion,
        optimizer,
        num_classes=5,
        output_file=file_path,
        gpu=0 if torch.cuda.is_available() else None,
        amp=torch.cuda.is_available(),
    )
    _test_trainer(learner, num_it, "2.weight", None)


def test_detection_trainer(tmpdir_factory):
    folder = tmpdir_factory.mktemp("checkpoints")
    file_path = str(folder.join("tmp.pt"))

    num_it = 10
    batch_size = 2
    # Generate all dependencies
    weights = get_model_weights("mobilenet_v3_large").DEFAULT
    model = get_model("fasterrcnn_mobilenet_v3_large_320_fpn", weights_backbone=weights, num_classes=10)
    train_loader = DataLoader(MockDetDataset(num_it * batch_size), batch_size=batch_size, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(model.parameters())

    learner = trainer.DetectionTrainer(
        model,
        train_loader,
        train_loader,
        None,
        optimizer,
        output_file=file_path,
        gpu=0 if torch.cuda.is_available() else None,
        gradient_clip=0.1,
    )

    _test_trainer(learner, num_it, "roi_heads.box_predictor.cls_score.weight", "backbone", 5e-4)
    # AMP
    learner = trainer.DetectionTrainer(
        model,
        train_loader,
        train_loader,
        None,
        optimizer,
        output_file=file_path,
        gpu=0 if torch.cuda.is_available() else None,
        amp=torch.cuda.is_available(),
        gradient_clip=0.1,
    )
    _test_trainer(learner, num_it, "roi_heads.box_predictor.cls_score.weight", "backbone", 5e-4)


def test_training_progress_obeys_refresh_interval(monkeypatch):
    bars = []

    class CountingProgress(tqdm):
        def __init__(self, *args, **kwargs):
            self.redraws = 0
            kwargs.update(mininterval=60, file=StringIO())
            super().__init__(*args, **kwargs)
            bars.append(self)

        def display(self, *args, **kwargs):
            self.redraws += 1
            return super().display(*args, **kwargs)

    monkeypatch.setattr("holocron.trainer.core.tqdm", CountingProgress)
    inputs = torch.ones(50, 1)
    learner = _linear_trainer(inputs, torch.zeros_like(inputs), gradient_acc=1)
    learner.scheduler = torch.optim.lr_scheduler.StepLR(learner.optimizer, step_size=1, gamma=1)
    learner._fit_epoch()
    assert learner.step == 50
    assert bars[0].redraws <= 3
