import copy
import gc
import json
import random
import shutil
import sys
import weakref
from argparse import Namespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision.datasets import ImageFolder

from holocron.experiments import Trial, imagefolder_manifest, sha256
from holocron.trainer import ClassificationTrainer
from references._common import run_training  # noqa: PLC2701
from references.classification import experiment


@pytest.fixture
def configuration(tmp_path, monkeypatch):
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()  # noqa: NPY002
    benchmark = torch.backends.cudnn.benchmark
    for offset, split in enumerate(("train", "validation", "test")):
        for index, label in enumerate(("a", "b")):
            directory = tmp_path / split / label
            directory.mkdir(parents=True)
            for sample in range(2):
                Image.new("RGB", (20, 20), (offset * 60 + index * 20 + sample, 30, 90)).save(
                    directory / f"{sample}.png"
                )
    config = {
        "schema_version": 1,
        "model": {"name": "convnext_atto", "initialization": {"kind": "random"}},
        "dataset": {"format": "imagefolder", "train": "train", "validation": "validation", "test": "test"},
        "training_device": "cpu",
        "deployment_target": "owner's CPU",
        "training": {"epochs": 2, "batch_size": 2, "workers": 0, "sched": "cosine", "mixup_alpha": 0},
        "preprocessing": {"train_crop_size": 16, "val_crop_size": 16, "val_resize_size": 20},
    }
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        experiment,
        "get_model",
        lambda *_args, **_kwargs: nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2)),
    )
    # The runner seeds process globals; keep later tests independent.
    with torch.random.fork_rng():
        yield path, config
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)  # noqa: NPY002
    torch.backends.cudnn.benchmark = benchmark


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        (None, "schema_version", True),
        (None, "schema_version", 2),
        (None, "wall_time_seconds", 10),
        (None, "training_device", "mps"),
        (None, "seed", -1),
        (None, "seed", 1.5),
        ("training", "epochs", 0),
        ("training", "epochs", True),
        ("training", "lr", float("nan")),
        ("training", "opt", "unknown"),
        ("training", "sched", "unknown"),
        ("training", "workers", -1),
        ("training", "amp", True),
        ("training", "resume", "old.pth"),
        ("preprocessing", "random_erase", 2),
        ("preprocessing", "mean", [0, 0, 0]),
        ("model", "name", "unknown"),
        ("model", "initialization", {"kind": "pretrained"}),
        ("dataset", "format", "cifar10"),
        ("dataset", "test", None),
    ],
)
def test_configuration_rejection(configuration, section, key, value):
    path, original = configuration
    config = copy.deepcopy(original)
    (config if section is None else config[section])[key] = value
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        experiment.run_experiment(path, path.parent / "rejected")
    assert not (path.parent / "rejected").exists()


@pytest.mark.parametrize("scheduler", ["cosine", "onecycle"])
def test_success_and_independent_reader(configuration, monkeypatch, scheduler):
    path, config = configuration
    config["training"]["sched"] = scheduler
    path.write_text(json.dumps(config), encoding="utf-8")
    original_getitem = ImageFolder.__getitem__

    def no_test_reads(dataset, index):
        assert dataset.root != str(path.parent / "test")
        return original_getitem(dataset, index)

    monkeypatch.setattr(ImageFolder, "__getitem__", no_test_reads)
    directory = path.parent / "trial"
    experiment.run_experiment(path, directory)
    # The reader uses only JSON and checkpoint bytes from the trial directory.
    config, provenance, data, result = [
        json.loads((directory / name).read_text())
        for name in ("config.json", "provenance.json", "data.json", "result.json")
    ]
    events = [json.loads(line) for line in (directory / "progress.jsonl").read_text().splitlines()]
    assert config["training"]["epochs"] == 2
    assert config["training"]["lr"] == experiment.get_parser().parse_args(["."]).lr
    assert config["dataset"]["train"] == str(path.parent / "train")
    assert config["deployment_target"] == "owner's CPU"
    assert data["class_to_idx"] == {"a": 0, "b": 1}
    assert len(data["splits"]["train"]["samples"]) == 4
    assert provenance["data"]["train"] == data["splits"]["train"]["sha256"]
    assert provenance["source"]["revision"]
    assert type(provenance["source"]["dirty"]) is bool
    assert provenance["packages"]["torch"]
    assert "RandomResizedCrop" in provenance["preprocessing"]["train"]
    assert len(provenance["initialization"]["state_dict_sha256"]) == 64
    assert result["state"] == "completed"
    assert result["exit_code"] == 0
    assert result["actual_device"] == "cpu"
    assert result["finished_at"]
    assert result["elapsed_seconds"] > 0
    assert [event["epoch"] for event in events] == [1, 2]
    assert all(metric["split"] == "validation" for event in events for metric in event["metrics"])
    assert result["final_epoch_metrics"] == events[-1]["metrics"]
    selected = result["selected_checkpoint"]
    assert selected["sha256"] == sha256(directory / selected["path"])
    checkpoint = torch.load(directory / selected["path"], map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == selected["epoch"]
    assert selected["metrics"] == events[selected["epoch"] - 1]["metrics"]
    assert selected["fully_resumable"] is False
    assert {metric["name"] for metric in selected["metrics"]} == {"val_loss", "acc1"}
    # A checkpoint from this contract can initialize another trial, without resume claims.
    raw = json.loads(path.read_text())
    raw["model"]["initialization"] = {"kind": "checkpoint", "path": str(directory / "checkpoint.pth")}
    path.write_text(json.dumps(raw), encoding="utf-8")
    experiment.run_experiment(path, path.parent / "warm-start")
    warm_start = json.loads((path.parent / "warm-start" / "provenance.json").read_text())
    assert warm_start["initialization"]["sha256"] == selected["sha256"]


@pytest.mark.parametrize(
    ("error", "state", "exit_code"),
    [
        (RuntimeError("training broke"), "failed", 1),
        (KeyboardInterrupt(), "interrupted", 130),
        (SystemExit(143), "interrupted", 143),
        (SystemExit(0), "interrupted", 0),
        (SystemExit(None), "interrupted", 0),
        (SystemExit("exit message"), "interrupted", 1),
    ],
)
def test_failure_and_interruption(configuration, monkeypatch, error, state, exit_code):
    path, _ = configuration

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(ClassificationTrainer, "fit_n_epochs", fail)
    directory = path.parent / "failed-trial"
    with pytest.raises(type(error)):
        experiment.run_experiment(path, directory)
    result = json.loads((directory / "result.json").read_text())
    assert result["state"] == state
    assert result["exit_code"] == exit_code
    assert result["error"]["type"] == type(error).__name__
    assert result["error"]["message"] == str(error)
    assert "fail" in result["error"]["traceback"]
    assert result["selected_checkpoint"] is None
    assert result["finished_at"]
    assert not (directory / "progress.jsonl").read_text()


def test_overwrite_protection(configuration):
    path, _ = configuration
    directory = path.parent / "existing"
    directory.mkdir()
    (directory / "result.json").write_text("do not replace", encoding="utf-8")
    with pytest.raises(FileExistsError):
        experiment.run_experiment(path, directory)
    assert (directory / "result.json").read_text() == "do not replace"
    assert list(directory.iterdir()) == [directory / "result.json"]


def test_existing_dangling_symlink_is_not_a_new_trial(configuration):
    path, _ = configuration
    target, alias = path.parent / "missing", path.parent / "existing-link"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        experiment.run_experiment(path, alias)
    assert alias.is_symlink()
    assert not target.exists()


def test_onecycle_rejects_single_warmup_update(configuration):
    path, config = configuration
    config["training"].update(sched="onecycle", epochs=5)
    path.write_text(json.dumps(config), encoding="utf-8")
    directory = path.parent / "invalid-schedule"
    with pytest.raises(ValueError, match="exactly one warmup update"):
        experiment.run_experiment(path, directory)
    result = json.loads((directory / "result.json").read_text())
    assert result["state"] == "failed"
    assert result["epoch"] == 0
    assert not (directory / "checkpoint.pth").exists()


def test_mixup_with_spawn_worker(configuration):
    path, config = configuration
    config["training"].update(workers=1, mixup_alpha=0.2, epochs=1)
    path.write_text(json.dumps(config), encoding="utf-8")
    experiment.run_experiment(path, path.parent / "worker-trial")
    assert json.loads((path.parent / "worker-trial/result.json").read_text())["state"] == "completed"


def test_trial_releases_trainer_without_garbage_collection(configuration, monkeypatch):
    path, _ = configuration
    references = []
    create_trainer = experiment.create_trainer

    def observe_trainer(*args, **kwargs):
        learner = create_trainer(*args, **kwargs)
        references.append(weakref.ref(learner))
        return learner

    monkeypatch.setattr(experiment, "create_trainer", observe_trainer)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        experiment.run_experiment(path, path.parent / "release-trial")
        assert references[0]() is None
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize("problem", ["overlap", "mapping"])
def test_split_rejection(configuration, problem):
    path, _ = configuration
    if problem == "overlap":
        shutil.copyfile(path.parent / "train/a/0.png", path.parent / "validation/b/renamed.png")
    else:
        (path.parent / "validation/b").rename(path.parent / "validation/c")
    with pytest.raises(ValueError, match=r"overlapping|inconsistent"):
        experiment.run_experiment(path, path.parent / "invalid-data")
    assert json.loads((path.parent / "invalid-data/result.json").read_text())["state"] == "failed"


def test_content_identity_and_selected_vs_final(configuration):
    path, _ = configuration
    splits = {split: ImageFolder(path.parent / split) for split in ("train", "validation")}
    before = imagefolder_manifest(splits)
    Image.new("RGB", (20, 20), (255, 0, 0)).save(path.parent / "train/a/0.png")
    assert imagefolder_manifest(splits)["splits"]["train"]["sha256"] != before["splits"]["train"]["sha256"]
    directory = path.parent / "metrics"
    with Trial(directory, {}) as trial:
        checkpoint = directory / "checkpoint.pth"
        checkpoint.write_bytes(b"best epoch weights")
        learner = Namespace(epoch=1, output_file=str(checkpoint))
        trial.record_epoch(learner, {"val_loss": 0.1, "acc1": 0.8})
        running = json.loads((directory / "result.json").read_text())
        assert running["state"] == "running"
        assert running["finished_at"] is None
        learner.epoch = 2
        trial.record_epoch(learner, {"val_loss": 0.3, "acc1": 0.6})
    result = json.loads((directory / "result.json").read_text())
    assert result["selected_checkpoint"]["epoch"] == 1
    assert result["final_epoch_metrics"][0]["epoch"] == 2
    assert result["selected_checkpoint"]["metrics"][0]["direction"] == "minimize"


def test_wandb_preserves_callback(monkeypatch):
    received, logged = [], []
    callback = received.append
    learner = Namespace(on_epoch_end=callback)
    learner.fit_n_epochs = lambda *_args, **_kwargs: learner.on_epoch_end({"val_loss": 0.2})
    run = Namespace(finish=lambda **_kwargs: None)
    monkeypatch.setitem(sys.modules, "wandb", Namespace(init=lambda **_kwargs: run, log=logged.append))
    args = experiment.get_parser().parse_args([".", "--wb"])
    run_training(learner, args, project="test", config={})
    assert received == logged == [{"val_loss": 0.2}]
    assert learner.on_epoch_end is callback
