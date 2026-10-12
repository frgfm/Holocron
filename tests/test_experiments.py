import json
import shutil
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from torchvision.datasets import ImageFolder

from holocron.experiments import Trial, imagefolder_manifest, sha256
from holocron.trainer import ClassificationTrainer
from references.classification import experiment, train


@pytest.fixture
def configuration(tmp_path, monkeypatch):
    for offset, split in enumerate(("train", "val", "test")):
        for index, label in enumerate(("a", "b")):
            directory = tmp_path / split / label
            directory.mkdir(parents=True)
            for sample in range(2):
                Image.new("RGB", (20, 20), (offset * 60 + index * 20 + sample, 30, 90)).save(
                    directory / f"{sample}.png"
                )
    config = {
        "schema_version": 1,
        "initialization": None,
        "test_split": "test",
        "deployment_target": "owner's CPU",
        "recipe": {
            "data_path": ".",
            "device": "cpu",
            "epochs": 2,
            "batch_size": 2,
            "workers": 0,
            "sched": "cosine",
            "mixup_alpha": 0,
            "train_crop_size": 16,
            "val_crop_size": 16,
            "val_resize_size": 20,
        },
    }
    monkeypatch.setattr(
        train,
        "get_model",
        lambda *_args, **_kwargs: nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2)),
    )
    return tmp_path / "experiment.json", config


def submit(path, config, trial="trial"):
    path.write_text(json.dumps(config), encoding="utf-8")
    experiment.run_experiment(path, path.parent / trial)
    return json.loads((path.parent / trial / "result.json").read_text())


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        (None, "schema_version", True),
        (None, "schema_version", 2),
        (None, "wall_time_seconds", 10),
        ("recipe", "device", "meta"),
        ("recipe", "device", 0),
        (None, "initialization", "missing.pth"),
        (None, "deployment_target", ""),
        ("recipe", "seed", -1),
        ("recipe", "seed", 1.5),
        ("recipe", "epochs", 0),
        ("recipe", "epochs", True),
        ("recipe", "lr", float("nan")),
        ("recipe", "opt", "unknown"),
        ("recipe", "sched", "unknown"),
        ("recipe", "workers", -1),
        ("recipe", "amp", "yes"),
        ("recipe", "resume", "old.pth"),
        ("recipe", "random_erase", 2),
        ("recipe", "mean", [0, 0, 0]),
        ("recipe", "arch", "unknown"),
        (None, "test_split", ""),
    ],
)
def test_configuration_rejection(configuration, section, key, value):
    path, config = configuration
    (config if section is None else config[section])[key] = value
    with pytest.raises(ValueError):
        submit(path, config)
    assert not (path.parent / "trial").exists()


@pytest.mark.parametrize("scheduler", ["cosine", "onecycle"])
def test_success_and_independent_reader(configuration, monkeypatch, scheduler):
    path, config = configuration
    config["recipe"]["sched"] = scheduler
    original_getitem = ImageFolder.__getitem__

    def no_test_reads(dataset, index):
        assert dataset.root != str(path.parent / "test")
        return original_getitem(dataset, index)

    def worsening_metrics(learner):
        running = json.loads(Path(learner.output_file).with_name("result.json").read_text())
        assert (running["state"], running["actual_device"]) == ("running", "cpu")
        return {"val_loss": float(learner.epoch), "acc1": 0.5, "acc5": 0.0}

    monkeypatch.setattr(ImageFolder, "__getitem__", no_test_reads)
    monkeypatch.setattr(ClassificationTrainer, "evaluate", worsening_metrics)
    result = submit(path, config)
    directory = path.parent / "trial"
    resolved, provenance, data = [
        json.loads((directory / f"{name}.json").read_text()) for name in ("config", "provenance", "data")
    ]
    events = [json.loads(line) for line in (directory / "progress.jsonl").read_text().splitlines()]
    model = result["models"][0]
    assert resolved["recipe"]["lr"] == train.get_parser().get_default("lr")
    assert resolved["recipe"]["data_path"] == str(path.parent)
    assert resolved["deployment_target"] == config["deployment_target"]
    assert data["class_to_idx"] == {"a": 0, "b": 1}
    assert all(len(split["samples"]) == 4 for split in data["splits"].values())
    assert provenance["data"]["train"] == data["splits"]["train"]["sha256"]
    assert provenance["source"]["revision"]
    assert type(provenance["source"]["dirty"]) is bool
    assert provenance["packages"]["torch"]
    assert "RandomResizedCrop" in provenance["preprocessing"]["train"]
    assert len(provenance["initialization"]["state_dict_sha256"]) == 64
    assert (result["state"], result["exit_code"], result["actual_device"]) == ("completed", 0, "cpu")
    assert result["finished_at"]
    assert result["elapsed_seconds"] > 0
    assert [(event["epoch"], event["split"]) for event in events] == [(1, "validation"), (2, "validation")]
    assert result["metrics_split"] == "validation"
    assert result["metric_directions"] == {"val_loss": "minimize", "acc1": "maximize"}
    assert (model["final_epoch"], model["selected_epoch"]) == (2, 1)
    for field, epoch in (("final_metrics", -1), ("selected_metrics", 0)):
        assert model[field] == {key: events[epoch][key] for key in result["metric_directions"]}
    assert model["checkpoint_sha256"] == sha256(directory / model["checkpoint"])
    assert torch.load(directory / model["checkpoint"], map_location="cpu", weights_only=True)["epoch"] == 1
    assert model["fully_resumable"] is False
    config["initialization"] = str(directory / model["checkpoint"])
    submit(path, config, "warm-start")
    assert (
        json.loads((path.parent / "warm-start/provenance.json").read_text())["initialization"]["sha256"]
        == model["checkpoint_sha256"]
    )


@pytest.mark.parametrize(
    ("error", "state", "code", "after_epoch"),
    [
        (RuntimeError("training broke"), "failed", 1, False),
        (RuntimeError("training broke"), "failed", 1, True),
        (KeyboardInterrupt(), "interrupted", 130, False),
        (SystemExit(143), "interrupted", 143, False),
        (SystemExit(0), "interrupted", 0, False),
        (SystemExit(None), "interrupted", 0, False),
        (SystemExit("exit message"), "interrupted", 1, False),
    ],
)
def test_failure_and_interruption(configuration, monkeypatch, error, state, code, after_epoch):
    path, config = configuration
    fit = ClassificationTrainer.fit_n_epochs

    def fail(learner, _epochs, *args, **kwargs):
        learners.append(learner)
        if after_epoch:
            fit(learner, 1, *args, **kwargs)
        raise error

    learners = []
    monkeypatch.setattr(ClassificationTrainer, "fit_n_epochs", fail)
    with pytest.raises(type(error)):
        submit(path, config)
    result = json.loads((path.parent / "trial/result.json").read_text())
    assert (result["state"], result["exit_code"]) == (state, code)
    assert result["error"]["type"] == type(error).__name__
    assert result["error"]["message"] == str(error)
    assert "fail" in result["error"]["traceback"]
    assert result["finished_at"]
    assert learners[0].on_epoch_end is None
    if after_epoch:
        model = result["models"][0]
        assert model["selected_epoch"] == result["epoch"] == 1
        assert model["checkpoint_sha256"] == sha256(path.parent / "trial" / model["checkpoint"])
    else:
        assert "checkpoint" not in result["models"][0]
        assert not (path.parent / "trial/progress.jsonl").read_text()


@pytest.mark.parametrize("symlink", [False, True])
def test_overwrite_protection(configuration, symlink):
    path, config = configuration
    directory = path.parent / "trial"
    if symlink:
        directory.symlink_to(path.parent / "missing", target_is_directory=True)
    else:
        directory.mkdir()
        (directory / "result.json").write_text("do not replace", encoding="utf-8")
    with pytest.raises(FileExistsError):
        submit(path, config)
    assert directory.is_symlink() if symlink else (directory / "result.json").read_text() == "do not replace"


@pytest.mark.parametrize("problem", ["overlap", "mapping", "content"])
def test_data_identity(configuration, problem):
    path, config = configuration
    splits = {name: ImageFolder(path.parent / split) for name, split in (("train", "train"), ("validation", "val"))}
    before = imagefolder_manifest(splits)
    if problem == "content":
        Image.new("RGB", (20, 20), (255, 0, 0)).save(path.parent / "train/a/0.png")
        assert imagefolder_manifest(splits)["splits"]["train"]["sha256"] != before["splits"]["train"]["sha256"]
    else:
        if problem == "overlap":
            shutil.copyfile(path.parent / "train/a/0.png", path.parent / "val/b/renamed.png")
        else:
            (path.parent / "val/b").rename(path.parent / "val/c")
        with pytest.raises(ValueError, match=r"overlapping|inconsistent"):
            submit(path, config)
        assert json.loads((path.parent / "trial/result.json").read_text())["state"] == "failed"


@pytest.mark.parametrize("dirty", [False, True])
def test_source_identity_precedes_trial_artifacts(tmp_path, monkeypatch, dirty):
    directory = tmp_path / "unignored-trial"

    def git_read(command, **_kwargs):
        assert not directory.exists()
        return "a" * 40 if command[1] == "rev-parse" else "?? owner.py" if dirty else ""

    monkeypatch.setattr("holocron.experiments.subprocess.check_output", git_read)
    Trial(directory, {"schema_version": 1})
    source = json.loads((directory / "provenance.json").read_text())["source"]
    assert source["dirty"] is dirty
    assert source["revision"] == "a" * 40
