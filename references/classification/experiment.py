# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Run a version-1 local ImageFolder experiment from JSON."""

import hashlib
import json
import math
import random
import re
import signal
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder

from holocron.experiments import Trial, imagefolder_manifest, sha256, write_json
from holocron.models import get_model, get_model_info
from references._common import create_loader, run_training
from references.classification.train import (
    collate_mixup,
    create_trainer,
    get_parser,
    imagefolder_transforms,
    scheduler_kwargs,
    worker_init_fn,
)

TRAINING_KEYS = (
    "epochs",
    "lr",
    "batch_size",
    "workers",
    "grad_acc",
    "opt",
    "sched",
    "weight_decay",
    "norm_wd",
    "label_smoothing",
    "mixup_alpha",
    "amp",
)
PREPROCESSING_KEYS = ("train_crop_size", "val_resize_size", "val_crop_size", "random_erase")


def _object(value, allowed, required=()):
    if not isinstance(value, dict) or value.keys() - set(allowed) or set(required) - value.keys():
        raise ValueError(f"expected object with fields {allowed}; required: {required}")
    return value


def _number(value, name, lower=0, upper=math.inf, *, integer=False):
    if (
        type(value) not in ((int,) if integer else (int, float))
        or not math.isfinite(value)
        or not lower <= value <= upper
    ):
        raise ValueError(f"invalid {name}: {value!r}")


def _resolve_model(value, base):
    model = dict(_object(value, ("name", "initialization"), ("name", "initialization")))
    if not isinstance(model["name"], str) or get_model_info(model["name"]).task != "classification":
        raise ValueError("model must be a catalog classification model")
    initialization = dict(_object(model["initialization"], ("kind", "path"), ("kind",)))
    if initialization["kind"] == "random":
        if set(initialization) != {"kind"}:
            raise ValueError("random initialization does not accept a path")
    elif initialization["kind"] == "checkpoint":
        path = initialization.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("checkpoint initialization requires a path")
        initialization["path"] = str((base / path).resolve())
        if not Path(initialization["path"]).is_file():
            raise ValueError("initialization checkpoint does not exist")
    else:
        raise ValueError("initialization kind must be random or checkpoint (weights only)")
    model["initialization"] = initialization
    return model


def _resolve_dataset(value, base):
    dataset = dict(_object(value, ("format", "train", "validation", "test"), ("format", "train", "validation")))
    if dataset["format"] != "imagefolder":
        raise ValueError("only imagefolder is supported")
    for split in dataset.keys() - {"format"}:
        if not isinstance(dataset[split], str) or not dataset[split]:
            raise ValueError(f"{split} must be a directory path")
        dataset[split] = str((base / dataset[split]).resolve())
    return dataset


def _validate_settings(training, preprocessing):
    for key in ("epochs", "batch_size", "grad_acc"):
        _number(training[key], key, 1, integer=True)
    _number(training["workers"], "workers", integer=True)
    _number(training["lr"], "lr", 0)
    if training["lr"] == 0:
        raise ValueError("lr must be positive")
    for key in ("weight_decay", "mixup_alpha"):
        _number(training[key], key)
    if training["norm_wd"] is not None:
        _number(training["norm_wd"], "norm_wd")
    _number(training["label_smoothing"], "label_smoothing", 0, 1)
    for key in ("train_crop_size", "val_resize_size", "val_crop_size"):
        _number(preprocessing[key], key, 1, integer=True)
    _number(preprocessing["random_erase"], "random_erase", 0, 1)
    if not isinstance(training["opt"], str) or training["opt"] not in {
        "sgd",
        "radam",
        "adamw",
        "adamp",
        "adabelief",
        "ademamix",
    }:
        raise ValueError("unsupported optimizer")
    if not isinstance(training["sched"], str) or training["sched"] not in {"onecycle", "cosine"}:
        raise ValueError("unsupported scheduler")


def _validate_device(device, amp):
    if not isinstance(device, str) or (device != "cpu" and not re.fullmatch(r"cuda:\d+", device)):
        raise ValueError("training_device must be cpu or cuda:N")
    if device != "cpu" and (not torch.cuda.is_available() or int(device[5:]) >= torch.cuda.device_count()):
        raise ValueError("requested CUDA device is unavailable")
    if type(amp) is not bool or (amp and device == "cpu"):
        raise ValueError("amp must be a boolean and requires CUDA")


def resolve_config(raw, base):
    """Validate input and resolve settings from the existing CLI parser.

    Returns:
        Resolved version-1 configuration with absolute data and checkpoint paths.

    Raises:
        ValueError: If a setting is invalid or unsupported.
    """
    required = ("schema_version", "model", "dataset", "training_device", "deployment_target")
    _object(raw, (*required, "training", "preprocessing", "seed", "tracking"), required)
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    model = _resolve_model(raw["model"], base)
    dataset = _resolve_dataset(raw["dataset"], base)
    defaults = vars(get_parser().parse_args(["."]))
    settings = {}
    for section, keys in (("training", TRAINING_KEYS), ("preprocessing", PREPROCESSING_KEYS)):
        overrides = _object(raw.get(section, {}), keys)
        settings[section] = {key: overrides.get(key, defaults[key]) for key in keys}
    training, preprocessing = settings["training"], settings["preprocessing"]
    _validate_settings(training, preprocessing)
    device = raw["training_device"]
    _validate_device(device, training["amp"])
    target = raw["deployment_target"]
    if target is not None and (not isinstance(target, str) or not target.strip()):
        raise ValueError("deployment_target must be null or a nonempty description")
    seed = raw.get("seed", defaults["seed"])
    _number(seed, "seed", 0, 2**32 - 1, integer=True)
    tracking = {"wb": defaults["wb"], "name": defaults["name"], **_object(raw.get("tracking", {}), ("wb", "name"))}
    if type(tracking["wb"]) is not bool or (tracking["name"] is not None and not isinstance(tracking["name"], str)):
        raise ValueError("invalid tracking settings")
    return {
        "schema_version": 1,
        "model": model,
        "dataset": dataset,
        **settings,
        "training_device": device,
        "deployment_target": target,
        "seed": seed,
        "tracking": tracking,
    }


def run_experiment(config_path, directory):
    """Run one trial, propagating exceptions and interruption exit codes.

    Raises:
        ValueError: If settings or data are invalid, or no full training batch exists.
        RuntimeError: If training ends without a selected checkpoint or all epochs.
    """
    config_path = Path(config_path).resolve()
    config = resolve_config(json.loads(config_path.read_text(encoding="utf-8")), config_path.parent)
    args = get_parser().parse_args(["."])
    vars(args).update(config["training"], **config["preprocessing"], **config["tracking"])
    args.arch, args.seed = config["model"]["name"], config["seed"]
    args.device = None if config["training_device"] == "cpu" else int(config["training_device"][5:])
    with Trial(Path(directory), config) as trial:
        trial.record_environment()
        random.seed(args.seed)
        np.random.seed(args.seed)  # noqa: NPY002
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        train_transform, val_transform = imagefolder_transforms(args)
        trial.record_provenance(
            preprocessing={
                "recipe": "imagenette-v1",
                "loader": "PIL RGB",
                "train": repr(train_transform),
                "validation": repr(val_transform),
            }
        )
        splits = {
            split: ImageFolder(root, train_transform if split == "train" else val_transform)
            for split, root in config["dataset"].items()
            if split != "format"
        }
        data = imagefolder_manifest(splits)
        write_json(trial.directory / "data.json", data)
        trial.record_provenance(data={split: manifest["sha256"] for split, manifest in data["splits"].items()})
        num_classes = len(splits["train"].classes)
        collate = default_collate
        if args.mixup_alpha > 0:
            collate = partial(collate_mixup, num_classes=num_classes, alpha=args.mixup_alpha)
        train_loader = create_loader(
            splits["train"], args, training=True, worker_init_fn=worker_init_fn, collate_fn=collate
        )
        val_loader = create_loader(splits["validation"], args, training=False, worker_init_fn=worker_init_fn)
        if not len(train_loader):
            raise ValueError("training split must contain at least one full batch")
        model = get_model(args.arch, pretrained=False, num_classes=num_classes)
        initialization = dict(config["model"]["initialization"])
        if initialization["kind"] == "checkpoint":
            path = Path(initialization["path"])
            initialization["sha256"] = sha256(path)
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state.get("model", state))
        digest = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode("utf-8"))
            digest.update(tensor.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        initialization["state_dict_sha256"] = digest.hexdigest()
        trial.record_provenance(initialization=initialization)
        args.output_file = str(trial.directory / "checkpoint.pth")
        trainer = create_trainer(model, train_loader, val_loader, args)
        trial.actual_device = str(next(trainer.model.parameters()).device)
        trial.record_provenance(
            actual_device=trial.actual_device,
            threads=torch.get_num_threads(),
            cudnn_benchmark=False,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            loader={
                "train_drop_last": True,
                "validation_drop_last": False,
                "train_batches": len(train_loader),
                "validation_batches": len(val_loader),
            },
            optimizer=trainer.optimizer.defaults,
            scheduler={
                "name": args.sched,
                "kwargs": scheduler_kwargs(args),
            },
        )
        trainer.on_epoch_end = lambda metrics: trial.record_epoch(
            trainer, {key: value for key, value in metrics.items() if key != "acc5" or num_classes >= 5}
        )
        run_training(
            trainer,
            args,
            project="holocron-image-classification",
            config=config,
            **scheduler_kwargs(args),
        )
        if trial.selected is None or trial.epoch != args.epochs:
            raise RuntimeError("training ended without the requested epochs and a selected checkpoint")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    parser.add_argument("trial_directory", type=Path)
    args = parser.parse_args()

    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        run_experiment(args.configuration, args.trial_directory)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
