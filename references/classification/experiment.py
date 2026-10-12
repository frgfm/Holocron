# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Run a version-1 local ImageFolder experiment from JSON."""

import json
import math
import random
import signal
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from holocron.experiments import Trial
from holocron.models import get_model_info
from holocron.trainer import resolve_device
from references._common import OPTIMIZERS
from references.classification._benchmark import run_benchmark
from references.classification.train import _train, get_parser

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


def _validate_settings(settings):
    for key in ("epochs", "batch_size", "grad_acc", *PREPROCESSING_KEYS[:3]):
        _number(settings[key], key, 1, integer=True)
    _number(settings["workers"], "workers", integer=True)
    for key in ("lr", "weight_decay", "mixup_alpha", "norm_wd"):
        if key != "norm_wd" or settings[key] is not None:
            _number(settings[key], key)
    if settings["lr"] == 0:
        raise ValueError("lr must be positive")
    for key in ("label_smoothing", "random_erase"):
        _number(settings[key], key, 0, 1)
    if not isinstance(settings["opt"], str) or settings["opt"] not in OPTIMIZERS:
        raise ValueError("unsupported optimizer")
    if not isinstance(settings["sched"], str) or settings["sched"] not in {"onecycle", "cosine"}:
        raise ValueError("unsupported scheduler")


def _resolve_config(raw, base, defaults):
    required = ("schema_version", "model", "dataset", "training_device", "deployment_target")
    _object(raw, (*required, "training", "preprocessing", "seed", "tracking"), required)
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    model = _resolve_model(raw["model"], base)
    dataset = _resolve_dataset(raw["dataset"], base)
    settings = {}
    for section, keys in (("training", TRAINING_KEYS), ("preprocessing", PREPROCESSING_KEYS)):
        overrides = _object(raw.get(section, {}), keys)
        settings[section] = {key: overrides.get(key, defaults[key]) for key in keys}
    training, preprocessing = settings["training"], settings["preprocessing"]
    _validate_settings({**training, **preprocessing})
    if not isinstance(raw["training_device"], str) or type(training["amp"]) is not bool:
        raise ValueError("training_device must be a device name; amp must be a boolean")
    device = str(resolve_device(raw["training_device"]))
    target = raw["deployment_target"]
    if target is not None and (not isinstance(target, str) or not target.strip()):
        raise ValueError("deployment_target must be null or a nonempty description")
    seed = raw.get("seed", defaults["seed"])
    _number(seed, "seed", 0, 2**32 - 1, integer=True)
    tracking = {"wb": defaults["wb"], "name": defaults["name"], **_object(raw.get("tracking", {}), ("wb", "name"))}
    if type(tracking["wb"]) is not bool or (tracking["name"] is not None and not isinstance(tracking["name"], str)):
        raise ValueError("invalid tracking settings")
    return {
        **raw,
        "training_device": device,
        "model": model,
        "dataset": dataset,
        **settings,
        "seed": seed,
        "tracking": tracking,
    }


def run_experiment(config_path, directory):
    """Validate JSON and run the existing benchmark training flow in one new trial."""
    config_path = Path(config_path).resolve()
    args = get_parser().parse_args(["."])
    config = _resolve_config(json.loads(config_path.read_text(encoding="utf-8")), config_path.parent, vars(args))
    vars(args).update(config["training"], **config["preprocessing"], **config["tracking"])
    args.arch, args.seed = config["model"]["name"], config["seed"]
    args.device = config["training_device"]
    args.data_path = config["dataset"]["train"]
    args.split_paths = config["dataset"]
    with Trial(Path(directory), config) as trial:
        args.output_file = str(trial.directory / "checkpoint.pth")
        random.seed(args.seed)
        np.random.seed(args.seed)  # noqa: NPY002
        run_benchmark(args, _train, trial=trial)


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
