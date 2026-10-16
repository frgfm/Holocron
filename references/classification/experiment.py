# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Submit a local ImageFolder experiment to the classification benchmark."""

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
from references.classification._benchmark import run_benchmark
from references.classification.train import _train, get_parser

RECIPE_KEYS = [
    "data_path",
    "arch",
    "device",
    "seed",
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
    "train_crop_size",
    "val_resize_size",
    "val_crop_size",
    "random_erase",
    "wb",
    "name",
]


def _object(value, allowed, required=()):
    if not isinstance(value, dict) or value.keys() - set(allowed) or set(required) - value.keys():
        raise ValueError(f"expected fields {allowed}; required: {required}")
    return value


def _recipe(raw, parser):
    overrides = _object(raw, RECIPE_KEYS, ("data_path",))
    actions = {action.dest: action for action in parser._actions}
    for key, value in overrides.items():
        action = actions[key]
        expected = action.type or type(action.default)
        if value is None and action.default is None:
            continue
        if type(value) not in ((int, float) if expected is float else (expected,)):
            raise ValueError(f"invalid type for {key}")
        if action.choices is not None and value not in action.choices:
            raise ValueError(f"unsupported {key}: {value}")
    recipe = {key: overrides.get(key, parser.get_default(key)) for key in RECIPE_KEYS}
    for key, value in recipe.items():
        if type(value) in {int, float}:
            lower = (
                1
                if key in {"epochs", "batch_size", "grad_acc", "train_crop_size", "val_crop_size", "val_resize_size"}
                else 0
            )
            upper = {"seed": 2**32 - 1, "label_smoothing": 1, "random_erase": 1}.get(key, math.inf)
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"invalid {key}: {value}")
    if recipe["lr"] == 0 or recipe["sched"] not in {"onecycle", "cosine"}:
        raise ValueError("lr must be positive; sched must be onecycle or cosine")
    if get_model_info(recipe["arch"]).task != "classification":
        raise ValueError("arch must be a catalog classification model")
    return recipe


def run_experiment(config_path, directory):
    """Validate JSON before allocating a trial, then use the existing benchmark.

    Raises:
        ValueError: If configuration fields or values are unsupported.
    """
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fields = ("schema_version", "initialization", "deployment_target", "recipe")
    _object(config, (*fields, "test_split"), fields)
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    parser = get_parser()
    config["recipe"] = _recipe(config["recipe"], parser)
    for section, key in ((config["recipe"], "data_path"), (config, "initialization"), (config, "test_split")):
        value = section.get(key)
        if value is None and key != "data_path":
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key} must be a nonempty path")
        section[key] = str((config_path.parent / value).resolve())
    if config["initialization"] is not None and not Path(config["initialization"]).is_file():
        raise ValueError("initialization checkpoint does not exist")
    config["recipe"]["device"] = str(resolve_device(config["recipe"]["device"]))
    target = config["deployment_target"]
    if target is not None and (not isinstance(target, str) or not target.strip()):
        raise ValueError("deployment_target must be null or a nonempty description")
    args = parser.parse_args([config["recipe"]["data_path"]])
    vars(args).update(config["recipe"])
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
