# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Local trial records. Recipe defaults and training remain in the references."""

import hashlib
import json
import subprocess  # noqa: S404
import time
import traceback
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import torch
from torchvision.datasets import ImageFolder

from holocron.trainer import Trainer


def sha256(path: Path) -> str:
    """Hash raw file bytes.

    Returns:
        SHA-256 digest.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Replace a JSON record atomically within its destination directory."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def imagefolder_manifest(splits: dict[str, ImageFolder]) -> dict[str, Any]:
    """Record exact ImageFolder membership and reject leakage between splits.

    Returns:
        Class mapping and ordered, content-hashed split manifests.

    Raises:
        ValueError: If class mappings differ or splits share file content.
    """
    mapping = splits["train"].class_to_idx
    manifests = {}
    seen: dict[str, str] = {}
    for split, dataset in splits.items():
        if dataset.class_to_idx != mapping:
            raise ValueError(f"inconsistent class mappings in {split}")
        root = Path(dataset.root).resolve()
        samples = []
        for filename, target in dataset.samples:
            path = Path(filename)
            digest = sha256(path)
            if digest in seen and seen[digest] != split:
                raise ValueError(f"overlapping sample content in {seen[digest]} and {split}: {path}")
            seen[digest] = split
            samples.append({"path": path.relative_to(dataset.root).as_posix(), "class_index": target, "sha256": digest})
        identity = json.dumps(samples, sort_keys=True, separators=(",", ":")).encode("utf-8")
        manifests[split] = {"root": str(root), "samples": samples, "sha256": hashlib.sha256(identity).hexdigest()}
    return {"schema_version": 1, "class_to_idx": mapping, "splits": manifests}


def checkpoint_record(path: Path, history: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the selected checkpoint and its original epoch metrics.

    Returns:
        Saved state and benchmark-compatible checkpoint metadata.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    metrics = {
        key: value for key, value in history[state["epoch"] - 1].items() if key not in {"epoch", "elapsed_seconds"}
    }
    return state, {
        "checkpoint": path.name,
        "checkpoint_sha256": sha256(path),
        "selected_epoch": state["epoch"],
        "selected_metrics": metrics,
        "fully_resumable": False,
    }


class Trial:
    """Record a trial in a new directory; never overwrite an existing path."""

    def __init__(self, directory: Path, configuration: dict[str, Any]) -> None:
        self.configuration = configuration
        self.provenance: dict[str, Any] = {"schema_version": 1}
        self._record_environment()
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory.resolve()
        self.started = time.monotonic()
        self.result: dict[str, Any] = {
            "schema_version": 1,
            "state": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "finished_at": None,
            "epoch": 0,
            "actual_device": None,
            "exit_code": None,
            "error": None,
        }
        write_json(self.directory / "config.json", configuration)
        write_json(self.directory / "provenance.json", self.provenance)
        (self.directory / "progress.jsonl").touch()
        self._write_result()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        state, code, error = "completed", 0, None
        if exc is not None:
            state = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            code = 130 if isinstance(exc, KeyboardInterrupt) else 1
            if isinstance(exc, SystemExit):
                code = int(exc.code) if isinstance(exc.code, int) else int(exc.code is not None)
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc)),
            }
        self.result.update(state=state, exit_code=code, error=error, finished_at=datetime.now(UTC).isoformat())
        self._write_result()

    def prepare(self, trainer: Trainer) -> None:
        """Fingerprint the reference's actual data, transforms and initial weights.

        Raises:
            ValueError: If no full training batch exists or splits are inconsistent.
        """
        self.result["actual_device"] = str(trainer.device)
        splits = {"train": trainer.train_loader.dataset, "validation": trainer.val_loader.dataset}
        if self.configuration.get("test_split") is not None:
            splits["test"] = ImageFolder(self.configuration["test_split"])
        data = imagefolder_manifest(splits)
        write_json(self.directory / "data.json", data)
        if not len(trainer.train_loader):
            raise ValueError("training split must contain at least one full batch")
        initialization = {"checkpoint": self.configuration["initialization"]}
        if initialization["checkpoint"] is not None:
            path = Path(initialization["checkpoint"])
            initialization["sha256"] = sha256(path)
            state = torch.load(path, map_location="cpu", weights_only=True)
            trainer.model.load_state_dict(state.get("model", state))
        digest = hashlib.sha256()
        for name, tensor in sorted(trainer.model.state_dict().items()):
            digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode("utf-8"))
            digest.update(tensor.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        initialization["state_dict_sha256"] = digest.hexdigest()
        self.provenance.update(
            data={split: manifest["sha256"] for split, manifest in data["splits"].items()},
            initialization=initialization,
            preprocessing={split: repr(dataset.transform) for split, dataset in splits.items() if split != "test"},
            optimizer=trainer.optimizer.defaults,
        )
        write_json(self.directory / "provenance.json", self.provenance)
        self._write_result()

    def _record_environment(self) -> None:
        """Capture source and package versions before creating trial artifacts."""
        root = Path(__file__).resolve().parents[1]
        source = {"root": str(root), "revision": None, "dirty": None}
        options = {"cwd": root, "text": True, "stderr": subprocess.PIPE}
        try:
            source["revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], **options).strip()  # noqa: S607
            source["dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], **options))  # noqa: S607
        except (OSError, subprocess.CalledProcessError):
            pass  # Installed wheels may have no Git metadata; unknown is not clean.
        packages = {}
        for name in ("pylocron", "torch", "torchvision", "numpy", "Pillow", "wandb"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        self.provenance.update(source=source, packages=packages)

    def record_epoch(self, trainer: Trainer, event: dict[str, Any]) -> None:
        """Append the benchmark's epoch event and record the trainer's saved selection."""
        model = self.result["models"][0]
        model["final_epoch"] = self.result["epoch"] = event["epoch"]
        model["final_metrics"] = {key: value for key, value in event.items() if key not in {"epoch", "elapsed_seconds"}}
        if event["val_loss"] == trainer.min_loss:
            _, selected = checkpoint_record(Path(trainer.output_file), model["history"])
            model.update(selected)
        with (self.directory / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"schema_version": 1, "split": "validation", **event}, allow_nan=False) + "\n")
        self._write_result()

    def _write_result(self) -> None:
        self.result["elapsed_seconds"] = time.monotonic() - self.started
        write_json(self.directory / "result.json", self.result)
