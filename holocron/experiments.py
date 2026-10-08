# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Local trial records. Recipe defaults and training remain in the references."""

import hashlib
import json
import math
import platform
import subprocess  # noqa: S404
import time
import traceback
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from torchvision.datasets import ImageFolder

from holocron.trainer import Trainer


def sha256(path: Path) -> str:
    """Hash file bytes with SHA-256.

    Returns:
        Lowercase hexadecimal digest.
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


class Trial:
    """Own one new directory and record a trial's lifecycle and epoch callback.

    Args:
        directory: New output directory; an existing path is always rejected.
        configuration: Resolved JSON configuration.
    """

    def __init__(self, directory: Path, configuration: dict[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory.resolve()
        self.started_at = datetime.now(UTC).isoformat()
        self.started = time.monotonic()
        self.epoch = 0
        self.final_metrics: list[dict[str, Any]] = []
        self.selected: dict[str, Any] | None = None
        self.actual_device: str | None = None
        write_json(self.directory / "config.json", configuration)
        self.provenance: dict[str, Any] = {"schema_version": 1, "started_at": self.started_at}
        write_json(self.directory / "provenance.json", self.provenance)
        (self.directory / "progress.jsonl").touch()
        self._status("running")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        state, code = "completed", 0
        error = None
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
        self._status(state, exit_code=code, error=error)

    def record_provenance(self, **records: Any) -> None:
        """Atomically add preparation records and refresh running status."""
        self.provenance.update(records)
        write_json(self.directory / "provenance.json", self.provenance)
        self._status("running")

    def record_environment(self) -> None:
        """Record the loaded source checkout and relevant installed package versions."""
        root = Path(__file__).resolve().parents[1]
        source = {"root": str(root), "revision": None, "dirty": None}
        try:
            source["revision"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],  # noqa: S607
                cwd=root,
                text=True,
                stderr=subprocess.PIPE,
            ).strip()
            source["dirty"] = bool(
                subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.PIPE)  # noqa: S607
            )
        except (OSError, subprocess.CalledProcessError):
            # Installed wheels may have no Git checkout; unknown is not clean.
            pass
        packages = {}
        for name in ("pylocron", "torch", "torchvision", "numpy", "Pillow", "wandb"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        self.record_provenance(
            source=source, python=platform.python_version(), platform=platform.platform(), packages=packages
        )

    def record_epoch(self, trainer: Trainer, metrics: dict[str, float]) -> None:
        """Record validation metrics after the trainer has selected its checkpoint.

        Raises:
            ValueError: If any recorded metric is not finite.
        """
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("non-finite epoch metrics")
        self.epoch = trainer.epoch
        self.final_metrics = [
            {
                "name": name,
                "value": value,
                "split": "validation",
                "epoch": self.epoch,
                "direction": "minimize" if name == "val_loss" else "maximize",
            }
            for name, value in metrics.items()
        ]
        if self.selected is None or metrics["val_loss"] < self.selected["validation_loss"]:
            self.selected = {
                "path": "checkpoint.pth",
                "sha256": sha256(Path(trainer.output_file)),
                "epoch": self.epoch,
                "validation_loss": metrics["val_loss"],
                "metrics": self.final_metrics,
                "fully_resumable": False,
            }
        event = {
            "schema_version": 1,
            "event": "epoch_end",
            "timestamp": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.monotonic() - self.started,
            "epoch": self.epoch,
            "metrics": self.final_metrics,
            "selected_checkpoint": self.selected,
        }
        with (self.directory / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        self._status("running")

    def _status(self, state: str, **fields: Any) -> None:
        now = datetime.now(UTC).isoformat()
        write_json(
            self.directory / "result.json",
            {
                "schema_version": 1,
                "state": state,
                "started_at": self.started_at,
                "updated_at": now,
                "finished_at": None if state == "running" else now,
                "elapsed_seconds": time.monotonic() - self.started,
                "epoch": self.epoch,
                "actual_device": self.actual_device,
                "final_epoch_metrics": self.final_metrics,
                "selected_checkpoint": self.selected,
                "artifacts": {
                    "configuration": "config.json",
                    "provenance": "provenance.json",
                    "data": "data.json",
                    "progress": "progress.jsonl",
                },
                "exit_code": None,
                "error": None,
                **fields,
            },
        )
