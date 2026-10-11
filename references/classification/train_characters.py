# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Train a character classifier from live synthetic images."""

import argparse
import hashlib
import json
import math
import os
import platform
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from matplotlib.ft2font import FT2Font
from PIL import Image, ImageFont
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler, SequentialSampler
from torchvision.models import alexnet
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2.functional import InterpolationMode, to_pil_image

from holocron.models import classification
from holocron.trainer import ClassificationTrainer, resolve_device
from holocron.utils import find_fonts, render_text

DEFAULT_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
FONT_EXTENSIONS = {".otf", ".ttc", ".ttf"}
MODEL_NAMES = (
    "alexnet",
    *sorted(
        name
        for name, value in vars(classification).items()
        if name.islower() and not name.startswith("_") and callable(value)
    ),
)


@dataclass(frozen=True)
class FontRecord:
    """Resolved local font and its optional manifest metadata."""

    path: Path
    family: str
    style: str | None = None
    sha256: str | None = None
    source: str | None = None


def _validate_alphabet(alphabet: Sequence[str]) -> tuple[str, ...]:
    classes = tuple(alphabet)
    if not classes:
        raise ValueError("alphabet must not be empty")
    if any(not isinstance(character, str) or len(character) != 1 for character in classes):
        raise ValueError("alphabet entries must each be one Unicode code point")
    if len(set(classes)) != len(classes):
        raise ValueError("alphabet must not contain duplicate characters")
    if any(not character.isprintable() or character.isspace() for character in classes):
        raise ValueError("alphabet must contain visible characters only")
    return classes


def _as_font_record(record: FontRecord | str | Path) -> FontRecord:
    if isinstance(record, FontRecord):
        return record
    path = Path(record).expanduser().resolve()
    return FontRecord(path, str(path))


def _inspect_fonts(
    classes: Sequence[str], fonts: Sequence[FontRecord | str | Path], render_size: int
) -> tuple[tuple[FontRecord, ...], list[list[FontRecord]]]:
    records = tuple(_as_font_record(record) for record in fonts)
    if not records:
        raise ValueError("fonts must not be empty")
    if len({record.path for record in records}) != len(records):
        raise ValueError("fonts must not contain duplicate paths")

    compatible: list[list[FontRecord]] = [[] for _ in classes]
    validated_records = []
    for record in records:
        if not record.path.is_file():
            raise FileNotFoundError(f"font file does not exist: {record.path}")
        try:
            font_info = FT2Font(str(record.path))
            codepoints = font_info.get_charmap()
            font = ImageFont.truetype(str(record.path), render_size)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"unable to load font: {record.path}") from exc

        resolved_record = replace(record, family=font_info.family_name, style=font_info.style_name)
        validated_records.append(resolved_record)
        for class_index, character in enumerate(classes):
            if ord(character) not in codepoints:
                continue
            try:
                render_text(character, font)
            except ValueError:
                continue
            compatible[class_index].append(resolved_record)
    return tuple(validated_records), compatible


def _group_fonts(
    classes: Sequence[str], records: Sequence[FontRecord], compatible: Sequence[Sequence[FontRecord]]
) -> tuple[list[tuple[tuple[FontRecord, ...], ...]], list[tuple[FontRecord, ...]], tuple[FontRecord, ...]]:
    font_groups = []
    font_choices = []
    for character, choices in zip(classes, compatible, strict=True):
        if not choices:
            raise ValueError(f"no visible font supports character {character!r}")
        families: dict[str, list[FontRecord]] = {}
        for record in choices:
            families.setdefault(record.family, []).append(record)
        groups = tuple(tuple(styles) for styles in families.values())
        font_groups.append(groups)
        font_choices.append(tuple(record for group in groups for record in group))

    selected_paths = {record.path for choices in compatible for record in choices}
    selected_records = tuple(record for record in records if record.path in selected_paths)
    return font_groups, font_choices, selected_records


class SyntheticCharacterDataset(Dataset[tuple[Tensor, int]]):
    """Map-style dataset that renders one live character image per sample."""

    def __init__(
        self,
        alphabet: Sequence[str],
        fonts: Sequence[FontRecord | str | Path],
        render_size: int,
        samples_per_epoch: int,
        transform: Callable[[Image.Image], Tensor],
        *,
        deterministic: bool = False,
    ) -> None:
        self.classes = _validate_alphabet(alphabet)
        if isinstance(render_size, bool) or not isinstance(render_size, int) or render_size <= 0:
            raise ValueError("render_size must be a positive integer")
        if isinstance(samples_per_epoch, bool) or not isinstance(samples_per_epoch, int) or samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be a positive integer")
        if not callable(transform):
            raise TypeError("transform must be callable")

        records, compatible = _inspect_fonts(self.classes, fonts, render_size)
        self._font_groups, self._font_choices, self.fonts = _group_fonts(self.classes, records, compatible)
        self.class_to_idx = {character: index for index, character in enumerate(self.classes)}
        self.render_size = render_size
        self.samples_per_epoch = samples_per_epoch
        self.transform = transform
        self.deterministic = deterministic
        self._font_cache: dict[Path, ImageFont.FreeTypeFont] = {}
        self._font_cache_pid: int | None = os.getpid()

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getstate__(self) -> dict[str, Any]:
        state = vars(self).copy()
        state["_font_cache"] = {}
        state["_font_cache_pid"] = None
        return state

    def _select_font(self, index: int, class_index: int) -> FontRecord:
        if self.deterministic:
            choices = self._font_choices[class_index]
            return choices[(index // len(self.classes)) % len(choices)]

        families = self._font_groups[class_index]
        family = families[torch.randint(len(families), ()).item()]
        return family[torch.randint(len(family), ()).item()]

    def _get_font(self, path: Path) -> ImageFont.FreeTypeFont:
        process_id = os.getpid()
        if process_id != self._font_cache_pid:
            self._font_cache.clear()
            self._font_cache_pid = process_id
        font = self._font_cache.get(path)
        if font is None:
            font = ImageFont.truetype(str(path), self.render_size)
            self._font_cache[path] = font
        return font

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        class_index = index % len(self.classes)
        record = self._select_font(index, class_index)
        image = render_text(self.classes[class_index], self._get_font(record.path))
        return self.transform(image), class_index


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _manifest_font_records(
    font_dir: Path, manifest_path: Path, sources: dict[str, Any], font_entries: Sequence[Any]
) -> tuple[FontRecord, ...]:
    records = []
    for entry in font_entries:
        try:
            filename = entry["filename"]
            family = entry["family"]
            expected_hash = entry["sha256"].lower()
            source = entry["source"]
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid font entry in manifest: {manifest_path}") from exc
        if not all(isinstance(value, str) for value in (filename, family, expected_hash, source)):
            raise ValueError(f"invalid font entry in manifest: {manifest_path}")
        if source not in sources:
            raise ValueError(f"unknown font source in manifest: {source}")
        try:
            valid_checksum = len(bytes.fromhex(expected_hash)) == 32
        except ValueError:
            valid_checksum = False
        if not valid_checksum:
            raise ValueError(f"invalid font checksum: {filename}")
        if Path(filename).name != filename:
            raise ValueError(f"invalid font filename: {filename}")
        path = (font_dir / filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"font file does not exist: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"font checksum mismatch: {path}")
        records.append(FontRecord(path, family, entry.get("style"), actual_hash, source))
    return tuple(records)


def resolve_font_records(
    alphabet: Sequence[str], font_dir: Path | None, manifest_path: Path | None
) -> tuple[tuple[FontRecord, ...], dict[str, Any] | None]:
    """Resolve local font files and optional manifest provenance.

    Returns:
        font records and optional manifest metadata

    Raises:
        NotADirectoryError: if the requested font directory does not exist
        TypeError: if a manifest source has an invalid type
        ValueError: if the inputs, manifest, checksums, or discovered corpus are invalid
    """
    if manifest_path is not None:
        if font_dir is None:
            raise ValueError("--manifest requires --font-dir")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") != 1:
                raise ValueError("unsupported font manifest version")
            sources = manifest["sources"]
            font_entries = manifest["fonts"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid font manifest: {manifest_path}") from exc
        if not isinstance(sources, dict) or not isinstance(font_entries, list):
            raise TypeError(f"invalid font manifest structure: {manifest_path}")
        if any(not isinstance(source, dict) for source in sources.values()):
            raise TypeError(f"invalid font source in manifest: {manifest_path}")

        records = _manifest_font_records(font_dir, manifest_path, sources, font_entries)
        manifest_info = {
            "path": str(manifest_path.resolve()),
            "version": manifest["version"],
            "source_revisions": {name: source.get("revision") for name, source in sources.items()},
        }
        return records, manifest_info

    if font_dir is not None:
        if not font_dir.is_dir():
            raise NotADirectoryError(f"font directory does not exist: {font_dir}")
        paths = sorted(
            path.resolve() for path in font_dir.rglob("*") if path.is_file() and path.suffix.lower() in FONT_EXTENSIONS
        )
    else:
        paths = [Path(path).resolve() for path in find_fonts("".join(alphabet))]

    if not paths:
        raise ValueError("no local font files found")
    return tuple(FontRecord(path, str(path)) for path in paths), None


def square_pad(image: Image.Image, margin: int) -> Image.Image:
    """Add a margin and center a tight glyph in a square without resizing it.

    Returns:
        square grayscale image
    """
    image = F.pad(image, [margin, margin, margin, margin], fill=0)
    width, height = image.size
    delta = abs(width - height)
    padding = [0, delta // 2, 0, delta - delta // 2] if width > height else [delta // 2, 0, delta - delta // 2, 0]
    return F.pad(image, padding, fill=0)


def build_transforms(args: argparse.Namespace) -> tuple[T.Compose, T.Compose]:
    finalize = [
        T.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.LANCZOS, antialias=True),
        T.Grayscale(num_output_channels=3),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    pad = partial(square_pad, margin=args.margin)
    train_transform = T.Compose([
        pad,
        T.RandomAffine(
            args.rotation,
            translate=(args.translate, args.translate),
            scale=(1 - args.scale_jitter, 1 + args.scale_jitter),
            shear=args.shear,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        ),
        T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=args.blur_prob),
        T.RandomInvert(p=args.invert_prob),
        *finalize,
    ])
    return train_transform, T.Compose([pad, *finalize])


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def build_loader(
    dataset: SyntheticCharacterDataset,
    batch_size: int,
    workers: int,
    sampler: Sampler[int],
    seed: int,
    *,
    pin_memory: bool,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "drop_last": False,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "generator": _generator(seed),
    }
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **kwargs)


def save_sample_grid(loader: DataLoader, classes: Sequence[str], output_path: Path) -> None:
    images, targets = next(iter(loader))
    count = min(32, len(targets))
    columns = min(8, count)
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= count:
            continue
        image = (images[index] * 0.5 + 0.5).clamp(0, 1)
        axis.imshow(to_pil_image(image))
        axis.set_title(classes[targets[index].item()])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved {count} samples to {output_path}")


def benchmark_loader(loader: DataLoader, warmup_batches: int, measured_batches: int) -> None:
    required_batches = warmup_batches + measured_batches
    if required_batches > len(loader):
        raise ValueError(
            f"benchmark needs {required_batches} batches, but the loader only has {len(loader)}; "
            "increase --samples-per-epoch"
        )

    iterator = iter(loader)
    for _ in range(warmup_batches):
        next(iterator)
    start = time.perf_counter()
    sample_count = 0
    image_size = None
    for _ in range(measured_batches):
        images, targets = next(iterator)
        sample_count += len(targets)
        image_size = images.shape[-1]
    elapsed = time.perf_counter() - start
    dataset = loader.dataset
    print(f"Hardware: {platform.processor() or platform.machine()} ({platform.platform()})")
    print(
        f"Loader: workers={loader.num_workers}, batch_size={loader.batch_size}, "
        f"render_size={dataset.render_size}, image_size={image_size}, font_cache=worker-local"
    )
    print(
        f"Measured {sample_count} samples after {warmup_batches} warm-up batches and "
        f"{measured_batches} measured batches: {sample_count / elapsed:.1f} samples/s"
    )


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "alexnet":
        model = alexnet(weights=None, num_classes=num_classes)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.classifier = nn.Linear(256, num_classes)
        return model
    return getattr(classification, name)(False, num_classes=num_classes)


def _json_value(value: Any) -> Any:
    return str(value) if isinstance(value, Path) else value


def write_provenance(
    output_path: Path,
    args: argparse.Namespace,
    dataset: SyntheticCharacterDataset,
    records: Sequence[FontRecord],
    manifest_info: dict[str, Any] | None,
) -> None:
    font_root = args.font_dir.resolve() if args.font_dir is not None else None
    fonts = []
    for record in records:
        try:
            path = record.path.relative_to(font_root) if font_root is not None else record.path
        except ValueError:
            path = record.path
        fonts.append({
            "path": str(path),
            "family": record.family,
            "style": record.style,
            "sha256": record.sha256 or _sha256(record.path),
            "source": record.source,
        })
    payload = {
        "version": 1,
        "alphabet": list(dataset.classes),
        "class_to_index": dataset.class_to_idx,
        "seed": args.seed,
        "image_size": args.image_size,
        "render_size": args.render_size,
        "architecture": args.arch,
        "configuration": {key: _json_value(value) for key, value in vars(args).items()},
        "fonts": fonts,
        "manifest": manifest_info,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_resume_metadata(resume_path: Path, architecture: str, classes: Sequence[str]) -> None:
    metadata_path = resume_path.with_suffix(".json")
    if not metadata_path.is_file():
        warnings.warn(
            f"resume metadata not found at {metadata_path}; architecture and alphabet cannot be verified",
            stacklevel=2,
        )
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("architecture") != architecture:
        raise ValueError("resume checkpoint architecture does not match --arch")
    if metadata.get("alphabet") != list(classes):
        raise ValueError("resume checkpoint alphabet does not match --alphabet")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative number")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive number")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("expected a probability between 0 and 1")
    return parsed


def _scale_jitter(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed < 1:
        raise argparse.ArgumentTypeError("expected a scale jitter between 0 and 1")
    return parsed


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    data = parser.add_argument_group("Data and model")
    data.add_argument("--font-dir", type=Path, help="local font directory; system fonts are used when omitted")
    data.add_argument("--manifest", type=Path, help="optional version-1 font manifest")
    data.add_argument("--alphabet", default=DEFAULT_ALPHABET, help="ordered character classes")
    data.add_argument("--arch", choices=MODEL_NAMES, default="convnext_atto", help="classifier architecture")
    data.add_argument("--image-size", type=_positive_int, default=64, help="square model input size")
    data.add_argument("--render-size", type=_positive_int, default=128, help="font rasterization size")
    data.add_argument("--samples-per-epoch", type=_positive_int, default=62_000)
    data.add_argument("--validation-fonts-per-class", type=_positive_int, default=5)

    loading = parser.add_argument_group("Data loading")
    loading.add_argument("--batch-size", type=_positive_int, default=256)
    loading.add_argument("--workers", type=_nonnegative_int, default=min(os.cpu_count() or 1, 8))
    loading.add_argument("--seed", type=int, default=0)

    augmentation = parser.add_argument_group("Augmentation")
    augmentation.add_argument("--margin", type=_nonnegative_int, default=8)
    augmentation.add_argument("--rotation", type=_nonnegative_float, default=8.0)
    augmentation.add_argument("--translate", type=_probability, default=0.08)
    augmentation.add_argument("--scale-jitter", type=_scale_jitter, default=0.1)
    augmentation.add_argument("--shear", type=_nonnegative_float, default=5.0)
    augmentation.add_argument("--blur-prob", type=_probability, default=0.1)
    augmentation.add_argument("--invert-prob", type=_probability, default=0.5)

    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument("--epochs", type=_positive_int, default=10)
    optimization.add_argument("--lr", type=_positive_float, default=1e-3)
    optimization.add_argument("--weight-decay", type=_nonnegative_float, default=1e-4)
    optimization.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    optimization.add_argument("--amp", action="store_true", help="use automatic mixed precision")
    optimization.add_argument("--output-dir", type=Path, default=Path("checkpoints/characters"))
    optimization.add_argument("--resume", type=Path, help="checkpoint to resume")

    actions = parser.add_argument_group("Actions")
    action = actions.add_mutually_exclusive_group()
    action.add_argument("--show-samples", type=Path, metavar="PATH", help="save an augmented grid and exit")
    action.add_argument("--benchmark-loader", action="store_true", help="benchmark live image loading and exit")
    action.add_argument("--check-setup", action="store_true", help="run the trainer's one-batch overfit check")
    actions.add_argument("--benchmark-warmup-batches", type=_nonnegative_int, default=10)
    actions.add_argument("--benchmark-batches", type=_positive_int, default=100)
    return parser


def main(args: argparse.Namespace) -> None:
    if args.render_size < args.image_size:
        raise ValueError("--render-size must be at least --image-size")
    if args.arch == "alexnet" and args.image_size < 63:
        raise ValueError("AlexNet requires --image-size of at least 63")
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    args.device = str(device)

    alphabet = tuple(args.alphabet)
    records, manifest_info = resolve_font_records(alphabet, args.font_dir, args.manifest)
    train_transform, val_transform = build_transforms(args)
    train_set = SyntheticCharacterDataset(alphabet, records, args.render_size, args.samples_per_epoch, train_transform)
    train_sampler = RandomSampler(train_set, generator=_generator(args.seed))
    train_loader = build_loader(
        train_set,
        min(args.batch_size, 32) if args.show_samples is not None else args.batch_size,
        args.workers,
        train_sampler,
        args.seed + 1,
        pin_memory=device.type == "cuda",
    )

    if args.show_samples is not None:
        save_sample_grid(train_loader, train_set.classes, args.show_samples)
        return
    if args.benchmark_loader:
        benchmark_loader(train_loader, args.benchmark_warmup_batches, args.benchmark_batches)
        return

    val_set = SyntheticCharacterDataset(
        alphabet,
        records,
        args.render_size,
        len(alphabet) * args.validation_fonts_per_class,
        val_transform,
        deterministic=True,
    )
    val_loader = build_loader(
        val_set,
        args.batch_size,
        args.workers,
        SequentialSampler(val_set),
        args.seed + 2,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.arch, len(alphabet))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_path = args.output_dir / "checkpoint.pth"
    best_loss = math.inf

    def save_metadata(metrics: dict[str, float]) -> None:
        nonlocal best_loss
        if metrics["val_loss"] < best_loss:
            write_provenance(args.output_dir / "checkpoint.json", args, train_set, train_set.fonts, manifest_info)
            best_loss = metrics["val_loss"]

    trainer = ClassificationTrainer(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        output_file=str(checkpoint_path),
        device=device,
        amp=args.amp,
        on_epoch_end=save_metadata,
    )

    if args.resume is not None:
        validate_resume_metadata(args.resume, args.arch, train_set.classes)
        trainer.load(torch.load(args.resume, map_location="cpu"))
    best_loss = trainer.min_loss
    if args.check_setup:
        trainer.check_setup(lr=args.lr, num_it=100)
        return

    trainer.fit_n_epochs(args.epochs, args.lr, sched_type="onecycle")


if __name__ == "__main__":
    main(get_parser().parse_args())
