# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Train reproducible character pretraining or variable-width CTC OCR."""

import argparse
import json
import math
import platform
import time
from itertools import pairwise
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from holocron.models.recognition import CharacterClassifier, CTCRecognizer
from holocron.trainer import resolve_device
from holocron.utils import CTCCodec
from references.classification.train_characters import _sha256, resolve_font_records
from references.recognition.data import ALPHABET, SyntheticTextDataset, collate_lines, inspect_fonts, split_fonts
from references.recognition.evaluate import evaluate_characters, evaluate_lines


def ctc_loss(probabilities, lengths, texts, codec):
    targets = [codec.encode(text) for text in texts]
    required = [len(text) + sum(a == b for a, b in pairwise(text)) for text in texts]
    if any(need > length for need, length in zip(required, lengths.tolist(), strict=True)):
        raise ValueError("rendered width is too short for CTC including repeated characters")
    # MPS has no CTC loss kernel; the differentiable transfer retains MPS model gradients.
    if probabilities.device.type == "mps":
        probabilities = probabilities.cpu()
    return nn.functional.ctc_loss(
        probabilities,
        torch.cat(targets).to(probabilities.device),
        lengths,
        torch.tensor([len(target) for target in targets]),
        blank=0,
    )


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--task", choices=["characters", "lines"], default="lines")
    parser.add_argument("--font-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--held-out", nargs="+", default=["Noto Serif", "Ubuntu"])
    parser.add_argument("--alphabet", default=ALPHABET)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deskew", action="store_true", help="normalize small line rotations before recognition")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--stop-after-epochs", type=int, help="end this phase early while preserving the full cosine schedule"
    )
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:N, mps, or auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--init", type=Path, help="weights-only warm start; character checkpoints transfer the backbone"
    )
    parser.add_argument(
        "--resume", type=Path, help="resume last.pth with optimizer and RNG state; use the same configuration"
    )
    return parser


def main(args):
    if (
        min(
            args.epochs, args.samples_per_epoch, args.validation_samples, args.max_length, args.batch_size, args.threads
        )
        <= 0
        or args.workers < 0
        or not math.isfinite(args.lr)
        or args.lr <= 0
    ):
        raise ValueError("invalid training counts or learning rate")
    if args.init is not None and args.resume is not None:
        raise ValueError("--init and --resume are mutually exclusive")
    if args.stop_after_epochs is not None and not 0 < args.stop_after_epochs <= args.epochs:
        raise ValueError("--stop-after-epochs must be between one and --epochs")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    args.device = str(resolve_device(args.device))
    codec = CTCCodec(args.alphabet)
    records, manifest = resolve_font_records(codec.alphabet.replace(" ", ""), args.font_dir, args.manifest)
    records = inspect_fonts(records, codec.alphabet)
    train_fonts, held_fonts = split_fonts(records, args.held_out)
    train_set = SyntheticTextDataset(
        train_fonts,
        codec,
        args.samples_per_epoch,
        seed=args.seed,
        augment=args.augment,
        task=args.task,
        max_length=args.max_length,
        deskew=args.deskew,
    )
    val_set = SyntheticTextDataset(
        train_fonts,
        codec,
        args.validation_samples,
        seed=args.seed + 100_000,
        augment=True,
        task=args.task,
        max_length=args.max_length,
        deskew=args.deskew,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate_lines)
    model = (
        CharacterClassifier(len(codec.alphabet)) if args.task == "characters" else CTCRecognizer(len(codec.alphabet))
    )
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    fonts = [
        {"family": record.family, "file": record.path.name, "sha256": _sha256(record.path)} for record in train_fonts
    ]
    held_fonts = [
        {"family": record.family, "file": record.path.name, "sha256": _sha256(record.path)} for record in held_fonts
    ]
    initialized_from = None
    if args.init is not None:
        initial = torch.load(args.init, map_location="cpu", weights_only=True)
        if initial["alphabet"] != codec.alphabet:
            raise ValueError("initial checkpoint alphabet does not match")
        initial_families = {record["family"] for record in initial["fonts"]}
        if initial_families & {record.family for record in records if record.family in args.held_out}:
            raise ValueError("initial checkpoint was trained on a held-out font family")
        if initial["task"] == args.task:
            model.load_state_dict(initial["model"])
        elif initial["task"] == "characters" and args.task == "lines":
            model.backbone.load_state_dict({
                key.removeprefix("backbone."): value
                for key, value in initial["model"].items()
                if key.startswith("backbone.")
            })
        else:
            raise ValueError("only character-to-line or same-task initialization is supported")
        initialized_from = {"sha256": _sha256(args.init), "task": initial["task"], "epoch": initial["epoch"]}
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.1)
    start, best, history = 0, float("inf"), []
    if args.resume is not None:
        last = torch.load(args.resume, map_location="cpu", weights_only=True)
        ignored = {"output_dir", "resume", "init", "workers", "threads", "device", "stop_after_epochs"}
        if (
            any(
                last["configuration"].get(key, False if key == "deskew" else None) != value
                for key, value in configuration.items()
                if key not in ignored
            )
            or last["fonts"] != fonts
            or last["held_out_fonts"] != held_fonts
        ):
            raise ValueError("resume training configuration or font checksums changed")
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        torch.set_rng_state(last["rng"])
        if str(args.device).startswith("cuda"):
            torch.cuda.set_rng_state_all(last["cuda_rng"])
        elif args.device == "mps" and last.get("mps_rng") is not None:
            torch.mps.set_rng_state(last["mps_rng"])
        start, best, history = last["epoch"], last["best_cer"], last["history"]
        initialized_from = last["initialized_from"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task": args.task,
        "alphabet": codec.alphabet,
        "configuration": configuration,
        "fonts": fonts,
        "held_out_fonts": held_fonts,
        "manifest": manifest,
        "initialized_from": initialized_from,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "runtime": {
            "torch": str(torch.__version__),
            "python": platform.python_version(),
            "device": args.device,
            "threads": args.threads,
        },
    }
    print(json.dumps(metadata), flush=True)
    for epoch in range(start, args.stop_after_epochs or args.epochs):
        started = time.perf_counter()
        train_set.epoch = epoch
        loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=collate_lines,
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        model.train()
        total_loss = 0.0
        for images, lengths, texts in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images.to(args.device), lengths)
            if args.task == "characters":
                targets = torch.tensor([codec.indices[text] - 1 for text in texts], device=args.device)
                loss = nn.functional.cross_entropy(outputs, targets)
            else:
                loss = ctc_loss(outputs, lengths, texts, codec)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss; CTC samples are not silently dropped")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(texts)
        evaluate = evaluate_characters if args.task == "characters" else evaluate_lines
        metrics = evaluate(model, codec, val_loader, args.device)
        summary = {
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_set),
            "validation_cer": metrics["cer"],
            "validation_exact_match": metrics["exact_match"],
            "seconds": time.perf_counter() - started,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(summary)
        print(json.dumps(summary), flush=True)
        scheduler.step()
        improved = metrics["cer"] < best
        best = min(best, metrics["cer"])
        checkpoint = {
            **metadata,
            "model": model.state_dict(),
            "epoch": epoch + 1,
            "best_cer": best,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if str(args.device).startswith("cuda") else [],
            "mps_rng": torch.mps.get_rng_state() if args.device == "mps" else None,
            "history": history,
        }
        torch.save(checkpoint, args.output_dir / "last.pth")
        if improved:
            torch.save(checkpoint, args.output_dir / "best.pth")
        (args.output_dir / "history.json").write_text(
            json.dumps({**metadata, "history": history}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main(get_parser().parse_args())
