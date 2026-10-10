# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Run issue #499's four scratch-training comparisons using the existing reference recipe."""

import argparse
import copy
import hashlib
import json
import platform
import threading
import time
from pathlib import Path

import torch
from torch import nn

from holocron.trainer import resolve_device
from scripts.eval_latency import measure, peak_rss_mib, synchronize


def write_json(path, value):
    """Replace a benchmark record only after finite JSON has been written."""
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def count_macs(model):
    """Count convolution and linear multiply-accumulates for one 224px image.

    Returns:
        Number of convolution and linear MACs.
    """
    total = 0

    def count(module, _inputs, output):
        nonlocal total
        if isinstance(module, nn.Conv2d):
            total += (
                output.numel() * module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
            )
        elif isinstance(module, nn.Linear):
            total += output.numel() * module.in_features

    handles = [
        module.register_forward_hook(count) for module in model.modules() if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    try:
        with torch.inference_mode():
            model(torch.zeros(1, 3, 224, 224))
    finally:
        for handle in handles:
            handle.remove()
    return total


def main(args):
    from references.classification.train import get_parser  # noqa: PLC0415
    from references.classification.train import main as train  # noqa: PLC0415

    device = resolve_device(args.device)
    if args.epochs < 1 or args.workers < 0 or args.threads < 1:
        raise ValueError("Epochs and threads must be positive; workers cannot be negative")
    if args.output_dir.is_dir() and any(args.output_dir.iterdir()):
        raise ValueError("Use a fresh output directory to preserve previous results")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not (args.data_path / "train").is_dir() or not (args.data_path / "val").is_dir():
        raise ValueError("Expected Imagenette train and val directories")
    torch.set_num_threads(args.threads)
    report = {
        "scope": "Controlled Imagenette scratch training; not ImageNet or paper accuracy reproduction",
        "runtime": {
            "device": str(device),
            "torch": str(torch.__version__),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "threads": args.threads,
        },
        "recipe": {
            "epochs": args.epochs,
            "seed": args.seed,
            "physical_batch_size": args.batch_size,
            "gradient_accumulation": 32 // args.batch_size,
            "effective_batch_size": 32,
            "workers": args.workers,
            "train_crop_size": 176,
            "validation_resize_size": 232,
            "validation_crop_size": 224,
            "optimizer": "AdamP",
            "learning_rate": 0.001,
            "scheduler": "OneCycle",
            "mixup_alpha": 0.2,
            "label_smoothing": 0.1,
            "pretrained": False,
            "amp": args.amp,
            "autocast_dtype": str(torch.get_autocast_dtype(device.type)) if args.amp else "torch.float32",
            "selection": "minimum validation loss; final and selected metrics reported separately",
        },
        "dataset": {"path": str(args.data_path)},
        "macs_scope": "Convolution and linear MACs at batch 1, 224x224; normalization and activations excluded",
        "timing_scope": "Wall time includes dataset/model setup, training and validation; excludes deployment profiling",
        "models": [],
    }
    archive = args.data_path.parent / "imagenette2-320.tgz"
    if archive.is_file():
        with archive.open("rb") as stream:
            report["dataset"]["archive_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    for arch in args.arch:
        checkpoint = args.output_dir / f"{arch}.pth"
        if checkpoint.exists():
            raise ValueError(f"Refusing to overwrite {checkpoint}")
        history = []
        memory = {"sampled_peak_mps_tensor_bytes": 0, "sampled_peak_mps_driver_bytes": 0}
        done = threading.Event()
        start = time.perf_counter()

        def sample_memory(memory=memory, done=done):
            while not done.is_set():
                if device.type == "mps":
                    memory["sampled_peak_mps_tensor_bytes"] = max(
                        memory["sampled_peak_mps_tensor_bytes"], torch.mps.current_allocated_memory()
                    )
                    memory["sampled_peak_mps_driver_bytes"] = max(
                        memory["sampled_peak_mps_driver_bytes"], torch.mps.driver_allocated_memory()
                    )
                done.wait(0.05)

        def record(metrics, history=history, arch=arch, start=start):
            history.append({"epoch": len(history) + 1, **metrics, "elapsed_seconds": time.perf_counter() - start})
            write_json(args.output_dir / f"{arch}-progress.json", history)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sampler = threading.Thread(target=sample_memory, daemon=True)
        synchronize(device)
        sampler.start()
        options = get_parser().parse_args([str(args.data_path)])
        vars(options).update(
            arch=arch,
            device=str(device),
            epochs=args.epochs,
            seed=args.seed,
            batch_size=args.batch_size,
            grad_acc=32 // args.batch_size,
            workers=args.workers,
            output_file=str(checkpoint),
            amp=args.amp,
        )
        try:
            learner = train(options, on_epoch_end=record)
            synchronize(device)
            elapsed = time.perf_counter() - start
        finally:
            done.set()
            sampler.join()
        report["dataset"].update({
            "train_samples": len(learner.train_loader.dataset),
            "validation_samples": len(learner.val_loader.dataset),
            "class_to_index": learner.train_loader.dataset.class_to_idx,
        })
        result = {
            "architecture": arch,
            "training_elapsed_seconds": elapsed,
            "training_samples_per_wall_second": args.epochs * len(learner.train_loader) * args.batch_size / elapsed,
            "history": history,
            "final_metrics": {key: history[-1][key] for key in ("val_loss", "acc1", "acc5")},
            "memory": {
                **(memory if device.type == "mps" else {}),
                "process_lifetime_peak_rss_mib": peak_rss_mib(),
                "sampling_interval_seconds": 0.05 if device.type == "mps" else None,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            },
        }
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        learner.model.load_state_dict(state["model"])
        result["selected_epoch"] = state["epoch"]
        result["selected_metrics"] = learner.evaluate()
        with checkpoint.open("rb") as stream:
            result["checkpoint_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        unfused = copy.deepcopy(learner.model).cpu().eval()
        fused = copy.deepcopy(unfused)
        fused.reparametrize()
        with torch.inference_mode():
            example = torch.randn(1, 3, 224, 224)
            original_output, fused_output = unfused(example), fused(example)
            torch.testing.assert_close(original_output, fused_output, atol=1e-4, rtol=1e-4)
            result["fusion_cpu_max_absolute_error"] = (original_output - fused_output).abs().max().item()
        result["parameters"] = {
            "training": sum(parameter.numel() for parameter in unfused.parameters()),
            "deployment": sum(parameter.numel() for parameter in fused.parameters()),
        }
        result["macs"] = {"training": count_macs(unfused), "deployment": count_macs(fused)}
        fused = fused.to(device)
        learner.model = fused
        result["deployment_metrics"] = learner.evaluate()
        image = torch.randn(1, 3, 224, 224, device=device)
        with torch.inference_mode(), torch.autocast(device.type, enabled=args.amp):
            result["deployment_latency"], _ = measure(
                lambda model=fused, image=image: model(image), 1, 100, 20, sync=lambda: synchronize(device)
            )
        report["models"].append(result)
        write_json(args.output_dir / "results.json", report)
        print(
            json.dumps({"architecture": arch, "selected": result["selected_metrics"], "seconds": elapsed}), flush=True
        )
        del learner, unfused, fused, state
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--arch",
        nargs="+",
        choices=["repvit_m0_9", "repvit_m1_0", "repvit_m1_1", "mobileone_s2"],
        default=["repvit_m0_9", "repvit_m1_0", "repvit_m1_1", "mobileone_s2"],
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, choices=[2, 4, 8, 16, 32], default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    main(parser.parse_args())
