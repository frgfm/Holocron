# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Measure a standard classification training run; no separate recipe or CLI."""

import copy
import json
import math
import platform
import threading
import time
from pathlib import Path

import torch
from torch import nn

from holocron.experiments import checkpoint_record, sha256, write_json
from holocron.trainer import resolve_device
from scripts.eval_latency import measure, peak_rss_mib, synchronize


def count_macs(model, shape=(1, 3, 224, 224)):
    """Count convolution and linear multiply-accumulates for one input.

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
            model(torch.zeros(shape))
    finally:
        for handle in handles:
            handle.remove()
    return total


def run_benchmark(args, train, *, trial=None):
    device = resolve_device(args.device)
    if min(args.epochs, args.batch_size, args.grad_acc) < 1 or args.workers < 0:
        raise ValueError("Epochs, batch size, and gradient accumulation must be positive; workers cannot be negative")
    if any(
        getattr(args, key)
        for key in ("resume", "test_only", "find_lr", "find_size", "check_setup", "show_samples", "plot_loss")
    ):
        raise ValueError("Benchmark mode requires fresh training, without diagnostic actions")
    checkpoint = Path(args.output_file)
    output = checkpoint.with_suffix(checkpoint.suffix + ".json")
    progress = checkpoint.with_name(checkpoint.stem + "-progress.json")
    if any(path.exists() for path in (checkpoint, output, progress)):
        raise ValueError("Refusing to overwrite benchmark checkpoint, results, or progress")
    if (
        args.dataset.lower() == "imagenette"
        and not hasattr(args, "split_paths")
        and not all((Path(args.data_path) / split).is_dir() for split in ("train", "val"))
    ):
        raise ValueError("Expected Imagenette train and val directories")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Controlled classification training; not paper accuracy reproduction",
        "runtime": {
            "device": str(device),
            "torch": str(torch.__version__),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "threads": torch.get_num_threads(),
        },
        "recipe": {
            **vars(args),
            "effective_batch_size": args.batch_size * args.grad_acc,
            "autocast_dtype": str(torch.get_autocast_dtype(device.type)) if args.amp else "torch.float32",
            "selection": "minimum validation loss; final and selected metrics reported separately",
        },
        "dataset": {"path": str(args.data_path)},
        "macs_scope": "Convolution and linear MACs at the validation input shape; normalization and activations excluded",
        "timing_scope": "Wall time includes dataset/model setup, training and validation; excludes deployment profiling",
        "models": [],
        "metrics_split": "validation",
        "metric_directions": {"val_loss": "minimize", "acc1": "maximize", "acc5": "maximize"},
    }
    archive = Path(args.data_path).parent / "imagenette2-320.tgz"
    if trial is None and args.dataset.lower() == "imagenette" and archive.is_file():
        report["dataset"]["archive_sha256"] = sha256(archive)
    history = []
    result = {"architecture": args.arch, "history": history}
    report["models"].append(result)
    if trial is not None:
        trial.result.update(report)
    learner = None
    memory = {"sampled_peak_mps_tensor_bytes": 0, "sampled_peak_mps_driver_bytes": 0}
    done = threading.Event()
    start = time.perf_counter()

    def sample_memory():
        while not done.is_set():
            if device.type == "mps":
                memory["sampled_peak_mps_tensor_bytes"] = max(
                    memory["sampled_peak_mps_tensor_bytes"], torch.mps.current_allocated_memory()
                )
                memory["sampled_peak_mps_driver_bytes"] = max(
                    memory["sampled_peak_mps_driver_bytes"], torch.mps.driver_allocated_memory()
                )
            done.wait(0.05)

    def setup(instance, scheduler_kwargs):
        nonlocal learner
        learner = instance
        report["recipe"]["scheduler_kwargs"] = scheduler_kwargs
        report["runtime"].update(
            device=str(learner.device),
            pin_memory=learner.train_loader.pin_memory,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        )
        if len(learner.val_loader.dataset.classes) < 5:
            report["metric_directions"].pop("acc5")
        if trial is not None:
            trial.prepare(learner)

    def record(metrics):
        metrics = {
            key: value
            for key, value in metrics.items()
            if key != "acc5" or len(learner.val_loader.dataset.classes) >= 5
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("non-finite epoch metrics")
        history.append({"epoch": len(history) + 1, **metrics, "elapsed_seconds": time.perf_counter() - start})
        if trial is None:
            write_json(progress, history)
        else:
            trial.record_epoch(learner, history[-1])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sampler = threading.Thread(target=sample_memory, daemon=True)
    synchronize(device)
    sampler.start()
    try:
        learner = train(args, on_epoch_end=record, on_setup=setup)
        synchronize(device)
        elapsed = time.perf_counter() - start
    finally:
        done.set()
        sampler.join()
        if learner is not None and learner.on_epoch_end is record:
            learner.on_epoch_end = None
    if learner.epoch != args.epochs:
        raise RuntimeError("training ended without the requested epochs")
    report["dataset"].update({
        "train_samples": len(learner.train_loader.dataset),
        "validation_samples": len(learner.val_loader.dataset),
        "class_to_index": learner.train_loader.dataset.class_to_idx,
    })
    shape = (1, *learner.val_loader.dataset[0][0].shape)
    result.update({
        "input_shape": shape,
        "training_elapsed_seconds": elapsed,
        "training_samples_per_wall_second": args.epochs * len(learner.train_loader) * args.batch_size / elapsed,
        "final_epoch": learner.epoch,
        "final_metrics": {key: value for key, value in history[-1].items() if key not in {"epoch", "elapsed_seconds"}},
        "memory": {
            **(memory if device.type == "mps" else {}),
            "process_lifetime_peak_rss_mib": peak_rss_mib(),
            "sampling_interval_seconds": 0.05 if device.type == "mps" else None,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        },
    })
    if trial is not None:
        if "checkpoint" not in result:
            raise RuntimeError("training ended without a selected checkpoint")
        report["dataset"].update(trial.provenance["data"])
        return
    state, selected = checkpoint_record(checkpoint, history)
    result.update(selected)
    learner.model.load_state_dict(state["model"])
    unfused = copy.deepcopy(learner.model).cpu().eval()
    fused = copy.deepcopy(unfused)
    if hasattr(fused, "reparametrize"):
        fused.reparametrize()
    with torch.inference_mode():
        example = torch.randn(shape)
        original_output, fused_output = unfused(example), fused(example)
        torch.testing.assert_close(original_output, fused_output, atol=1e-4, rtol=1e-4)
        result["fusion_cpu_max_absolute_error"] = (original_output - fused_output).abs().max().item()
    result["parameters"] = {
        "training": sum(parameter.numel() for parameter in unfused.parameters()),
        "deployment": sum(parameter.numel() for parameter in fused.parameters()),
    }
    result["macs"] = {"training": count_macs(unfused, shape), "deployment": count_macs(fused, shape)}
    learner.model = fused.to(device)
    result["deployment_metrics"] = learner.evaluate()
    image = torch.randn(shape, device=device)
    with torch.inference_mode(), torch.amp.autocast(device.type, enabled=args.amp):
        result["deployment_latency"], _ = measure(
            lambda: learner.model(image), 1, 100, 20, sync=lambda: synchronize(device)
        )
    write_json(output, report)
    print(
        json.dumps({"architecture": args.arch, "selected": result["selected_metrics"], "seconds": elapsed}), flush=True
    )
