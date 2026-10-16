# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import copy
import platform
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn

__all__ = ["export_coreml"]

_RTOL, _ATOL = 1e-3, 3e-5


def _check_prediction(actual: torch.Tensor, expected: torch.Tensor, stage: str, index: int) -> None:
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise RuntimeError(f"{stage} verification input {index}: non-finite logits")
    try:
        torch.testing.assert_close(actual, expected, rtol=_RTOL, atol=_ATOL)
    except AssertionError as exc:
        raise RuntimeError(f"{stage} verification input {index}: {exc}") from exc


def _check_input(model: nn.Module, example_input: torch.Tensor, path: Path, verify: bool) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite Core ML destination: {path}")
    if path.suffix != ".mlpackage":
        raise ValueError("Core ML destination must end in .mlpackage")
    if example_input.ndim != 4 or example_input.shape[0] != 1 or min(example_input.shape) < 1:
        raise ValueError("Core ML input must have fixed shape (1, channels, height, width)")
    if example_input.dtype != torch.float32 or not torch.isfinite(example_input).all():
        raise ValueError("Core ML input must contain finite FP32 values")
    if verify and (
        platform.system() != "Darwin" or platform.machine() != "arm64" or int(platform.mac_ver()[0].split(".")[0]) < 12
    ):
        raise RuntimeError(
            "Core ML verification requires Apple Silicon macOS 12+; use verify=False for unverified export"
        )
    if any(
        tensor.is_floating_point() and tensor.dtype != torch.float32
        for tensor in (*model.parameters(), *model.buffers())
    ):
        raise ValueError("Core ML export requires FP32 model parameters and buffers")


@torch.no_grad()
def export_coreml(model: nn.Module, example_input: torch.Tensor, path: str | Path, *, verify: bool = True) -> Path:
    """Save a new ML Program package from a batch-one, FP32 NCHW tensor.

    The model must return FP32 logits shaped (1, classes). Uses an independent CPU
    evaluation copy; the caller's model and input are untouched. Checks tracing on
    the example, zeros and deterministic random input. By default, also requires
    actual Core ML CPU prediction parity on Apple Silicon macOS 12+; verify=False
    explicitly permits unverified conversion. The destination must end in .mlpackage
    and its parent must exist. Existing paths are refused; failures leave them intact.

    Validated architectures: ResNet18 and MobileOne-S0, including reparameterized models.

    Returns:
        The saved package path.

    Raises:
        ValueError: Incompatible input, model or output.
        ImportError: Missing coremltools.
        FileExistsError: Existing destination.
        RuntimeError: Tracing/conversion/verification failure or unavailable runtime.
    """
    path = Path(path)
    _check_input(model, example_input, path, verify)
    if not verify:
        warnings.warn("Core ML inference was not checked (verify=False). Only tracing is verified.", stacklevel=2)
    try:
        import coremltools as ct  # ty: ignore[unresolved-import]  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Install the optional Core ML dependencies: pip install 'pylocron[coreml]' (Python 3.11-3.13)"
        ) from exc

    model = copy.deepcopy(model).cpu().eval()
    images = example_input.detach().cpu().clone()
    samples = (
        images,
        torch.zeros_like(images),
        torch.rand(images.shape, dtype=torch.float32, device="cpu", generator=torch.Generator().manual_seed(0)),
    )
    references = []
    for sample in samples:
        output = model(sample.clone())
        if (
            not isinstance(output, torch.Tensor)
            or output.ndim != 2
            or output.shape[0] != 1
            or output.shape[1] < 1
            or output.dtype != torch.float32
        ):
            raise ValueError("Core ML classifier must return FP32 logits with shape (1, classes)")
        references.append(output.detach().clone())
    try:
        traced = torch.jit.trace(model, images.clone(), check_trace=False)
        for index, (sample, expected) in enumerate(zip(samples, references, strict=True)):
            _check_prediction(traced(sample.clone()), expected, "Tracing", index)
    except Exception as exc:
        raise RuntimeError(f"Core ML tracing failed: {exc}") from exc
    with TemporaryDirectory(dir=path.parent) as directory:
        candidate = Path(directory) / path.name
        try:
            converted = ct.convert(
                traced,
                source="pytorch",
                convert_to="mlprogram",
                inputs=[ct.TensorType(name="images", shape=tuple(images.shape), dtype=np.float32)],
                outputs=[ct.TensorType(name="logits", dtype=np.float32)],
                compute_precision=ct.precision.FLOAT32,
                compute_units=ct.ComputeUnit.CPU_ONLY,
                minimum_deployment_target=ct.target.iOS15,
                skip_model_load=True,
            )
            converted.save(str(candidate))
        except Exception as exc:
            raise RuntimeError(f"Core ML conversion failed: {exc}") from exc
        if verify:
            try:
                runtime = ct.models.MLModel(str(candidate), compute_units=ct.ComputeUnit.CPU_ONLY)
                for index, (sample, expected) in enumerate(zip(samples, references, strict=True)):
                    actual = torch.from_numpy(runtime.predict({"images": sample.numpy()})["logits"])
                    _check_prediction(actual, expected, "Core ML CPU", index)
            except Exception as exc:
                raise RuntimeError(f"Core ML runtime verification failed: {exc}") from exc
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite Core ML destination: {path}")
        candidate.rename(path)
    return path
