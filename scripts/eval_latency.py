# Copyright (C) 2019-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
Holocron model latency benchmark
"""

import argparse
import time

import numpy as np
import onnxruntime
import torch

from holocron import models


@torch.inference_mode()
def run_evaluation(
    model: torch.nn.Module, img_tensor: torch.Tensor, num_it: int = 100, warmup_it: int = 10
) -> np.array:
    # Warmup
    for _ in range(warmup_it):
        _ = model(img_tensor)

    timings = []

    # Evaluation runs
    for _ in range(num_it):
        start_ts = time.perf_counter()
        _ = model(img_tensor)
        timings.append(time.perf_counter() - start_ts)

    return np.array(timings)


def run_onnx_evaluation(
    model: onnxruntime.InferenceSession, img_tensor: np.array, num_it: int = 100, warmup_it: int = 10
) -> np.array:
    # Set input
    ort_input = {model.get_inputs()[0].name: img_tensor}
    # Warmup
    for _ in range(warmup_it):
        _ = model.run(None, ort_input)

    timings = []

    # Evaluation runs
    for _ in range(num_it):
        start_ts = time.perf_counter()
        _ = model.run(None, ort_input)
        timings.append(time.perf_counter() - start_ts)

    return np.array(timings)


@torch.inference_mode()
def main(args):
    # Pretrained imagenet model
    model = models.__dict__[args.arch](pretrained=args.pretrained).eval()
    # Reparametrizable models
    if args.arch.startswith("repvgg") or args.arch.startswith("mobileone"):
        model.reparametrize()

    # Input
    img_tensor = torch.rand((1, 3, args.size, args.size))

    _timings = run_evaluation(model, img_tensor, args.it)
    cpu_str = f"mean {1000 * _timings.mean():.2f}ms, std {1000 * _timings.std():.2f}ms"

    # ONNX
    torch.onnx.export(
        model,
        img_tensor,
        "tmp.onnx",
        export_params=True,
        opset_version=14,
    )
    onnx_session = onnxruntime.InferenceSession("tmp.onnx")
    npy_tensor = img_tensor.numpy()
    _timings = run_onnx_evaluation(onnx_session, npy_tensor, args.it)
    onnx_str = f"mean {1000 * _timings.mean():.2f}ms, std {1000 * _timings.std():.2f}ms"

    # GPU
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.device == "cpu":
        gpu_str = "N/A"
    else:
        device = torch.device(args.device)
        model = model.to(device=device)

        # Input
        img_tensor = img_tensor.to(device=device)
        _timings = run_evaluation(model, img_tensor, args.it)
        gpu_str = f"mean {1000 * _timings.mean():.2f}ms, std {1000 * _timings.std():.2f}ms"

    print(f"{args.arch} ({args.it} runs on ({args.size}, {args.size}) inputs)")
    print(f"CPU - {cpu_str}\nONNX - {onnx_str}\nGPU - {gpu_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Holocron model latency benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("arch", type=str, help="Architecture to use")
    parser.add_argument("--size", type=int, default=224, help="The image input size")
    parser.add_argument("--device", type=str, default=None, help="Default device to perform computation on")
    parser.add_argument("--it", type=int, default=100, help="Number of iterations to run")
    parser.add_argument("--warmup", type=int, default=10, help="Number of iterations for warmup")
    parser.add_argument(
        "--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo", action="store_true"
    )
    args = parser.parse_args()

    main(args)
