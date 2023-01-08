# Copyright (C) 2019-2023, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
Holocron model latency benchmark
"""

import argparse
import time

import numpy as np
import torch

from holocron import models


@torch.inference_mode()
def main(args):

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(args.device)

    # Pretrained imagenet model
    model = models.__dict__[args.arch](pretrained=args.pretrained).eval().to(device=device)

    # RepVGG
    if args.arch.startswith("repvgg") or args.arch.startswith("mobileone"):
        model.reparametrize()

    # Input
    img_tensor = torch.rand((1, 3, args.size, args.size)).to(device=device)

    # Warmup
    for _ in range(10):
        _ = model(img_tensor)

    timings = []

    # Evaluation runs
    for _ in range(args.it):
        start_ts = time.perf_counter()
        _ = model(img_tensor)
        timings.append(time.perf_counter() - start_ts)

    _timings = np.array(timings)
    print(f"{args.arch} ({args.it} runs on ({args.size}, {args.size}) inputs)")
    print(f"mean {1000 * _timings.mean():.2f}ms, std {1000 * _timings.std():.2f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Holocron model latency benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("arch", type=str, help="Architecture to use")
    parser.add_argument("--size", type=int, default=224, help="The image input size")
    parser.add_argument("--device", type=str, default=None, help="Default device to perform computation on")
    parser.add_argument("--it", type=int, default=100, help="Number of iterations to run")
    parser.add_argument(
        "--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo", action="store_true"
    )
    args = parser.parse_args()

    main(args)
