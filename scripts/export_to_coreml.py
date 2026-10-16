# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Holocron classification checkpoint to Core ML export."""

import argparse

import torch

from holocron import models
from holocron.models.coreml import export_coreml


def main(args):
    model = models.get_model(args.arch, pretrained=False, num_classes=args.num_classes).eval()
    if args.reparameterized:
        if args.arch != "mobileone_s0":
            raise ValueError("--reparameterized is only supported for mobileone_s0 checkpoints")
        model.reparametrize()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model", state), strict=True)
    images = torch.rand(1, 3, args.height, args.width, device="cpu", generator=torch.Generator().manual_seed(42))
    export_coreml(model, images, args.path, verify=not args.unverified)
    status = (
        "Core ML CPU predictions match PyTorch on three inputs"
        if not args.unverified
        else "Core ML inference NOT checked"
    )
    print(f"Exported {args.path}; {status}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("arch", choices=("resnet18", "mobileone_s0"), help="Architecture to restore")
    parser.add_argument("--checkpoint", required=True, help="Raw state_dict or trainer checkpoint containing model")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes used during training")
    parser.add_argument("--height", type=int, default=224, help="Fixed input height")
    parser.add_argument("--width", type=int, default=224, help="Fixed input width")
    parser.add_argument("--path", default="model.mlpackage", help="New output package path")
    parser.add_argument(
        "--reparameterized", action="store_true", help="Load an already reparameterized MobileOne checkpoint"
    )
    parser.add_argument("--unverified", action="store_true", help="Allow conversion without Core ML runtime checks")
    main(parser.parse_args())
