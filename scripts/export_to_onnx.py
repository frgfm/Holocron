# Copyright (C) 2022-2023, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
Holocron model ONNX export
"""

import argparse

import torch

from holocron import models


@torch.inference_mode()
def main(args):

    is_pretrained = args.pretrained and not isinstance(args.checkpoint, str)
    # Pretrained imagenet model
    model = models.__dict__[args.arch](pretrained=is_pretrained).eval()

    # Load the checkpoint
    if isinstance(args.checkpoint, str):
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    # RepVGG
    if args.arch.startswith("repvgg") or args.arch.startswith("mobileone"):
        model.reparametrize()

    # Input
    img_tensor = torch.rand((args.batch_size, args.in_channels, args.height, args.width))

    # ONNX export
    torch.onnx.export(
        model,
        img_tensor,
        args.path,
        export_params=True,
        opset_version=14,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Holocron model ONNX export", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("arch", type=str, help="Architecture to use")
    parser.add_argument("--height", type=int, default=224, help="The height of the input image")
    parser.add_argument("--width", type=int, default=224, help="The width of the input image")
    parser.add_argument("--in-channels", type=int, default=3, help="The number of channels of the input image")
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size used for the model")
    parser.add_argument("--path", type=str, default="./model.onnx", help="The path of the output file")
    parser.add_argument("--checkpoint", type=str, default=None, help="The checkpoint to restore")
    parser.add_argument(
        "--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo", action="store_true"
    )
    args = parser.parse_args()

    main(args)
