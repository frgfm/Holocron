# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import hashlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import torch


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"]
    torch.save(checkpoint, args.outfile, _use_new_zipfile_serialization=False)

    sha_hash = hashlib.sha256(Path(args.outfile).read_bytes()).hexdigest()
    print(f"Checkpoint saved to {args.outfile} with hash: {sha_hash[:8]}")


def parse_args():
    parser = ArgumentParser(description="Training checkpoint cleanup", formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("checkpoint", type=str, help="path to the training checkpoint")
    parser.add_argument("outfile", type=str, help="model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
