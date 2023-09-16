# Copyright (C) 2019-2023, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import hashlib
from pathlib import Path

import torch


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")["model"]
    torch.save(checkpoint, args.outfile, _use_new_zipfile_serialization=False)

    with Path(args.outfile).open("rb") as f:
        sha_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"Checkpoint saved to {args.outfile} with hash: {sha_hash[:8]}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Training checkpoint cleanup", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("checkpoint", type=str, help="path to the training checkpoint")
    parser.add_argument("outfile", type=str, help="model")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
