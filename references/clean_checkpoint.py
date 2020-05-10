#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Checkpoint cleanup
'''

import hashlib
import torch


def main(args):

    checkpoint = torch.load(args.checkpoint, map_location='cpu')['model']
    torch.save(checkpoint, args.outfile)

    with open(args.outfile, 'rb') as f:
        sha_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"Checkpoint saved to {args.outfile} with hash: {sha_hash[:8]}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Training checkpoint cleanup',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('checkpoint', type=str, help='path to the training checkpoint')
    parser.add_argument('outfile', type=str, help='model')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
