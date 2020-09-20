#!/usr/bin/env python

import unittest
import torch
import torch.nn as nn
from holocron import trainer


class UtilsTester(unittest.TestCase):

    def test_freeze_bn(self):

        # Simple module with BN
        mod = nn.Sequential(nn.Conv2d(3, 32, 3), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        nb = mod[1].num_batches_tracked.clone()
        rm = mod[1].running_mean.clone()
        rv = mod[1].running_var.clone()
        # Freeze & forward
        for p in mod.parameters():
            p.requires_grad_(False)
        mod = trainer.freeze_bn(mod)
        for _ in range(10):
            out = mod(torch.rand((1, 3, 32, 32)))
        # Check that stats were not updated
        self.assertTrue(torch.equal(mod[1].num_batches_tracked, nb))
        self.assertTrue(torch.equal(mod[1].running_mean, rm))
        self.assertTrue(torch.equal(mod[1].running_var, rv))

    def test_freeze_model(self):

        # Simple model
        mod = nn.Sequential(nn.Conv2d(3, 32, 3), nn.ReLU(inplace=True), nn.Conv2d(32, 64, 3), nn.ReLU(inplace=True))
        mod = trainer.freeze_model(mod, '0')
        # Check that the correct layers were frozen
        self.assertFalse(any(p.requires_grad for p in mod[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in mod[2].parameters()))


if __name__ == '__main__':
    unittest.main()
