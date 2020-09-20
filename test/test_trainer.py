#!/usr/bin/env python

import unittest
from tempfile import NamedTemporaryFile
import torch
import torch.nn as nn
from holocron.nn import GlobalAvgPool2d
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from holocron import trainer


class MockClassificationDataset(torch.utils.data.Dataset):
    """Mock dataset generating a random sample and a fixed zero target"""
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __getitem__(self, idx):
        return torch.rand((3, 32, 32)), 0

    def __len__(self):
        return self.n


class MockSegDataset(torch.utils.data.Dataset):
    """Mock dataset generating a random sample and a fixed zero target"""
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __getitem__(self, idx):
        return torch.rand((3, 32, 32)), torch.zeros((32, 32), dtype=torch.long)

    def __len__(self):
        return self.n


class MockDetDataset(torch.utils.data.Dataset):
    """Mock dataset generating a random sample and a fixed zero target"""
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __getitem__(self, idx):
        boxes = torch.tensor([[0, 0, 1, 1], [0.25, 0.25, 0.75, 0.75]], dtype=torch.float32)
        return torch.rand((3, 416, 416)), dict(boxes=boxes, labels=torch.zeros(2, dtype=torch.long))

    def __len__(self):
        return self.n


def collate_fn(batch):
    imgs, target = zip(*batch)
    return imgs, target


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
            _ = mod(torch.rand((1, 3, 32, 32)))
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
        self.assertRaises(ValueError, trainer.freeze_model, mod, 'wrong_layer')


class CoreTester(unittest.TestCase):

    def test_classification_trainer(self):

        num_it = 100
        batch_size = 8
        # Generate all dependencies
        model = nn.Sequential(nn.Conv2d(3, 32, 3), nn.ReLU(inplace=True),
                              GlobalAvgPool2d(flatten=True), nn.Linear(32, 5))
        train_loader = torch.utils.data.DataLoader(MockClassificationDataset(num_it * batch_size),
                                                   batch_size=batch_size)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        self.assertRaises(ValueError if torch.cuda.is_available() else AssertionError,
                          trainer.ClassificationTrainer,
                          model, train_loader, train_loader, criterion, optimizer, gpu=7)

        with NamedTemporaryFile() as tf:
            learner = trainer.ClassificationTrainer(model, train_loader, train_loader, criterion, optimizer,
                                                    output_file=tf.name, gpu=0 if torch.cuda.is_available() else None)
            learner.save(tf.name)
            checkpoint = torch.load(tf.name, map_location='cpu')
            model_w = learner.model[-1].weight.data.clone()
            # Check setup
            self.assertTrue(learner.check_setup(num_it=num_it))

            # LR Find
            learner.load(checkpoint)

            self.assertRaises(AssertionError, learner.plot_recorder, block=False)
            learner.lr_find(num_it=num_it)
            self.assertEqual(len(learner.lr_recorder), len(learner.loss_recorder))
            learner.plot_recorder(block=False)

            # Training
            # Perform the iterations
            learner.load(checkpoint)
            self.assertRaises(ValueError, learner.fit_n_epochs, 1, 1e-3, sched_type='my_scheduler')
            learner.fit_n_epochs(1, 1e-3)
            # Check that params were updated
            self.assertFalse(torch.equal(model[-1].weight.data, model_w))
            learner.load(checkpoint)
            learner.fit_n_epochs(1, 1e-3, sched_type='cosine')
            # Check that params were updated
            self.assertFalse(torch.equal(model[-1].weight.data, model_w))

    def test_segmentation_trainer(self):

        num_it = 100
        batch_size = 8
        # Generate all dependencies
        model = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 5, 3, padding=1))
        train_loader = torch.utils.data.DataLoader(MockSegDataset(num_it * batch_size), batch_size=batch_size)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        with NamedTemporaryFile() as tf:
            learner = trainer.SegmentationTrainer(model, train_loader, train_loader, criterion, optimizer,
                                                  output_file=tf.name, gpu=0 if torch.cuda.is_available() else None)
            learner.save(tf.name)
            checkpoint = torch.load(tf.name, map_location='cpu')
            model_w = learner.model[-1].weight.data.clone()
            # Check setup
            self.assertTrue(learner.check_setup(num_it=num_it))

            # LR Find
            learner.load(checkpoint)
            learner.lr_find(num_it=num_it)
            learner.plot_recorder(block=False)

            # Training
            # Perform the iterations
            learner.load(checkpoint)
            learner.fit_n_epochs(1, 1e-3)
            # Check that params were updated
            self.assertFalse(torch.equal(model[-1].weight.data, model_w))
            learner.load(checkpoint)
            learner.fit_n_epochs(1, 1e-3, sched_type='cosine')
            # Check that params were updated
            self.assertFalse(torch.equal(model[-1].weight.data, model_w))

    def test_detection_trainer(self):

        num_it = 10
        batch_size = 2
        # Generate all dependencies
        model = fasterrcnn_resnet50_fpn(pretrained_backbone=True, num_classes=10)
        train_loader = torch.utils.data.DataLoader(MockDetDataset(num_it * batch_size),
                                                   batch_size=batch_size, collate_fn=collate_fn)
        optimizer = torch.optim.Adam(model.parameters())

        with NamedTemporaryFile() as tf:
            learner = trainer.DetectionTrainer(model, train_loader, train_loader, None, optimizer,
                                               output_file=tf.name, gpu=0 if torch.cuda.is_available() else None)
            learner.save(tf.name)
            checkpoint = torch.load(tf.name, map_location='cpu')
            model_w = learner.model.roi_heads.box_predictor.cls_score.weight.data.clone()
            # Check setup
            self.assertTrue(learner.check_setup('backbone', num_it=num_it))

            # LR Find
            learner.load(checkpoint)
            learner.lr_find('backbone', num_it=num_it)
            learner.plot_recorder(block=False)

            # Training
            # Perform the iterations
            learner.load(checkpoint)
            learner.fit_n_epochs(1, 5e-4, 'backbone')
            # Check that params were updated
            self.assertFalse(torch.equal(learner.model.roi_heads.box_predictor.cls_score.weight.data, model_w))
            learner.load(checkpoint)
            learner.fit_n_epochs(1, 5e-4, 'backbone', sched_type='cosine')
            # Check that params were updated
            self.assertFalse(torch.equal(learner.model.roi_heads.box_predictor.cls_score.weight.data, model_w))


if __name__ == '__main__':
    unittest.main()
