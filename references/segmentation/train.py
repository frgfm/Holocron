# Copyright (C) 2019-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

'''
Training script for semantic segmentation
'''

import os
import datetime
import time
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms as T
from torchvision.datasets import VOCSegmentation
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms import functional as F
from torch.utils.data import RandomSampler, SequentialSampler

import holocron
from holocron.models import segmentation
from holocron.trainer import SegmentationTrainer
from transforms import Compose, RandomResize, RandomCrop, RandomHorizontalFlip, SampleTransform, ToTensor


VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']


def worker_init_fn(worker_id: int) -> None:
    np.random.seed((worker_id + torch.initial_seed()) % np.iinfo(np.int32).max)


def plot_samples(images, targets, ignore_index=None):
    # Unnormalize image
    nb_samples = 4
    _, axes = plt.subplots(2, nb_samples, figsize=(20, 5))
    for idx in range(nb_samples):
        img = images[idx]
        img *= torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img += torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        img = F.to_pil_image(img)
        target = targets[idx]
        if isinstance(ignore_index, int):
            target[target == ignore_index] = 0

        axes[0][idx].imshow(img)
        axes[0][idx].axis('off')
        axes[1][idx].imshow(target)
        axes[1][idx].axis('off')
    plt.show()


def main(args):

    print(args)

    torch.backends.cudnn.benchmark = True

    # Data loading
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    base_size = 520
    crop_size = 480
    min_size, max_size = int(0.5 * base_size), int(2.0 * base_size)

    train_loader, val_loader = None, None
    if not args.test_only:
        st = time.time()
        train_set = VOCSegmentation(
            args.data_path,
            image_set='train',
            download=True,
            transforms=Compose([
                RandomResize(min_size, max_size),
                RandomHorizontalFlip(0.5),
                RandomCrop(crop_size),
                ToTensor(),
                SampleTransform(normalize)
            ])
        )

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, drop_last=True,
            sampler=RandomSampler(train_set), num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn)

        print(f"Training set loaded in {time.time() - st:.2f}s "
              f"({len(train_set)} samples in {len(train_loader)} batches)")

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target, ignore_index=255)
        return

    if not (args.lr_finder or args.check_setup):
        st = time.time()
        val_set = VOCSegmentation(
            args.data_path,
            image_set='val',
            download=True,
            transforms=Compose([
                RandomResize(base_size, base_size),
                ToTensor(),
                SampleTransform(normalize)
            ])
        )

        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=args.batch_size, drop_last=False,
            sampler=SequentialSampler(val_set), num_workers=args.workers, pin_memory=True,
            worker_init_fn=worker_init_fn)

        print(f"Validation set loaded in {time.time() - st:.2f}s ({len(val_set)} samples in {len(val_loader)} batches)")

    model = segmentation.__dict__[args.model](args.pretrained, num_classes=len(VOC_CLASSES))

    # Loss setup
    loss_weight = torch.ones(len(VOC_CLASSES))
    loss_weight[0] = 0.1
    if args.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss(weight=loss_weight, ignore_index=255)
    elif args.loss == 'label_smoothing':
        criterion = holocron.nn.LabelSmoothingCrossEntropy(weight=loss_weight, ignore_index=255)
    elif args.loss == 'focal':
        criterion = holocron.nn.FocalLoss(weight=loss_weight, ignore_index=255)

    # Optimizer setup
    model_params = [p for p in model.parameters() if p.requires_grad]
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model_params, args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model_params, args.lr,
                                     betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay)
    elif args.opt == 'radam':
        optimizer = holocron.optim.RAdam(model_params, args.lr,
                                         betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay)
    elif args.opt == 'ranger':
        optimizer = Lookahead(holocron.optim.RAdam(model_params, args.lr,
                                                   betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay))
    elif args.opt == 'tadam':
        optimizer = holocron.optim.TAdam(model_params, args.lr,
                                         betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay)

    trainer = SegmentationTrainer(model, train_loader, val_loader, criterion, optimizer,
                                  args.device, args.output_file)
    if args.resume:
        print(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        trainer.load(checkpoint)

    if args.test_only:
        print("Running evaluation")
        eval_metrics = trainer.evaluate()
        print(f"Validation loss: {eval_metrics['val_loss']:.4} (Mean IoU: {eval_metrics['mean_iou']:.2%})")
        return

    if args.lr_finder:
        print("Looking for optimal LR")
        trainer.lr_find(args.freeze_until, num_it=min(len(train_loader), 100))
        trainer.plot_recorder()
        return

    if args.check_setup:
        print("Checking batch overfitting")
        is_ok = trainer.check_setup(args.freeze_until, args.lr, num_it=min(len(train_loader), 100))
        print(is_ok)
        return

    print("Start training")
    start_time = time.time()
    trainer.fit_n_epochs(args.epochs, args.lr, args.freeze_until, args.sched)
    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {total_time_str}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Holocron Segmentation Training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data_path', type=str, help='path to dataset folder')
    parser.add_argument('--model', default='unet3p', help='model')
    parser.add_argument('--freeze-until', default=None, type=str, help='Last layer to freeze')
    parser.add_argument('--device', default=None, type=int, help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=min(os.cpu_count(), 16), type=int,
                        help='number of data loading workers')
    parser.add_argument('--loss', default='crossentropy', type=str, help='loss')
    parser.add_argument('--opt', default='adam', type=str, help='optimizer')
    parser.add_argument('--sched', default='onecycle', type=str, help='scheduler')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float, help='weight decay', dest='weight_decay')
    parser.add_argument("--lr-finder", dest='lr_finder', action='store_true', help="Should you run LR Finder")
    parser.add_argument("--check-setup", dest='check_setup', action='store_true', help="Check your training setup")
    parser.add_argument('--output-file', default='./model.pth', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--show-samples", dest='show_samples', action='store_true',
                        help="Whether training samples should be displayed")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo",
                        action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
