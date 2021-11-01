# Copyright (C) 2019-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

'''
Training script for object detection
'''

import datetime
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from fastprogress import master_bar, progress_bar
from matplotlib.patches import Rectangle
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision import transforms as T
from torchvision.datasets import VOCDetection
from torchvision.ops.boxes import box_iou
from torchvision.transforms import functional as F
from transforms import (CenterCrop, Compose, ImageTransform, RandomHorizontalFlip, RandomResizedCrop, Resize,
                        VOCTargetTransform, convert_to_relative)

import holocron
from holocron.models import detection
from holocron.trainer import DetectionTrainer

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']


def worker_init_fn(worker_id: int) -> None:
    np.random.seed((worker_id + torch.initial_seed()) % np.iinfo(np.int32).max)


def collate_fn(batch):
    imgs, target = zip(*batch)
    return imgs, target


def plot_samples(images, targets):
    # Unnormalize image
    nb_samples = 4
    _, axes = plt.subplots(1, nb_samples, figsize=(20, 5))
    for idx in range(nb_samples):
        img = images[idx]
        img *= torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img += torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        img = F.to_pil_image(img)

        axes[idx].imshow(img)
        axes[idx].axis('off')
        for box, label in zip(targets[idx]['boxes'], targets[idx]['labels']):
            xmin = int(box[0] * images[idx].shape[-1])
            ymin = int(box[1] * images[idx].shape[-2])
            xmax = int(box[2] * images[idx].shape[-1])
            ymax = int(box[3] * images[idx].shape[-2])

            rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             linewidth=2, edgecolor='lime', facecolor='none')
            axes[idx].add_patch(rect)
            axes[idx].text(xmin, ymin, VOC_CLASSES[label.item()], color='lime', fontsize=12)

    plt.show()


def main(args):

    print(args)

    torch.backends.cudnn.benchmark = True

    # Data loading
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    crop_pct = 0.875
    scale_size = int(math.floor(args.img_size / crop_pct))

    train_loader, val_loader = None, None

    if not args.test_only:
        st = time.time()
        train_set = VOCDetection(
            datadir,
            image_set='train',
            download=True,
            transforms=Compose([
                VOCTargetTransform(VOC_CLASSES),
                RandomResizedCrop((args.img_size, args.img_size), scale=(0.3, 1.0)),
                RandomHorizontalFlip(),
                convert_to_relative,
                ImageTransform(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02)),
                ImageTransform(T.ToTensor()),
                ImageTransform(normalize)
            ])
        )

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, drop_last=True, collate_fn=collate_fn,
            sampler=RandomSampler(train_set), num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn)

        print(f"Training set loaded in {time.time() - st:.2f}s "
              f"({len(train_set)} samples in {len(train_loader)} batches)")

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target)
        return

    if not (args.lr_finder or args.check_setup):
        st = time.time()
        val_set = VOCDetection(
            datadir,
            image_set='val',
            download=True,
            transforms=Compose([
                VOCTargetTransform(VOC_CLASSES),
                Resize(scale_size),
                CenterCrop(args.img_size),
                convert_to_relative,
                ImageTransform(T.ToTensor()),
                ImageTransform(normalize)
            ])
        )

        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=args.batch_size, drop_last=False, collate_fn=collate_fn,
            sampler=SequentialSampler(val_set), num_workers=args.workers, pin_memory=True,
            worker_init_fn=worker_init_fn)

        print(f"Validation set loaded in {time.time() - st:.2f}s ({len(val_set)} samples in {len(val_loader)} batches)")

    model = detection.__dict__[args.model](args.pretrained, num_classes=len(VOC_CLASSES),
                                           pretrained_backbone=True)

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

    trainer = DetectionTrainer(model, train_loader, val_loader, None, optimizer,
                               args.device, args.output_file)

    if args.resume:
        print(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        trainer.load(checkpoint)

    if args.test_only:
        print("Running evaluation")
        eval_metrics = trainer.evaluate()
        print(f"Loc error: {eval_metrics['loc_err']:.2%} | Clf error: {eval_metrics['clf_err']:.2%} | "
              f"Det error: {eval_metrics['det_err']:.2%}")
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
    parser = argparse.ArgumentParser(description='Holocron Detection Training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data_path', type=str, help='path to dataset folder')
    parser.add_argument('--model', default='yolov2', help='model')
    parser.add_argument('--freeze-until', default='backbone', type=str, help='Last layer to freeze')
    parser.add_argument('--device', default=None, type=int, help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=min(os.cpu_count(), 16), type=int,
                        help='number of data loading workers')
    parser.add_argument('--img-size', default=416, type=int, help='image size')
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
