#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Training script for semantic segmentation
'''

import math
import datetime
import time
from pathlib import Path
from fastprogress import master_bar, progress_bar
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms import functional as F
from contiguous_params import ContiguousParams

import holocron
from transforms import (Compose, Resize, ImageTransform, CenterCrop, RandomResizedCrop,
                        RandomHorizontalFlip, convert_to_relative)


VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def train_one_batch(model, x, target, optimizer, criterion, device=None):

    x, target = x.to(device), target.to(device)
    out = model(x)
    batch_loss = criterion(out, target)

    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    return batch_loss.item()


def train_one_epoch(model, optimizer, criterion, scheduler, data_loader, device, master_bar):
    model.train()

    for x, target in progress_bar(data_loader, parent=master_bar):

        x, target = x.to(device), target.to(device)
        out = model(x)
        batch_loss = criterion(out, target)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        master_bar.child.comment = f"Training loss: {batch_loss.item():.4}"


def evaluate(model, data_loader, criterion, device, ignore_index=255):
    model.eval()

    val_loss, mean_iou = 0, 0
    with torch.no_grad():
        for x, target in data_loader:
            x, target = x.to(device), target.to(device)
            out = model(x)

            val_loss += criterion(out, target).item()
            pred = out.argmax(dim=1)
            tmp_iou, num_seg = 0, 0
            for class_idx in torch.unique(target):
                if class_idx != ignore_index:
                    inter = (pred[target == class_idx] == class_idx).sum().item()
                    tmp_iou += inter / ((pred == class_idx) | (target == class_idx)).sum().item()
                    num_seg += 1
            mean_iou += tmp_iou / num_seg

    val_loss /= len(data_loader)
    mean_iou /= len(data_loader)

    return val_loss, mean_iou


def load_data(datadir):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    base_size = 320
    crop_size = 256

    min_size = int(0.5 * base_size)
    max_size = int(2.0 * base_size)

    print("Loading training data")
    st = time.time()
    dataset = VOCSegmentation(datadir, image_set='train', download=True,
                              transforms=Compose([RandomResize(min_size, max_size),
                                                  RandomCrop(crop_size),
                                                  RandomHorizontalFlip(0.5),
                                                  SampleTransform(transforms.ColorJitter(brightness=0.3,
                                                                                         contrast=0.3,
                                                                                         saturation=0.1,
                                                                                         hue=0.02)),
                                                  ToTensor(),
                                                  SampleTransform(normalize)]))

    print("Took", time.time() - st)

    print("Loading validation data")
    st = time.time()
    dataset_test = VOCSegmentation(datadir, image_set='val', download=True,
                                   transforms=Compose([RandomResize(base_size, base_size),
                                                       ToTensor(),
                                                       SampleTransform(normalize)]))

    print("Took", time.time() - st)
    print("Creating data loaders")
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def plot_lr_finder(train_batch, model, data_loader, optimizer, criterion, device,
                   start_lr=1e-7, end_lr=1):

    lrs, losses = holocron.utils.lr_finder(train_batch, model, data_loader,
                                           optimizer, criterion, device, start_lr=start_lr, end_lr=end_lr,
                                           stop_threshold=10, beta=0.95)
    # Plot Loss vs LR
    plt.plot(lrs[10:-5], losses[10:-5])
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Training loss')
    plt.grid(True, linestyle='--', axis='x')
    plt.show()


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

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    dataset, dataset_test, train_sampler, test_sampler = load_data(args.data_path)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                               sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target, ignore_index=255)
        return

    val_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                             sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    kwargs = {}
    if args.freeze_backbone:
        kwargs['norm_layer'] = FrozenBatchNorm2d
    model = holocron.models.__dict__[args.model](args.pretrained, num_classes=len(classes), in_channels=3, **kwargs)
    # Backbone freezing
    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
    model.to(device)

    model_params = ContiguousParams([p for p in model.parameters() if p.requires_grad])
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model_params.contiguous(), args.lr, betas=(0.95, 0.99), eps=1e-6,
                                     weight_decay=args.weight_decay)
    elif args.opt == 'radam':
        optimizer = holocron.optim.RAdam(model_params.contiguous(), args.lr, betas=(0.95, 0.99), eps=1e-6,
                                         weight_decay=args.weight_decay)
    elif args.opt == 'ranger':
        optimizer = Lookahead(holocron.optim.RAdam(model_params.contiguous(), args.lr, betas=(0.95, 0.99), eps=1e-6,
                                                   weight_decay=args.weight_decay))

    loss_weight = torch.ones(len(classes))
    loss_weight[0] = 0.1
    criterion = torch.nn.CrossEntropyLoss(weight=loss_weight, ignore_index=255).to(device)

    if args.lr_finder:
        plot_lr_finder(train_one_batch, model, train_loader, optimizer, criterion, device,
                       start_lr=1e-7, end_lr=1)
        return

    if args.sched == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  patience=3, threshold=5e-3)
    elif args.sched == 'onecycle':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                           total_steps=args.epochs * len(train_loader),
                                                           cycle_momentum=False, div_factor=25, final_div_factor=25e4)

    best_loss = math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['val_loss']

    if args.test_only:
        val_loss, mean_iou = evaluate(model, val_loader, criterion, device=device)
        print(f"Validation loss: {val_loss:.4} | Mean IoU: {mean_iou:.2%}")
        return

    print("Start training")
    start_time = time.time()
    mb = master_bar(range(args.start_epoch, args.epochs))
    for epoch in mb:
        train_one_epoch(model, optimizer, criterion, lr_scheduler, train_loader, device, mb)
        # Check that the optimizer only applies valid ops.
        model_params.assert_buffer_is_valid()
        val_loss, mean_iou = evaluate(model, val_loader, criterion, device=device)
        mb.main_bar.comment = f"Epoch {args.start_epoch+epoch+1}/{args.start_epoch+args.epochs}"
        mb.write(f"Epoch {args.start_epoch+epoch+1}/{args.start_epoch+args.epochs} - "
                 f"Validation loss: {val_loss:.4} | Mean IoU: {mean_iou:.2%}")
        if args.sched == 'plateau':
            lr_scheduler.step(val_loss)
        if val_loss < best_loss:
            if args.output_dir:
                print(f"Validation loss decreased {best_loss:.4} --> {val_loss:.4}: saving state...")
                torch.save(dict(model=model.state_dict(),
                                optimizer=optimizer.state_dict(),
                                lr_scheduler=lr_scheduler.state_dict(),
                                epoch=epoch,
                                val_loss=val_loss),
                           Path(args.output_dir, f"{args.checkpoint}_best_state.pth"))
            best_loss = val_loss

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Holocron Segmentation Training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data_path', type=str, help='path to dataset folder')
    parser.add_argument('--model', default='unet3p', help='model')
    parser.add_argument("--freeze-backbone", dest='freeze_backbone', action='store_true',
                        help="Should the backbone be frozen")
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--loss', default='crossentropy', type=str, help='loss')
    parser.add_argument('--opt', default='adam', type=str, help='optimizer')
    parser.add_argument('--sched', default='onecycle', type=str, help='scheduler')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument("--lr-finder", dest='lr_finder', action='store_true',
                        help="Should you run LR Finder")
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--checkpoint', default='model', help='checkpoint name')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
