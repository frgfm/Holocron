#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Training script for object detection
'''

import math
import datetime
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
from fastprogress import master_bar, progress_bar
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torchvision.ops.boxes import box_iou
from torchvision.transforms import functional as F

import holocron
from transforms import (Compose, VOCTargetTransform, Resize, ImageTransform, CenterCrop, RandomResizedCrop,
                        RandomHorizontalFlip, convert_to_relative)


VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def collate_fn(batch):
    imgs, target = zip(*batch)
    return imgs, target


def train_one_batch(model, x, target, optimizer, criterion=None, device=None):

    x = [_x.to(device) for _x in x]
    target = [{k: v.to(device) for k, v in t.items()} for t in target]
    loss_dict = model(x, [t['boxes'] for t in target], [t['labels'] for t in target])
    batch_loss = sum(loss_dict.values())

    optimizer.zero_grad()
    batch_loss.backward()
    # Safeguard for Gradient explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), .25)
    optimizer.step()

    return batch_loss.item()


def train_one_epoch(model, criterion, optimizer, scheduler, data_loader, device, master_bar):
    model.train()

    for x, target in progress_bar(data_loader, parent=master_bar):

        x = [_x.to(device) for _x in x]
        target = [{k: v.to(device) for k, v in t.items()} for t in target]
        loss_dict = model(x, target)
        batch_loss = sum(loss_dict.values())

        optimizer.zero_grad()
        batch_loss.backward()
        # Safeguard for Gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), .25)
        optimizer.step()
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        master_bar.child.comment = '|'.join(f"{k}: {v.item():.4}" for k, v in loss_dict.items())


def assign_iou(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Assigns boxes by IoU"""
    iou = box_iou(gt_boxes, pred_boxes)
    iou = iou.max(dim=1)
    gt_kept = iou.values >= iou_threshold
    assign_unique = torch.unique(iou.indices[gt_kept])
    # Filter
    if iou.indices[gt_kept].shape[0] == assign_unique.shape[0]:
        return torch.arange(gt_boxes.shape[0])[gt_kept], iou.indices[gt_kept]
    else:
        gt_indices, pred_indices = [], []
        for pred_idx in assign_unique:
            selection = iou.values[gt_kept][iou.indices[gt_kept] == pred_idx].argmax()
            gt_indices.append(torch.arange(gt_boxes.shape[0])[gt_kept][selection].item())
            pred_indices.append(iou.indices[gt_kept][selection].item())
        return gt_indices, pred_indices


def evaluate(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    loc_assigns = 0
    correct, clf_error, loc_fn, loc_fp, nb_boxes = 0, 0, 0, 0, 0
    with torch.no_grad():
        for x, target in data_loader:
            x = [_x.to(device) for _x in x]
            detections = model(x)

            for dets, t in zip(detections, target):
                if t['boxes'].shape[0] > 0 and dets['boxes'].shape[0] > 0:
                    t = {k: v.to(device) for k, v in t.items()}
                    gt_indices, pred_indices = assign_iou(t['boxes'], dets['boxes'], iou_threshold)
                    loc_assigns += len(gt_indices)
                    _correct = (t['labels'][gt_indices] == dets['labels'][pred_indices]).sum().item()
                else:
                    gt_indices, pred_indices = [], []
                    _correct = 0
                correct += _correct
                clf_error += len(gt_indices) - _correct
                loc_fn += t['boxes'].shape[0] - len(gt_indices)
                loc_fp += dets['boxes'].shape[0] - len(pred_indices)
            nb_boxes += sum(t['boxes'].shape[0] for t in target)

    nb_preds = nb_boxes - loc_fn + loc_fp
    # Localization
    loc_err = 1 - 2 * loc_assigns / (nb_preds + nb_boxes) if nb_preds + nb_boxes > 0 else 1.
    # Classification
    clf_err = 1 - correct / loc_assigns if loc_assigns > 0 else 1.
    # End-to-end
    det_err = 1 - 2 * correct / (nb_preds + nb_boxes) if nb_preds + nb_boxes > 0 else 1.
    return loc_err, clf_err, det_err


def load_data(datadir):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("Loading training data")
    st = time.time()
    dataset = VOCDetection(datadir, image_set='train', download=True,
                           transforms=Compose([VOCTargetTransform(classes),
                                              Resize(512), RandomResizedCrop(416), RandomHorizontalFlip(),
                                              convert_to_relative,
                                              ImageTransform(transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                                                                    saturation=0.1, hue=0.02)),
                                              ImageTransform(transforms.ToTensor()), ImageTransform(normalize)]))

    print("Took", time.time() - st)

    print("Loading validation data")
    st = time.time()
    dataset_test = VOCDetection(datadir, image_set='val', download=True,
                                transforms=Compose([VOCTargetTransform(classes),
                                                    Resize(416), CenterCrop(416),
                                                    convert_to_relative,
                                                    ImageTransform(transforms.ToTensor()), ImageTransform(normalize)]))

    print("Took", time.time() - st)
    print("Creating data loaders")
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def plot_lr_finder(train_batch, model, data_loader, optimizer, criterion, device,
                   start_lr=1e-7, end_lr=1, loss_margin=1e-2):

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
    sys.exit()


def plot_samples(images, targets):
    # Unnormalize image
    nb_samples = 4
    fig, axes = plt.subplots(1, nb_samples, figsize=(20, 5))
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
            axes[idx].text(xmin, ymin, classes[label.item()], color='lime', fontsize=12)

    plt.show()


def main(args):

    print(args)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    dataset, dataset_test, train_sampler, test_sampler = load_data(args.data_path)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate_fn,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target)
        return

    val_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, collate_fn=collate_fn,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    model = holocron.models.__dict__[args.model](args.pretrained, num_classes=len(classes))
    model.to(device)

    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.95, 0.99), eps=1e-6,
                                     weight_decay=args.weight_decay)
    elif args.opt == 'radam':
        optimizer = holocron.optim.RAdam(model.parameters(), args.lr, betas=(0.95, 0.99), eps=1e-6,
                                         weight_decay=args.weight_decay)
    elif args.opt == 'ranger':
        optimizer = Lookahead(holocron.optim.RAdam(model.parameters(), args.lr, betas=(0.95, 0.99), eps=1e-6,
                                                   weight_decay=args.weight_decay))

    if args.lr_finder:
        plot_lr_finder(train_one_batch, model, train_loader, optimizer, device,
                       start_lr=1e-7, end_lr=1)
        return

    if args.sched == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  patience=3, threshold=5e-3)
    elif args.sched == 'onecycle':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                           total_steps=args.epochs * len(train_loader),
                                                           cycle_momentum=False, div_factor=25, final_div_factor=25e4)

    best_error = math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        best_error = checkpoint['det_err']

    if args.test_only:
        recall, precision = evaluate(model, val_loader, device=device)
        print(f"Recall: {recall:.2%} | Precision: {precision:.2%}")
        return

    print("Start training")
    start_time = time.time()
    mb = master_bar(range(args.start_epoch, args.epochs))
    for epoch in mb:
        train_one_epoch(model, optimizer, lr_scheduler, train_loader, device, mb)
        loc_err, clf_err, det_err = evaluate(model, val_loader, device=device)
        mb.main_bar.comment = f"Epoch {args.start_epoch+epoch+1}/{args.start_epoch+args.epochs}"
        mb.write(f"Epoch {args.start_epoch+epoch+1}/{args.start_epoch+args.epochs} - "
                 f"Loc error: {loc_err:.2%} | Clf error: {clf_err:.2%} | Det error: {det_err:.2%}")
        if args.sched == 'plateau':
            lr_scheduler.step(det_err)
        if det_err < best_error:
            if args.output_dir:
                print(f"Validation loss decreased {best_error:.4} --> {det_err:.4}: saving state...")
                torch.save(dict(model=model.state_dict(),
                                optimizer=optimizer.state_dict(),
                                lr_scheduler=lr_scheduler.state_dict(),
                                epoch=epoch,
                                det_err=det_err),
                           Path(args.output_dir, f"{args.checkpoint}_best_state.pth"))
            best_error = det_err

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data_path', type=str, help='path to dataset folder')
    parser.add_argument('--model', default='darknet19', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--loss', default='crossentropy', type=str, help='loss')
    parser.add_argument('--opt', default='adam', type=str, help='optimizer')
    parser.add_argument('--sched', default='plateau', type=str, help='scheduler')
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
