#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Training script for image classification
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

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms

import holocron


def train_one_batch(model, x, target, optimizer, criterion, device):

    x, target = x.to(device), target.to(device)
    output = model(x)
    batch_loss = criterion(output, target)

    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    return batch_loss.item()


def train_one_epoch(model, criterion, optimizer, scheduler, data_loader, device, master_bar):
    model.train()

    for x, target in progress_bar(data_loader, parent=master_bar):

        x, target = x.to(device), target.to(device)
        output = model(x)
        batch_loss = criterion(output, target)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        master_bar.child.comment = f"Training loss: {batch_loss.item():.4}"


def evaluate(model, criterion, data_loader, device):
    model.eval()
    val_loss, top1, top5, nb_imgs = 0, 0, 0, 0
    with torch.no_grad():
        for x, target in tqdm(data_loader):
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(x)
            batch_loss = criterion(output, target)

            val_loss += batch_loss.item()

            pred = output.topk(5, dim=1)[1]
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            top1 += correct[:, 0].sum().item()
            top5 += correct.any(dim=1).sum().item()
            nb_imgs += x.shape[0]

    val_loss /= len(data_loader)

    return val_loss, top1 / nb_imgs, top5 / nb_imgs


def load_data(traindir, valdir, img_size=224, crop_pct=0.875):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    scale_size = min(int(math.floor(img_size / crop_pct)), 320)

    print("Loading training data")
    st = time.time()
    dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02),
            transforms.ToTensor(), normalize,
            transforms.RandomErasing(p=0.9, value='random')
        ]))
    print("Took", time.time() - st)

    print("Loading validation data")
    eval_tf = []
    if scale_size < 320:
        eval_tf.append(transforms.Resize(scale_size))
    eval_tf.extend([transforms.CenterCrop(img_size), transforms.ToTensor(), normalize])
    dataset_test = torchvision.datasets.ImageFolder(valdir, transforms.Compose(eval_tf))

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


def main(args):

    print(args)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, img_size=args.img_size)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    model = holocron.models.__dict__[args.model](args.pretrained, num_classes=len(dataset.classes))
    model.to(device)

    if args.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'label_smoothing':
        criterion = holocron.nn.LabelSmoothingCrossEntropy()

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
        plot_lr_finder(train_one_batch, model, train_loader, optimizer, criterion, device,
                       start_lr=1e-7, end_lr=1)

    if args.sched == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  patience=3, threshold=5e-3)
    elif args.sched == 'onecycle':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.epochs * len(train_loader),
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
        val_loss, acc1, acc5 = evaluate(model, criterion, val_loader, device=device)
        print(f"Validation loss: {val_loss:.4} (Acc@1: {acc1:.2%}, Acc@5: {acc5:.2%})")
        return

    print("Start training")
    start_time = time.time()
    mb = master_bar(range(args.start_epoch, args.epochs))
    for epoch in mb:
        train_one_epoch(model, criterion, optimizer, lr_scheduler, train_loader, device, mb)
        val_loss, acc1, acc5 = evaluate(model, criterion, val_loader, device=device)
        if args.sched == 'plateau':
            lr_scheduler.step(val_loss)
        mb.first_bar.comment = f"Epoch {args.start_epoch+epoch+1}/{args.start_epoch+args.epochs}"
        mb.write(f"Epoch {args.start_epoch+epoch+1}/{args.start_epoch+args.epochs} - "
                 f"Validation loss: {val_loss:.4} "
                 f"(Acc@1: {acc1:.2%}, Acc@5: {acc5:.2%})")
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
    parser.add_argument('--img-size', default=224, type=int, help='image size')
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
