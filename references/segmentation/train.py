# Copyright (C) 2019-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
Training script for semantic segmentation
"""

import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import wandb
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision import transforms as T
from torchvision.datasets import VOCSegmentation
from torchvision.models import segmentation as tv_segmentation
from torchvision.transforms.functional import InterpolationMode, to_pil_image
from transforms import Compose, ImageTransform, RandomCrop, RandomHorizontalFlip, RandomResize, Resize, ToTensor

import holocron
from holocron.models import segmentation
from holocron.trainer import SegmentationTrainer

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def worker_init_fn(worker_id: int) -> None:
    np.random.seed((worker_id + torch.initial_seed()) % np.iinfo(np.int32).max)


def plot_samples(images, targets, ignore_index=None):
    # Unnormalize image
    nb_samples = 4
    _, axes = plt.subplots(2, nb_samples, figsize=(20, 5))
    for idx in range(nb_samples):
        img = images[idx]
        img *= torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img += torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        img = to_pil_image(img)
        target = targets[idx]
        if isinstance(ignore_index, int):
            target[target == ignore_index] = 0

        axes[0][idx].imshow(img)
        axes[0][idx].axis("off")
        axes[0][idx].set_title("Input image")
        axes[1][idx].imshow(target)
        axes[1][idx].axis("off")
        axes[1][idx].set_title("Target")
    plt.show()


def plot_predictions(images, preds, targets, ignore_index=None):
    # Unnormalize image
    nb_samples = 4
    _, axes = plt.subplots(3, nb_samples, figsize=(20, 5))
    for idx in range(nb_samples):
        img = images[idx]
        img *= torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img += torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        img = to_pil_image(img)
        # Target
        target = targets[idx]
        if isinstance(ignore_index, int):
            target[target == ignore_index] = 0
        # Prediction
        pred = preds[idx].detach().cpu().argmax(dim=0)

        axes[0][idx].imshow(img)
        axes[0][idx].axis("off")
        axes[0][idx].set_title("Input image")
        axes[1][idx].imshow(target)
        axes[1][idx].axis("off")
        axes[1][idx].set_title("Target")
        axes[2][idx].imshow(pred)
        axes[2][idx].axis("off")
        axes[2][idx].set_title("Prediction")
    plt.show()


def main(args):

    print(args)

    torch.backends.cudnn.benchmark = True

    # Data loading
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    base_size = 320
    crop_size = 256
    min_size, max_size = int(0.5 * base_size), int(2.0 * base_size)

    interpolation_mode = InterpolationMode.BILINEAR

    train_loader, val_loader = None, None
    if not args.test_only:
        st = time.time()
        train_set = VOCSegmentation(
            args.data_path,
            image_set="train",
            download=True,
            transforms=Compose(
                [
                    RandomResize(min_size, max_size, interpolation_mode),
                    RandomCrop(crop_size),
                    RandomHorizontalFlip(0.5),
                    ImageTransform(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02)),
                    ToTensor(),
                    ImageTransform(normalize),
                ]
            ),
        )

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            drop_last=True,
            sampler=RandomSampler(train_set),
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

        print(
            f"Training set loaded in {time.time() - st:.2f}s "
            f"({len(train_set)} samples in {len(train_loader)} batches)"
        )

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target, ignore_index=255)
        return

    if not (args.find_lr or args.check_setup):
        st = time.time()
        val_set = VOCSegmentation(
            args.data_path,
            image_set="val",
            download=True,
            transforms=Compose(
                [Resize((crop_size, crop_size), interpolation_mode), ToTensor(), ImageTransform(normalize)]
            ),
        )

        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            drop_last=False,
            sampler=SequentialSampler(val_set),
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

        print(f"Validation set loaded in {time.time() - st:.2f}s ({len(val_set)} samples in {len(val_loader)} batches)")

    if args.source.lower() == "holocron":
        model = segmentation.__dict__[args.arch](args.pretrained, num_classes=len(VOC_CLASSES))
    elif args.source.lower() == "torchvision":
        model = tv_segmentation.__dict__[args.arch](args.pretrained, num_classes=len(VOC_CLASSES))

    # Loss setup
    loss_weight = None
    if isinstance(args.bg_factor, float) and args.bg_factor != 1:
        loss_weight = torch.ones(len(VOC_CLASSES))
        loss_weight[0] = args.bg_factor
    if args.loss == "crossentropy":
        criterion = nn.CrossEntropyLoss(weight=loss_weight, ignore_index=255, label_smoothing=args.label_smoothing)
    elif args.loss == "focal":
        criterion = holocron.nn.FocalLoss(weight=loss_weight, ignore_index=255)
    elif args.loss == "mc":
        criterion = holocron.nn.MutualChannelLoss(weight=loss_weight, ignore_index=255, xi=3)

    # Optimizer setup
    model_params = [p for p in model.parameters() if p.requires_grad]
    if args.opt == "sgd":
        optimizer = torch.optim.SGD(model_params, args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.opt == "radam":
        optimizer = holocron.optim.RAdam(
            model_params, args.lr, betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay
        )
    elif args.opt == "adamp":
        optimizer = holocron.optim.AdamP(
            model_params, args.lr, betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay
        )
    elif args.opt == "adabelief":
        optimizer = holocron.optim.AdaBelief(
            model_params, args.lr, betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay
        )

    log_wb = lambda metrics: wandb.log(metrics) if args.wb else None
    trainer = SegmentationTrainer(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        args.device,
        args.output_file,
        num_classes=len(VOC_CLASSES),
        amp=args.amp,
        on_epoch_end=log_wb,
    )
    if args.resume:
        print(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        trainer.load(checkpoint)

    if args.show_preds:
        x, target = next(iter(train_loader))
        with torch.no_grad():
            if isinstance(args.device, int):
                x = x.cuda()
            trainer.model.eval()
            preds = trainer.model(x)
        plot_predictions(x.cpu(), preds.cpu(), target, ignore_index=255)
        return

    if args.test_only:
        print("Running evaluation")
        eval_metrics = trainer.evaluate()
        print(f"Validation loss: {eval_metrics['val_loss']:.4} (Mean IoU: {eval_metrics['mean_iou']:.2%})")
        return

    if args.find_lr:
        print("Looking for optimal LR")
        trainer.find_lr(args.freeze_until, norm_weight_decay=args.norm_weight_decay, num_it=min(len(train_loader), 100))
        trainer.plot_recorder()
        return

    if args.check_setup:
        print("Checking batch overfitting")
        is_ok = trainer.check_setup(
            args.freeze_until, args.lr, norm_weight_decay=args.norm_weight_decay, num_it=min(len(train_loader), 100)
        )
        print(is_ok)
        return

    # Training monitoring
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.arch}-{current_time}" if args.name is None else args.name

    # W&B
    if args.wb:

        run = wandb.init(
            name=exp_name,
            project="holocron-semantic-segmentation",
            config={
                "learning_rate": args.lr,
                "scheduler": args.sched,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "architecture": args.arch,
                "source": args.source,
                "input_size": 256,
                "optimizer": args.opt,
                "dataset": "Pascal VOC2012 Segmentation",
                "loss": args.loss,
            },
        )

    print("Start training")
    start_time = time.time()
    trainer.fit_n_epochs(args.epochs, args.lr, args.freeze_until, args.sched, norm_weight_decay=args.norm_weight_decay)
    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {total_time_str}")

    if args.wb:
        run.finish()


def get_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Holocron Segmentation Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("data_path", type=str, help="path to dataset folder")
    parser.add_argument("--name", type=str, default=None, help="Name of your training experiment")
    parser.add_argument("--arch", type=str, default="unet3p", help="architecture to use")
    parser.add_argument("--source", type=str, default="holocron", help="where should the architecture be taken from")
    parser.add_argument("--freeze-until", default=None, type=str, help="Last layer to freeze")
    parser.add_argument("--device", default=None, type=int, help="device")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
    parser.add_argument("--epochs", default=20, type=int, help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=min(os.cpu_count(), 16), type=int, help="number of data loading workers"
    )
    parser.add_argument("--loss", default="crossentropy", type=str, help="loss")
    parser.add_argument("--label-smoothing", default=0.1, type=float, help="label smoothing")
    parser.add_argument("--bg-factor", default=1, type=float, help="Class weight of background in the loss")
    parser.add_argument("--opt", default="adam", type=str, help="optimizer")
    parser.add_argument("--sched", default="onecycle", type=str, help="scheduler")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--wd", "--weight-decay", default=0, type=float, help="weight decay", dest="weight_decay")
    parser.add_argument("--norm-wd", default=None, type=float, help="weight decay of norm parameters")
    parser.add_argument("--find-lr", action="store_true", help="Should you run LR Finder")
    parser.add_argument("--check-setup", action="store_true", help="Check your training setup")
    parser.add_argument("--output-file", default="./model.pth", help="path where to save")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--show-samples", action="store_true", help="Whether training samples should be displayed")
    parser.add_argument("--show-preds", action="store_true", help="Whether one batch predictions should be displayed")
    parser.add_argument("--test-only", action="store_true", help="Only test the model")
    parser.add_argument("--pretrained", action="store_true", help="Use pre-trained models from the model zoo")
    parser.add_argument("--amp", action="store_true", help="Use Automatic Mixed Precision")
    parser.add_argument("--wb", action="store_true", help="Log to Weights & Biases")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
