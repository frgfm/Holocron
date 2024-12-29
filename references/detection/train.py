# Copyright (C) 2019-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
Training script for object detection
"""

import datetime
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import wandb
from codecarbon import track_emissions
from matplotlib.patches import Rectangle
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision import transforms as T
from torchvision.datasets import VOCDetection
from torchvision.models import detection as tv_detection
from torchvision.transforms.functional import InterpolationMode, to_pil_image
from transforms import Compose, ImageTransform, RandomHorizontalFlip, Resize, VOCTargetTransform, convert_to_relative

import holocron
from holocron.models import detection
from holocron.trainer import DetectionTrainer
from holocron.utils.misc import find_image_size

VOC_CLASSES = [
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
    np.random.default_rng((worker_id + torch.initial_seed()) % np.iinfo(np.int32).max)


def collate_fn(batch):
    imgs, target = zip(*batch, strict=False)
    return imgs, target


def plot_samples(images, targets, num_samples=8):
    # Unnormalize image
    nb_samples = min(num_samples, len(images))
    num_cols = min(nb_samples, 4)
    num_rows = int(math.ceil(nb_samples / num_cols))
    _, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5))
    for idx in range(nb_samples):
        img = images[idx]
        img *= torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img += torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        img = to_pil_image(img)

        _row = int(idx / num_cols)
        _col = idx - _row * num_cols

        axes[_row][_col].imshow(img)
        axes[_row][_col].axis("off")
        for box, label in zip(targets[idx]["boxes"], targets[idx]["labels"], strict=False):
            xmin = int(box[0] * images[idx].shape[-1])
            ymin = int(box[1] * images[idx].shape[-2])
            xmax = int(box[2] * images[idx].shape[-1])
            ymax = int(box[3] * images[idx].shape[-2])

            rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor="lime", facecolor="none")
            axes[_row][_col].add_patch(rect)
            axes[_row][_col].text(xmin, ymin, VOC_CLASSES[label.item()], color="lime", fontsize=12)

    plt.show()


@track_emissions()
def main(args):
    print(args)

    torch.backends.cudnn.benchmark = True

    # Data loading
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader, val_loader = None, None

    interpolation_mode = InterpolationMode.BILINEAR

    if not args.test_only:
        st = time.time()
        train_set = VOCDetection(
            args.data_path,
            image_set="train",
            download=True,
            transforms=Compose([
                VOCTargetTransform(VOC_CLASSES),
                Resize((args.img_size, args.img_size), interpolation=interpolation_mode),
                RandomHorizontalFlip(),
                convert_to_relative if args.source == "holocron" else lambda x, y: (x, y),
                ImageTransform(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02)),
                ImageTransform(T.PILToTensor()),
                ImageTransform(T.ConvertImageDtype(torch.float32)),
                ImageTransform(normalize),
            ]),
        )

        # Suggest size
        if args.find_size:
            print("Looking for optimal image size")
            find_image_size(train_set)
            return

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=RandomSampler(train_set),
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

        print(
            f"Training set loaded in {time.time() - st:.2f}s ({len(train_set)} samples in {len(train_loader)} batches)"
        )

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target)
        return

    if not (args.find_lr or args.check_setup):
        st = time.time()
        val_set = VOCDetection(
            args.data_path,
            image_set="val",
            download=True,
            transforms=Compose([
                VOCTargetTransform(VOC_CLASSES),
                Resize((args.img_size, args.img_size), interpolation=interpolation_mode),
                convert_to_relative if args.source == "holocron" else lambda x, y: (x, y),
                ImageTransform(T.PILToTensor()),
                ImageTransform(T.ConvertImageDtype(torch.float32)),
                ImageTransform(normalize),
            ]),
        )

        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            drop_last=False,
            collate_fn=collate_fn,
            sampler=SequentialSampler(val_set),
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

        print(f"Validation set loaded in {time.time() - st:.2f}s ({len(val_set)} samples in {len(val_loader)} batches)")

    if args.source.lower() == "holocron":
        model = detection.__dict__[args.arch](args.pretrained, num_classes=len(VOC_CLASSES))
    elif args.source.lower() == "torchvision":
        model = tv_detection.__dict__[args.arch](args.pretrained, num_classes=len(VOC_CLASSES))

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
    trainer = DetectionTrainer(
        model,
        train_loader,
        val_loader,
        None,
        optimizer,
        args.device,
        args.output_file,
        amp=args.amp,
        skip_nan_loss=True,
        gradient_clip=0.1,
        gradient_acc=args.grad_acc,
        on_epoch_end=log_wb,
    )

    if args.resume:
        print(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        trainer.load(checkpoint)

    if args.test_only:
        print("Running evaluation")
        eval_metrics = trainer.evaluate()
        print(trainer._eval_metrics_str(eval_metrics))
        return

    if args.find_lr:
        print("Looking for optimal LR")
        trainer.find_lr(args.freeze_until, norm_weight_decay=args.norm_wd, num_it=min(len(train_loader), 100))
        trainer.plot_recorder()
        return

    if args.check_setup:
        print("Checking batch overfitting")
        trainer.check_setup(
            args.freeze_until, args.lr, norm_weight_decay=args.norm_wd, num_it=min(len(train_loader), 100)
        )
        return

    # Training monitoring
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.arch}-{current_time}" if args.name is None else args.name

    # W&B
    if args.wb:
        run = wandb.init(
            name=exp_name,
            project="holocron-object-detection",
            config={
                "learning_rate": args.lr,
                "scheduler": args.sched,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "architecture": args.arch,
                "source": args.source,
                "input_size": args.img_size,
                "optimizer": args.opt,
                "dataset": "PASCAL VOC2012 Detection",
            },
        )

    print("Start training")
    start_time = time.time()
    trainer.fit_n_epochs(args.epochs, args.lr, args.freeze_until, args.sched, norm_weight_decay=args.norm_wd)
    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {total_time_str}")

    if args.wb:
        run.finish()


def get_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Holocron Detection Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data & model
    group = parser.add_argument_group("Data & model")
    group.add_argument("data_path", type=str, help="path to dataset folder")
    group.add_argument("--arch", default="yolov2", type=str, help="architecture to use")
    group.add_argument("--source", type=str, default="holocron", help="where should the architecture be taken from")
    group.add_argument("--pretrained", action="store_true", help="Use pre-trained models from the modelzoo")
    group.add_argument("--output-file", default="./checkpoints/model.pth", help="path where to save")
    group.add_argument("--resume", default="", help="resume from checkpoint")
    # Hardware
    group = parser.add_argument_group("Hardware")
    group.add_argument("--device", default=None, type=int, help="device")
    group.add_argument("--amp", help="Use Automatic Mixed Precision", action="store_true")
    # Data loading
    group = parser.add_argument_group("Data loading")
    group.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
    group.add_argument(
        "-j", "--workers", default=min(os.cpu_count(), 16), type=int, help="number of data loading workers"
    )
    # Transformations
    group = parser.add_argument_group("Transformations")
    group.add_argument("--img-size", default=416, type=int, help="image size")
    # Optimization
    group = parser.add_argument_group("Optimization")
    group.add_argument("--epochs", default=20, type=int, help="number of total epochs to run")
    group.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    group.add_argument("--freeze-until", default=None, type=str, help="Last layer to freeze")
    group.add_argument("--grad-acc", default=1, type=int, help="Number of batches to accumulate the gradient of")
    group.add_argument("--opt", default="adamp", type=str, help="optimizer")
    group.add_argument("--sched", default="onecycle", type=str, help="Scheduler to be used")
    group.add_argument("--wd", "--weight-decay", default=0, type=float, help="weight decay", dest="weight_decay")
    group.add_argument("--norm-wd", default=None, type=float, help="weight decay of norm parameters")
    # Actions
    group = parser.add_argument_group("Actions")
    group.add_argument("--find-lr", action="store_true", help="Should you run LR Finder")
    group.add_argument("--find-size", dest="find_size", action="store_true", help="Should you run Image size Finder")
    group.add_argument("--check-setup", action="store_true", help="Check your training setup")
    group.add_argument("--show-samples", action="store_true", help="Whether training samples should be displayed")
    group.add_argument("--test-only", help="Only test the model", action="store_true")
    # Experiment tracking
    group = parser.add_argument_group("Experiment tracking")
    group.add_argument("--wb", action="store_true", help="Log to Weights & Biases")
    group.add_argument("--name", type=str, default=None, help="Name of your training experiment")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
