# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Training script for object detection"""

import math
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
from codecarbon import track_emissions
from torchvision.datasets import VOCDetection
from torchvision.models import detection as tv_detection
from torchvision.transforms import v2 as T
from torchvision.transforms.v2.functional import InterpolationMode, to_pil_image

if __package__:
    from .transforms import VOCTargetTransform, convert_to_relative
else:
    from transforms import VOCTargetTransform, convert_to_relative

from holocron.models import detection
from holocron.trainer import DetectionTrainer
from holocron.utils.misc import find_image_size
from references._common import (
    add_loading_args,
    create_loader,
    create_optimizer,
    load_checkpoint,
    run_training,
)

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
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.patches import Rectangle  # noqa: PLC0415

    nb_samples = min(num_samples, len(images))
    num_cols = min(nb_samples, 4)
    num_rows = math.ceil(nb_samples / num_cols)
    _, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5))
    for idx in range(nb_samples):
        img = images[idx]
        img *= torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img += torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        img = to_pil_image(img)

        row = int(idx / num_cols)
        col = idx - row * num_cols

        axes[row][col].imshow(img)
        axes[row][col].axis("off")
        for box, label in zip(targets[idx]["boxes"], targets[idx]["labels"], strict=False):
            xmin = int(box[0] * images[idx].shape[-1])
            ymin = int(box[1] * images[idx].shape[-2])
            xmax = int(box[2] * images[idx].shape[-1])
            ymax = int(box[3] * images[idx].shape[-2])

            rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor="lime", facecolor="none")
            axes[row][col].add_patch(rect)
            axes[row][col].text(xmin, ymin, VOC_CLASSES[label.item()], color="lime", fontsize=12)

    plt.show()


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
            transforms=T.Compose([
                VOCTargetTransform(VOC_CLASSES),
                T.Resize((args.img_size, args.img_size), interpolation=interpolation_mode),
                T.RandomHorizontalFlip(),
                *([convert_to_relative] if args.source == "holocron" else []),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                normalize,
                T.ToPureTensor(),
            ]),
        )

        # Suggest size
        if args.find_size:
            print("Looking for optimal image size")
            find_image_size(train_set)
            return

        train_loader = create_loader(
            train_set, args, training=True, collate_fn=collate_fn, worker_init_fn=worker_init_fn
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
            transforms=T.Compose([
                VOCTargetTransform(VOC_CLASSES),
                T.Resize((args.img_size, args.img_size), interpolation=interpolation_mode),
                *([convert_to_relative] if args.source == "holocron" else []),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                normalize,
                T.ToPureTensor(),
            ]),
        )

        val_loader = create_loader(val_set, args, training=False, collate_fn=collate_fn, worker_init_fn=worker_init_fn)

        print(f"Validation set loaded in {time.time() - st:.2f}s ({len(val_set)} samples in {len(val_loader)} batches)")

    if args.source.lower() == "holocron":
        model = detection.__dict__[args.arch](args.pretrained, num_classes=len(VOC_CLASSES))
    elif args.source.lower() == "torchvision":
        model = tv_detection.__dict__[args.arch](args.pretrained, num_classes=len(VOC_CLASSES))

    optimizer = create_optimizer(model, args)

    trainer = DetectionTrainer(
        model,
        train_loader,
        val_loader,
        None,
        optimizer,
        output_file=args.output_file,
        device=args.device,
        amp=args.amp,
        skip_nan_loss=True,
        gradient_clip=0.1,
        gradient_acc=args.grad_acc,
    )

    load_checkpoint(trainer, args.resume)

    run_training(
        trainer,
        args,
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


def get_parser():
    parser = ArgumentParser(description="Holocron Detection Training", formatter_class=ArgumentDefaultsHelpFormatter)

    # Data & model
    group = parser.add_argument_group("Data & model")
    group.add_argument("data_path", type=str, help="path to dataset folder")
    group.add_argument("--arch", default="yolov2", type=str, help="architecture to use")
    group.add_argument("--source", type=str, default="holocron", help="where should the architecture be taken from")
    group.add_argument("--pretrained", action="store_true", help="Use pre-trained models from the modelzoo")
    group.add_argument("--output-file", default="./checkpoints/model.pth", help="path where to save")
    group.add_argument("--resume", default="", help="resume from checkpoint")
    add_loading_args(parser)
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
    group.add_argument("--momentum", default=0.9, type=float, help="SGD momentum")
    group.add_argument("--sched", default="onecycle", type=str, help="Scheduler to be used")
    group.add_argument("--wd", "--weight-decay", default=0, type=float, help="weight decay", dest="weight_decay")
    group.add_argument("--norm-wd", default=None, type=float, help="weight decay of norm parameters")
    # Actions
    group = parser.add_argument_group("Actions")
    group.add_argument("--find-lr", action="store_true", help="Should you run LR Finder")
    group.add_argument("--find-lr-start", default=1e-7, type=float, help="initial LR for LR Finder")
    group.add_argument("--find-lr-end", default=1, type=float, help="final LR for LR Finder")
    group.add_argument("--find-size", dest="find_size", action="store_true", help="Should you run Image size Finder")
    group.add_argument("--check-setup", action="store_true", help="Check your training setup")
    group.add_argument("--show-samples", action="store_true", help="Whether training samples should be displayed")
    group.add_argument("--test-only", help="Only test the model", action="store_true")
    # Experiment tracking
    group = parser.add_argument_group("Experiment tracking")
    group.add_argument("--wb", action="store_true", help="Log to Weights & Biases")
    group.add_argument("--name", type=str, default=None, help="Name of your training experiment")
    group.add_argument("--verbose-codecarbon", action="store_true", help="Show CodeCarbon informational logs")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    track_emissions(log_level="info" if args.verbose_codecarbon else "error")(main)(args)
