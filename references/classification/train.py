# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Training script for image classification"""

import logging
import math
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.transforms import v2 as T
from torchvision.transforms.v2.functional import InterpolationMode, to_pil_image

from holocron.models import classification
from holocron.models.presets import CIFAR10 as CIF10
from holocron.models.presets import IMAGENETTE
from holocron.trainer import ClassificationTrainer
from holocron.utils.data import Mixup
from holocron.utils.misc import find_image_size
from references._common import (
    add_loading_args,
    create_loader,
    create_optimizer,
    load_checkpoint,
    run_training,
)

# Prevent the annoying console log of codecarbon
logger = logging.getLogger("codecarbon")
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(name)s %(levelname)s %(asctime)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.ERROR)


def worker_init_fn(worker_id: int) -> None:
    np.random.default_rng((worker_id + torch.initial_seed()) % np.iinfo(np.int32).max)


def collate_mixup(batch, *, num_classes, alpha):
    return Mixup(num_classes, alpha=alpha)(*default_collate(batch))


def scheduler_kwargs(args):
    return {"div_factor": 100, "pct_start": 0.1} if args.sched == "onecycle" else {}


def create_trainer(model, train_loader, val_loader, args, on_epoch_end=None):
    return ClassificationTrainer(
        model,
        train_loader,
        val_loader,
        nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
        create_optimizer(model, args),
        output_file=args.output_file,
        device=args.device,
        gradient_acc=args.grad_acc,
        amp=args.amp,
        on_epoch_end=on_epoch_end,
    )


def plot_samples(images, targets, num_samples=8):
    # Unnormalize image
    import matplotlib.pyplot as plt  # noqa: PLC0415

    nb_samples = min(num_samples, images.shape[0])
    num_cols = min(nb_samples, 4)
    num_rows = math.ceil(nb_samples / num_cols)
    _, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5))
    for idx in range(nb_samples):
        img = images[idx]
        img *= torch.tensor(IMAGENETTE.std).view(-1, 1, 1)
        img += torch.tensor(IMAGENETTE.mean).view(-1, 1, 1)
        img = to_pil_image(img)

        row = int(idx / num_cols)
        col = idx - row * num_cols

        axes[row][col].imshow(img)
        axes[row][col].axis("off")
        if targets.ndim == 1:
            axes[row][col].set_title(IMAGENETTE.classes[targets[idx].item()])
        else:
            class_idcs = torch.where(targets[idx] > 0)[0]
            info = [f"{IMAGENETTE.classes[idx_.item()]} ({targets[idx, idx_]:.2f})" for idx_ in class_idcs]
            axes[row][col].set_title(" ".join(info))

    plt.show()


def imagefolder_transforms(args):
    normalize = T.Normalize(mean=IMAGENETTE.mean, std=IMAGENETTE.std)
    interpolation = InterpolationMode.BILINEAR
    return (
        T.Compose([
            T.RandomResizedCrop(args.train_crop_size, scale=(0.3, 1.0), interpolation=interpolation),
            T.RandomHorizontalFlip(),
            T.TrivialAugmentWide(interpolation=interpolation),
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float32),
            normalize,
            T.RandomErasing(p=args.random_erase, scale=(0.02, 0.2), value="random"),
        ]),
        T.Compose([
            T.Resize(args.val_resize_size, interpolation=interpolation),
            T.CenterCrop(args.val_crop_size),
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float32),
            normalize,
        ]),
    )


def main(args):
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    if args.benchmark:
        from references.classification._benchmark import run_benchmark  # noqa: PLC0415

        return run_benchmark(args, _train)
    return _train(args)


def _train(args, on_epoch_end=None):
    print(args)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Data loading
    train_loader, val_loader = None, None

    normalize = T.Compose([
        T.Lambda(torch.Tensor.contiguous),
        T.Normalize(
            mean=IMAGENETTE.mean if args.dataset.lower() == "imagenette" else CIF10.mean,
            std=IMAGENETTE.std if args.dataset.lower() == "imagenette" else CIF10.std,
        ),
    ])

    interpolation = InterpolationMode.BILINEAR
    train_transform, val_transform = imagefolder_transforms(args)

    num_classes = None
    if not args.test_only:
        st = time.time()
        if args.dataset.lower() == "imagenette":
            train_set = ImageFolder(
                Path(args.data_path).joinpath("train"),
                train_transform,
            )
        else:
            cifar_version = CIFAR100 if args.dataset.lower() == "cifar100" else CIFAR10
            train_set = cifar_version(
                args.data_path,
                True,
                T.Compose([
                    T.RandomHorizontalFlip(),
                    T.TrivialAugmentWide(interpolation=interpolation),
                    T.PILToTensor(),
                    T.ConvertImageDtype(torch.float32),
                    normalize,
                    T.RandomErasing(p=args.random_erase, value="random"),
                ]),
                download=True,
            )

        # Suggest size
        if args.find_size:
            print("Looking for optimal image size")
            find_image_size(train_set)
            return None

        num_classes = len(train_set.classes)
        collate_fn = default_collate
        if args.mixup_alpha > 0:
            collate_fn = Mixup(len(train_set.classes), alpha=args.mixup_alpha).collate
        train_loader = create_loader(
            train_set, args, training=True, worker_init_fn=worker_init_fn, collate_fn=collate_fn
        )

        print(
            f"Training set loaded in {time.time() - st:.2f}s ({len(train_set)} samples in {len(train_loader)} batches)"
        )

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target)
        return None

    if not (args.find_lr or args.check_setup):
        st = time.time()
        if args.dataset.lower() == "imagenette":
            val_set = ImageFolder(
                Path(args.data_path).joinpath("val"),
                val_transform,
            )
        else:
            cifar_version = CIFAR100 if args.dataset.lower() == "cifar100" else CIFAR10
            val_set = cifar_version(
                args.data_path,
                False,
                T.Compose([T.PILToTensor(), T.ConvertImageDtype(torch.float32), normalize]),
                download=True,
            )
        num_classes = len(val_set.classes)

        val_loader = create_loader(val_set, args, training=False, worker_init_fn=worker_init_fn)

        print(f"Validation set loaded in {time.time() - st:.2f}s ({len(val_set)} samples in {len(val_loader)} batches)")

    model = classification.__dict__[args.arch](args.pretrained, num_classes=num_classes)

    trainer = create_trainer(model, train_loader, val_loader, args, on_epoch_end)
    load_checkpoint(trainer, args.resume)

    if args.plot_loss and not args.test_only:
        print("Checking top losses")
        trainer.plot_top_losses(IMAGENETTE["mean"], IMAGENETTE["std"], IMAGENETTE["classes"])
        return None

    run_training(
        trainer,
        args,
        project="holocron-image-classification",
        config={
            "learning_rate": args.lr,
            "scheduler": args.sched,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "architecture": args.arch,
            "train_crop_size": args.train_crop_size,
            "val_resize_size": args.val_resize_size,
            "val_crop_size": args.val_crop_size,
            "optimizer": args.opt,
            "dataset": args.dataset,
            "loss": "crossentropy",
            "label_smoothing": args.label_smoothing,
            "mixup_alpha": args.mixup_alpha,
            "seed": args.seed,
        },
        **scheduler_kwargs(args),
    )
    return trainer


def get_parser():
    parser = ArgumentParser(
        description="Holocron Classification Training", formatter_class=ArgumentDefaultsHelpFormatter
    )

    # Data & model
    group = parser.add_argument_group("Data & model")
    group.add_argument("data_path", type=str, help="path to dataset folder")
    group.add_argument("--dataset", default="imagenette", type=str, help="dataset to train on")
    group.add_argument("--arch", default="darknet19", type=str, help="architecture to use")
    group.add_argument("--pretrained", action="store_true", help="Use pre-trained models from the modelzoo")
    group.add_argument("--output-file", default="./checkpoints/checkpoint.pth", help="path where to save")
    group.add_argument("--resume", default="", help="resume from checkpoint")
    group.add_argument("--seed", default=0, type=int, help="random seed")
    add_loading_args(parser)
    parser.add_argument("--threads", type=int, help="CPU intra-op threads; defaults to PyTorch's current setting")
    # Transformations
    group = parser.add_argument_group("Transformations")
    group.add_argument("--train-crop-size", default=176, type=int, help="training image size")
    group.add_argument("--val-resize-size", default=232, type=int, help="validation image resize size")
    group.add_argument("--val-crop-size", default=224, type=int, help="validation image size")
    group.add_argument("--random-erase", default=0.0, type=float, help="probability to do random erasing")
    group.add_argument("--mixup-alpha", default=0.2, type=float, help="Mixup alpha factor")
    # Optimization
    group = parser.add_argument_group("Optimization")
    group.add_argument("--epochs", default=20, type=int, help="number of total epochs to run")
    group.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
    group.add_argument("--freeze-until", default=None, type=str, help="Last layer to freeze")
    group.add_argument("--grad-acc", default=1, type=int, help="Number of batches to accumulate the gradient of")
    group.add_argument("--opt", default="adamp", type=str, help="optimizer")
    group.add_argument("--sched", default="onecycle", type=str, help="Scheduler to be used")
    group.add_argument("--wd", "--weight-decay", default=0, type=float, help="weight decay", dest="weight_decay")
    group.add_argument("--norm-wd", default=None, type=float, help="weight decay of norm parameters")
    group.add_argument("--label-smoothing", default=0.1, type=float, help="label smoothing to apply")
    # Actions
    group = parser.add_argument_group("Actions")
    group.add_argument("--find-lr", action="store_true", help="Should you run LR Finder")
    group.add_argument("--find-size", dest="find_size", action="store_true", help="Should you run Image size Finder")
    group.add_argument("--check-setup", action="store_true", help="Check your training setup")
    group.add_argument("--show-samples", action="store_true", help="Whether training samples should be displayed")
    group.add_argument("--test-only", help="Only test the model", action="store_true")
    group.add_argument("--plot-loss", help="Check the top losses of the model", action="store_true")
    group.add_argument(
        "--benchmark",
        action="store_true",
        help="Record training metrics and deployment profiling beside the checkpoint",
    )
    # Experiment tracking
    group = parser.add_argument_group("Experiment tracking")
    group.add_argument("--wb", action="store_true", help="Log to Weights & Biases")
    group.add_argument("--name", type=str, default=None, help="Name of your training experiment")

    return parser


if __name__ == "__main__":
    from codecarbon import track_emissions

    args = get_parser().parse_args()
    track_emissions()(main)(args)
