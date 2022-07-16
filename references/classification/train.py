# Copyright (C) 2019-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
Training script for image classification
"""

import datetime
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.transforms import autoaugment as A
from torchvision.transforms import transforms as T
from torchvision.transforms.functional import InterpolationMode, to_pil_image

from holocron.models import classification
from holocron.models.presets import CIFAR10 as CIF10
from holocron.models.presets import IMAGENETTE
from holocron.optim import AdaBelief, AdamP
from holocron.trainer import ClassificationTrainer
from holocron.utils.data import Mixup


def worker_init_fn(worker_id: int) -> None:
    np.random.seed((worker_id + torch.initial_seed()) % np.iinfo(np.int32).max)


def plot_samples(images, targets, num_samples=4):
    # Unnormalize image
    nb_samples = min(num_samples, images.shape[0])
    _, axes = plt.subplots(1, nb_samples, figsize=(20, 5))
    for idx in range(nb_samples):
        img = images[idx]
        img *= torch.tensor(IMAGENETTE["std"]).view(-1, 1, 1)
        img += torch.tensor(IMAGENETTE["mean"]).view(-1, 1, 1)
        img = to_pil_image(img)

        axes[idx].imshow(img)
        axes[idx].axis("off")
        if targets.ndim == 1:
            axes[idx].set_title(IMAGENETTE["classes"][targets[idx].item()])
        else:
            class_idcs = torch.where(targets[idx] > 0)[0]
            _info = [f"{IMAGENETTE['classes'][_idx.item()]} ({targets[idx, _idx]:.2f})" for _idx in class_idcs]
            axes[idx].set_title(" ".join(_info))

    plt.show()


def main(args):

    print(args)

    torch.backends.cudnn.benchmark = True

    # Data loading
    train_loader, val_loader = None, None

    normalize = T.Normalize(
        mean=IMAGENETTE["mean"] if args.dataset.lower() == "imagenette" else CIF10["mean"],
        std=IMAGENETTE["std"] if args.dataset.lower() == "imagenette" else CIF10["std"],
    )

    interpolation = InterpolationMode.BILINEAR

    num_classes = None
    if not args.test_only:
        st = time.time()
        if args.dataset.lower() == "imagenette":

            train_set = ImageFolder(
                os.path.join(args.data_path, "train"),
                T.Compose(
                    [
                        T.RandomResizedCrop(args.img_size, scale=(0.3, 1.0), interpolation=interpolation),
                        T.RandomHorizontalFlip(),
                        A.TrivialAugmentWide(interpolation=interpolation),
                        T.PILToTensor(),
                        T.ConvertImageDtype(torch.float32),
                        normalize,
                        T.RandomErasing(p=0.9, scale=(0.02, 0.2), value="random"),
                    ]
                ),
            )
        else:
            cifar_version = CIFAR100 if args.dataset.lower() == "cifar100" else CIFAR10
            train_set = cifar_version(
                args.data_path,
                True,
                T.Compose(
                    [
                        T.RandomHorizontalFlip(),
                        A.TrivialAugmentWide(interpolation=interpolation),
                        T.PILToTensor(),
                        T.ConvertImageDtype(torch.float32),
                        normalize,
                        T.RandomErasing(p=0.9, value="random"),
                    ]
                ),
                download=True,
            )

        num_classes = len(train_set.classes)
        collate_fn = default_collate
        if args.mixup_alpha > 0:
            mix = Mixup(len(train_set.classes), alpha=args.mixup_alpha)
            collate_fn = lambda batch: mix(*default_collate(batch))  # noqa: E731
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            drop_last=True,
            sampler=RandomSampler(train_set),
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

        print(
            f"Training set loaded in {time.time() - st:.2f}s "
            f"({len(train_set)} samples in {len(train_loader)} batches)"
        )

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target)
        return

    if not (args.find_lr or args.check_setup):
        st = time.time()
        if args.dataset.lower() == "imagenette":
            eval_tf = []
            crop_pct = 0.875
            scale_size = min(int(math.floor(args.img_size / crop_pct)), 320)
            if scale_size < 320:
                eval_tf.append(T.Resize(scale_size))
            eval_tf.extend(
                [T.CenterCrop(args.img_size), T.PILToTensor(), T.ConvertImageDtype(torch.float32), normalize]
            )
            val_set = ImageFolder(os.path.join(args.data_path, "val"), T.Compose(eval_tf))
        else:
            cifar_version = CIFAR100 if args.dataset.lower() == "cifar100" else CIFAR10
            val_set = cifar_version(
                args.data_path,
                False,
                T.Compose([T.PILToTensor(), T.ConvertImageDtype(torch.float32), normalize]),
                download=True,
            )
        num_classes = len(val_set.classes)

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

    model = classification.__dict__[args.arch](args.pretrained, num_classes=num_classes)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Create the contiguous parameters.
    model_params = [p for p in model.parameters() if p.requires_grad]
    if args.opt == "sgd":
        optimizer = torch.optim.SGD(model_params, args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.opt == "radam":
        optimizer = torch.optim.RAdam(
            model_params, args.lr, betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay
        )
    elif args.opt == "adamp":
        optimizer = AdamP(model_params, args.lr, betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay)
    elif args.opt == "adabelief":
        optimizer = AdaBelief(model_params, args.lr, betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay)

    log_wb = lambda metrics: wandb.log(metrics) if args.wb else None
    trainer = ClassificationTrainer(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        args.device,
        args.output_file,
        amp=args.amp,
        on_epoch_end=log_wb,
    )
    if args.resume:
        print(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        trainer.load(checkpoint)

    if args.test_only:
        print("Running evaluation")
        eval_metrics = trainer.evaluate()
        print(
            f"Validation loss: {eval_metrics['val_loss']:.4} "
            f"(Acc@1: {eval_metrics['acc1']:.2%}, Acc@5: {eval_metrics['acc5']:.2%})"
        )
        return

    if args.find_lr:
        print("Looking for optimal LR")
        trainer.find_lr(args.freeze_until, num_it=min(len(train_loader), 100), norm_weight_decay=args.norm_wd)
        trainer.plot_recorder()
        return

    if args.check_setup:
        print("Checking batch overfitting")
        is_ok = trainer.check_setup(
            args.freeze_until, args.lr, norm_weight_decay=args.norm_wd, num_it=min(len(train_loader), 100)
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
            project="holocron-image-classification",
            config={
                "learning_rate": args.lr,
                "scheduler": args.sched,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "architecture": args.arch,
                "input_size": args.img_size,
                "optimizer": args.opt,
                "dataset": args.dataset,
                "loss": "crossentropy",
                "label_smoothing": args.label_smoothing,
                "mixup_alpha": args.mixup_alpha,
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
        description="Holocron Classification Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("data_path", type=str, help="path to dataset folder")
    parser.add_argument("--name", type=str, default=None, help="Name of your training experiment")
    parser.add_argument("--arch", default="darknet19", type=str, help="architecture to use")
    parser.add_argument("--dataset", default="imagenette", type=str, help="dataset to train on")
    parser.add_argument("--freeze-until", default=None, type=str, help="Last layer to freeze")
    parser.add_argument("--device", default=None, type=int, help="device")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
    parser.add_argument("--epochs", default=20, type=int, help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=min(os.cpu_count(), 16), type=int, help="number of data loading workers"
    )
    parser.add_argument("--img-size", default=224, type=int, help="image size")
    parser.add_argument("--label-smoothing", default=0.1, type=float, help="label smoothing to apply")
    parser.add_argument("--opt", default="adam", type=str, help="optimizer")
    parser.add_argument("--sched", default="onecycle", type=str, help="Scheduler to be used")
    parser.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
    parser.add_argument("--wd", "--weight-decay", default=0, type=float, help="weight decay", dest="weight_decay")
    parser.add_argument("--norm-wd", default=None, type=float, help="weight decay of norm parameters")
    parser.add_argument("--mixup-alpha", default=0, type=float, help="Mixup alpha factor")
    parser.add_argument("--find-lr", action="store_true", help="Should you run LR Finder")
    parser.add_argument("--check-setup", action="store_true", help="Check your training setup")
    parser.add_argument("--show-samples", action="store_true", help="Whether training samples should be displayed")
    parser.add_argument("--output-file", default="./model.pth", help="path where to save")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--test-only", help="Only test the model", action="store_true")
    parser.add_argument("--pretrained", action="store_true", help="Use pre-trained models from the modelzoo")
    parser.add_argument("--amp", help="Use Automatic Mixed Precision", action="store_true")
    parser.add_argument("--wb", action="store_true", help="Log to Weights & Biases")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
