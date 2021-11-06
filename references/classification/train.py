# Copyright (C) 2019-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

'''
Training script for image classification
'''

import datetime
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.transforms import transforms as T
from torchvision.transforms.functional import to_pil_image

import holocron
from holocron.trainer import ClassificationTrainer
from holocron.utils.data import Mixup

IMAGENETTE_CLASSES = [
    'tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump',
    'golf ball', 'parachute',
]


def worker_init_fn(worker_id: int) -> None:
    np.random.seed((worker_id + torch.initial_seed()) % np.iinfo(np.int32).max)


def plot_samples(images, targets, num_samples=4):
    # Unnormalize image
    nb_samples = min(num_samples, images.shape[0])
    _, axes = plt.subplots(1, nb_samples, figsize=(20, 5))
    for idx in range(nb_samples):
        img = images[idx]
        img *= torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img += torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        img = to_pil_image(img)

        axes[idx].imshow(img)
        axes[idx].axis('off')
        if targets.ndim == 1:
            axes[idx].set_title(IMAGENETTE_CLASSES[targets[idx].item()])
        else:
            class_idcs = torch.where(targets[idx] > 0)[0]
            _info = [f"{IMAGENETTE_CLASSES[_idx.item()]} ({targets[idx, _idx]:.2f})" for _idx in class_idcs]
            axes[idx].set_title(" ".join(_info))

    plt.show()


def main(args):

    print(args)

    torch.backends.cudnn.benchmark = True

    # Data loading
    train_loader, val_loader = None, None

    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406] if args.dataset.lower() == "imagenette" else [0.5071, 0.4866, 0.4409],
        std=[0.229, 0.224, 0.225] if args.dataset.lower() == "imagenette" else [0.2673, 0.2564, 0.2761]
    )

    if not args.test_only:
        st = time.time()
        if args.dataset.lower() == "imagenette":

            train_set = ImageFolder(
                os.path.join(args.data_path, 'train'),
                T.Compose([
                    T.RandomResizedCrop(args.img_size, scale=(0.3, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02),
                    T.ToTensor(), normalize,
                    T.RandomErasing(p=0.9, value='random')
                ]))
        else:
            cifar_version = CIFAR100 if args.dataset.lower() == "cifar100" else CIFAR10
            train_set = cifar_version(
                data_dir,
                True,
                T.Compose([
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02),
                    T.ToTensor(), normalize,
                    T.RandomErasing(p=0.9, value='random')
                ]))

        collate_fn = default_collate
        if args.mixup_alpha > 0:
            mix = Mixup(len(train_set.classes), alpha=0.2)
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

        print(f"Training set loaded in {time.time() - st:.2f}s "
              f"({len(train_set)} samples in {len(train_loader)} batches)")

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target)
        return

    if not (args.lr_finder or args.check_setup):
        st = time.time()
        if args.dataset.lower() == "imagenette":
            eval_tf = []
            crop_pct = 0.875
            scale_size = min(int(math.floor(args.img_size / crop_pct)), 320)
            if scale_size < 320:
                eval_tf.append(T.Resize(scale_size))
            eval_tf.extend([T.CenterCrop(args.img_size), T.ToTensor(), normalize])
            val_set = ImageFolder(
                os.path.join(args.data_path, 'val'),
                T.Compose(eval_tf)
            )
        else:
            val_set = CIFAR100(data_dir, False, T.Compose([T.ToTensor(), normalize]))

        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=args.batch_size, drop_last=False,
            sampler=SequentialSampler(val_set), num_workers=args.workers, pin_memory=True,
            worker_init_fn=worker_init_fn)

        print(f"Validation set loaded in {time.time() - st:.2f}s ({len(val_set)} samples in {len(val_loader)} batches)")

    model = holocron.models.__dict__[args.model](args.pretrained, num_classes=len(train_set.classes))

    if args.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'label_smoothing':
        criterion = holocron.nn.LabelSmoothingCrossEntropy()

    # Create the contiguous parameters.
    model_params = [p for p in model.parameters() if p.requires_grad]
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model_params, args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model_params, args.lr,
                                     betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay)
    elif args.opt == 'radam':
        optimizer = torch.optim.RAdam(model_params, args.lr,
                                      betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay)
    elif args.opt == 'ranger':
        optimizer = Lookahead(torch.optim.RAdam(model_params, args.lr,
                                                betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay))
    elif args.opt == 'tadam':
        optimizer = holocron.optim.TAdam(model_params, args.lr,
                                         betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay)

    trainer = ClassificationTrainer(model, train_loader, val_loader, criterion, optimizer,
                                    args.device, args.output_file, amp=args.amp)
    if args.resume:
        print(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        trainer.load(checkpoint)

    if args.test_only:
        print("Running evaluation")
        eval_metrics = trainer.evaluate()
        print(f"Validation loss: {eval_metrics['val_loss']:.4} "
              f"(Acc@1: {eval_metrics['acc1']:.2%}, Acc@5: {eval_metrics['acc5']:.2%})")
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
    parser = argparse.ArgumentParser(description='Holocron Classification Training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data_path', type=str, help='path to dataset folder')
    parser.add_argument('--model', default='darknet19', type=str, help='model')
    parser.add_argument('--dataset', default='imagenette', type=str, help='dataset to train on')
    parser.add_argument('--freeze-until', default=None, type=str, help='Last layer to freeze')
    parser.add_argument('--device', default=None, type=int, help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=min(os.cpu_count(), 16), type=int,
                        help='number of data loading workers')
    parser.add_argument('--img-size', default=224, type=int, help='image size')
    parser.add_argument('--loss', default='crossentropy', type=str, help='loss')
    parser.add_argument('--opt', default='adam', type=str, help='optimizer')
    parser.add_argument('--sched', default='onecycle', type=str, help='Scheduler to be used')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float, help='weight decay', dest='weight_decay')
    parser.add_argument('--mixup-alpha', default=0, type=float, help='Mixup alpha factor')
    parser.add_argument("--lr-finder", dest='lr_finder', action='store_true', help="Should you run LR Finder")
    parser.add_argument("--check-setup", dest='check_setup', action='store_true', help="Check your training setup")
    parser.add_argument("--show-samples", dest='show_samples', action='store_true',
                        help="Whether training samples should be displayed")
    parser.add_argument('--output-file', default='./model.pth', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo",
                        action="store_true")
    parser.add_argument("--amp", dest="amp", help="Use Automatic Mixed Precision", action="store_true")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
