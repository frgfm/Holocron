# Copyright (C) 2019-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
Transformation for object detection
"""

import random

import torch
from torchvision.transforms import functional as F
from torchvision.transforms import transforms


class VOCTargetTransform:
    def __init__(self, classes):
        self.class_map = {label: idx for idx, label in enumerate(classes)}

    def __call__(self, image, target):
        # Format boxes properly
        boxes = torch.tensor(
            [
                [
                    int(obj["bndbox"]["xmin"]),
                    int(obj["bndbox"]["ymin"]),
                    int(obj["bndbox"]["xmax"]),
                    int(obj["bndbox"]["ymax"]),
                ]
                for obj in target["annotation"]["object"]
            ],
            dtype=torch.float32,
        )
        # Encode class labels
        labels = torch.tensor([self.class_map[obj["name"]] for obj in target["annotation"]["object"]], dtype=torch.long)

        return image, dict(boxes=boxes, labels=labels)


class Compose(transforms.Compose):
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ImageTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, target):
        image = self.transform.__call__(image)
        return image, target

    def __repr__(self):
        return self.transform.__repr__()


class CenterCrop(transforms.CenterCrop):
    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        x = int(image.size[0] / 2 - self.size[0] / 2)
        y = int(image.size[1] / 2 - self.size[1] / 2)
        # Crop
        target["boxes"][:, [0, 2]] = target["boxes"][:, [0, 2]].clamp_(x, x + self.size[0])
        target["boxes"][:, [1, 3]] = target["boxes"][:, [1, 3]].clamp_(y, y + self.size[1])
        target["boxes"][:, [0, 2]] -= x
        target["boxes"][:, [1, 3]] -= y

        return image, target


class Resize(transforms.Resize):
    def __call__(self, image, target):
        if isinstance(self.size, int):
            if image.size[1] < image.size[0]:
                target["boxes"] *= self.size / image.size[1]
            else:
                target["boxes"] *= self.size / image.size[0]
        elif isinstance(self.size, tuple):
            target["boxes"][:, [0, 2]] *= self.size[0] / image.size[0]
            target["boxes"][:, [1, 3]] *= self.size[1] / image.size[1]
        return F.resize(image, self.size, self.interpolation), target


class RandomResizedCrop(transforms.RandomResizedCrop):
    def __call__(self, image, target):
        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        image = F.resized_crop(image, i, j, h, w, self.size, self.interpolation)
        # Crop
        target["boxes"][:, [0, 2]] = target["boxes"][:, [0, 2]].clamp_(j, j + w)
        target["boxes"][:, [1, 3]] = target["boxes"][:, [1, 3]].clamp_(i, i + h)
        # Reset origin
        target["boxes"][:, [0, 2]] -= j
        target["boxes"][:, [1, 3]] -= i
        # Remove targets that are out of crop
        target_filter = (target["boxes"][:, 0] != target["boxes"][:, 2]) & (
            target["boxes"][:, 1] != target["boxes"][:, 3]
        )
        target["boxes"] = target["boxes"][target_filter]
        target["labels"] = target["labels"][target_filter]
        # Resize
        target["boxes"][:, [0, 2]] *= self.size[0] / w
        target["boxes"][:, [1, 3]] *= self.size[1] / h

        return image, target


def convert_to_relative(image, target):

    target["boxes"][:, [0, 2]] /= image.size[0]
    target["boxes"][:, [1, 3]] /= image.size[1]

    # Clip
    target["boxes"][:, [0, 2]] = target["boxes"][:, [0, 2]].clamp_(0, 1)
    target["boxes"][:, [1, 3]] = target["boxes"][:, [1, 3]].clamp_(0, 1)

    return image, target


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, image, target):
        if random.random() < self.p:
            _, width = image.size
            image = F.hflip(image)
            target["boxes"][:, [0, 2]] = width - target["boxes"][:, [0, 2]]
            # Reorder them correctly
            target["boxes"] = target["boxes"][:, [2, 1, 0, 3]]
        return image, target
