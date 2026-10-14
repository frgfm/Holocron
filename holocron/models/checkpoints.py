# Copyright (C) 2023-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import logging
from dataclasses import dataclass
from enum import StrEnum

import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode

__all__ = [
    "Checkpoint",
    "Dataset",
    "Evaluation",
    "LoadingMeta",
    "Metric",
    "PreProcessing",
    "TrainingRecipe",
]

logger = logging.getLogger(__name__)


@dataclass
class TrainingRecipe:
    """Implements a training recipe.

    Args:
        commit_hash: the commit that was used to train the model.
        args: the argument values that were passed to the reference script to train this.
    """

    commit: str | None
    script: str | None
    args: str | None


class Metric(StrEnum):
    """Evaluation metric"""

    TOP1_ACC = "top1-accuracy"
    TOP5_ACC = "top5-accuracy"


class Dataset(StrEnum):
    """Training/evaluation dataset"""

    IMAGENET1K = "imagenet-1k"
    IMAGENETTE = "imagenette"
    CIFAR10 = "cifar10"


@dataclass
class Evaluation:
    """Results of model evaluation"""

    dataset: Dataset
    results: dict[Metric, float]


@dataclass
class LoadingMeta:
    """Metadata to load the model"""

    url: str
    sha256: str
    size: int
    arch: str
    num_params: int
    categories: list[str]


@dataclass
class PreProcessing:
    """Preprocessing metadata for the model"""

    input_shape: tuple[int, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    interpolation: InterpolationMode = InterpolationMode.BILINEAR
    # Shorter-side resize before the evaluation center crop; None resizes straight to input_shape
    resize_size: int | None = None


@dataclass
class Checkpoint:
    """Data required to run a model in the exact same condition than the checkpoint"""

    # What to expect
    evaluation: Evaluation
    # How to load it
    meta: LoadingMeta
    # How to use it
    pre_processing: PreProcessing
    # How to reproduce
    recipe: TrainingRecipe

    def transforms(self, train: bool = False) -> T.Compose:
        """Build the image transform matching this checkpoint.

        Args:
            train: whether to use random resized crops and horizontal flips, a fine-tuning starter rather than the
                full reference recipe, instead of the evaluation resize and crop.

        Returns:
            a transform that maps a PIL image to a normalized float tensor.
        """
        pre = self.pre_processing
        size = pre.input_shape[1:]
        if train:
            resize = [T.RandomResizedCrop(size, interpolation=pre.interpolation), T.RandomHorizontalFlip()]
        elif pre.resize_size is None:
            resize = [T.Resize(size, interpolation=pre.interpolation)]
        else:
            resize = [T.Resize(pre.resize_size, interpolation=pre.interpolation), T.CenterCrop(size)]
        return T.Compose([
            *resize,
            T.PILToTensor(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(pre.mean, pre.std),
        ])


def _handle_legacy_pretrained(
    pretrained: bool = False,
    checkpoint: Checkpoint | None = None,
    default_checkpoint: Checkpoint | None = None,
) -> Checkpoint | None:
    checkpoint = checkpoint or (default_checkpoint if pretrained else None)

    if pretrained and checkpoint is None:
        logger.warning("Invalid model URL, using default initialization.")

    return checkpoint
