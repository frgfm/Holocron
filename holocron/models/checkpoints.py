# Copyright (C) 2023, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Union

from torchvision.transforms.functional import InterpolationMode

__all__ = [
    "Checkpoint",
    "TrainingRecipe",
    "Metric",
    "EvalResult",
    "Dataset",
    "Evaluation",
    "LoadingMeta",
    "PreProcessing",
]


@dataclass
class TrainingRecipe:
    """Implements a training recipe.

    Args:
        commit_hash: the commit that was used to train the model.
        args: the argument values that were passed to the reference script to train this.
    """

    commit: str
    script: str
    args: str


class Metric(str, Enum):
    """Evaluation metric"""

    TOP1_ACC = "top1-accuracy"
    TOP5_ACC = "top5-accuracy"


@dataclass
class EvalResult:
    metric: Metric
    val: float


class Dataset(str, Enum):
    """Evaluation dataset"""

    IMAGENET1K = "imagenet-1k"
    IMAGENETTE = "imagenette"
    CIFAR10 = "cifar10"


@dataclass
class Evaluation:
    dataset: Dataset
    results: List[EvalResult]


@dataclass
class LoadingMeta:
    url: str
    sha256: str
    size: int
    arch: str
    num_params: int
    categories: List[str]


@dataclass
class PreProcessing:
    input_shape: Tuple[int, ...]
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    interpolation: InterpolationMode = InterpolationMode.BILINEAR


@dataclass
class Checkpoint:
    # What to expect
    evaluation: Evaluation
    # How to load it
    meta: LoadingMeta
    # How to use it
    pre_processing: PreProcessing
    # How to reproduce
    recipe: TrainingRecipe


def _handle_legacy_pretrained(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    default_checkpoint: Union[Checkpoint, None] = None,
) -> Union[Checkpoint, None]:

    checkpoint = checkpoint or (default_checkpoint if pretrained else None)

    if pretrained and checkpoint is None:
        logging.warning("Invalid model URL, using default initialization.")

    return checkpoint
