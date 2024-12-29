# Copyright (C) 2023-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Union

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

    commit: Union[str, None]
    script: Union[str, None]
    args: Union[str, None]


class Metric(str, Enum):
    """Evaluation metric"""

    TOP1_ACC = "top1-accuracy"
    TOP5_ACC = "top5-accuracy"


class Dataset(str, Enum):
    """Training/evaluation dataset"""

    IMAGENET1K = "imagenet-1k"
    IMAGENETTE = "imagenette"
    CIFAR10 = "cifar10"


@dataclass
class Evaluation:
    """Results of model evaluation"""

    dataset: Dataset
    results: Dict[Metric, float]


@dataclass
class LoadingMeta:
    """Metadata to load the model"""

    url: str
    sha256: str
    size: int
    arch: str
    num_params: int
    categories: List[str]


@dataclass
class PreProcessing:
    """Preprocessing metadata for the model"""

    input_shape: Tuple[int, ...]
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    interpolation: InterpolationMode = InterpolationMode.BILINEAR


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


def _handle_legacy_pretrained(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    default_checkpoint: Union[Checkpoint, None] = None,
) -> Union[Checkpoint, None]:
    checkpoint = checkpoint or (default_checkpoint if pretrained else None)

    if pretrained and checkpoint is None:
        logger.warning("Invalid model URL, using default initialization.")

    return checkpoint
