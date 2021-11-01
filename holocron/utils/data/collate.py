# Copyright (C) 2019-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data._utils.collate import default_collate

__all__ = ['mixup_collate']


def mixup_collate(data: List[Tuple[Tensor, int]], alpha: float = 0.1) -> Tuple[Tensor, Tensor, Tensor, float]:
    """Implements a batch collate function with MixUp strategy from
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/pdf/1710.09412.pdf>`_

    Args:
        data: list of elements
        alpha: mixup factor

    Returns:
        interpolated input
        original target
        resorted target
        interpolation factor

    Example::
        >>> import torch
        >>> from holocron import utils
        >>> loader = torch.utils.data.DataLoader(dataset, batch_size, collate_fn=utils.data.mixup_collate)
    """

    inputs, targets = default_collate(data)

    # Sample lambda
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    # Mix batch indices
    batch_size = inputs.size()[0]
    index = torch.randperm(batch_size)

    # Create the new input and targets
    inputs = lam * inputs + (1 - lam) * inputs[index, :]
    targets_a, targets_b = targets, targets[index]

    return inputs, targets_a, targets_b, lam
