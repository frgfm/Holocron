# Copyright (C) 2019-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import one_hot

__all__ = ["Mixup"]


class Mixup(torch.nn.Module):
    """Implements a batch collate function with MixUp strategy from
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/pdf/1710.09412.pdf>`_

    Args:
        num_classes: number of expected classes
        alpha: mixup factor

    >>> import torch
    >>> from torch.utils.data._utils.collate import default_collate
    >>> from holocron.utils.data import Mixup
    >>> mix = Mixup(num_classes=10, alpha=0.4)
    >>> loader = torch.utils.data.DataLoader(dataset, batch_size, collate_fn=lambda b: mix(*default_collate(b)))
    """

    def __init__(self, num_classes: int, alpha: float = 0.2) -> None:
        super().__init__()
        self.num_classes = num_classes
        if alpha < 0:
            raise ValueError("`alpha` only takes positive values")
        self.alpha = alpha

    def forward(self, inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:

        # Convert target to one-hot
        if targets.ndim == 1:
            # (N,) --> (N, C)
            if self.num_classes > 1:
                targets = one_hot(targets, num_classes=self.num_classes)
            elif self.num_classes == 1:
                targets = targets.unsqueeze(1)
        targets = targets.to(dtype=inputs.dtype)

        # Sample lambda
        if self.alpha == 0:
            return inputs, targets
        lam = np.random.beta(self.alpha, self.alpha)

        # Mix batch indices
        batch_size = inputs.size()[0]
        index = torch.randperm(batch_size)

        # Create the new input and targets
        mixed_input, mixed_target = inputs[index, :], targets[index]
        mixed_input.mul_(1 - lam)
        inputs.mul_(lam).add_(mixed_input)
        mixed_target.mul_(1 - lam)
        targets.mul_(lam).add_(mixed_target)

        return inputs, targets
