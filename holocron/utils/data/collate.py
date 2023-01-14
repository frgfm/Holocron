# Copyright (C) 2019-2023, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import math
from typing import Tuple

import torch
from torch import Tensor
from torch.nn.functional import one_hot

__all__ = ["Mixup", "CutMix"]


class Mixup(torch.nn.Module):
    """Implements a batch collate function with MixUp strategy from
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/pdf/1710.09412.pdf>`_.

    >>> import torch
    >>> from torch.utils.data._utils.collate import default_collate
    >>> from holocron.utils.data import Mixup
    >>> mix = Mixup(num_classes=10, alpha=0.4)
    >>> loader = torch.utils.data.DataLoader(dataset, batch_size, collate_fn=lambda b: mix(*default_collate(b)))

    Args:
        num_classes: number of expected classes
        alpha: mixup factor
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
        lam = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])

        # Mix batch indices (faster to roll than shuffle)
        mixed_input, mixed_target = inputs.roll(1, 0), targets.roll(1, 0)
        # Create the new input and targets
        mixed_input.mul_(1 - lam)
        inputs.mul_(lam).add_(mixed_input)
        mixed_target.mul_(1 - lam)
        targets.mul_(lam).add_(mixed_target)

        return inputs, targets


class CutMix(torch.nn.Module):
    """Implements a batch collate function with MixUp strategy from `"CutMix: Regularization Strategy to Train Strong
    Classifiers with Localizable Features" <https://arxiv.org/abs/1905.04899>`_.

    >>> import torch
    >>> from torch.utils.data._utils.collate import default_collate
    >>> from holocron.utils.data import CutMix
    >>> mix = CutMix(num_classes=10, alpha=0.4)
    >>> loader = torch.utils.data.DataLoader(dataset, batch_size, collate_fn=lambda b: mix(*default_collate(b)))

    Args:
        num_classes: number of expected classes
        alpha: cutmix factor
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
        lam = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])

        # Mix batch indices (faster to roll than shuffle)
        mixed_input, mixed_target = inputs.roll(1, 0), targets.roll(1, 0)

        # Cf. paper page 12
        # borrowed from https://github.com/pytorch/vision/blob/main/references/classification/transforms.py
        _, _, h, w = inputs.shape
        r_x = torch.randint(w, (1,))
        r_y = torch.randint(h, (1,))
        r = 0.5 * math.sqrt(1.0 - lam)
        r_w_half = int(r * w)
        r_h_half = int(r * h)
        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=w))
        y2 = int(torch.clamp(r_y + r_h_half, max=h))

        # Create the new input and targets
        inputs[:, :, y1:y2, x1:x2] = mixed_input[:, :, y1:y2, x1:x2]
        lam = float(1.0 - (x2 - x1) * (y2 - y1) / (w * h))
        mixed_target.mul_(1 - lam)
        targets.mul_(lam).add_(mixed_target)

        return inputs, targets
