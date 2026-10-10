# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.


import torch
from torch import Tensor
from torch.distributions.beta import Beta
from torch.nn.functional import one_hot
from torch.utils.data import default_collate

__all__ = ["Mixup"]


class Mixup(torch.nn.Module):
    """Implements a batch collate function with MixUp strategy from
    ["mixup: Beyond Empirical Risk Minimization"](https://arxiv.org/pdf/1710.09412.pdf).

    Example:
        ```python
        import torch
        from holocron.utils.data import Mixup
        mix = Mixup(num_classes=10, alpha=0.4)
        loader = torch.utils.data.DataLoader(dataset, batch_size, collate_fn=mix.collate)
        ```

    Args:
        num_classes: number of expected classes
        alpha: mixup factor
    """

    def __init__(self, num_classes: int, alpha: float = 0.2) -> None:
        super().__init__()
        self.num_classes: int = num_classes
        if alpha < 0:
            raise ValueError("`alpha` only takes positive values")
        self.alpha: float = alpha

    def forward(self, inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:  # noqa: D102
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
        lam = Beta(self.alpha, self.alpha).sample()

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

    def collate(self, batch: list[tuple[Tensor, int]]) -> tuple[Tensor, Tensor]:
        """Collate and mix a batch using a worker-picklable bound method.

        Args:
            batch: dataset images and class indices

        Returns:
            Collated mixed images and soft class targets.
        """
        return self(*default_collate(batch))
