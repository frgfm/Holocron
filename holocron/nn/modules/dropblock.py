# Copyright (C) 2019-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import torch.nn as nn
from torch import Tensor

from .. import functional as F

__all__ = ["DropBlock2d"]


class DropBlock2d(nn.Module):
    """Implements the DropBlock module from `"DropBlock: A regularization method for convolutional networks"
    <https://arxiv.org/pdf/1810.12890.pdf>`_

    .. image:: https://github.com/frgfm/Holocron/releases/download/v0.1.3/dropblock.png
        :align: center

    Args:
        p (float, optional): probability of dropping activation value
        block_size (int, optional): size of each block that is expended from the sampled mask
        inplace (bool, optional): whether the operation should be done inplace
    """

    def __init__(self, p: float = 0.1, block_size: int = 7, inplace: bool = False) -> None:
        super().__init__()
        self.p = p
        self.block_size = block_size
        self.inplace = inplace

    @property
    def drop_prob(self) -> float:
        return self.p / self.block_size**2

    def forward(self, x: Tensor) -> Tensor:
        return F.dropblock2d(x, self.drop_prob, self.block_size, self.inplace, self.training)

    def extra_repr(self) -> str:
        return f"p={self.p}, block_size={self.block_size}, inplace={self.inplace}"
