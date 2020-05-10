# -*- coding: utf-8 -*-

"""
Collate functions
"""

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


__all__ = ['mixup_collate']


def mixup_collate(data, alpha=0.1):
    """Implements a batch collate function with MixUp strategy from
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/pdf/1710.09412.pdf>`_

    Args:
        data (list): list of elements
        alpha (float, optional): mixup factor

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
