# Copyright (C) 2019-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import math
from typing import Tuple

import torch
from torch import Tensor
from torchvision.ops.boxes import box_area, box_iou

__all__ = ["box_giou", "diou_loss", "ciou_loss"]


def _box_iou(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    # from https://github.com/facebookresearch/detr/blob/master/util/box_ops.py
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def box_giou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    r"""Computes the Generalized-IoU as described in `"Generalized Intersection over Union: A Metric and A Loss
    for Bounding Box Regression" <https://arxiv.org/pdf/1902.09630.pdf>`_. This implementation was adapted
    from https://github.com/facebookresearch/detr/blob/master/util/box_ops.py

    The generalized IoU is defined as follows:

    .. math::
        GIoU = IoU - \frac{|C - A \cup B|}{|C|}

    where :math:`IoU` is the Intersection over Union,
    :math:`A \cup B` is the area of the boxes' union,
    and :math:`C` is the area of the smallest enclosing box covering the two boxes.

    Args:
        boxes1 (torch.Tensor[M, 4]): bounding boxes
        boxes2 (torch.Tensor[N, 4]): bounding boxes

    Returns:
        torch.Tensor[M, N]: Generalized-IoU
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if torch.any(boxes1[:, 2:] < boxes1[:, :2]) or torch.any(boxes2[:, 2:] < boxes2[:, :2]):
        raise AssertionError("Incorrect coordinate format")
    iou, union = _box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def iou_penalty(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Implements the penalty term for the Distance-IoU loss

    Args:
        boxes1 (torch.Tensor[M, 4]): bounding boxes
        boxes2 (torch.Tensor[N, 4]): bounding boxes

    Returns:
        torch.Tensor[M, N]: penalty term
    """

    # Diagonal length of the smallest enclosing box
    c2 = torch.zeros((boxes1.shape[0], boxes2.shape[0], 2), device=boxes1.device)
    # Assign bottom right coords
    c2[..., 0] = torch.max(boxes1[:, 2].unsqueeze(-1), boxes2[:, 2].unsqueeze(-2))
    c2[..., 1] = torch.max(boxes1[:, 3].unsqueeze(-1), boxes2[:, 3].unsqueeze(-2))
    # Subtract top left coords
    c2[..., 0].sub_(torch.min(boxes1[:, 0].unsqueeze(-1), boxes2[:, 0].unsqueeze(-2)))
    c2[..., 1].sub_(torch.min(boxes1[:, 1].unsqueeze(-1), boxes2[:, 1].unsqueeze(-2)))

    c2.pow_(2)
    c2 = c2.sum(dim=-1)

    # L2 - distance between box centers
    center_dist2 = torch.zeros((boxes1.shape[0], boxes2.shape[0], 2), device=boxes1.device)
    # Centers of boxes1
    center_dist2[..., 0] = boxes1[:, [0, 2]].sum(dim=1).unsqueeze(1)
    center_dist2[..., 1] = boxes1[:, [1, 3]].sum(dim=1).unsqueeze(1)
    # Centers of boxes2
    center_dist2[..., 0].sub_(boxes2[:, [0, 2]].sum(dim=1).unsqueeze(0))
    center_dist2[..., 1].sub_(boxes2[:, [1, 3]].sum(dim=1).unsqueeze(0))

    center_dist2.pow_(2)
    center_dist2 = center_dist2.sum(dim=-1) / 4

    return center_dist2 / c2


def diou_loss(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    r"""Computes the Distance-IoU loss as described in `"Distance-IoU Loss: Faster and Better Learning for
    Bounding Box Regression" <https://arxiv.org/pdf/1911.08287.pdf>`_.

    The loss is defined as follows:

    .. math::
        \mathcal{L}_{DIoU} = 1 - IoU + \frac{\rho^2(b, b^{GT})}{c^2}

    where :math:`IoU` is the Intersection over Union,
    :math:`b` and :math:`b^{GT}` are the centers of the box and the ground truth box respectively,
    :math:`c` c is the diagonal length of the smallest enclosing box covering the two boxes,
    and :math:`\rho(.)` is the Euclidean distance.

    .. image:: https://github.com/frgfm/Holocron/releases/download/v0.1.3/diou_loss.png
        :align: center

    Args:
        boxes1 (torch.Tensor[M, 4]): bounding boxes
        boxes2 (torch.Tensor[N, 4]): bounding boxes

    Returns:
        torch.Tensor[M, N]: Distance-IoU loss
    """

    return 1 - box_iou(boxes1, boxes2) + iou_penalty(boxes1, boxes2)


def aspect_ratio(boxes: Tensor) -> Tensor:
    """Computes the aspect ratio of boxes

    Args:
        boxes (torch.Tensor[N, 4]): bounding boxes

    Returns:
        torch.Tensor[N]: aspect ratio
    """

    return torch.atan((boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1]))


def aspect_ratio_consistency(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Computes the aspect ratio consistency from the complete IoU loss

    Args:
        boxes1 (torch.Tensor[M, 4]): bounding boxes
        boxes2 (torch.Tensor[N, 4]): bounding boxes

    Returns:
        torch.Tensor[M, N]: aspect ratio consistency
    """

    v = aspect_ratio(boxes1).unsqueeze(-1) - aspect_ratio(boxes2).unsqueeze(-2)
    v.pow_(2)
    v.mul_(4 / math.pi**2)

    return v


def ciou_loss(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    r"""Computes the Complete IoU loss as described in `"Distance-IoU Loss: Faster and Better Learning for
    Bounding Box Regression" <https://arxiv.org/pdf/1911.08287.pdf>`_.

    The loss is defined as follows:

    .. math::
        \mathcal{L}_{CIoU} = 1 - IoU + \frac{\rho^2(b, b^{GT})}{c^2} + \alpha v

    where :math:`IoU` is the Intersection over Union,
    :math:`b` and :math:`b^{GT}` are the centers of the box and the ground truth box respectively,
    :math:`c` c is the diagonal length of the smallest enclosing box covering the two boxes,
    :math:`\rho(.)` is the Euclidean distance,
    :math:`\alpha` is a positive trade-off parameter,
    and :math:`v` is the aspect ratio consistency.

    More specifically:

    .. math::
        v = \frac{4}{\pi^2} \Big(\arctan{\frac{w^{GT}}{h^{GT}}} - \arctan{\frac{w}{h}}\Big)^2

    and

    .. math::
        \alpha = \frac{v}{(1 - IoU) + v}

    Args:
        boxes1 (torch.Tensor[M, 4]): bounding boxes
        boxes2 (torch.Tensor[N, 4]): bounding boxes

    Returns:
        torch.Tensor[M, N]: Complete IoU loss

    Example:
        >>> import torch
        >>> from holocron.ops.boxes import box_ciou
        >>> boxes1 = torch.tensor([[0, 0, 100, 100], [100, 100, 200, 200]], dtype=torch.float32)
        >>> boxes2 = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)
        >>> box_ciou(boxes1, boxes2)
    """

    iou = box_iou(boxes1, boxes2)
    v = aspect_ratio_consistency(boxes1, boxes2)

    ciou_loss = 1 - iou + iou_penalty(boxes1, boxes2)

    # Check
    _filter = (v != 0) & (iou != 0)
    ciou_loss[_filter].addcdiv_(v[_filter], 1 - iou[_filter] + v[_filter])

    return ciou_loss
