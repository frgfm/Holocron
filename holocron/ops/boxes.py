# -*- coding: utf-8 -*-

'''
Bounding box operations
'''

import math
import torch
from torchvision.ops.boxes import box_iou


__all__ = ['box_diou', 'box_ciou']


def iou_penalty(boxes1, boxes2):
    """Implements the penalty term for the Distance-IoU loss

    Args:
        boxes1 (torch.Tensor[M, 4]): bounding boxes
        boxes2 (torch.Tensor[N, 4]): bounding boxes

    Returns:
        torch.Tensor[M, N]: penalty term
    """

    # Diagonal length of the smallest enclosing box
    c2 = torch.zeros((boxes1.shape[0], boxes2.shape[0], 2))
    # Assign bottom right coords
    c2[..., 0] = torch.max(boxes1[:, 2].unsqueeze(-1), boxes2[:, 2].unsqueeze(-2))
    c2[..., 1] = torch.max(boxes1[:, 3].unsqueeze(-1), boxes2[:, 3].unsqueeze(-2))
    # Subtract top left coords
    c2[..., 0].sub_(torch.min(boxes1[:, 0].unsqueeze(-1), boxes2[:, 0].unsqueeze(-2)))
    c2[..., 1].sub_(torch.min(boxes1[:, 1].unsqueeze(-1), boxes2[:, 1].unsqueeze(-2)))

    c2.pow_(2)
    c2 = c2.sum(axis=-1)

    # L2 - distance between box centers
    center_dist2 = torch.zeros((boxes1.shape[0], boxes2.shape[0], 2))
    # Centers of boxes1
    center_dist2[..., 0] = boxes1[:, [0, 2]].sum(dim=1).unsqueeze(1)
    center_dist2[..., 1] = boxes1[:, [1, 3]].sum(dim=1).unsqueeze(1)
    # Centers of boxes2
    center_dist2[..., 0].sub_(boxes2[:, [0, 2]].sum(dim=1).unsqueeze(0))
    center_dist2[..., 1].sub_(boxes2[:, [1, 3]].sum(dim=1).unsqueeze(0))

    center_dist2.pow_(2)
    center_dist2 = center_dist2.sum(axis=-1) / 4

    return center_dist2 / c2


def box_diou(boxes1, boxes2):
    """Computes the Distance-IoU loss as described in `"Distance-IoU Loss: Faster and Better Learning for
    Bounding Box Regression" <https://arxiv.org/pdf/1911.08287.pdf>`_

    Args:
        boxes1 (torch.Tensor[M, 4]): bounding boxes
        boxes2 (torch.Tensor[N, 4]): bounding boxes

    Returns:
        torch.Tensor[M, N]: Distance-IoU loss
    """

    return 1 - box_iou(boxes1, boxes2) + iou_penalty(boxes1, boxes2)


def aspect_ratio(boxes):
    """Computes the aspect ratio of boxes

    Args:
        boxes (torch.Tensor[N, 4]): bounding boxes

    Returns:
        torch.Tensor[N]: aspect ratio
    """

    return torch.atan((boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1]))


def aspect_ratio_consistency(boxes1, boxes2):
    """Computes the aspect ratio consistency from the complete IoU loss

    Args:
        boxes1 (torch.Tensor[M, 4]): bounding boxes
        boxes2 (torch.Tensor[N, 4]): bounding boxes

    Returns:
        torch.Tensor[M, N]: aspect ratio consistency
    """

    v = aspect_ratio(boxes1).unsqueeze(-1) - aspect_ratio(boxes2).unsqueeze(-2)
    v.pow_(2)
    v.mul_(4 / math.pi ** 2)

    return v


def box_ciou(boxes1, boxes2):
    """Computes the Complete IoU loss as described in `"Distance-IoU Loss: Faster and Better Learning for
    Bounding Box Regression" <https://arxiv.org/pdf/1911.08287.pdf>`_

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
    ciou_loss[_filter].addcdiv_(1, v[_filter] ** 2, 1 - iou[_filter] + v[_filter])

    return ciou_loss
