# Copyright (C) 2020-2023, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops.boxes import box_iou, nms

from holocron.nn.init import init_module

from ..classification.darknet import DarknetBodyV1
from ..classification.darknet import default_cfgs as dark_cfgs
from ..utils import conv_sequence, load_pretrained_params

__all__ = ["YOLOv1", "yolov1"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "yolov1": {"arch": "YOLOv1", "backbone": dark_cfgs["darknet24"], "url": None},
}


class _YOLO(nn.Module):
    def __init__(
        self,
        num_classes: int = 20,
        rpn_nms_thresh: float = 0.7,
        box_score_thresh: float = 0.05,
        lambda_obj: float = 1,
        lambda_noobj: float = 0.5,
        lambda_class: float = 1,
        lambda_coords: float = 5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.rpn_nms_thresh = rpn_nms_thresh
        self.box_score_thresh = box_score_thresh
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
        self.lambda_coords = lambda_coords

    def _compute_losses(
        self,
        pred_boxes: Tensor,
        pred_o: Tensor,
        pred_scores: Tensor,
        target: List[Dict[str, Tensor]],
        ignore_high_iou: bool = False,
    ) -> Dict[str, Tensor]:
        """Computes the detector losses as described in `"You Only Look Once: Unified, Real-Time Object Detection"
        <https://pjreddie.com/media/files/papers/yolo_1.pdf>`_

        Args:
            pred_boxes (torch.Tensor[N, H, W, num_anchors, 4]): relative coordinates in format (xc, yc, w, h)
            pred_o (torch.Tensor[N, H, W, num_anchors]): objectness scores
            pred_scores (torch.Tensor[N, H, W, num_anchors, num_classes]): classification probabilities
            target (list<dict>, optional): list of targets
            ignore_high_iou (bool): ignore the intersections with high IoUs in the noobj penalty term

        Returns:
            dict: dictionary of losses
        """
        gt_boxes = [t["boxes"] for t in target]
        gt_labels = [t["labels"] for t in target]

        # GT xmin, ymin, xmax, ymax
        if not all(torch.all(boxes >= 0) and torch.all(boxes <= 1) for boxes in gt_boxes):
            raise ValueError("Ground truth boxes are expected to have values between 0 and 1.")

        b, h, w, _, num_classes = pred_scores.shape
        # Convert from (xcenter, ycenter, w, h) to (xmin, ymin, xmax, ymax)
        pred_xyxy = self.to_isoboxes(pred_boxes, (h, w), clamp=False)
        pred_xy = (pred_xyxy[..., [0, 1]] + pred_xyxy[..., [2, 3]]) / 2

        # Initialize losses
        obj_loss = torch.zeros(1, device=pred_boxes.device)
        noobj_loss = torch.zeros(1, device=pred_boxes.device)
        bbox_loss = torch.zeros(1, device=pred_boxes.device)
        clf_loss = torch.zeros(1, device=pred_boxes.device)

        is_noobj = torch.ones_like(pred_o, dtype=torch.bool)

        for idx in range(b):
            gt_xy = (gt_boxes[idx][:, :2] + gt_boxes[idx][:, 2:]) / 2
            gt_wh = gt_boxes[idx][:, 2:] - gt_boxes[idx][:, :2]
            gt_centers = torch.stack(
                (gt_boxes[idx][:, [0, 2]].mean(dim=-1) * w, gt_boxes[idx][:, [1, 3]].mean(dim=-1) * h), dim=1
            )
            gt_idcs = gt_centers.to(dtype=torch.long)

            # Assign GT to anchors
            for _idx in range(gt_boxes[idx].shape[0]):
                # Assign the anchor inside the cell
                _iou = box_iou(gt_boxes[idx][_idx].unsqueeze(0), pred_xyxy[idx, gt_idcs[_idx, 1], gt_idcs[_idx, 0]])
                iou, anchor_idx = _iou.squeeze(0).max(dim=0)
                # Flag that there is an object here
                is_noobj[idx, gt_idcs[_idx, 1], gt_idcs[_idx, 0], anchor_idx] = False
                # Classification loss
                gt_scores = torch.zeros_like(pred_scores[idx, gt_idcs[_idx, 1], gt_idcs[_idx, 0]])
                gt_scores[:, gt_labels[idx][_idx]] = 1
                clf_loss += (gt_scores - pred_scores[idx, gt_idcs[_idx, 1], gt_idcs[_idx, 0]]).pow(2).sum()
                # Objectness loss
                obj_loss += (iou - pred_o[idx, gt_idcs[_idx, 1], gt_idcs[_idx, 0], anchor_idx]).pow(2)
                # Bbox loss
                bbox_loss += (
                    (gt_xy[_idx] - pred_xy[idx, gt_idcs[_idx, 1], gt_idcs[_idx, 0], anchor_idx, :2]).pow(2).sum()
                )
                bbox_loss += (
                    (gt_wh.sqrt() - pred_boxes[idx, gt_idcs[_idx, 1], gt_idcs[_idx, 0], anchor_idx, 2:].sqrt())
                    .pow(2)
                    .sum()
                )

            # Ignore high ious
            if ignore_high_iou:
                _iou = box_iou(pred_xyxy[idx].reshape(-1, 4), gt_boxes[idx]).max(dim=-1).values.reshape(h, w, -1)
                is_noobj[idx, _iou >= 0.5] = False
        # Non-objectness loss
        noobj_loss += pred_o[is_noobj].pow(2).sum()

        return {
            "obj_loss": self.lambda_obj * obj_loss / pred_boxes.shape[0],
            "noobj_loss": self.lambda_noobj * noobj_loss / pred_boxes.shape[0],
            "bbox_loss": self.lambda_coords * bbox_loss / pred_boxes.shape[0],
            "clf_loss": self.lambda_class * clf_loss / pred_boxes.shape[0],
        }

    @staticmethod
    def to_isoboxes(b_coords: Tensor, grid_shape: Tuple[int, int], clamp: bool = False) -> Tensor:
        # Cell offset
        c_x = torch.arange(grid_shape[1], dtype=torch.float, device=b_coords.device)
        c_y = torch.arange(grid_shape[0], dtype=torch.float, device=b_coords.device)
        # Box coordinates
        b_x = (b_coords[..., 0] + c_x.reshape(1, 1, -1, 1)) / grid_shape[1]
        b_y = (b_coords[..., 1] + c_y.reshape(1, -1, 1, 1)) / grid_shape[0]
        xy = torch.stack((b_x, b_y), dim=-1)
        wh = b_coords[..., 2:]
        pred_xyxy = torch.cat((xy - wh / 2, xy + wh / 2), dim=-1).reshape(*b_coords.shape)
        if clamp:
            pred_xyxy.clamp_(0, 1)

        return pred_xyxy

    def post_process(
        self,
        b_coords: Tensor,
        b_o: Tensor,
        b_scores: Tensor,
        grid_shape: Tuple[int, int],
        rpn_nms_thresh: float = 0.7,
        box_score_thresh: float = 0.05,
    ) -> List[Dict[str, Tensor]]:
        """Perform final filtering to produce detections

        Args:
            b_coords (torch.Tensor[N, H * W * num_anchors, 4]): relative coordinates in format (x, y, w, h)
            b_o (torch.Tensor[N, H * W * num_anchors]): objectness scores
            b_scores (torch.Tensor[N, H * W * num_anchors, num_classes]): classification scores
            grid_shape (Tuple[int, int]): the size of the grid
            rpn_nms_thresh (float, optional): IoU threshold for NMS
            box_score_thresh (float, optional): minimum classification confidence threshold

        Returns:
            list<dict>: detections dictionary
        """
        # Convert box coords
        pred_xyxy = self.to_isoboxes(
            b_coords.reshape(-1, *grid_shape, self.num_anchors, 4),  # type: ignore[call-overload]
            grid_shape,
            clamp=True,
        ).reshape(b_o.shape[0], -1, 4)

        detections = []
        for idx in range(b_coords.shape[0]):
            coords = torch.zeros((0, 4), dtype=b_o.dtype, device=b_o.device)
            scores = torch.zeros(0, dtype=b_o.dtype, device=b_o.device)
            labels = torch.zeros(0, dtype=torch.long, device=b_o.device)

            # Objectness filter
            obj_mask = b_o[idx] >= 0.5
            if torch.any(obj_mask):
                coords = pred_xyxy[idx, obj_mask]
                scores, labels = b_scores[idx, obj_mask].max(dim=-1)
                # Multiply by the objectness
                scores.mul_(b_o[idx, obj_mask])

                # Confidence threshold
                coords = coords[scores >= box_score_thresh]
                labels = labels[scores >= box_score_thresh]
                scores = scores[scores >= box_score_thresh]

                # NMS
                kept_idxs = nms(coords, scores, iou_threshold=rpn_nms_thresh)
                coords = coords[kept_idxs]
                scores = scores[kept_idxs]
                labels = labels[kept_idxs]

            detections.append({"boxes": coords, "scores": scores, "labels": labels})

        return detections


class YOLOv1(_YOLO):
    def __init__(
        self,
        layout: List[List[int]],
        num_classes: int = 20,
        in_channels: int = 3,
        stem_channels: int = 64,
        num_anchors: int = 2,
        lambda_obj: float = 1,
        lambda_noobj: float = 0.5,
        lambda_class: float = 1,
        lambda_coords: float = 5.0,
        rpn_nms_thresh: float = 0.7,
        box_score_thresh: float = 0.05,
        head_hidden_nodes: int = 512,  # In the original paper, 4096
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
        backbone_norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__(
            num_classes, rpn_nms_thresh, box_score_thresh, lambda_obj, lambda_noobj, lambda_class, lambda_coords
        )

        if act_layer is None:
            act_layer = nn.LeakyReLU(0.1, inplace=True)

        if backbone_norm_layer is None and norm_layer is not None:
            backbone_norm_layer = norm_layer

        self.backbone = DarknetBodyV1(layout, in_channels, stem_channels, act_layer, backbone_norm_layer)

        self.block4 = nn.Sequential(
            *conv_sequence(
                1024,
                1024,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=1,
                bias=(norm_layer is None),
            ),
            *conv_sequence(
                1024,
                1024,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=1,
                stride=2,
                bias=(norm_layer is None),
            ),
            *conv_sequence(
                1024,
                1024,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=1,
                bias=(norm_layer is None),
            ),
            *conv_sequence(
                1024,
                1024,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=1,
                bias=(norm_layer is None),
            ),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7**2, head_hidden_nodes),
            act_layer,
            nn.Dropout(0.5),
            nn.Linear(head_hidden_nodes, 7**2 * (num_anchors * 5 + num_classes)),
        )
        self.num_anchors = num_anchors

        init_module(self.block4, "leaky_relu")
        init_module(self.classifier, "leaky_relu")

    def _format_outputs(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Formats convolutional layer output

        Args:
            x (torch.Tensor[N, num_anchors * (5 + num_classes) * H * W]): output tensor

        Returns:
            torch.Tensor[N, H * W, num_anchors, 4]: relative coordinates in format (x, y, w, h)
            torch.Tensor[N, H * W, num_anchors]: objectness scores
            torch.Tensor[N, H * W, num_anchors, num_classes]: classification scores
        """
        b, _ = x.shape
        h, w = 7, 7
        # (B, H * W * (num_anchors * 5 + num_classes)) --> (B, H, W, num_anchors * 5 + num_classes)
        x = x.reshape(b, h, w, self.num_anchors * 5 + self.num_classes)
        # Classification scores
        b_scores = x[..., -self.num_classes :]
        # Repeat for anchors to keep compatibility across YOLO versions
        b_scores = F.softmax(b_scores.unsqueeze(3), dim=-1)
        #  (B, H, W, num_anchors * 5 + num_classes) -->  (B, H, W, num_anchors, 5)
        x = torch.sigmoid(x[..., : self.num_anchors * 5].reshape(b, h, w, self.num_anchors, 5))
        # Box coordinates
        b_coords = x[..., :4]
        # Objectness
        b_o = x[..., 4]

        return b_coords, b_o, b_scores

    def _forward(self, x: Tensor) -> Tensor:
        out = self.backbone(x)
        out = self.block4(out)
        out = self.classifier(out)

        return out

    def forward(
        self, x: Tensor, target: Optional[List[Dict[str, Tensor]]] = None
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """Perform detection on an image tensor and returns either the loss dictionary in training mode
        or the list of detections in eval mode.

        Args:
            x (torch.Tensor[N, 3, H, W]): input image tensor
            target (list<dict>, optional): each dict must have two keys `boxes` of type torch.Tensor[-1, 4]
            and `labels` of type torch.Tensor[-1]
        """
        if self.training and target is None:
            raise ValueError("`target` needs to be specified in training mode")

        if isinstance(x, (list, tuple)):
            x = torch.stack(x, dim=0)

        out = self._forward(x)

        # (N, H * W * (num_anchors * 5 + num_classes) -> (N, H, W, num_anchors)
        b_coords, b_o, b_scores = self._format_outputs(out)

        if self.training:
            # Update losses
            return self._compute_losses(b_coords, b_o, b_scores, target)  # type: ignore[arg-type]

        # (B, H * W * num_anchors)
        b_coords = b_coords.reshape(b_coords.shape[0], -1, 4)
        b_o = b_o.reshape(b_o.shape[0], -1)
        # Repeat for each anchor box
        b_scores = b_scores.repeat_interleave(self.num_anchors, dim=3)
        b_scores = b_scores.contiguous().reshape(b_scores.shape[0], -1, self.num_classes)

        # Stack detections into a list
        return self.post_process(b_coords, b_o, b_scores, (7, 7), self.rpn_nms_thresh, self.box_score_thresh)


def _yolo(
    arch: str, pretrained: bool, progress: bool, pretrained_backbone: bool, layout: List[List[int]], **kwargs: Any
) -> YOLOv1:
    if pretrained:
        pretrained_backbone = False

    # Build the model
    model = YOLOv1(layout, **kwargs)
    # Load backbone pretrained parameters
    if pretrained_backbone:
        load_pretrained_params(
            model.backbone,
            default_cfgs[arch]["backbone"]["url"],
            progress,
            key_replacement=("features.", ""),
            key_filter="features.",
        )
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"], progress)

    return model


def yolov1(pretrained: bool = False, progress: bool = True, pretrained_backbone: bool = True, **kwargs: Any) -> YOLOv1:
    r"""YOLO model from
    `"You Only Look Once: Unified, Real-Time Object Detection" <https://pjreddie.com/media/files/papers/yolo_1.pdf>`_.

    YOLO's particularity is to make predictions in a grid (same size as last feature map). For each grid cell,
    the model predicts classification scores and a fixed number of boxes (default: 2). Each box in the cell gets
    5 predictions: an objectness score, and 4 coordinates. The 4 coordinates are composed of: the 2-D coordinates of
    the predicted box center (relative to the cell), and the width and height of the predicted box (relative to
    the whole image).

    For training, YOLO uses a multi-part loss whose components are computed by:

    .. math::
        \mathcal{L}_{coords} = \sum\limits_{i=0}^{S^2} \sum\limits_{j=0}^{B}
        \mathbb{1}_{ij}^{obj} \Big[
        (x_{ij} - \hat{x}_{ij})² + (y_{ij} - \hat{y}_{ij})² +
        (\sqrt{w_{ij}} - \sqrt{\hat{w}_{ij}})² + (\sqrt{h_{ij}} - \sqrt{\hat{h}_{ij}})²
        \Big]

    where :math:`S` is size of the output feature map (7 for an input size :math:`(448, 448)`),
    :math:`B` is the number of anchor boxes per grid cell (default: 2),
    :math:`\mathbb{1}_{ij}^{obj}` equals to 1 if a GT center falls inside the i-th grid cell and among the
    anchor boxes of that cell, has the highest IoU with the j-th box else 0,
    :math:`(x_{ij}, y_{ij}, w_{ij}, h_{ij})` are the coordinates of the ground truth assigned to
    the j-th anchor box of the i-th grid cell,
    and :math:`(\hat{x}_{ij}, \hat{y}_{ij}, \hat{w}_{ij}, \hat{h}_{ij})` are the coordinate predictions
    for the j-th anchor box of the i-th grid cell.

    .. math::
        \mathcal{L}_{objectness} = \sum\limits_{i=0}^{S^2} \sum\limits_{j=0}^{B}
        \Big[ \mathbb{1}_{ij}^{obj} \Big(C_{ij} - \hat{C}_{ij} \Big)^2
        + \lambda_{noobj} \mathbb{1}_{ij}^{noobj} \Big(C_{ij} - \hat{C}_{ij} \Big)^2
        \Big]

    where :math:`\lambda_{noobj}` is a positive coefficient (default: 0.5),
    :math:`\mathbb{1}_{ij}^{noobj} = 1 - \mathbb{1}_{ij}^{obj}`,
    :math:`C_{ij}` equals the Intersection Over Union between the j-th anchor box in the i-th grid cell and its
    matched ground truth box if that box is matched with a ground truth else 0,
    and :math:`\hat{C}_{ij}` is the objectness score of the j-th anchor box in the i-th grid cell..

    .. math::
        \mathcal{L}_{classification} = \sum\limits_{i=0}^{S^2}
        \mathbb{1}_{i}^{obj} \sum\limits_{c \in classes}
        (p_i(c) - \hat{p}_i(c))^2

    where :math:`\mathbb{1}_{i}^{obj}` equals to 1 if a GT center falls inside the i-th grid cell else 0,
    :math:`p_i(c)` equals 1 if the assigned ground truth to the i-th cell is classified as class :math:`c`,
    and :math:`\hat{p}_i(c)` is the predicted probability of class :math:`c` in the i-th cell.

    And the full loss is given by:

    .. math::
        \mathcal{L}_{YOLOv1} = \lambda_{coords} \cdot \mathcal{L}_{coords} +
        \mathcal{L}_{objectness} + \mathcal{L}_{classification}

    where :math:`\lambda_{coords}` is a positive coefficient (default: 5).

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet
        progress (bool, optional): If True, displays a progress bar of the download to stderr
        pretrained_backbone (bool, optional): If True, backbone parameters will have been pretrained on Imagenette
        kwargs: keyword args of _yolo

    Returns:
        torch.nn.Module: detection module
    """
    return _yolo(
        "yolov1",
        pretrained,
        progress,
        pretrained_backbone,
        [[192], [128, 256, 256, 512], [*([256, 512] * 4), 512, 1024], [512, 1024] * 2],
        **kwargs,
    )
