# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops.boxes import batched_nms, box_iou

from holocron.nn.init import init_module

from ..classification.darknet import DarknetBodyV1
from ..classification.darknet import default_cfgs as dark_cfgs
from ..utils import conv_sequence, load_pretrained_params

__all__ = ["YOLOv1", "yolov1"]


default_cfgs: dict[str, dict[str, Any]] = {
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
        self.num_classes: int = num_classes
        self.rpn_nms_thresh: float = rpn_nms_thresh
        self.box_score_thresh: float = box_score_thresh
        self.lambda_obj: float = lambda_obj
        self.lambda_noobj: float = lambda_noobj
        self.lambda_class: float = lambda_class
        self.lambda_coords: float = lambda_coords

    def _compute_losses(
        self,
        pred_boxes: Tensor,
        pred_o: Tensor,
        pred_scores: Tensor,
        target: list[dict[str, Tensor]],
        ignore_high_iou: bool = False,
    ) -> dict[str, Tensor]:
        """Computes the detector losses as described in `"You Only Look Once: Unified, Real-Time Object Detection"
        <https://pjreddie.com/media/files/papers/yolo_1.pdf>`_

        Args:
            pred_boxes (torch.Tensor[N, H, W, num_anchors, 4]): relative coordinates in format (xc, yc, w, h)
            pred_o (torch.Tensor[N, H, W, num_anchors]): objectness scores
            pred_scores (torch.Tensor[N, H, W, num_anchors, num_classes]): classification probabilities
            target (list<dict>, optional): list of targets
            ignore_high_iou (bool): ignore the intersections with high IoUs in the noobj penalty term

        Returns:
            dict[str, Tensor]: dictionary of losses

        Raises:
            ValueError: if `target` is not specified in training mode
        """
        gt_boxes = [t["boxes"] for t in target]
        gt_labels = [t["labels"] for t in target]

        # GT xmin, ymin, xmax, ymax
        if not all(torch.all(boxes >= 0) and torch.all(boxes <= 1) for boxes in gt_boxes):
            raise ValueError("Ground truth boxes are expected to have values between 0 and 1.")

        b, h, w, _, _ = pred_scores.shape
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
            for idx_ in range(gt_boxes[idx].shape[0]):
                # Assign the anchor inside the cell
                iou_ = box_iou(gt_boxes[idx][idx_].unsqueeze(0), pred_xyxy[idx, gt_idcs[idx_, 1], gt_idcs[idx_, 0]])
                iou, anchor_idx = iou_.squeeze(0).max(dim=0)
                # Flag that there is an object here
                is_noobj[idx, gt_idcs[idx_, 1], gt_idcs[idx_, 0], anchor_idx] = False
                # Classification loss
                gt_scores = torch.zeros_like(pred_scores[idx, gt_idcs[idx_, 1], gt_idcs[idx_, 0]])
                gt_scores[:, gt_labels[idx][idx_]] = 1
                clf_loss += (gt_scores - pred_scores[idx, gt_idcs[idx_, 1], gt_idcs[idx_, 0]]).pow(2).sum()
                # Objectness loss
                obj_loss += (iou - pred_o[idx, gt_idcs[idx_, 1], gt_idcs[idx_, 0], anchor_idx]).pow(2)
                # Bbox loss
                bbox_loss += (
                    (gt_xy[idx_] - pred_xy[idx, gt_idcs[idx_, 1], gt_idcs[idx_, 0], anchor_idx, :2]).pow(2).sum()
                )
                bbox_loss += (
                    (gt_wh.sqrt() - pred_boxes[idx, gt_idcs[idx_, 1], gt_idcs[idx_, 0], anchor_idx, 2:].sqrt())
                    .pow(2)
                    .sum()
                )

            # Ignore high ious
            if ignore_high_iou:
                iou_ = box_iou(pred_xyxy[idx].reshape(-1, 4), gt_boxes[idx]).max(dim=-1).values.reshape(h, w, -1)
                is_noobj[idx, iou_ >= 0.5] = False
        # Non-objectness loss
        noobj_loss += pred_o[is_noobj].pow(2).sum()

        return {
            "obj_loss": self.lambda_obj * obj_loss / pred_boxes.shape[0],
            "noobj_loss": self.lambda_noobj * noobj_loss / pred_boxes.shape[0],
            "bbox_loss": self.lambda_coords * bbox_loss / pred_boxes.shape[0],
            "clf_loss": self.lambda_class * clf_loss / pred_boxes.shape[0],
        }

    @staticmethod
    def to_isoboxes(b_coords: Tensor, grid_shape: tuple[int, int], clamp: bool = False) -> Tensor:
        """Converts xywh boxes to xyxy format.

        Args:
            b_coords: tensor of shape (..., 4) where the last dimension is xcenter,ycenter,w,h
            grid_shape: the size of the grid
            clamp: whether the coords should be clamped to the extreme values

        Returns:
            tensor with the boxes using relative coords
        """
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
        grid_shape: tuple[int, int],
        rpn_nms_thresh: float = 0.7,
        box_score_thresh: float = 0.05,
    ) -> list[dict[str, Tensor]]:
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
            b_coords.reshape(-1, *grid_shape, self.num_anchors, 4),
            grid_shape,
            clamp=True,
        ).reshape(b_o.shape[0], -1, 4)

        return _post_process(pred_xyxy, b_o, b_scores, rpn_nms_thresh, box_score_thresh)


def _post_process(
    boxes: Tensor,
    b_o: Tensor,
    b_scores: Tensor,
    rpn_nms_thresh: float = 0.7,
    box_score_thresh: float = 0.05,
) -> list[dict[str, Tensor]]:
    if boxes.dtype in {torch.float16, torch.bfloat16}:
        boxes = boxes.float()
    b_o = b_o.to(dtype=boxes.dtype)
    b_scores = b_scores.to(dtype=boxes.dtype)
    detections: list[dict[str, Tensor]] = []
    for idx in range(boxes.shape[0]):
        scores, labels = b_scores[idx].max(dim=-1)
        scores *= b_o[idx]
        score_mask = scores >= box_score_thresh
        coords, scores, labels = boxes[idx, score_mask], scores[score_mask], labels[score_mask]
        kept_idxs = batched_nms(coords, scores, labels, iou_threshold=rpn_nms_thresh)
        detections.append({"boxes": coords[kept_idxs], "scores": scores[kept_idxs], "labels": labels[kept_idxs]})

    return detections


class YOLOv1(_YOLO):
    def __init__(
        self,
        layout: list[list[int]],
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
        act_layer: nn.Module | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
        drop_layer: Callable[..., nn.Module] | None = None,
        conv_layer: Callable[..., nn.Module] | None = None,
        backbone_norm_layer: Callable[[int], nn.Module] | None = None,
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
        self.num_anchors: int = num_anchors

        init_module(self.block4, "leaky_relu")
        init_module(self.classifier, "leaky_relu")

    def _format_outputs(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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
        return self.classifier(out)

    def forward(  # ty: ignore[invalid-mutable-override]
        self, x: Tensor, target: list[dict[str, Tensor]] | None = None
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """Perform detection on an image tensor and returns either the loss dictionary in training mode
        or the list of detections in eval mode.

        Args:
            x: input image tensor of shape (N, 3, H, W)
            target: each dict must have two keys `boxes` of type torch.Tensor[-1, 4] and `labels` of type torch.Tensor[-1]

        Returns:
            loss dictionary in training mode or list of detections in eval mode

        Raises:
            ValueError: if `target` is not specified in training mode
        """
        if self.training and target is None:
            raise ValueError("`target` needs to be specified in training mode")

        if isinstance(x, (list, tuple)):
            x = torch.stack(x, dim=0)  # ty: ignore[invalid-argument-type]

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
    arch: str, pretrained: bool, progress: bool, pretrained_backbone: bool, layout: list[list[int]], **kwargs: Any
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
    ["You Only Look Once: Unified, Real-Time Object Detection"](https://pjreddie.com/media/files/papers/yolo_1.pdf).

    YOLO's particularity is to make predictions in a grid (same size as last feature map). For each grid cell,
    the model predicts classification scores and a fixed number of boxes (default: 2). Each box in the cell gets
    5 predictions: an objectness score, and 4 coordinates. The 4 coordinates are composed of: the 2-D coordinates of
    the predicted box center (relative to the cell), and the width and height of the predicted box (relative to
    the whole image).

    For training, YOLO uses a multi-part loss whose components are computed by:

    $$
    \mathcal{L}_{coords} = \sum\limits_{i=0}^{S^2} \sum\limits_{j=0}^{B}
    \mathbb{1}_{ij}^{obj} \Big[
    (x_{ij} - \hat{x}_{ij})² + (y_{ij} - \hat{y}_{ij})² +
    (\sqrt{w_{ij}} - \sqrt{\hat{w}_{ij}})² + (\sqrt{h_{ij}} - \sqrt{\hat{h}_{ij}})²
    \Big]
    $$

    where:
    $S$ is size of the output feature map (7 for an input size $(448, 448)$),
    $B$ is the number of anchor boxes per grid cell (default: 2),
    $\mathbb{1}_{ij}^{obj}$ equals to 1 if a GT center falls inside the i-th grid cell and among the
    anchor boxes of that cell, has the highest IoU with the j-th box else 0,
    $(x_{ij}, y_{ij}, w_{ij}, h_{ij})$ are the coordinates of the ground truth assigned to
    the j-th anchor box of the i-th grid cell,
    $(\hat{x}_{ij}, \hat{y}_{ij}, \hat{w}_{ij}, \hat{h}_{ij})$ are the coordinate predictions
    for the j-th anchor box of the i-th grid cell.

    $$
    \mathcal{L}_{objectness} = \sum\limits_{i=0}^{S^2} \sum\limits_{j=0}^{B}
    \Big[ \mathbb{1}_{ij}^{obj} \Big(C_{ij} - \hat{C}_{ij} \Big)^2
    + \lambda_{noobj} \mathbb{1}_{ij}^{noobj} \Big(C_{ij} - \hat{C}_{ij} \Big)^2
    \Big]
    $$

    where $\lambda_{noobj}$ is a positive coefficient (default: 0.5),
    $\mathbb{1}_{ij}^{noobj} = 1 - \mathbb{1}_{ij}^{obj}$,
    $C_{ij}$ equals the Intersection Over Union between the j-th anchor box in the i-th grid cell and its
    matched ground truth box if that box is matched with a ground truth else 0,
    and $\hat{C}_{ij}$ is the objectness score of the j-th anchor box in the i-th grid cell..

    $$
    \mathcal{L}_{classification} = \sum\limits_{i=0}^{S^2}
    \mathbb{1}_{i}^{obj} \sum\limits_{c \in classes}
    (p_i(c) - \hat{p}_i(c))^2
    $$

    where $\mathbb{1}_{i}^{obj}$ equals to 1 if a GT center falls inside the i-th grid cell else 0,
    $p_i(c)$ equals 1 if the assigned ground truth to the i-th cell is classified as class $c$,
    and $\hat{p}_i(c)$ is the predicted probability of class $c$ in the i-th cell.

    And the full loss is given by:

    $$
    \mathcal{L}_{YOLOv1} = \lambda_{coords} \cdot \mathcal{L}_{coords} +
    \mathcal{L}_{objectness} + \mathcal{L}_{classification}
    $$

    where $\lambda_{coords}$ is a positive coefficient (default: 5).

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        progress: If True, displays a progress bar of the download to stderr
        pretrained_backbone: If True, backbone parameters will have been pretrained on Imagenette
        kwargs: keyword args of [`YOLOv1`][holocron.models.detection.yolo.YOLOv1]

    Returns:
        detection module
    """
    return _yolo(
        "yolov1",
        pretrained,
        progress,
        pretrained_backbone,
        [[192], [128, 256, 256, 512], [*([256, 512] * 4), 512, 1024], [512, 1024] * 2],
        **kwargs,
    )
