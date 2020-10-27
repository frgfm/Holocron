import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.boxes import box_iou, nms
from typing import Dict, Optional, Any, Callable, List, Tuple, Union

from ..utils import conv_sequence, load_pretrained_params
from ..darknet import DarknetBodyV1, default_cfgs as dark_cfgs
from holocron.nn.init import init_module


__all__ = ['YOLOv1', 'yolov1']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'yolov1': {'arch': 'YOLOv1', 'backbone': dark_cfgs['darknet24'],
               'url': None},
}


class _YOLO(nn.Module):
    def __init__(
        self,
        rpn_nms_thresh: float = 0.7,
        box_score_thresh: float = 0.05
    ) -> None:
        super().__init__()
        self.rpn_nms_thresh = rpn_nms_thresh
        self.box_score_thresh = box_score_thresh

    def _compute_losses(
        self,
        pred_boxes: Tensor,
        pred_o: Tensor,
        pred_scores: Tensor,
        target: List[Dict[str, Tensor]],
        ignore_high_iou=False
    ) -> Dict[str, Tensor]:
        """Computes the detector losses as described in `"You Only Look Once: Unified, Real-Time Object Detection"
        <https://pjreddie.com/media/files/papers/yolo_1.pdf>`_

        Args:
            pred_boxes (torch.Tensor[N, H, W, num_anchors, 4]): relative coordinates in format (xc, yc, w, h)
            pred_o (torch.Tensor[N, H, W, num_anchors]): objectness scores
            pred_scores (torch.Tensor[N, H, W, num_anchors, num_classes]): classification probabilities
            target (list<dict>, optional): list of targets

        Returns:
            dict: dictionary of losses
        """

        gt_boxes = [t['boxes'] for t in target]
        gt_labels = [t['labels'] for t in target]

        if not all(torch.all(boxes >= 0) and torch.all(boxes <= 1) for boxes in gt_boxes):
            raise ValueError("Ground truth boxes are expected to have values between 0 and 1.")

        b, h, w, _, num_classes = pred_scores.shape
        # Pred scores of YOLOv1 do not have the anchor dimension properly sized (only for broadcasting)
        num_anchors = pred_boxes.shape[3]
        # Initialize losses
        obj_loss = torch.zeros(1, device=pred_boxes.device)
        noobj_loss = torch.zeros(1, device=pred_boxes.device)
        bbox_loss = torch.zeros(1, device=pred_boxes.device)
        clf_loss = torch.zeros(1, device=pred_boxes.device)

        # Convert from (xcenter, ycenter, w, h) to (xmin, ymin, xmax, ymax)
        wh = pred_boxes[..., 2:]
        pred_xyxy = torch.cat((pred_boxes[..., :2] - wh / 2, pred_boxes[..., :2] + wh / 2), dim=-1)

        # B * cells * predictors * info
        for idx in range(b):

            # Match the anchor boxes
            is_matched = torch.arange(0)
            not_matched = torch.arange(h * w * num_anchors)
            if gt_boxes[idx].shape[0] > 0:
                # Locate the cell of each GT
                gt_centers = torch.stack((gt_boxes[idx][:, [0, 2]].mean(dim=-1) * w,
                                          gt_boxes[idx][:, [1, 3]].mean(dim=-1) * h), dim=1).to(dtype=torch.long)
                cell_idxs = gt_centers[:, 1] * w + gt_centers[:, 0]
                # Assign the best anchor in each corresponding cell
                iou_mat = box_iou(pred_xyxy[idx].view(-1, 4), gt_boxes[idx]).view(h * w, num_anchors, -1)
                iou_max = iou_mat[cell_idxs, :, range(gt_boxes[idx].shape[0])].max(dim=1)
                box_idxs = iou_max.indices
                # Keep IoU for loss computation
                selection_iou = iou_max.values
                # Update anchor box matching
                box_selection = torch.zeros((h * w, num_anchors), dtype=torch.bool)
                box_selection[cell_idxs, box_idxs] = True
                is_matched = torch.arange(h * w * num_anchors).view(-1, num_anchors)[cell_idxs, box_idxs]
                not_matched = not_matched[box_selection.view(-1)]

            # Update losses for boxes without any object
            if not_matched.shape[0] > 0:
                # SSE between objectness and IoU
                selection_o = pred_o.view(b, -1)[idx, not_matched]
                if ignore_high_iou and gt_boxes[idx].shape[0] > 0:
                    # Don't penalize anchor boxes with high IoU with GTs
                    selection_o = selection_o[iou_mat.view(h * w * num_anchors)[not_matched].max(dim=1).values < 0.5]
                # Update loss (target = 0)
                noobj_loss += selection_o.pow(2).sum()

            # Update loss for boxes with an object
            if is_matched.shape[0] > 0:
                # Get prediction assignment
                selection_o = pred_o.view(b, -1)[idx, is_matched]
                pred_filter = cell_idxs if (pred_scores.shape[3] == 1) else is_matched
                selected_scores = pred_scores.reshape(b, -1, num_classes)[idx, pred_filter].view(-1, num_classes)
                selected_boxes = pred_boxes.view(b, -1, 4)[idx, is_matched].view(-1, 4)
                # Convert GT --> xc, yc, w, h
                gt_wh = gt_boxes[idx][:, 2:] - gt_boxes[idx][:, :2]
                gt_centers = (gt_boxes[idx][:, 2:] + gt_boxes[idx][:, :2]) / 2
                # Make xy relative to cell
                gt_centers[:, 0] *= w
                gt_centers[:, 1] *= h
                gt_centers -= gt_centers.floor()
                selected_boxes[:, 0] *= w
                selected_boxes[:, 1] *= h
                selected_boxes[:, :2] -= selected_boxes[:, :2].floor()
                gt_probs = torch.zeros_like(selected_scores)
                gt_probs[range(gt_labels[idx].shape[0]), gt_labels[idx]] = 1

                # Localization
                # cf. YOLOv1 loss: SSE of xy preds, SSE of squared root of wh
                bbox_loss += F.mse_loss(selected_boxes[:, :2], gt_centers, reduction='sum')
                bbox_loss += F.mse_loss(selected_boxes[:, 2:].sqrt(), gt_wh.sqrt(), reduction='sum')
                # Objectness
                obj_loss += F.mse_loss(selection_o, selection_iou, reduction='sum')
                # Classification
                clf_loss += F.mse_loss(selected_scores, gt_probs, reduction='sum')

        return dict(obj_loss=obj_loss / pred_boxes.shape[0],
                    noobj_loss=self.lambda_noobj * noobj_loss / pred_boxes.shape[0],
                    bbox_loss=self.lambda_coords * bbox_loss / pred_boxes.shape[0],
                    clf_loss=clf_loss / pred_boxes.shape[0])

    @staticmethod
    def post_process(
        b_coords: Tensor,
        b_o: Tensor,
        b_scores: Tensor,
        rpn_nms_thresh=0.7,
        box_score_thresh=0.05
    ) -> List[Dict[str, Tensor]]:
        """Perform final filtering to produce detections

        Args:
            b_coords (torch.Tensor[N, H * W * num_anchors, 4]): relative coordinates in format (x, y, w, h)
            b_o (torch.Tensor[N, H * W * num_anchors]): objectness scores
            b_scores (torch.Tensor[N, H * W * num_anchors, num_classes]): classification scores
            rpn_nms_thresh (float, optional): IoU threshold for NMS
            box_score_thresh (float, optional): minimum classification confidence threshold

        Returns:
            list<dict>: detections dictionary
        """

        detections = []
        for idx in range(b_coords.shape[0]):

            coords = torch.zeros((0, 4), dtype=torch.float, device=b_o.device)
            scores = torch.zeros(0, dtype=torch.float, device=b_o.device)
            labels = torch.zeros(0, dtype=torch.long, device=b_o.device)

            # Objectness filter
            if torch.any(b_o[idx] >= 0.5):
                coords = b_coords[idx, b_o[idx] >= 0.5]
                scores, labels = b_scores[idx, b_o[idx] >= 0.5].max(dim=-1)
                # Multiply by the objectness
                scores.mul_(b_o[idx, b_o[idx] >= 0.5])

                # Confidence threshold
                coords = coords[scores >= box_score_thresh]
                labels = labels[scores >= box_score_thresh]
                scores = scores[scores >= box_score_thresh]

                # Switch to xmin, ymin, xmax, ymax coords
                wh = coords[..., 2:]
                coords = torch.cat((coords[..., :2] - wh / 2, coords[..., :2] + wh / 2), dim=1)
                coords = coords.clamp_(0, 1)
                # NMS
                kept_idxs = nms(coords, scores, iou_threshold=rpn_nms_thresh)
                coords = coords[kept_idxs]
                scores = scores[kept_idxs]
                labels = labels[kept_idxs]

            detections.append(dict(boxes=coords, scores=scores, labels=labels))

        return detections


class YOLOv1(_YOLO):
    def __init__(
        self,
        layout: List[List[int]],
        num_classes: int = 20,
        in_channels: int = 3,
        stem_channels: int = 64,
        num_anchors: int = 2,
        lambda_noobj: float = 0.5,
        lambda_coords: float = 5.,
        rpn_nms_thresh: float = 0.7,
        box_score_thresh: float = 0.05,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
        backbone_norm_layer: Optional[Callable[[int], nn.Module]] = None
    ) -> None:

        super().__init__(rpn_nms_thresh, box_score_thresh)

        if act_layer is None:
            act_layer = nn.LeakyReLU(0.1, inplace=True)

        if backbone_norm_layer is None and norm_layer is not None:
            backbone_norm_layer = norm_layer

        self.backbone = DarknetBodyV1(layout, in_channels, stem_channels, act_layer, backbone_norm_layer)

        self.block4 = nn.Sequential(
            *conv_sequence(1024, 1024, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=1, bias=False),
            *conv_sequence(1024, 1024, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=1, stride=2, bias=False),
            *conv_sequence(1024, 1024, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=1, bias=False),
            *conv_sequence(1024, 1024, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=1, bias=False))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 ** 2, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 7 ** 2 * (num_anchors * 5 + num_classes)))
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        # Loss coefficients
        self.lambda_noobj = lambda_noobj
        self.lambda_coords = lambda_coords

        init_module(self.block4, 'leaky_relu')
        init_module(self.classifier, 'leaky_relu')

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
        # B * (H * W * (num_anchors * 5 + num_classes)) --> B * H * W * (num_anchors * 5 + num_classes)
        x = x.view(b, h, w, self.num_anchors * 5 + self.num_classes)
        # Classification scores
        b_scores = x[..., -self.num_classes:]
        # Repeat for anchors to keep compatibility across YOLO versions
        b_scores = F.softmax(b_scores.unsqueeze(3), dim=-1)
        #  B * H * W * (num_anchors * 5 + num_classes) -->  B * H * W * num_anchors * 5
        x = x[..., :self.num_anchors * 5].view(b, h, w, self.num_anchors, 5)
        # Cell offset
        c_x = torch.arange(w, dtype=torch.float, device=x.device)
        c_y = torch.arange(h, dtype=torch.float, device=x.device)
        # Box coordinates
        b_x = (torch.sigmoid(x[..., 0]) + c_x.view(1, 1, -1, 1)) / w
        b_y = (torch.sigmoid(x[..., 1]) + c_y.view(1, -1, 1, 1)) / h
        b_w = torch.sigmoid(x[..., 2])
        b_h = torch.sigmoid(x[..., 3])
        # B * H * W * num_anchors * 4
        b_coords = torch.stack((b_x, b_y, b_w, b_h), dim=4)
        # Objectness
        b_o = torch.sigmoid(x[..., 4])

        return b_coords, b_o, b_scores

    def _forward(self, x: Tensor) -> Tensor:

        out = self.backbone(x)
        out = self.block4(out)
        out = self.classifier(out)

        return out

    def forward(
        self,
        x: Tensor,
        target: Optional[List[Dict[str, Tensor]]] = None
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

        # B * (H * W) * num_anchors
        b_coords, b_o, b_scores = self._format_outputs(out)

        if self.training:
            # Update losses
            return self._compute_losses(b_coords, b_o, b_scores, target)  # type: ignore[arg-type]
        else:
            # B * (H * W * num_anchors)
            b_coords = b_coords.view(b_coords.shape[0], -1, 4)
            b_o = b_o.view(b_o.shape[0], -1)
            # Repeat for each anchor box
            b_scores = b_scores.repeat_interleave(self.num_anchors, dim=3)
            b_scores = b_scores.contiguous().view(b_scores.shape[0], -1, self.num_classes)

            # Stack detections into a list
            return self.post_process(b_coords, b_o, b_scores, self.rpn_nms_thresh, self.box_score_thresh)


def _yolo(arch: str, pretrained: bool, progress: bool, pretrained_backbone: bool, **kwargs: Any) -> YOLOv1:

    if pretrained:
        pretrained_backbone = False

    # Build the model
    model = YOLOv1(default_cfgs[arch]['backbone']['layout'], **kwargs)
    # Load backbone pretrained parameters
    if pretrained_backbone:
        load_pretrained_params(model.backbone, default_cfgs[arch]['backbone']['url'], progress,
                               key_replacement=('features.', ''), key_filter='features.')
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'], progress)

    return model


def yolov1(pretrained: bool = False, progress: bool = True, pretrained_backbone: bool = True, **kwargs: Any) -> YOLOv1:
    """YOLO model from
    `"You Only Look Once: Unified, Real-Time Object Detection" <https://pjreddie.com/media/files/papers/yolo_1.pdf>`_.

    YOLO's particularity is to make predictions in a grid (same size as last feature map). For each grid cell,
    the model predicts classification scores and a fixed number of boxes (default: 2). Each box in the cell gets
    5 predictions: an objectness score, and 4 coordinates. The 4 coordinates are composed of: the 2-D coordinates of
    the predicted box center (relative to the cell), and the width and height of the predicted box (relative to
    the whole image).

    For training, YOLO uses a multi-part loss whose components are computed by:

    .. math::
        \\mathcal{L}_{coords} = \\sum\\limits_{i=0}^{S^2} \\sum\\limits_{j=0}^{B}
        \\mathbb{1}_{ij}^{obj} \\Big[
        (x_{ij} - \\hat{x}_{ij})² + (y_{ij} - \\hat{y}_{ij})² +
        (\\sqrt{w_{ij}} - \\sqrt{\\hat{w}_{ij}})² + (\\sqrt{h_{ij}} - \\sqrt{\\hat{h}_{ij}})²
        \\Big]

    where :math:`S` is size of the output feature map (7 for an input size :math:`(448, 448)`),
    :math:`B` is the number of anchor boxes per grid cell (default: 2),
    :math:`\\mathbb{1}_{ij}^{obj}` equals to 1 if a GT center falls inside the i-th grid cell and among the
    anchor boxes of that cell, has the highest IoU with the j-th box else 0,
    :math:`(x_{ij}, y_{ij}, w_{ij}, h_{ij})` are the coordinates of the ground truth assigned to
    the j-th anchor box of the i-th grid cell,
    and :math:`(\\hat{x}_{ij}, \\hat{y}_{ij}, \\hat{w}_{ij}, \\hat{h}_{ij})` are the coordinate predictions
    for the j-th anchor box of the i-th grid cell.

    .. math::
        \\mathcal{L}_{objectness} = \\sum\\limits_{i=0}^{S^2} \\sum\\limits_{j=0}^{B}
        \\Big[ \\mathbb{1}_{ij}^{obj} \\Big(C_{ij} - \\hat{C}_{ij} \\Big)^2
        + \\lambda_{noobj} \\mathbb{1}_{ij}^{noobj} \\Big(C_{ij} - \\hat{C}_{ij} \\Big)^2
        \\Big]

    where :math:`\\lambda_{noobj}` is a positive coefficient (default: 0.5),
    :math:`\\mathbb{1}_{ij}^{noobj} = 1 - \\mathbb{1}_{ij}^{obj}`,
    :math:`C_{ij}` equals the Intersection Over Union between the j-th anchor box in the i-th grid cell and its
    matched ground truth box if that box is matched with a ground truth else 0,
    and :math:`\\hat{C}_{ij}` is the objectness score of the j-th anchor box in the i-th grid cell..

    .. math::
        \\mathcal{L}_{classification} = \\sum\\limits_{i=0}^{S^2}
        \\mathbb{1}_{i}^{obj} \\sum\\limits_{c \\in classes}
        (p_i(c) - \\hat{p}_i(c))^2

    where :math:`\\mathbb{1}_{i}^{obj}` equals to 1 if a GT center falls inside the i-th grid cell else 0,
    :math:`p_i(c)` equals 1 if the assigned ground truth to the i-th cell is classified as class :math:`c`,
    and :math:`\\hat{p}_i(c)` is the predicted probability of class :math:`c` in the i-th cell.

    And the full loss is given by:

    .. math::
        \\mathcal{L}_{YOLOv1} = \\lambda_{coords} \\cdot \\mathcal{L}_{coords} +
        \\mathcal{L}_{objectness} + \\mathcal{L}_{classification}

    where :math:`\\lambda_{coords}` is a positive coefficient (default: 5).

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet
        progress (bool, optional): If True, displays a progress bar of the download to stderr
        pretrained_backbone (bool, optional): If True, backbone parameters will have been pretrained on Imagenette

    Returns:
        torch.nn.Module: detection module
    """

    return _yolo('yolov1', pretrained, progress, pretrained_backbone, **kwargs)  # type: ignore[return-value]
