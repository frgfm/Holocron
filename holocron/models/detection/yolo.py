# -*- coding: utf-8 -*-

"""
Personal implementation of YOLOv2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.boxes import box_iou, nms

from ...nn import ConcatDownsample2d
from ..darknet import conv1x1, conv3x3, DarknetBodyV2


__all__ = ['YOLOv2', 'yolov2']


class YOLOv2(nn.Module):

    def __init__(self, layout, num_classes=20, anchors=None):

        super().__init__()

        # Priors computed using K-means
        if anchors is None:
            anchors = torch.tensor([[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]])
        self.num_classes = num_classes

        self.backbone = DarknetBodyV2(layout, passthrough=True)

        self.reorg_layer = ConcatDownsample2d(scale_factor=2)

        self.block5 = nn.Sequential(
            conv3x3(layout[-1][0], layout[-1][0]),
            nn.BatchNorm2d(layout[-1][0]),
            nn.LeakyReLU(0.1, inplace=True),
            conv3x3(layout[-1][0], layout[-1][0]),
            nn.BatchNorm2d(layout[-1][0]),
            nn.LeakyReLU(0.1, inplace=True))

        self.block6 = nn.Sequential(
            conv3x3(layout[-1][0] + layout[-2][0] * 2 ** 2, layout[-1][0]),
            nn.BatchNorm2d(layout[-1][0]),
            nn.LeakyReLU(0.1, inplace=True))

        # Each box has P_objectness, 4 coords, and score for each class
        self.head = conv1x1(layout[-1][0], anchors.shape[0] * (5 + num_classes))

        # Register losses
        self.register_buffer('anchors', anchors)

    @property
    def num_anchors(self):
        return self.anchors.shape[0]

    def _format_outputs(self, x, img_h, img_w):
        """Formats convolutional layer output

        Args:
            x (torch.Tensor[N, num_anchors * (5 + num_classes), H, W]): output tensor
            img_h (int): input image height
            img_w (int): input image width

        Returns:
            torch.Tensor[N, H * W, num_anchors, 4]: relative coordinates in format (x, y, w, h)
            torch.Tensor[N, H * W, num_anchors]: objectness scores
            torch.Tensor[N, H * W, num_anchors, num_classes]: classification scores
        """

        b, c, h, w = x.shape
        # B * C * H * W --> B * H * W * num_anchors * (5 + num_classes)
        x = x.view(b, self.num_anchors, 5 + self.num_classes, h, w).permute(0, 3, 4, 1, 2)
        # Cell offset
        c_x = torch.arange(0, w, dtype=torch.float) * img_w / w
        c_y = torch.arange(0, h, dtype=torch.float) * img_h / h
        # Box coordinates
        b_x = (torch.sigmoid(x[..., 0]) + c_x.view(1, 1, -1, 1)).view(b, -1, self.anchors.shape[0])
        b_y = (torch.sigmoid(x[..., 1]) + c_y.view(1, -1, 1, 1)).view(b, -1, self.anchors.shape[0])
        # B * H * W * num_anchors * (5 + num_classes) --> B * (H * W) * num_anchors * (5 + num_classes)
        x = x.view(b, h * w, self.num_anchors, -1)
        b_w = img_w / w * (self.anchors[:, 0].view(1, 1, -1) * torch.exp(x[..., 2]))
        b_h = img_h / h * (self.anchors[:, 1].view(1, 1, -1) * torch.exp(x[..., 3]))
        # B * (H * W) * num_anchors * 4
        b_coords = torch.stack((b_x, b_y, b_w, b_h), dim=3)
        # Objectness
        b_o = torch.sigmoid(x[..., 4])
        # Classification scores
        b_scores = x[..., 5:]

        return b_coords, b_o, b_scores

    def _compute_losses(self, pred_boxes, pred_o, pred_scores, gt_boxes, gt_labels):
        """Computes the detector losses as described in `"You Only Look Once: Unified, Real-Time Object Detection"
        <https://pjreddie.com/media/files/papers/yolo_1.pdf>`_

        Args:
            pred_boxes (torch.Tensor[N, H * W, num_anchors, 4]): relative coordinates in format (x, y, w, h)
            pred_o (torch.Tensor[N, H * W, num_anchors]): objectness scores
            pred_scores (torch.Tensor[N, H * W, num_anchors, num_classes]): classification scores
            gt_boxes (list<torch.Tensor[-1, 4]>): ground truth boxes
            gt_labels (list<torch.Tensor>): ground truth labels

        Returns:
            dict: dictionary of losses
        """

        # Reset losses
        objectness_loss = torch.zeros(1).to(pred_boxes.device)
        bbox_loss = torch.zeros(1).to(pred_boxes.device)
        clf_loss = torch.zeros(1).to(pred_boxes.device)
        # Convert from x, y, w, h to xmin, ymin, xmax, ymax
        pred_boxes[..., 2:] += pred_boxes[..., :2]
        # B * cells * predictors * info
        for idx in range(pred_boxes.shape[0]):

            # cells * predictors * num_gt
            iou_mat = box_iou(pred_boxes[idx].view(-1, 4), gt_boxes[idx])
            # Assign in each cell the best box predictors
            # Compute max IoU for each predictor, take the highest predictor in each cell
            cell_selection = iou_mat.view(pred_boxes.shape[1], -1).max(dim=1).values > 0
            if torch.any(cell_selection):
                # S * predictors * num_gt
                iou_mat = iou_mat.view(pred_boxes.shape[1], self.num_anchors, -1)[cell_selection]
                anchor_selection = iou_mat.max(dim=2).values.argmax(dim=1)
                selection = [idx * self.num_anchors + anchor.item() for idx, anchor in enumerate(anchor_selection)]
                # Predictor selection
                selected_pred_boxes = pred_boxes[idx, cell_selection].view(-1, 4)[selection]
                selected_o = pred_o[idx, cell_selection].view(-1)[selection]
                selected_scores = pred_scores[idx, cell_selection].view(-1, self.num_classes)[selection]
                # GT selection
                max_iou = iou_mat.view(-1, gt_boxes[idx].shape[0])[selection].max(dim=1)
                selected_gt_boxes = gt_boxes[idx][max_iou.indices]
                select_gt_labels = gt_labels[idx][max_iou.indices]

                # Objectness loss for cells where any object was detected
                objectness_loss += F.mse_loss(selected_o, max_iou.values, reduction='sum')
                # Regression loss
                # cf. YOLOv1 loss: SSE of xy preds, SSE of squared root of wh
                bbox_loss += F.mse_loss(selected_pred_boxes[..., :2], selected_gt_boxes[..., :2], reduction='sum')
                bbox_loss += F.mse_loss(selected_pred_boxes[..., 2:].sqrt(), selected_gt_boxes[..., 2:].sqrt(),
                                        reduction='sum')
                # Classification loss
                clf_loss += F.cross_entropy(selected_scores, select_gt_labels, reduction='sum')

            # Objectness loss for cells where no object was detected
            empty_cell_o = pred_o[idx][~cell_selection].max(dim=1).values
            objectness_loss += 0.5 * F.mse_loss(empty_cell_o, torch.zeros_like(empty_cell_o), reduction='sum')

        return dict(objectness_loss=objectness_loss,
                    bbox_loss=bbox_loss,
                    clf_loss=clf_loss)

    @staticmethod
    def post_process(b_coords, b_o, b_scores, rpn_nms_thresh=0.7, box_score_thresh=0.05):
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

            coords = torch.zeros((0, 4), dtype=torch.float).to(device=b_o.device)
            scores = torch.zeros(0, dtype=torch.float).to(device=b_o.device)
            labels = torch.zeros(0, dtype=torch.long).to(device=b_o.device)

            # Objectness filter
            if torch.any(b_o[idx] >= 0.5):
                coords = b_coords[idx, b_o[idx] >= 0.5]
                scores = b_scores[idx, b_o[idx] >= 0.5].max(dim=-1)
                labels = scores.indices
                scores = scores.values

                # NMS
                # Switch to xmin, ymin, xmax, ymax coords
                coords[..., -2:] += coords[..., :2]
                coords = coords.clamp_(0, 1)
                is_kept = nms(coords, scores, iou_threshold=rpn_nms_thresh)
                coords = coords[is_kept]
                scores = scores[is_kept]
                labels = labels[is_kept]

                # Confidence threshold
                coords = coords[scores >= box_score_thresh]
                labels = labels[scores >= box_score_thresh]
                scores = scores[scores >= box_score_thresh]

            detections.append(dict(boxes=coords, scores=scores, labels=labels))

        return detections

    def forward(self, x, gt_boxes=None, gt_labels=None):
        """Perform detection on an image tensor and returns either the loss dictionary in training mode
        or the list of detections in eval mode.

        Args:
            x (torch.Tensor[N, 3, H, W]): input image tensor
            gt_boxes (list<torch.Tensor[-1, 4]>, optional): ground truth boxes relative coordinates
            in format [xmin, ymin, xmax, ymax]
            gt_labels (list<torch.Tensor[-1]>, optional): ground truth labels
        """

        if self.training and (gt_boxes is None or gt_labels is None):
            raise ValueError("`gt_boxes` and `gt_labels` need to be specified in training mode")

        img_h, img_w = x.shape[-2:]
        x, passthrough = self.backbone(x)
        # Downsample the feature map by stacking adjacent features on the channel dimension
        passthrough = self.reorg_layer(passthrough)

        x = self.block5(x)
        # Stack the downsampled feature map on the channel dimension
        x = torch.cat((passthrough, x), 1)
        x = self.block6(x)

        x = self.head(x)

        # B * (H * W) * num_anchors
        b_coords, b_o, b_scores = self._format_outputs(x, img_h, img_w)

        if self.training:
            # Update losses
            return self._compute_losses(b_coords, b_o, b_scores, gt_boxes, gt_labels)
        else:
            # B * (H * W * num_anchors)
            b_coords = b_coords.view(b_coords.shape[0], -1, 4)
            b_o = b_o.view(b_o.shape[0], -1)
            b_scores = b_scores.contiguous().view(b_scores.shape[0], -1, self.num_classes)

            # Stack detections into a list
            return self.post_process(b_coords, b_o, b_scores)


def yolov2(num_classes=20, anchors=None):
    """YOLOv2 model from
    `"YOLO9000: Better, Faster, Stronger" <https://pjreddie.com/media/files/papers/YOLO9000.pdf>`_

    Args:
        num_classes (int, optional): number of output classes
        anchors (torch.Tensor[N, 2]): priors for anchors

    Returns:
        torch.nn.Module: detection module
    """

    return YOLOv2([(128, 1), (256, 1), (512, 2), (1024, 2)], num_classes, anchors)
