# -*- coding: utf-8 -*-

"""
Personal implementation of YOLO models
"""

import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.boxes import box_iou, nms
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.resnet import conv1x1, conv3x3

from ...nn import ConcatDownsample2d
from ...nn.init import init_module
from ..darknet import DarknetBodyV1, DarknetBodyV2, default_cfgs as dark_cfgs


__all__ = ['YOLOv1', 'YOLOv2', 'yolov1', 'yolov2']


default_cfgs = {
    'yolov1': {'arch': 'YOLOv1', 'layout': dark_cfgs['darknet24']['layout'],
               'url': None},
    'yolov2': {'arch': 'YOLOv2', 'layout': dark_cfgs['darknet19']['layout'],
               'url': None},
}


class _YOLO(nn.Module):
    @staticmethod
    def _compute_losses(pred_boxes, pred_o, pred_scores, gt_boxes, gt_labels):
        """Computes the detector losses as described in `"You Only Look Once: Unified, Real-Time Object Detection"
        <https://pjreddie.com/media/files/papers/yolo_1.pdf>`_

        Args:
            pred_boxes (torch.Tensor[N, H, W, num_anchors, 4]): relative coordinates in format (xc, yc, w, h)
            pred_o (torch.Tensor[N, H, W, num_anchors]): objectness scores
            pred_scores (torch.Tensor[N, H, W, num_anchors, num_classes]): classification scores
            gt_boxes (list<torch.Tensor[-1, 4]>): ground truth boxes in format (xmin, ymin, xmax, ymax)
            gt_labels (list<torch.Tensor>): ground truth labels

        Returns:
            dict: dictionary of losses
        """

        b, h, w, num_anchors = pred_o.shape
        # Reset losses
        objectness_loss = torch.zeros(w * h, device=pred_boxes.device)
        bbox_loss = torch.zeros(w * h, device=pred_boxes.device)
        clf_loss = torch.zeros(w * h, device=pred_boxes.device)

        # Convert from x, y, w, h to xmin, ymin, xmax, ymax
        pred_wh = pred_boxes[..., 2:]
        pred_boxes[..., 2:] = pred_boxes[..., :2] + pred_wh
        pred_boxes[..., :2] -= pred_wh / 2

        # B * cells * predictors * info
        for idx in range(b):

            # Locate grid cells where there is an object
            cell_selection = torch.zeros(h * w, dtype=torch.bool)
            # Selection the anchor boxes
            box_selection = torch.zeros((h * w, num_anchors), dtype=torch.bool)
            if gt_boxes[idx].shape[0] > 0:
                gt_centers = (torch.stack((gt_boxes[idx][:, [0, 2]].sum(dim=-1) * w,
                                           gt_boxes[idx][:, [1, 3]].sum(dim=-1) * h), dim=1) / 2).to(dtype=torch.long)
                cell_idxs = gt_centers[:, 1] * w + gt_centers[:, 0]
                iou_mat = box_iou(pred_boxes[idx].view(-1, 4), gt_boxes[idx]).view(h * w, num_anchors, -1)
                iou_max = iou_mat[cell_idxs, :, range(gt_boxes[idx].shape[0])].max(dim=1)
                box_idxs = iou_max.indices
                selection_iou = iou_max.values

                cell_selection[cell_idxs] = True
                box_selection[cell_idxs, box_idxs] = True

            # Update losses for cells without any object
            if torch.any(~cell_selection):
                # SSE between objectness and IoU
                selection_o = pred_o.view(b, h * w, -1)[idx, ~cell_selection].max(dim=-1).values
                # Update loss
                objectness_loss[~cell_selection] += 0.5 * F.mse_loss(selection_o, torch.zeros_like(selection_o),
                                                                     reduction='none')

            # Update loss for cells with an object
            if torch.any(cell_selection):
                # Get prediction assignment
                selection_o = pred_o.view(b, h * w, -1)[idx, cell_idxs, box_idxs].view(-1)
                selected_scores = pred_scores.view(b, h * w, num_anchors, -1)[idx, cell_idxs, box_idxs]
                selected_scores = selected_scores.view(-1, pred_scores.shape[-1])
                selected_boxes = pred_boxes.view(b, h * w, num_anchors, -1)[idx, cell_idxs, box_idxs].view(-1, 4)
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

                # Localization
                # cf. YOLOv1 loss: SSE of xy preds, SSE of squared root of wh
                bbox_loss[cell_idxs] += F.mse_loss(selected_boxes[:, :2], gt_centers,
                                                        reduction='none').sum(dim=-1)
                bbox_loss[cell_idxs] += F.mse_loss(selected_boxes[:, 2:].sqrt(), gt_wh.sqrt(),
                                                        reduction='none').sum(dim=-1)
                # Objectness
                objectness_loss[cell_idxs] += F.mse_loss(selection_o, selection_iou, reduction='none')
                # Classification
                clf_loss[cell_idxs] += F.cross_entropy(selected_scores, gt_labels[idx], reduction='none')

        return dict(objectness_loss=objectness_loss.sum() / pred_boxes.shape[0],
                    bbox_loss=bbox_loss.sum() / pred_boxes.shape[0],
                    clf_loss=clf_loss.sum() / pred_boxes.shape[0])

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

            coords = torch.zeros((0, 4), dtype=torch.float, device=b_o.device)
            scores = torch.zeros(0, dtype=torch.float, device=b_o.device)
            labels = torch.zeros(0, dtype=torch.long, device=b_o.device)

            # Objectness filter
            if torch.any(b_o[idx] >= 0.5):
                coords = b_coords[idx, b_o[idx] >= 0.5]
                scores = b_scores[idx, b_o[idx] >= 0.5].max(dim=-1)
                labels = scores.indices
                scores = scores.values

                # NMS
                # Switch to xmin, ymin, xmax, ymax coords
                wh = coords[..., 2:]
                coords[..., 2:] /= 2
                coords[..., 2:] += coords[..., :2]
                coords[..., :2] -= wh / 2
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


class YOLOv1(_YOLO):

    def __init__(self, layout, num_classes=20, num_anchors=2):

        super().__init__()

        self.backbone = DarknetBodyV1(layout)

        self.block4 = nn.Sequential(
            conv3x3(1024, 1024),
            conv3x3(1024, 1024, stride=2),
            conv3x3(1024, 1024),
            conv3x3(1024, 1024))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 ** 2, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, 7 ** 2 * (num_anchors * 5 + num_classes)))
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        init_module(self, 'leaky_relu')

    def _format_outputs(self, x, img_h, img_w):
        """Formats convolutional layer output

        Args:
            x (torch.Tensor[N, num_anchors * (5 + num_classes) * H * W]): output tensor
            img_h (int): input image height
            img_w (int): input image width

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
        b_scores = b_scores.unsqueeze(3).repeat_interleave(self.num_anchors, dim=3)
        #  B * H * W * (num_anchors * 5 + num_classes) -->  B * H * W * num_anchors * 5
        x = x[..., :self.num_anchors * 5].view(b, h, w, self.num_anchors, 5)
        # Cell offset
        c_x = torch.arange(0, w, dtype=torch.float, device=x.device) / w
        c_y = torch.arange(0, h, dtype=torch.float, device=x.device) / h
        # Box coordinates
        b_x = torch.sigmoid(x[..., 0]) / w + c_x.view(1, 1, -1, 1)
        b_y = torch.sigmoid(x[..., 1]) / h + c_y.view(1, -1, 1, 1)
        b_w = torch.sigmoid(x[..., 2])
        b_h = torch.sigmoid(x[..., 3])
        # B * H * W * num_anchors * 4
        b_coords = torch.stack((b_x, b_y, b_w, b_h), dim=4)
        # Objectness
        b_o = torch.sigmoid(x[..., 4])

        return b_coords, b_o, b_scores

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

        if isinstance(x, (list, tuple)):
            x = torch.stack(x, dim=0)

        img_h, img_w = x.shape[-2:]
        x = self.backbone(x)
        x = self.block4(x)
        x = self.classifier(x)

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


class YOLOv2(_YOLO):

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

        init_module(self, 'leaky_relu')

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
            torch.Tensor[N, H, W, num_anchors, 4]: relative coordinates in format (x, y, w, h)
            torch.Tensor[N, H, W, num_anchors]: objectness scores
            torch.Tensor[N, H, W, num_anchors, num_classes]: classification scores
        """

        b, _, h, w = x.shape
        # B * C * H * W --> B * H * W * num_anchors * (5 + num_classes)
        x = x.view(b, self.num_anchors, 5 + self.num_classes, h, w).permute(0, 3, 4, 1, 2)
        # Cell offset
        c_x = torch.arange(0, w, dtype=torch.float, device=x.device) / w
        c_y = torch.arange(0, h, dtype=torch.float, device=x.device) / h
        # Box coordinates
        b_x = torch.sigmoid(x[..., 0]) / w + c_x.view(1, 1, -1, 1)
        b_y = torch.sigmoid(x[..., 1]) / h + c_y.view(1, -1, 1, 1)
        b_w = self.anchors[:, 0].view(1, 1, 1, -1) / w * torch.exp(x[..., 2])
        b_h = self.anchors[:, 1].view(1, 1, 1, -1) / h * torch.exp(x[..., 3])
        # B * H * W * num_anchors * 4
        b_coords = torch.stack((b_x, b_y, b_w, b_h), dim=4)
        # Objectness
        b_o = torch.sigmoid(x[..., 4])
        # Classification scores
        b_scores = x[..., 5:]

        return b_coords, b_o, b_scores

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

        if isinstance(x, (list, tuple)):
            x = torch.stack(x, dim=0)

        img_h, img_w = x.shape[-2:]
        x, passthrough = self.backbone(x)
        # Downsample the feature map by stacking adjacent features on the channel dimension
        passthrough = self.reorg_layer(passthrough)

        x = self.block5(x)
        # Stack the downsampled feature map on the channel dimension
        x = torch.cat((passthrough, x), 1)
        x = self.block6(x)

        x = self.head(x)

        # B * H * W * num_anchors
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


def _yolo(arch, pretrained, progress, **kwargs):

    # Retrieve the correct Darknet layout type
    yolo_type = sys.modules[__name__].__dict__[default_cfgs[arch]['arch']]
    # Build the model
    model = yolo_type(default_cfgs[arch]['layout'], **kwargs)
    # Load pretrained parameters
    if pretrained:
        if default_cfgs[arch]['url'] is None:
            logging.warning(f"Invalid model URL for {arch}, using default initialization.")
        else:
            state_dict = load_state_dict_from_url(default_cfgs[arch]['url'],
                                                  progress=progress)
            model.load_state_dict(state_dict)

    return model


def yolov1(pretrained=False, progress=True, **kwargs):
    """YOLO model from
    `"You Only Look Once: Unified, Real-Time Object Detection" <https://pjreddie.com/media/files/papers/yolo_1.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: detection module
    """

    return _yolo('yolov1', pretrained, progress, **kwargs)


def yolov2(pretrained=False, progress=True, **kwargs):
    """YOLOv2 model from
    `"YOLO9000: Better, Faster, Stronger" <https://pjreddie.com/media/files/papers/YOLO9000.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: detection module
    """

    return _yolo('yolov2', pretrained, progress, **kwargs)
