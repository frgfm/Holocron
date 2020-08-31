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
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models.utils import load_state_dict_from_url

from ..utils import conv_sequence
from ..darknet import DarknetBodyV1, DarknetBodyV2, DarknetBodyV4, default_cfgs as dark_cfgs
from holocron.ops.boxes import ciou_loss
from holocron.nn import ConcatDownsample2d, Mish, DropBlock2d
from holocron.nn.init import init_module


__all__ = ['YOLOv1', 'YOLOv2', 'YOLOv4', 'yolov1', 'yolov2', 'yolov4', 'SPP', 'PAN', 'Neck']


default_cfgs = {
    'yolov1': {'arch': 'YOLOv1', 'backbone': dark_cfgs['darknet24'],
               'url': None},
    'yolov2': {'arch': 'YOLOv2', 'backbone': dark_cfgs['darknet19'],
               'url': None},
    'yolov4': {'arch': 'YOLOv4', 'backbone': dark_cfgs['cspdarknet53'],
               'url': None},
}


class _YOLO(nn.Module):
    def __init__(self, rpn_nms_thresh=0.7, box_score_thresh=0.05):
        super().__init__()
        self.rpn_nms_thresh = rpn_nms_thresh
        self.box_score_thresh = box_score_thresh

    def _compute_losses(self, pred_boxes, pred_o, pred_scores, gt_boxes, gt_labels, ignore_high_iou=False):
        """Computes the detector losses as described in `"You Only Look Once: Unified, Real-Time Object Detection"
        <https://pjreddie.com/media/files/papers/yolo_1.pdf>`_

        Args:
            pred_boxes (torch.Tensor[N, H, W, num_anchors, 4]): relative coordinates in format (xc, yc, w, h)
            pred_o (torch.Tensor[N, H, W, num_anchors]): objectness scores
            pred_scores (torch.Tensor[N, H, W, num_anchors, num_classes]): classification probabilities
            gt_boxes (list<torch.Tensor[-1, 4]>): ground truth boxes in format (xmin, ymin, xmax, ymax)
            gt_labels (list<torch.Tensor>): ground truth labels

        Returns:
            dict: dictionary of losses
        """

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
                gt_centers = (torch.stack((gt_boxes[idx][:, [0, 2]].sum(dim=-1) * w,
                                           gt_boxes[idx][:, [1, 3]].sum(dim=-1) * h), dim=1) / 2).to(dtype=torch.long)
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
                # Multiply by the objectness
                scores = scores.values * b_o[idx, b_o[idx] >= 0.5]

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
    def __init__(self, layout, num_classes=20, in_channels=3, stem_channels=64, num_anchors=2,
                 lambda_noobj=0.5, lambda_coords=5., rpn_nms_thresh=0.7, box_score_thresh=0.05,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None, backbone_norm_layer=None):

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

    def _format_outputs(self, x):
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

    def _forward(self, x):

        out = self.backbone(x)
        out = self.block4(out)
        out = self.classifier(out)

        return out

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

        out = self._forward(x)

        # B * (H * W) * num_anchors
        b_coords, b_o, b_scores = self._format_outputs(out)

        if self.training:
            # Update losses
            return self._compute_losses(b_coords, b_o, b_scores, gt_boxes, gt_labels)
        else:
            # B * (H * W * num_anchors)
            b_coords = b_coords.view(b_coords.shape[0], -1, 4)
            b_o = b_o.view(b_o.shape[0], -1)
            # Repeat for each anchor box
            b_scores = b_scores.repeat_interleave(self.num_anchors, dim=3)
            b_scores = b_scores.contiguous().view(b_scores.shape[0], -1, self.num_classes)

            # Stack detections into a list
            return self.post_process(b_coords, b_o, b_scores, self.rpn_nms_thresh, self.box_score_thresh)


class YOLOv2(_YOLO):

    def __init__(self, layout, num_classes=20, in_channels=3, stem_chanels=32, anchors=None, passthrough_ratio=8,
                 lambda_noobj=0.5, lambda_coords=5., rpn_nms_thresh=0.7, box_score_thresh=0.05,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None, backbone_norm_layer=None):

        super().__init__(rpn_nms_thresh, box_score_thresh)

        if act_layer is None:
            act_layer = nn.LeakyReLU(0.1, inplace=True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if backbone_norm_layer is None:
            backbone_norm_layer = norm_layer

        # Priors computed using K-means
        if anchors is None:
            anchors = torch.tensor([[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892],
                                    [9.47112, 4.84053], [11.2364, 10.0071]])
        self.num_classes = num_classes

        self.backbone = DarknetBodyV2(layout, in_channels, stem_chanels, True, act_layer,
                                      backbone_norm_layer, drop_layer, conv_layer)

        self.block5 = nn.Sequential(
            *conv_sequence(layout[-1][0], layout[-1][0], act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=1, bias=False),
            *conv_sequence(layout[-1][0], layout[-1][0], act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=1, bias=False))

        self.passthrough_layer = nn.Sequential(*conv_sequence(layout[-2][0], layout[-2][0] // passthrough_ratio,
                                                              act_layer, norm_layer, drop_layer, conv_layer,
                                                              kernel_size=1, bias=False),
                                               ConcatDownsample2d(scale_factor=2))

        self.block6 = nn.Sequential(
            *conv_sequence(layout[-1][0] + layout[-2][0] // passthrough_ratio * 2 ** 2, layout[-1][0],
                           act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=1, bias=False))

        # Each box has P_objectness, 4 coords, and score for each class
        self.head = nn.Conv2d(layout[-1][0], anchors.shape[0] * (5 + num_classes), 1)

        # Register losses
        self.register_buffer('anchors', anchors)

        # Loss coefficients
        self.lambda_noobj = lambda_noobj
        self.lambda_coords = lambda_coords

        init_module(self.block5, 'leaky_relu')
        init_module(self.passthrough_layer, 'leaky_relu')
        init_module(self.block6, 'leaky_relu')
        init_module(self.head, 'leaky_relu')

    @property
    def num_anchors(self):
        return self.anchors.shape[0]

    def _format_outputs(self, x):
        """Formats convolutional layer output

        Args:
            x (torch.Tensor[N, num_anchors * (5 + num_classes), H, W]): output tensor

        Returns:
            torch.Tensor[N, H, W, num_anchors, 4]: relative coordinates in format (x, y, w, h)
            torch.Tensor[N, H, W, num_anchors]: objectness scores
            torch.Tensor[N, H, W, num_anchors, num_classes]: classification scores
        """

        b, _, h, w = x.shape
        # B * C * H * W --> B * H * W * num_anchors * (5 + num_classes)
        x = x.view(b, self.num_anchors, 5 + self.num_classes, h, w).permute(0, 3, 4, 1, 2)
        # Cell offset
        c_x = torch.arange(w, dtype=torch.float, device=x.device)
        c_y = torch.arange(h, dtype=torch.float, device=x.device)
        # Box coordinates
        b_x = (torch.sigmoid(x[..., 0]) + c_x.view(1, 1, -1, 1)) / w
        b_y = (torch.sigmoid(x[..., 1]) + c_y.view(1, -1, 1, 1)) / h
        b_w = self.anchors[:, 0].view(1, 1, 1, -1) / w * torch.exp(x[..., 2])
        b_h = self.anchors[:, 1].view(1, 1, 1, -1) / h * torch.exp(x[..., 3])
        # B * H * W * num_anchors * 4
        b_coords = torch.stack((b_x, b_y, b_w, b_h), dim=4)
        # Objectness
        b_o = torch.sigmoid(x[..., 4])
        # Classification scores
        b_scores = F.softmax(x[..., 5:], dim=-1)

        return b_coords, b_o, b_scores

    def _forward(self, x):

        out, passthrough = self.backbone(x)
        # Downsample the feature map by stacking adjacent features on the channel dimension
        passthrough = self.passthrough_layer(passthrough)

        out = self.block5(out)
        # Stack the downsampled feature map on the channel dimension
        out = torch.cat((passthrough, out), 1)
        out = self.block6(out)

        out = self.head(out)

        return out

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

        out = self._forward(x)

        # B * H * W * num_anchors
        b_coords, b_o, b_scores = self._format_outputs(out)

        if self.training:
            # Update losses
            return self._compute_losses(b_coords, b_o, b_scores, gt_boxes, gt_labels)
        else:
            # B * (H * W * num_anchors)
            b_coords = b_coords.view(b_coords.shape[0], -1, 4)
            b_o = b_o.view(b_o.shape[0], -1)
            b_scores = b_scores.reshape(b_scores.shape[0], -1, self.num_classes)

            # Stack detections into a list
            return self.post_process(b_coords, b_o, b_scores, self.rpn_nms_thresh, self.box_score_thresh)


class SAM(nn.Module):
    """SAM layer from `"CBAM: Convolutional Block Attention Module" <https://arxiv.org/pdf/1807.06521.pdf>`_
    modified in `"YOLOv4: Optimal Speed and Accuracy of Object Detection" <https://arxiv.org/pdf/2004.10934.pdf>`_.

    Args:
        in_channels (int): input channels
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        return x * torch.sigmoid(self.conv(x))


class SPP(nn.ModuleList):
    """SPP layer from `"Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition"
    <https://arxiv.org/pdf/1406.4729.pdf>`_.

    Args:
        kernel_sizes (list<int>): kernel sizes of each pooling
    """

    def __init__(self, kernel_sizes):
        super().__init__([nn.MaxPool2d(k_size, stride=1, padding=k_size // 2)
                          for k_size in kernel_sizes])

    def forward(self, x):
        feats = [x]
        for pool_layer in self:
            feats.append(pool_layer(x))
        return torch.cat(feats, dim=1)


class PAN(nn.Module):
    """PAN layer from `"Path Aggregation Network for Instance Segmentation" <https://arxiv.org/pdf/1803.01534.pdf>`_.

    Args:
        in_channels (int): input channels
        act_layer (torch.nn.Module, optional): activation layer to be used
        norm_layer (callable, optional): normalization layer
        drop_layer (callable, optional): regularization layer
        conv_layer (callable, optional): convolutional layer
    """
    def __init__(self, in_channels, act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):
        super().__init__()

        self.conv1 = nn.Sequential(*conv_sequence(in_channels, in_channels // 2,
                                                  act_layer, norm_layer, drop_layer, conv_layer,
                                                  kernel_size=1, bias=False))
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv2 = nn.Sequential(*conv_sequence(in_channels, in_channels // 2,
                                                  act_layer, norm_layer, drop_layer, conv_layer,
                                                  kernel_size=1, bias=False))

        self.convs = nn.Sequential(
            *conv_sequence(in_channels, in_channels // 2, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=1, bias=False),
            *conv_sequence(in_channels // 2, in_channels, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=1, bias=False),
            *conv_sequence(in_channels, in_channels // 2, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=1, bias=False),
            *conv_sequence(in_channels // 2, in_channels, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=1, bias=False),
            *conv_sequence(in_channels, in_channels // 2, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=1, bias=False))

    def forward(self, x, up):
        out = self.conv1(x)

        out = torch.cat([self.conv2(up), self.up(out)], dim=1)

        return self.convs(out)


class Neck(nn.Module):
    def __init__(self, in_planes, act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):
        super().__init__()

        self.fpn = nn.Sequential(
            *conv_sequence(in_planes[0], in_planes[0] // 2, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=1, bias=False),
            *conv_sequence(in_planes[0] // 2, in_planes[0], act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=1, bias=False),
            *conv_sequence(in_planes[0], in_planes[0] // 2, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=1, bias=False),
            SPP([5, 9, 13]),
            *conv_sequence(4 * in_planes[0] // 2, in_planes[0] // 2, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=1, bias=False),
            *conv_sequence(in_planes[0] // 2, in_planes[0], act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=1, bias=False),
            *conv_sequence(in_planes[0], in_planes[0] // 2, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=1, bias=False)
        )

        self.pan1 = PAN(in_planes[1], act_layer, norm_layer, drop_layer, conv_layer)
        self.pan2 = PAN(in_planes[2], act_layer, norm_layer, drop_layer, conv_layer)

    def forward(self, feats):

        out = self.fpn(feats[2])

        aux1 = self.pan1(out, feats[1])
        aux2 = self.pan2(aux1, feats[0])

        return aux2, aux1, out


class YoloLayer(nn.Module):
    """Scale-specific part of YoloHead"""
    def __init__(self, anchors, num_classes=80, scale_xy=1, iou_thresh=0.213, eps=1e-16,
                 lambda_noobj=0.5, lambda_coords=5., rpn_nms_thresh=0.7, box_score_thresh=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('anchors', torch.tensor(anchors, dtype=torch.float32) / 608)

        self.rpn_nms_thresh = rpn_nms_thresh
        self.box_score_thresh = box_score_thresh
        self.lambda_noobj = lambda_noobj
        self.lambda_coords = lambda_coords

        # cf. https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg#L1150
        self.scale_xy = scale_xy
        # cf. https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg#L1151
        self.iou_thresh = iou_thresh

        self.eps = eps

    def _format_outputs(self, output):
        b, _, h, w = output.shape

        # B x (num_anchors * (5 + num_classes)) x H x W --> B x H x W x num_anchors x (5 + num_classes)
        output = output.reshape(b, len(self.scaled_anchors), 5 + self.num_classes, h, w).permute(0, 3, 4, 1, 2)

        # Box center
        c_x = torch.arange(w, dtype=torch.float32, device=output.device).view(1, 1, -1, 1)
        c_y = torch.arange(h, dtype=torch.float32, device=output.device).view(1, -1, 1, 1)

        b_xy = self.scale_xy * torch.sigmoid(output[..., :2]) - 0.5 * (self.scale_xy - 1)
        b_xy[..., 0] += c_x
        b_xy[..., 1] += c_y

        # Box dimension
        b_wh = torch.exp(output[..., 2:4]) * self.scaled_anchors.view(1, 1, 1, -1, 2)

        top_left = b_xy - 0.5 * b_wh
        bot_right = top_left + b_wh
        boxes = torch.cat((top_left, bot_right), dim=-1)
        # Coords relative to whole image
        boxes[..., [0, 2]] /= w
        boxes[..., [1, 3]] /= h
        boxes = boxes.clamp_(0, 1)

        # Objectness
        b_o = torch.sigmoid(output[..., 4])
        # Classification scores
        b_scores = torch.sigmoid(output[..., 5:])

        return boxes, b_o, b_scores

    @staticmethod
    def post_process(boxes, b_o, b_scores, rpn_nms_thresh=0.7, box_score_thresh=0.05):

        detections = []
        for idx in range(b_o.shape[0]):

            coords = torch.zeros((0, 4), dtype=torch.float32, device=b_o.device)
            scores = torch.zeros(0, dtype=torch.float32, device=b_o.device)
            labels = torch.zeros(0, dtype=torch.long, device=b_o.device)

            # Objectness filter
            if torch.any(b_o[idx] >= 0.5):
                coords = boxes[idx, b_o[idx] >= 0.5]
                scores = b_scores[idx, b_o[idx] >= 0.5].max(dim=-1)
                labels = scores.indices
                # Multiply by the objectness
                scores = scores.values * b_o[idx, b_o[idx] >= 0.5]

                # Confidence threshold
                coords = coords[scores >= box_score_thresh]
                labels = labels[scores >= box_score_thresh]
                scores = scores[scores >= box_score_thresh]
                coords = coords.clamp_(0, 1)
                # NMS
                kept_idxs = nms(coords, scores, iou_threshold=rpn_nms_thresh)
                coords = coords[kept_idxs]
                scores = scores[kept_idxs]
                labels = labels[kept_idxs]

            detections.append(dict(boxes=coords, scores=scores, labels=labels))

        return detections

    def _build_targets(self, pred_boxes, b_o, b_scores, gt_boxes, gt_labels):

        b, h, w, num_anchors = b_o.shape

        # Target formatting
        target_o = torch.zeros((b, h, w, num_anchors), device=b_o.device)
        target_scores = torch.zeros((b, h, w, num_anchors, self.num_classes), device=b_o.device)
        obj_mask = torch.zeros((b, h, w, num_anchors), dtype=torch.bool, device=b_o.device)
        noobj_mask = torch.ones((b, h, w, num_anchors), dtype=torch.bool, device=b_o.device)

        # GT coords --> left, top, width, height
        _boxes = torch.cat(gt_boxes, dim=0)
        gt_centers = _boxes[..., [0, 2, 1, 3]].view(-1, 2, 2).mean(dim=-1).to(dtype=torch.long)

        target_selection = torch.tensor([_idx for _idx, _boxes in enumerate(gt_boxes) for _ in range(_boxes.shape[0])],
                                        dtype=torch.long, device=b_o.device)
        if target_selection.shape[0] > 0:

            # Assign boxes
            obj_mask[target_selection, gt_centers[:, 1], gt_centers[:, 0]] = True
            noobj_mask[target_selection, gt_centers[:, 1], gt_centers[:, 0]] = False

            # B * cells * predictors * info
            for idx in range(b):

                if gt_boxes[idx].shape[0] > 0:

                    # IoU with cells that enclose the GT centers
                    gt_ious, gt_idxs = box_iou(pred_boxes[idx, obj_mask[idx]], gt_boxes[idx]).max(dim=1)

                    # Remove boxes with IoU below threshold
                    selection_mask = (gt_ious > 0).view(-1, 3)
                    if torch.any(~selection_mask):
                        _centers = gt_centers[target_selection == idx][gt_idxs[selection_mask]]
                        obj_mask[idx, _centers[..., 1], _centers[..., 0], torch.where(~selection_mask)[1]] = False

                    if torch.any(selection_mask):
                        # Objectness target
                        target_o[idx, obj_mask[idx]] = gt_ious[selection_mask.view(-1)]
                        # Classification target
                        target_scores[idx, obj_mask[idx], gt_idxs[selection_mask.view(-1)]] = 1

        return target_o, target_scores, obj_mask, noobj_mask

    def _compute_losses(self, pred_boxes, b_o, b_scores, gt_boxes, gt_labels, ignore_high_iou=False):

        target_o, target_scores, obj_mask, noobj_mask = self._build_targets(pred_boxes, b_o, b_scores,
                                                                            gt_boxes, gt_labels)

        # Bbox regression
        bbox_loss = torch.zeros(1, device=b_o.device)
        for idx, _target_boxes in enumerate(gt_boxes):
            if _target_boxes.shape[0] > 0 and torch.any(obj_mask[idx]):
                bbox_loss += ciou_loss(pred_boxes[idx, obj_mask[idx]], _target_boxes).min(dim=1).values.sum()

        return dict(obj_loss=F.mse_loss(b_o[obj_mask], target_o[obj_mask], reduction='sum'),
                    noobj_loss=self.lambda_noobj * b_o[noobj_mask].pow(2).sum(),
                    bbox_loss=self.lambda_coords * bbox_loss,
                    clf_loss=F.binary_cross_entropy(b_scores[obj_mask], target_scores[obj_mask], reduction='sum'))

    def forward(self, output, gt_boxes=None, gt_labels=None):

        if self.training and (gt_boxes is None or gt_labels is None):
            raise ValueError("`gt_boxes` and `gt_labels` need to be specified in training mode")

        pred_boxes, b_o, b_scores = self._format_outputs(output)

        if self.training:
            return self._compute_losses(pred_boxes, b_o, b_scores, gt_boxes, gt_labels)
        else:
            # cf. https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/tool/yolo_layer.py#L117
            return self.post_process(pred_boxes, b_o, b_scores, self.rpn_nms_thresh, self.box_score_thresh)


class Yolov4Head(nn.Module):
    def __init__(self, num_classes=80, anchors=None,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):

        # cf. https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg#L1143
        if anchors is None:
            anchors = torch.tensor([[[12, 16], [19, 36], [40, 28]],
                       [[36, 75], [76, 55], [72, 146]],
                       [[142, 110], [192, 243], [459, 401]]], dtype=torch.float32) / 608
        elif not isinstance(anchors, torch.Tensor):
            anchors = torch.tensor(anchors, dtype=torch.float32)

        if anchors.shape[0] != 3:
            raise AssertionError(f"The number of anchors is expected to be 3. received: {anchors.shape[0]}")

        super().__init__()

        self.head1 = nn.Sequential(
            *conv_sequence(128, 256, act_layer, norm_layer, None, conv_layer,
                           kernel_size=3, padding=1, bias=False),
            *conv_sequence(256, (5 + num_classes) * 3, None, None, None, conv_layer,
                           kernel_size=1, bias=True))

        self.yolo1 = YoloLayer(anchors[0], num_classes=num_classes, scale_xy=1.2)

        self.pre_head2 = nn.Sequential(*conv_sequence(128, 256, act_layer, norm_layer, drop_layer, conv_layer,
                                                      kernel_size=3, padding=1, stride=2, bias=False))
        self.head2_1 = nn.Sequential(
            *conv_sequence(512, 256, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=1, bias=False),
            *conv_sequence(256, 512, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=1, bias=False),
            *conv_sequence(512, 256, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=1, bias=False),
            *conv_sequence(256, 512, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=1, bias=False),
            *conv_sequence(512, 256, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=1, bias=False))
        self.head2_2 = nn.Sequential(
            *conv_sequence(256, 512, act_layer, norm_layer, None, conv_layer,
                           kernel_size=3, padding=1, bias=False),
            *conv_sequence(512, (5 + num_classes) * 3, None, None, None, conv_layer,
                           kernel_size=1, bias=True))

        self.yolo2 = YoloLayer(anchors[1], num_classes=num_classes, scale_xy=1.1)

        self.pre_head3 = nn.Sequential(*conv_sequence(256, 512, act_layer, norm_layer, drop_layer, conv_layer,
                                                      kernel_size=3, padding=1, stride=2, bias=False))
        self.head3 = nn.Sequential(
            *conv_sequence(1024, 512, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=1, bias=False),
            *conv_sequence(512, 1024, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=1, bias=False),
            *conv_sequence(1024, 512, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=1, bias=False),
            *conv_sequence(512, 1024, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=1, bias=False),
            *conv_sequence(1024, 512, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=1, bias=False),
            *conv_sequence(512, 1024, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=1, bias=False),
            *conv_sequence(1024, (5 + num_classes) * 3, None, None, None, conv_layer,
                           kernel_size=1, bias=True))

        self.yolo3 = YoloLayer(anchors[2], num_classes=num_classes, scale_xy=1.05)
        init_module(self, 'leaky_relu')
        # Zero init
        self.head1[-1].weight.data.zero_()
        self.head1[-1].bias.data.zero_()
        self.head2_2[-1].weight.data.zero_()
        self.head2_2[-1].bias.data.zero_()
        self.head3[-1].weight.data.zero_()
        self.head3[-1].bias.data.zero_()

    def forward(self, feats, gt_boxes=None, gt_labels=None):
        o1 = self.head1(feats[0])

        h2 = self.pre_head2(feats[0])
        h2 = torch.cat([h2, feats[1]], dim=1)
        h2 = self.head2_1(h2)
        o2 = self.head2_2(h2)

        h3 = self.pre_head3(h2)
        h3 = torch.cat([h3, feats[2]], dim=1)
        o3 = self.head3(h3)

        # YOLO output
        y1 = self.yolo1(o1, gt_boxes, gt_labels)
        y2 = self.yolo2(o2, gt_boxes, gt_labels)
        y3 = self.yolo3(o3, gt_boxes, gt_labels)

        if not self.training:

            detections = [dict(boxes=torch.cat((det1['boxes'], det2['boxes'], det3['boxes']), dim=0),
                               scores=torch.cat((det1['scores'], det2['scores'], det3['scores']), dim=0),
                               labels=torch.cat((det1['labels'], det2['labels'], det3['labels']), dim=0))
                          for det1, det2, det3 in zip(y1, y2, y3)]
            return detections

        else:

            return {k: y1[k] + y2[k] + y3[k] for k in y1.keys()}


class YOLOv4(nn.Module):
    def __init__(self, layout, num_classes=80, in_channels=3, stem_channels=32, anchors=None,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None, backbone_norm_layer=None):
        super().__init__()

        if act_layer is None:
            act_layer = nn.LeakyReLU(0.1, inplace=True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if backbone_norm_layer is None:
            backbone_norm_layer = norm_layer
        if drop_layer is None:
            drop_layer = DropBlock2d

        # backbone
        self.backbone = DarknetBodyV4(layout, in_channels, stem_channels, 3, Mish(),
                                      backbone_norm_layer, drop_layer, conv_layer)
        # neck
        self.neck = Neck([1024, 512, 256], act_layer, norm_layer, drop_layer, conv_layer)
        # head
        self.head = Yolov4Head(num_classes, anchors, act_layer, norm_layer, drop_layer, conv_layer)

        init_module(self.neck, 'leaky_relu')
        init_module(self.head, 'leaky_relu')

    def forward(self, x, gt_boxes=None, gt_labels=None):

        if not isinstance(x, torch.Tensor):
            x = torch.stack(x, dim=0)

        out = self.backbone(x)

        x20, x13, x6 = self.neck(out)

        return self.head((x20, x13, x6), gt_boxes, gt_labels)


def _yolo(arch, pretrained, progress, pretrained_backbone, **kwargs):

    if pretrained:
        pretrained_backbone = False

    # Retrieve the correct Darknet layout type
    yolo_type = sys.modules[__name__].__dict__[default_cfgs[arch]['arch']]
    # Build the model
    model = yolo_type(default_cfgs[arch]['backbone']['layout'], **kwargs)
    # Load backbone pretrained parameters
    if pretrained_backbone:
        if default_cfgs[arch]['backbone']['url'] is None:
            logging.warning(f"Invalid model URL for {arch}'s backbone, using default initialization.")
        else:
            state_dict = load_state_dict_from_url(default_cfgs[arch]['backbone']['url'],
                                                  progress=progress)
            state_dict = {k.replace('features.', ''): v
                          for k, v in state_dict.items() if k.startswith('features')}
            model.backbone.load_state_dict(state_dict)
    # Load pretrained parameters
    if pretrained:
        if default_cfgs[arch]['url'] is None:
            logging.warning(f"Invalid model URL for {arch}, using default initialization.")
        else:
            state_dict = load_state_dict_from_url(default_cfgs[arch]['url'],
                                                  progress=progress)
            model.load_state_dict(state_dict)

    return model


def yolov1(pretrained=False, progress=True, pretrained_backbone=True, **kwargs):
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

    return _yolo('yolov1', pretrained, progress, pretrained_backbone, **kwargs)


def yolov2(pretrained=False, progress=True, pretrained_backbone=True, **kwargs):
    """YOLOv2 model from
    `"YOLO9000: Better, Faster, Stronger" <https://pjreddie.com/media/files/papers/YOLO9000.pdf>`_.

    YOLOv2 improves upon YOLO by raising the number of boxes predicted by grid cell (default: 5), introducing
    bounding box priors and predicting class scores for each anchor box in the grid cell.

    For training, YOLOv2 uses the same multi-part loss as YOLO apart from its classification loss:

    .. math::
        \\mathcal{L}_{classification} = \\sum\\limits_{i=0}^{S^2}  \\sum\\limits_{j=0}^{B}
        \\mathbb{1}_{ij}^{obj} \\sum\\limits_{c \\in classes}
        (p_{ij}(c) - \\hat{p}_{ij}(c))^2

    where :math:`S` is size of the output feature map (13 for an input size :math:`(416, 416)`),
    :math:`B` is the number of anchor boxes per grid cell (default: 5),
    :math:`\\mathbb{1}_{ij}^{obj}` equals to 1 if a GT center falls inside the i-th grid cell and among the
    anchor boxes of that cell, has the highest IoU with the j-th box else 0,
    :math:`p_{ij}(c)` equals 1 if the assigned ground truth to the j-th anchor box of the i-th cell is classified
    as class :math:`c`,
    and :math:`\\hat{p}_{ij}(c)` is the predicted probability of class :math:`c` for the j-th anchor box
    in the i-th cell.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet
        progress (bool, optional): If True, displays a progress bar of the download to stderr
        pretrained_backbone (bool, optional): If True, backbone parameters will have been pretrained on Imagenette

    Returns:
        torch.nn.Module: detection module
    """

    if pretrained_backbone:
        kwargs['backbone_norm_layer'] = FrozenBatchNorm2d

    return _yolo('yolov2', pretrained, progress, pretrained_backbone, **kwargs)


def yolov4(pretrained=False, progress=True, pretrained_backbone=True, **kwargs):
    """YOLOv4 model from
    `"YOLOv4: Optimal Speed and Accuracy of Object Detection" <https://arxiv.org/pdf/2004.10934.pdf>`_.

    YOLOv4 is an improvement on YOLOv3 that includes many changes including: the usage of `DropBlock
    <https://arxiv.org/pdf/1810.12890.pdf>`_ regularization, `Mish <https://arxiv.org/pdf/1908.08681.pdf>`_ activation,
    `CSP <https://arxiv.org/pdf/2004.10934.pdf>`_ and `SAM <https://arxiv.org/pdf/1807.06521.pdf>`_ in the
    backbone, `SPP <https://arxiv.org/pdf/1406.4729.pdf>`_ and `PAN <https://arxiv.org/pdf/1803.01534.pdf>`_ in the
    neck.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet
        progress (bool, optional): If True, displays a progress bar of the download to stderr
        pretrained_backbone (bool, optional): If True, backbone parameters will have been pretrained on Imagenette

    Returns:
        torch.nn.Module: detection module
    """

    if pretrained_backbone:
        kwargs['backbone_norm_layer'] = FrozenBatchNorm2d

    return _yolo('yolov4', pretrained, progress, pretrained_backbone, **kwargs)
