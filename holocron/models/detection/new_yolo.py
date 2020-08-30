# -*- coding: utf-8 -*-

"""
Personal implementation of YOLO models
"""

import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.boxes import nms, box_iou
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models.utils import load_state_dict_from_url
from ..utils import conv_sequence
from ..darknet import DarknetBodyV4, default_cfgs as dark_cfgs
from holocron.ops.boxes import ciou_loss
from holocron.nn import Mish, DropBlock2d


__all__ = ['YOLOv4', 'yolov4', 'SPP', 'PAN', 'Neck']

default_cfgs = {
    'yolov4': {'arch': 'YOLOv4', 'backbone': dark_cfgs['cspdarknet53'],
               'url': None},
}


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
        self.head = Yolov4Head(num_classes, anchors)

    def forward(self, x, gt_boxes=None, gt_labels=None):

        if not isinstance(x, torch.Tensor):
            x = torch.stack(x, dim=0)

        out = self.backbone(x)

        x20, x13, x6 = self.neck(out)

        return self.head((x20, x13, x6), gt_boxes, gt_labels)


class SPP(nn.Module):
    """SPP layer from `"Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition"
    <https://arxiv.org/pdf/1406.4729.pdf>`_.

    Args:
        kernel_sizes (list<int>): kernel sizes of each pooling
    """

    def __init__(self, kernel_sizes):
        super().__init__()
        self.maxpools = nn.ModuleList([nn.MaxPool2d(k_size, stride=1, padding=k_size // 2)
                                       for k_size in kernel_sizes])

    def forward(self, x):
        feats = [x]
        for pool_layer in self.maxpools:
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


class YoloLayer(nn.Module):
    ''' Yolo layer
    model_out: while inference,is post-processing inside or outside the model
        true:outside
    '''
    def __init__(self, anchors=[], num_classes=80, stride=32, scale_xy=1, iou_thresh=0.213, eps=1e-16,
                 lambda_noobj=0.5, lambda_coords=5., rpn_nms_thresh=0.7, box_score_thresh=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('scaled_anchors',
                             torch.tensor([[e / stride for e in anchor] for anchor in anchors], dtype=torch.float32))
        self.stride = stride

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
        # Normalize
        b_xy[..., 0] /= w
        b_xy[..., 1] /= h

        # Box dimension
        b_wh = torch.exp(output[..., 2:4]) * self.scaled_anchors.view(1, 1, 1, -1, 2)
        # Normalize
        b_wh[..., 0] /= w
        b_wh[..., 1] /= h

        # Objectness
        b_o = torch.sigmoid(output[..., 4])
        # Classification scores
        b_scores = torch.sigmoid(output[..., 5:])

        return b_xy, b_wh, b_o, b_scores

    @staticmethod
    def post_process(b_xy, b_wh, b_o, b_scores, rpn_nms_thresh=0.7, box_score_thresh=0.05):

        top_left = b_xy - 0.5 * b_wh
        bot_right = top_left + b_wh
        boxes = torch.cat((top_left, bot_right), dim=-1)

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

    def _build_targets(self, b_xy, b_wh, b_o, b_scores, gt_boxes, gt_labels):

        b, h, w, num_anchors, num_classes = b_scores.shape

        # Target formatting
        target_xy = torch.zeros((b, h, w, num_anchors, 2), device=b_o.device)
        target_wh = torch.zeros((b, h, w, num_anchors, 2), device=b_o.device)
        target_o = torch.zeros((b, h, w, num_anchors), device=b_o.device)
        target_scores = torch.zeros((b, h, w, num_anchors, self.num_classes), device=b_o.device)
        obj_mask = torch.zeros((b, h, w, num_anchors), dtype=torch.bool, device=b_o.device)

        # GT coords --> left, top, width, height
        target_selection = torch.tensor([_idx for _idx, _boxes in enumerate(gt_boxes) for _ in range(_boxes.shape[0])],
                                        dtype=torch.long, device=b_o.device)
        if target_selection.shape[0] > 0:
            gt_boxes = torch.cat(gt_boxes, dim=0)
            gt_boxes[..., [0, 2]] *= w
            gt_boxes[..., [1, 3]] *= h
            gt_xy = 0.5 * (gt_boxes[..., 2:] + gt_boxes[..., :2]) / self.stride
            gt_idxs = gt_xy.to(dtype=torch.int16)
            gt_wh = (gt_boxes[..., 2:] - gt_boxes[..., :2]) / self.stride

            # Batched anchor selection
            anchor_selection = box_iou(torch.cat((torch.zeros_like(gt_wh), gt_wh), dim=-1),
                                       torch.cat((torch.zeros_like(self.scaled_anchors), self.scaled_anchors), dim=-1))
            anchor_selection = anchor_selection.argmax(dim=1)

            # Prediction coords --> left, top, right, bot
            top_left = b_xy - 0.5 * b_wh
            bot_right = top_left + b_wh
            pred_boxes = torch.cat((top_left, bot_right), dim=-1)

            # B * cells * predictors * info
            for idx in range(b):

                target_mask = target_selection == idx
                if torch.any(target_mask):

                    # CIoU
                    # gt_cious = ciou_loss(pred_boxes[idx].view(-1, 4), gt_boxes[target_mask]).max(dim=1).values

                    gt_ious, gt_idxs = box_iou(pred_boxes[idx].view(-1, 4), gt_boxes[target_mask]).max(dim=1)

                    # Assign boxes
                    _gt_mask = gt_ious > 0
                    pred_mask = _gt_mask.view(b_o.shape[1:])
                    obj_mask[idx, pred_mask] = True
                    gt_ious, gt_idxs = gt_ious[_gt_mask], gt_idxs[_gt_mask]
                    # Objectness target
                    target_o[idx, pred_mask] = gt_ious
                    # Boxes that are not matched --> 0
                    target_xy[idx, pred_mask] = gt_xy[target_mask][gt_idxs] - gt_xy[target_mask][gt_idxs].floor()
                    target_wh[idx, pred_mask] = torch.log(gt_wh[target_mask][gt_idxs] /
                                                          self.scaled_anchors[anchor_selection[target_mask][gt_idxs]] +
                                                          self.eps)
                    # Classification target
                    target_scores[idx, pred_mask, gt_labels[idx][gt_idxs]] = 1

        return target_xy, target_wh, target_o, target_scores, obj_mask

    def _compute_losses(self, b_xy, b_wh, b_o, b_scores, gt_boxes, gt_labels, ignore_high_iou=False):

        target_xy, target_wh, target_o, target_scores, obj_mask = self._build_targets(b_xy, b_wh, b_o, b_scores,
                                                                                      gt_boxes, gt_labels)

        # Replace with CIoU
        xy_loss = F.mse_loss(b_xy[obj_mask], target_xy[obj_mask], reduction='sum')
        wh_loss = F.mse_loss(b_wh[obj_mask], target_wh[obj_mask], reduction='sum')

        return dict(obj_loss=F.mse_loss(b_o[obj_mask], target_o[obj_mask], reduction='sum'),
                    noobj_loss=self.lambda_noobj * F.mse_loss(b_o[~obj_mask], target_o[~obj_mask], reduction='sum'),
                    bbox_loss=self.lambda_coords * (xy_loss + wh_loss),
                    clf_loss=F.binary_cross_entropy(b_scores[obj_mask], target_scores[obj_mask], reduction='sum'))

    def forward(self, output, gt_boxes=None, gt_labels=None):

        if self.training and (gt_boxes is None or gt_labels is None):
            raise ValueError("`gt_boxes` and `gt_labels` need to be specified in training mode")

        b_xy, b_wh, b_o, b_scores = self._format_outputs(output)

        if self.training:
            return self._compute_losses(b_xy, b_wh, b_o, b_scores, gt_boxes, gt_labels)
        else:
            # cf. https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/tool/yolo_layer.py#L117
            return self.post_process(b_xy, b_wh, b_o, b_scores, self.rpn_nms_thresh, self.box_score_thresh)


class Yolov4Head(nn.Module):
    def __init__(self, num_classes=80, anchors=None,
                 act_layer=None, norm_layer=None, drop_layer=None, conv_layer=None):

        # cf. https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg#L1143
        if anchors is None:
            anchors = [[[12, 16], [19, 36], [40, 28]],
                       [[36, 75], [76, 55], [72, 146]],
                       [[142, 110], [192, 243], [459, 401]]]
        if len(anchors) != 3:
            raise AssertionError(f"The number of anchors is expected to be 3. received: {len(anchors)}")
        self.anchors = anchors

        super().__init__()

        self.head1 = nn.Sequential(
            *conv_sequence(128, 256, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=1, bias=False),
            *conv_sequence(256, (5 + num_classes) * 3, act_layer, None, drop_layer, conv_layer,
                           kernel_size=1, bias=True))

        self.yolo1 = YoloLayer(self.anchors[0], num_classes=num_classes, stride=8, scale_xy=1.2)

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
            *conv_sequence(256, 512, act_layer, norm_layer, drop_layer, conv_layer,
                           kernel_size=3, padding=1, bias=False),
            *conv_sequence(512, (5 + num_classes) * 3, act_layer, None, drop_layer, conv_layer,
                           kernel_size=1, bias=True))

        self.yolo2 = YoloLayer(self.anchors[1], num_classes=num_classes, stride=16, scale_xy=1.1)

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
            *conv_sequence(1024, (5 + num_classes) * 3, act_layer, None, drop_layer, conv_layer,
                           kernel_size=1, bias=True))

        self.yolo3 = YoloLayer(self.anchors[2], num_classes=num_classes, stride=32, scale_xy=1.05)

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
