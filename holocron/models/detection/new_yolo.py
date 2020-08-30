# -*- coding: utf-8 -*-

"""
Personal implementation of YOLO models
"""

import sys
import logging
import torch
import torch.nn as nn
from torchvision.ops.boxes import nms
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models.utils import load_state_dict_from_url
from ..utils import conv_sequence
from ..darknet import DarknetBodyV4, default_cfgs as dark_cfgs
from holocron.ops.boxes import box_ciou


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

        # backbone
        self.backbone = DarknetBodyV4(layout, in_channels, stem_channels, 3, act_layer,
                                      backbone_norm_layer, drop_layer, conv_layer)
        # neck
        self.neck = Neck([1024, 512, 256], act_layer, norm_layer, drop_layer, conv_layer)
        # head
        # self.head = Yolov4Head(output_ch, n_classes, inference)
        self.head = Yolov4Head(num_classes, anchors)

    def forward(self, input):
        out = self.backbone(input)

        x20, x13, x6 = self.neck(out)

        return self.head((x20, x13, x6))


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
        self.register_buffer('scaled_anchors', torch.tensor([[e / stride for e in anchor] for anchor in anchors], dtype=torch.float32))
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
        b_xy = b_xy.view(b, h * w, -1, 2)
        # Normalize
        b_xy[..., 0] /= w
        b_xy[..., 1] /= h

        # Box dimension
        b_wh = torch.exp(output[..., 2:4]) * self.scaled_anchors.to(device=output.device).view(1, 1, 1, -1, 2)
        b_wh = b_wh.view(b, h * w, -1, 2)
        # Normalize
        b_wh[..., 0] /= w
        b_wh[..., 1] /= h

        # Objectness
        b_o = torch.sigmoid(output[..., 4]).view(b, h * w, -1)
        # Classification scores
        b_scores = torch.sigmoid(output[..., 5:]).view(b, h * w, -1, self.num_classes)

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

    def _build_targets(b_xy, b_wh, b_o, b_scores, gt_boxes, gt_labels):

        return

    def _compute_losses(self, b_xy, b_wh, b_o, b_scores, gt_boxes, gt_labels, ignore_high_iou=False):

        b, h, w, num_anchors, num_classes = b_scores.shape

        # Prediction coords --> left, top, right, bot
        top_left = b_xy - 0.5 * b_wh
        bot_right = top_left + b_wh
        boxes = torch.cat((top_left, bot_right), dim=-1)

        # GT coords --> left, top, width, height
        target_selection = torch.tensor([_idx for _idx, _boxes in gt_boxes for _ in range(_boxes.shape[0])],
                                        dtype=torch.long, device=b_o.device)
        gt_boxes = torch.cat(gt_boxes, dim=0)
        gt_xy = 0.5 * (gt_boxes[..., 2:] + gt_boxes[..., :2]) / self.stride
        gt_idxs = gt_wy.to(dtype=torch.int16)
        gt_wh = (gt_boxes[..., 2:] - gt_boxes[:2]) / self.stride

        # Target formatting
        target_xy = gt_xy - gt_xy.floor()

        # Initialize losses
        obj_loss = torch.zeros(1, device=b_o.device)
        noobj_loss = torch.zeros(1, device=b_o.device)
        bbox_loss = torch.zeros(1, device=b_o.device)
        clf_loss = torch.zeros(1, device=b_o.device)

        # Batched anchor selection
        anchor_selection = box_ciou(gt_wh, self.scaled_anchors.to(device=b_o.device)).argmax(dim=1)

        # B * cells * predictors * info
        for idx in range(b):

            target_mask = target_selection == idx
            # if gt_wh[idx].shape[0] > 0:
            if torch.any(target_mask):
                # Anchor selection
                # FIXME: can be batched
                # anchor_selection = box_ciou(gt_wh[idx], self.scaled_anchors.to(device=b_o.device)).argmax(dim=1)
                anchor_selection[target_mask]

                #
                pred_ious = box_ciou(boxes[idx].view(-1, 4), gt_boxes[idx]).max(dim=1).values
                ######################
                pred_mask = (pred_ious > self.iou_thresh).view(b_o.shape)

                # Format for YOLO loss

                # target_xy[] =
                target_xy[target_mask]
                target_wh = torch.log(gt_wh[target_mask] / self.scaled_anchors[anchor_selection[target_mask]] + self.eps)








        return dict(obj_loss=obj_loss / pred_boxes.shape[0],
                    noobj_loss=self.lambda_noobj * noobj_loss / pred_boxes.shape[0],
                    bbox_loss=self.lambda_coords * bbox_loss / pred_boxes.shape[0],
                    clf_loss=clf_loss / pred_boxes.shape[0])

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

    def forward(self, feats):
        o1 = self.head1(feats[0])

        h2 = self.pre_head2(feats[0])
        h2 = torch.cat([h2, feats[1]], dim=1)
        h2 = self.head2_1(h2)
        o2 = self.head2_2(h2)

        h3 = self.pre_head3(h2)
        h3 = torch.cat([h3, feats[2]], dim=1)
        o3 = self.head3(h3)


        # YOLO output
        y1 = self.yolo1(o1)
        y2 = self.yolo2(o2)
        y3 = self.yolo3(o3)

        if not self.training:

            detections = [dict(boxes=torch.cat((det1['boxes'], det2['boxes'], det3['boxes']), dim=0),
                               scores=torch.cat((det1['scores'], det2['scores'], det3['scores']), dim=0),
                               labels=torch.cat((det1['labels'], det2['labels'], det3['labels']), dim=0))
                          for det1, det2, det3 in zip(y1, y2, y3)]
            return detections

        else:

            return {k: y1[k] + y2[k] + y3[k] for k in y1.keys()}
