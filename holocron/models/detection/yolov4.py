# Copyright (C) 2020-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops.boxes import box_iou, nms
from torchvision.ops.misc import FrozenBatchNorm2d

from holocron.nn import SPP, DropBlock2d
from holocron.nn.init import init_module
from holocron.ops.boxes import ciou_loss

from ..classification.darknetv4 import DarknetBodyV4
from ..classification.darknetv4 import default_cfgs as dark_cfgs
from ..utils import conv_sequence, load_pretrained_params

__all__ = ["YOLOv4", "yolov4", "PAN"]


default_cfgs = {
    "yolov4": {"arch": "YOLOv4", "backbone": dark_cfgs["cspdarknet53_mish"], "url": None},
}


class PAN(nn.Module):
    """PAN layer from `"Path Aggregation Network for Instance Segmentation" <https://arxiv.org/pdf/1803.01534.pdf>`_.

    Args:
        in_channels (int): input channels
        act_layer (torch.nn.Module, optional): activation layer to be used
        norm_layer (callable, optional): normalization layer
        drop_layer (callable, optional): regularization layer
        conv_layer (callable, optional): convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            *conv_sequence(
                in_channels,
                in_channels // 2,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=1,
                bias=(norm_layer is None),
            )
        )
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv2 = nn.Sequential(
            *conv_sequence(
                in_channels,
                in_channels // 2,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=1,
                bias=(norm_layer is None),
            )
        )

        self.convs = nn.Sequential(
            *conv_sequence(
                in_channels,
                in_channels // 2,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=1,
                bias=(norm_layer is None),
            ),
            *conv_sequence(
                in_channels // 2,
                in_channels,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=1,
                bias=(norm_layer is None),
            ),
            *conv_sequence(
                in_channels,
                in_channels // 2,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=1,
                bias=(norm_layer is None),
            ),
            *conv_sequence(
                in_channels // 2,
                in_channels,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=1,
                bias=(norm_layer is None),
            ),
            *conv_sequence(
                in_channels,
                in_channels // 2,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=1,
                bias=(norm_layer is None),
            ),
        )

    def forward(self, x: Tensor, up: Tensor) -> Tensor:
        out = self.conv1(x)

        out = torch.cat([self.conv2(up), self.up(out)], dim=1)

        return self.convs(out)


class Neck(nn.Module):
    def __init__(
        self,
        in_planes: List[int],
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        self.fpn = nn.Sequential(
            *conv_sequence(
                in_planes[0],
                in_planes[0] // 2,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=1,
                bias=(norm_layer is None),
            ),
            *conv_sequence(
                in_planes[0] // 2,
                in_planes[0],
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=1,
                bias=(norm_layer is None),
            ),
            *conv_sequence(
                in_planes[0],
                in_planes[0] // 2,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=1,
                bias=(norm_layer is None),
            ),
            SPP([5, 9, 13]),
            *conv_sequence(
                4 * in_planes[0] // 2,
                in_planes[0] // 2,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=1,
                bias=(norm_layer is None),
            ),
            *conv_sequence(
                in_planes[0] // 2,
                in_planes[0],
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=1,
                bias=(norm_layer is None),
            ),
            *conv_sequence(
                in_planes[0],
                in_planes[0] // 2,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=1,
                bias=(norm_layer is None),
            ),
        )

        self.pan1 = PAN(in_planes[1], act_layer, norm_layer, drop_layer, conv_layer)
        self.pan2 = PAN(in_planes[2], act_layer, norm_layer, drop_layer, conv_layer)
        init_module(self, "leaky_relu")

    def forward(self, feats: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:

        out = self.fpn(feats[2])

        aux1 = self.pan1(out, feats[1])
        aux2 = self.pan2(aux1, feats[0])

        return aux2, aux1, out


class YoloLayer(nn.Module):
    """Scale-specific part of YoloHead"""

    def __init__(
        self,
        anchors: Tensor,
        num_classes: int = 80,
        scale_xy: float = 1.0,
        iou_thresh: float = 0.213,
        lambda_obj: float = 1,
        lambda_noobj: float = 0.001,
        lambda_class: float = 0.1,
        lambda_coords: float = 1.0,
        rpn_nms_thresh: float = 0.7,
        box_score_thresh: float = 0.05,
        ignore_thresh: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("anchors", anchors)

        self.rpn_nms_thresh = rpn_nms_thresh
        self.box_score_thresh = box_score_thresh
        self.ignore_thresh = ignore_thresh
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
        self.lambda_coords = lambda_coords

        # cf. https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg#L1150
        self.scale_xy = scale_xy
        # cf. https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg#L1151
        self.iou_thresh = iou_thresh

    def extra_repr(self) -> str:
        return f"num_classes={self.num_classes}, scale_xy={self.scale_xy}"

    def _format_outputs(self, output: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        b, _, h, w = output.shape

        self.anchors: Tensor
        # B x (num_anchors * (5 + num_classes)) x H x W --> B x H x W x num_anchors x (5 + num_classes)
        output = output.reshape(b, len(self.anchors), 5 + self.num_classes, h, w).permute(0, 3, 4, 1, 2)

        # Box center
        c_x = torch.arange(w, dtype=torch.float32, device=output.device).reshape(1, 1, -1, 1)
        c_y = torch.arange(h, dtype=torch.float32, device=output.device).reshape(1, -1, 1, 1)

        b_xy = self.scale_xy * torch.sigmoid(output[..., :2]) - 0.5 * (self.scale_xy - 1)
        b_xy[..., 0].add_(c_x)
        b_xy[..., 1].add_(c_y)
        b_xy[..., 0].div_(w)
        b_xy[..., 1].div_(h)

        # Box dimension
        b_wh = torch.exp(output[..., 2:4]) * self.anchors.view(1, 1, 1, -1, 2)
        # Fix overflow by clipping
        b_wh = b_wh.clamp_(0, 2)

        top_left = b_xy - 0.5 * b_wh
        bot_right = top_left + b_wh
        boxes = torch.cat((top_left, bot_right), dim=-1)

        # Objectness
        b_o = output[..., 4]
        # Classification scores
        b_scores = output[..., 5:]

        return boxes, b_o, b_scores

    @staticmethod
    def post_process(
        boxes: Tensor, b_o: Tensor, b_scores: Tensor, rpn_nms_thresh: float = 0.7, box_score_thresh: float = 0.05
    ) -> List[Dict[str, Tensor]]:

        b_o = torch.sigmoid(b_o)
        b_scores = torch.sigmoid(b_scores)

        boxes = boxes.clamp_(0, 1)
        detections = []
        for idx in range(b_o.shape[0]):

            coords = torch.zeros((0, 4), dtype=torch.float32, device=b_o.device)
            scores = torch.zeros(0, dtype=torch.float32, device=b_o.device)
            labels = torch.zeros(0, dtype=torch.long, device=b_o.device)

            # Objectness filter
            if torch.any(b_o[idx] >= 0.5):
                coords = boxes[idx, b_o[idx] >= 0.5]
                scores, labels = b_scores[idx, b_o[idx] >= 0.5].max(dim=-1)
                # Multiply by the objectness
                scores.mul_(b_o[idx, b_o[idx] >= 0.5])

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

    def _build_targets(
        self, pred_boxes: Tensor, b_o: Tensor, b_scores: Tensor, target: List[Dict[str, Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        b, h, w, num_anchors = b_o.shape

        # Target formatting
        target_o = torch.zeros((b, h, w, num_anchors), device=b_o.device)
        target_scores = torch.zeros((b, h, w, num_anchors, self.num_classes), device=b_o.device)
        obj_mask = torch.zeros((b, h, w, num_anchors), dtype=torch.bool, device=b_o.device)
        noobj_mask = torch.ones((b, h, w, num_anchors), dtype=torch.bool, device=b_o.device)

        gt_boxes = [t["boxes"] for t in target]
        gt_labels = [t["labels"] for t in target]

        # GT coords --> left, top, width, height
        _boxes = torch.cat(gt_boxes, dim=0)
        gt_centers = _boxes[..., [0, 2, 1, 3]].reshape(-1, 2, 2).mean(dim=-1)
        gt_centers[:, 0] *= w
        gt_centers[:, 1] *= h
        gt_centers = gt_centers.to(dtype=torch.long)

        target_selection = torch.tensor(
            [_idx for _idx, _boxes in enumerate(gt_boxes) for _ in range(_boxes.shape[0])],
            dtype=torch.long,
            device=b_o.device,
        )
        if target_selection.shape[0] > 0:

            # Anchors IoU
            gt_wh = _boxes[:, 2:] - _boxes[:, :2]
            anchor_idxs = box_iou(
                torch.cat((-gt_wh, gt_wh), dim=-1), torch.cat((-self.anchors, self.anchors), dim=-1)
            ).argmax(dim=1)

            # Assign boxes
            obj_mask[target_selection, gt_centers[:, 1], gt_centers[:, 0], anchor_idxs] = True
            noobj_mask[target_selection, gt_centers[:, 1], gt_centers[:, 0], :] = False

            # B * cells * predictors * info
            for idx in range(b):
                if gt_boxes[idx].shape[0] > 0:
                    # IoU with cells that enclose the GT centers
                    gt_ious, gt_idxs = box_iou(pred_boxes[idx, obj_mask[idx]], gt_boxes[idx]).max(dim=1)
                    # Objectness target
                    target_o[idx, obj_mask[idx]] = gt_ious
                    # Classification target
                    target_scores[idx, obj_mask[idx], gt_labels[idx][gt_idxs]] = 1.0
                    # Ignore predictions
                    gt_ious = box_iou(pred_boxes[idx, noobj_mask[idx]], gt_boxes[idx]).max(dim=1).values
                    noobj_mask[idx, noobj_mask[idx]][gt_ious >= self.ignore_thresh] = False

        return target_o, target_scores, obj_mask, noobj_mask

    def _compute_losses(
        self,
        pred_boxes: Tensor,
        b_o: Tensor,
        b_scores: Tensor,
        target: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:

        target_o, target_scores, obj_mask, noobj_mask = self._build_targets(pred_boxes, b_o, b_scores, target)

        # Bbox regression
        bbox_loss = torch.zeros(1, device=b_o.device)
        for idx, _target in enumerate(target):
            if _target["boxes"].shape[0] > 0 and torch.any(obj_mask[idx]):
                bbox_loss += ciou_loss(pred_boxes[idx, obj_mask[idx]], _target["boxes"]).min(dim=1).values.sum()

        b_o = torch.sigmoid(b_o)

        return dict(
            obj_loss=self.lambda_obj * F.mse_loss(b_o[obj_mask], target_o[obj_mask], reduction="sum") / b_o.shape[0],
            noobj_loss=self.lambda_noobj * b_o[noobj_mask].pow(2).sum() / b_o.shape[0],
            bbox_loss=self.lambda_coords * bbox_loss / b_o.shape[0],
            clf_loss=self.lambda_class
            * F.binary_cross_entropy_with_logits(
                b_scores[obj_mask],
                target_scores[obj_mask],
                reduction="none",
            )
            .mean(1)
            .sum(0)
            / b_o.shape[0],
        )

    def forward(
        self, x: Tensor, target: Optional[List[Dict[str, Tensor]]] = None
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """Perform detection on an image tensor and returns either the loss dictionary in training mode
        or the list of detections in eval mode.

        Args:
            x (torch.Tensor[N, 3, H, W]): input image tensor
            target (list<dict>, optional): each dict must have two keys `boxes` of type torch.Tensor[*, 4]
                and `labels` of type torch.Tensor[*]
        """

        if self.training and target is None:
            raise ValueError("`target` needs to be specified in training mode")

        pred_boxes, b_o, b_scores = self._format_outputs(x)

        if self.training:
            return self._compute_losses(pred_boxes, b_o, b_scores, target)  # type: ignore[arg-type]

        # cf. https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/tool/yolo_layer.py#L117
        return self.post_process(pred_boxes, b_o, b_scores, self.rpn_nms_thresh, self.box_score_thresh)


class Yolov4Head(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        anchors: Optional[Tensor] = None,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:

        # cf. https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg#L1143
        if anchors is None:
            anchors = (
                torch.tensor(
                    [
                        [[12, 16], [19, 36], [40, 28]],
                        [[36, 75], [76, 55], [72, 146]],
                        [[142, 110], [192, 243], [459, 401]],
                    ],
                    dtype=torch.float32,
                )
                / 608
            )
        elif not isinstance(anchors, torch.Tensor):
            anchors = torch.tensor(anchors, dtype=torch.float32)

        if anchors.shape[0] != 3:
            raise AssertionError(f"The number of anchors is expected to be 3. received: {anchors.shape[0]}")

        super().__init__()

        self.head1 = nn.Sequential(
            *conv_sequence(
                128, 256, act_layer, norm_layer, None, conv_layer, kernel_size=3, padding=1, bias=(norm_layer is None)
            ),
            *conv_sequence(256, (5 + num_classes) * 3, None, None, None, conv_layer, kernel_size=1, bias=True),
        )

        self.yolo1 = YoloLayer(anchors[0], num_classes=num_classes, scale_xy=1.2)

        self.pre_head2 = nn.Sequential(
            *conv_sequence(
                128,
                256,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=1,
                stride=2,
                bias=(norm_layer is None),
            )
        )
        self.head2_1 = nn.Sequential(
            *conv_sequence(
                512, 256, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=1, bias=(norm_layer is None)
            ),
            *conv_sequence(
                256,
                512,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=1,
                bias=(norm_layer is None),
            ),
            *conv_sequence(
                512, 256, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=1, bias=(norm_layer is None)
            ),
            *conv_sequence(
                256,
                512,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=1,
                bias=(norm_layer is None),
            ),
            *conv_sequence(
                512, 256, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=1, bias=(norm_layer is None)
            ),
        )
        self.head2_2 = nn.Sequential(
            *conv_sequence(
                256, 512, act_layer, norm_layer, None, conv_layer, kernel_size=3, padding=1, bias=(norm_layer is None)
            ),
            *conv_sequence(512, (5 + num_classes) * 3, None, None, None, conv_layer, kernel_size=1, bias=True),
        )

        self.yolo2 = YoloLayer(anchors[1], num_classes=num_classes, scale_xy=1.1)

        self.pre_head3 = nn.Sequential(
            *conv_sequence(
                256,
                512,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=1,
                stride=2,
                bias=(norm_layer is None),
            )
        )
        self.head3 = nn.Sequential(
            *conv_sequence(
                1024, 512, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=1, bias=(norm_layer is None)
            ),
            *conv_sequence(
                512,
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
                1024, 512, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=1, bias=(norm_layer is None)
            ),
            *conv_sequence(
                512,
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
                1024, 512, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=1, bias=(norm_layer is None)
            ),
            *conv_sequence(
                512,
                1024,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=1,
                bias=(norm_layer is None),
            ),
            *conv_sequence(1024, (5 + num_classes) * 3, None, None, None, conv_layer, kernel_size=1, bias=True),
        )

        self.yolo3 = YoloLayer(anchors[2], num_classes=num_classes, scale_xy=1.05)
        init_module(self, "leaky_relu")
        # Zero init
        self.head1[-1].weight.data.zero_()
        self.head1[-1].bias.data.zero_()
        self.head2_2[-1].weight.data.zero_()
        self.head2_2[-1].bias.data.zero_()
        self.head3[-1].weight.data.zero_()
        self.head3[-1].bias.data.zero_()

    def forward(
        self, feats: List[Tensor], target: Optional[List[Dict[str, Tensor]]] = None
    ) -> Union[List[Dict[str, Tensor]], Dict[str, Tensor]]:
        o1 = self.head1(feats[0])

        h2 = self.pre_head2(feats[0])
        h2 = torch.cat([h2, feats[1]], dim=1)
        h2 = self.head2_1(h2)
        o2 = self.head2_2(h2)

        h3 = self.pre_head3(h2)
        h3 = torch.cat([h3, feats[2]], dim=1)
        o3 = self.head3(h3)

        # YOLO output
        y1 = self.yolo1(o1, target)
        y2 = self.yolo2(o2, target)
        y3 = self.yolo3(o3, target)

        if not self.training:

            detections = [
                dict(
                    boxes=torch.cat((det1["boxes"], det2["boxes"], det3["boxes"]), dim=0),
                    scores=torch.cat((det1["scores"], det2["scores"], det3["scores"]), dim=0),
                    labels=torch.cat((det1["labels"], det2["labels"], det3["labels"]), dim=0),
                )
                for det1, det2, det3 in zip(y1, y2, y3)
            ]
            return detections

        return {k: y1[k] + y2[k] + y3[k] for k in y1.keys()}


class YOLOv4(nn.Module):
    def __init__(
        self,
        layout: List[Tuple[int, int]],
        num_classes: int = 80,
        in_channels: int = 3,
        stem_channels: int = 32,
        anchors: Optional[Tensor] = None,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
        backbone_norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()

        if act_layer is None:
            act_layer = nn.Mish(inplace=True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if backbone_norm_layer is None:
            backbone_norm_layer = norm_layer
        if drop_layer is None:
            drop_layer = DropBlock2d

        # backbone
        self.backbone = DarknetBodyV4(
            layout, in_channels, stem_channels, 3, act_layer, backbone_norm_layer, drop_layer, conv_layer
        )
        # neck
        self.neck = Neck([1024, 512, 256], act_layer, norm_layer, drop_layer, conv_layer)
        # head
        self.head = Yolov4Head(num_classes, anchors, act_layer, norm_layer, drop_layer, conv_layer)

        init_module(self.neck, "leaky_relu")
        init_module(self.head, "leaky_relu")

    def forward(
        self, x: Tensor, target: Optional[List[Dict[str, Tensor]]] = None
    ) -> Union[List[Dict[str, Tensor]], Dict[str, Tensor]]:

        if not isinstance(x, torch.Tensor):
            x = torch.stack(x, dim=0)

        out = self.backbone(x)

        x20, x13, x6 = self.neck(out)

        return self.head((x20, x13, x6), target)


def _yolo(
    arch: str,
    pretrained: bool,
    progress: bool,
    pretrained_backbone: bool,
    layout: List[Tuple[int, int]],
    **kwargs: Any,
) -> YOLOv4:

    if pretrained:
        pretrained_backbone = False

    # Build the model
    model = YOLOv4(layout, **kwargs)
    # Load backbone pretrained parameters
    if pretrained_backbone:
        load_pretrained_params(
            model.backbone,
            default_cfgs[arch]["backbone"]["url"],  # type: ignore[index]
            progress,
            key_replacement=("features.", ""),
            key_filter="features.",
        )
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"], progress)  # type: ignore[arg-type]

    return model


def yolov4(pretrained: bool = False, progress: bool = True, pretrained_backbone: bool = True, **kwargs: Any) -> YOLOv4:
    r"""YOLOv4 model from
    `"YOLOv4: Optimal Speed and Accuracy of Object Detection" <https://arxiv.org/pdf/2004.10934.pdf>`_.

    The architecture improves upon YOLOv3 by including: the usage of `DropBlock
    <https://arxiv.org/pdf/1810.12890.pdf>`_ regularization, `Mish <https://arxiv.org/pdf/1908.08681.pdf>`_ activation,
    `CSP <https://arxiv.org/pdf/2004.10934.pdf>`_ and `SAM <https://arxiv.org/pdf/1807.06521.pdf>`_ in the
    backbone, `SPP <https://arxiv.org/pdf/1406.4729.pdf>`_ and `PAN <https://arxiv.org/pdf/1803.01534.pdf>`_ in the
    neck.

    For training, YOLOv4 uses the same multi-part loss as YOLOv3 apart from its box coordinate loss:

    .. math::
        \mathcal{L}_{coords} = \sum\limits_{i=0}^{S^2}  \sum\limits_{j=0}^{B}
        \min\limits_{k \in [1, M]} C_{IoU}(\hat{loc}_{ij}, loc^{GT}_k)

    where :math:`S` is size of the output feature map (13 for an input size :math:`(416, 416)`),
    :math:`B` is the number of anchor boxes per grid cell (default: 3),
    :math:`M` is the number of ground truth boxes,
    :math:`C_{IoU}` is the complete IoU loss,
    :math:`\hat{loc}_{ij}` is the predicted bounding box for grid cell :math:`i` at anchor :math:`j`,
    and :math:`loc^{GT}_k` is the k-th ground truth bounding box.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet
        progress (bool, optional): If True, displays a progress bar of the download to stderr
        pretrained_backbone (bool, optional): If True, backbone parameters will have been pretrained on Imagenette

    Returns:
        torch.nn.Module: detection module
    """

    if pretrained_backbone:
        kwargs["backbone_norm_layer"] = FrozenBatchNorm2d

    return _yolo(
        "yolov4",
        pretrained,
        progress,
        pretrained_backbone,
        [(64, 1), (128, 2), (256, 8), (512, 8), (1024, 4)],
        **kwargs,
    )
