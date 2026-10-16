# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops.boxes import box_iou
from torchvision.ops.misc import FrozenBatchNorm2d

from holocron.nn import SPP, DropBlock2d
from holocron.nn.init import init_module
from holocron.ops.boxes import ciou_loss

from ..classification.darknetv4 import DarknetBodyV4
from ..classification.darknetv4 import default_cfgs as dark_cfgs
from ..utils import conv_sequence, load_pretrained_params
from .yolo import _post_process

__all__ = ["PAN", "YOLOv4", "yolov4"]


default_cfgs = {
    "yolov4": {"arch": "YOLOv4", "backbone": dark_cfgs["cspdarknet53_mish"], "url": None},
}


class PAN(nn.Module):
    """PAN layer from `"Path Aggregation Network for Instance Segmentation" <https://arxiv.org/pdf/1803.01534.pdf>`_.

    Args:
        in_channels: input channels
        act_layer: activation layer to be used
        norm_layer: normalization layer
        drop_layer: regularization layer
        conv_layer: convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        act_layer: nn.Module | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
        drop_layer: Callable[..., nn.Module] | None = None,
        conv_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()

        conv = partial(
            conv_sequence, act_layer=act_layer, norm_layer=norm_layer, drop_layer=drop_layer, conv_layer=conv_layer
        )

        self.conv1 = nn.Sequential(*conv(in_channels, in_channels // 2, kernel_size=1))
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv2 = nn.Sequential(*conv(in_channels, in_channels // 2, kernel_size=1))

        self.convs = nn.Sequential(
            *conv(in_channels, in_channels // 2, kernel_size=1),
            *conv(in_channels // 2, in_channels, kernel_size=3, padding=1),
            *conv(in_channels, in_channels // 2, kernel_size=1),
            *conv(in_channels // 2, in_channels, kernel_size=3, padding=1),
            *conv(in_channels, in_channels // 2, kernel_size=1),
        )

    def forward(self, x: Tensor, up: Tensor) -> Tensor:
        out = self.conv1(x)

        out = torch.cat([self.conv2(up), self.up(out)], dim=1)

        return self.convs(out)


class Neck(nn.Module):
    def __init__(
        self,
        in_planes: list[int],
        act_layer: nn.Module | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
        drop_layer: Callable[..., nn.Module] | None = None,
        conv_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()

        conv = partial(
            conv_sequence, act_layer=act_layer, norm_layer=norm_layer, drop_layer=drop_layer, conv_layer=conv_layer
        )

        self.fpn = nn.Sequential(
            *conv(in_planes[0], in_planes[0] // 2, kernel_size=1),
            *conv(in_planes[0] // 2, in_planes[0], kernel_size=3, padding=1),
            *conv(in_planes[0], in_planes[0] // 2, kernel_size=1),
            SPP([5, 9, 13]),
            *conv(4 * in_planes[0] // 2, in_planes[0] // 2, kernel_size=1),
            *conv(in_planes[0] // 2, in_planes[0], kernel_size=3, padding=1),
            *conv(in_planes[0], in_planes[0] // 2, kernel_size=1),
        )

        self.pan1 = PAN(in_planes[1], act_layer, norm_layer, drop_layer, conv_layer)
        self.pan2 = PAN(in_planes[2], act_layer, norm_layer, drop_layer, conv_layer)
        init_module(self, "leaky_relu")

    def forward(self, feats: list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
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
        lambda_noobj: float = 1,
        lambda_class: float = 1,
        lambda_coords: float = 0.07,
        rpn_nms_thresh: float = 0.7,
        box_score_thresh: float = 0.05,
        ignore_thresh: float = 0.7,
        all_anchors: Tensor | None = None,
        anchor_mask: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_classes: int = num_classes
        self.register_buffer("anchors", anchors)
        if all_anchors is None:
            all_anchors = anchors
        if anchor_mask is None:
            anchor_mask = torch.arange(len(anchors), device=anchors.device)
        self.register_buffer("all_anchors", all_anchors, persistent=False)
        self.register_buffer("anchor_mask", anchor_mask, persistent=False)

        self.rpn_nms_thresh: float = rpn_nms_thresh
        self.box_score_thresh: float = box_score_thresh
        self.ignore_thresh: float = ignore_thresh
        self.lambda_obj: float = lambda_obj
        self.lambda_noobj: float = lambda_noobj
        self.lambda_class: float = lambda_class
        self.lambda_coords: float = lambda_coords

        # cf. https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg#L1150
        self.scale_xy: float = scale_xy
        # cf. https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg#L1151
        self.iou_thresh: float = iou_thresh

    def extra_repr(self) -> str:
        return f"num_classes={self.num_classes}, scale_xy={self.scale_xy}"

    def _format_outputs(self, output: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, _, h, w = output.shape

        self.anchors: Tensor
        # B x (num_anchors * (5 + num_classes)) x H x W --> B x H x W x num_anchors x (5 + num_classes)
        output = output.reshape(b, self.anchors.shape[0], 5 + self.num_classes, h, w).permute(0, 3, 4, 1, 2)

        # Box center
        # Half precision can overflow CIoU gradients for small, elongated boxes.
        box_logits = output[..., :4]
        if box_logits.dtype in {torch.float16, torch.bfloat16}:
            box_logits = box_logits.float()
        c_x = torch.arange(w, dtype=torch.float32, device=output.device).reshape(1, 1, -1, 1)
        c_y = torch.arange(h, dtype=torch.float32, device=output.device).reshape(1, -1, 1, 1)

        b_xy = self.scale_xy * torch.sigmoid(box_logits[..., :2]) - 0.5 * (self.scale_xy - 1)
        b_xy[..., 0].add_(c_x)
        b_xy[..., 1].add_(c_y)
        b_xy[..., 0].div_(w)
        b_xy[..., 1].div_(h)

        # Box dimension
        anchors = self.anchors.to(dtype=box_logits.dtype).view(1, 1, 1, -1, 2)
        max_wh_logits = torch.log(2 / anchors)
        wh_logits = torch.minimum(box_logits[..., 2:4], max_wh_logits)
        b_wh = (torch.exp(wh_logits) * anchors).clamp_min_(2 * torch.finfo(box_logits.dtype).eps)

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
    ) -> list[dict[str, Tensor]]:
        return _post_process(
            boxes.clamp(0, 1).reshape(boxes.shape[0], -1, 4),
            torch.sigmoid(b_o).flatten(1),
            torch.sigmoid(b_scores).reshape(b_scores.shape[0], -1, b_scores.shape[-1]),
            rpn_nms_thresh,
            box_score_thresh,
        )

    def _build_targets(
        self, pred_boxes: Tensor, b_o: Tensor, b_scores: Tensor, target: list[dict[str, Tensor]]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        _b, h, w, num_anchors = b_o.shape

        target_boxes = torch.zeros_like(pred_boxes)
        target_scores = torch.zeros_like(b_scores)
        obj_mask = torch.zeros_like(b_o, dtype=torch.bool)
        noobj_mask = torch.ones_like(b_o, dtype=torch.bool)

        self.all_anchors: Tensor
        self.anchor_mask: Tensor
        anchor_boxes = torch.cat((-self.all_anchors / 2, self.all_anchors / 2), dim=-1)
        for batch_idx, target_ in enumerate(target):
            boxes, labels = target_["boxes"], target_["labels"]
            if boxes.shape[0] == 0:
                continue

            ignored = box_iou(pred_boxes[batch_idx].detach().reshape(-1, 4), boxes).amax(dim=1)
            noobj_mask[batch_idx] = ignored.reshape(h, w, num_anchors) <= self.ignore_thresh

            gt_wh = boxes[:, 2:] - boxes[:, :2]
            anchor_ious = box_iou(torch.cat((-gt_wh / 2, gt_wh / 2), dim=-1), anchor_boxes)
            selected = anchor_ious > self.iou_thresh
            selected.scatter_(1, anchor_ious.argmax(dim=1, keepdim=True), True)

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            grid_x = (centers[:, 0] * w).long().clamp_(0, w - 1)
            grid_y = (centers[:, 1] * h).long().clamp_(0, h - 1)
            for gt_idx, global_anchor_idx in selected.nonzero():
                local_anchor = (self.anchor_mask == global_anchor_idx).nonzero()
                if local_anchor.numel() == 0:
                    continue
                anchor_idx = local_anchor.item()
                y, x = grid_y[gt_idx], grid_x[gt_idx]
                obj_mask[batch_idx, y, x, anchor_idx] = True
                noobj_mask[batch_idx, y, x, anchor_idx] = False
                target_boxes[batch_idx, y, x, anchor_idx] = boxes[gt_idx]
                target_scores[batch_idx, y, x, anchor_idx] = 0
                target_scores[batch_idx, y, x, anchor_idx, labels[gt_idx]] = 1

        return target_boxes, target_scores, obj_mask, noobj_mask

    def _compute_losses(
        self,
        pred_boxes: Tensor,
        b_o: Tensor,
        b_scores: Tensor,
        target: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        target_boxes, target_scores, obj_mask, noobj_mask = self._build_targets(pred_boxes, b_o, b_scores, target)

        bbox_loss = pred_boxes.flatten()[:0].sum()
        if torch.any(obj_mask):
            matched_boxes = target_boxes[obj_mask]
            areas = (matched_boxes[:, 2] - matched_boxes[:, 0]) * (matched_boxes[:, 3] - matched_boxes[:, 1])
            bbox_loss = ((2 - areas) * ciou_loss(pred_boxes[obj_mask], matched_boxes).diagonal()).sum()

        return {
            "obj_loss": self.lambda_obj
            * F.binary_cross_entropy_with_logits(b_o[obj_mask], torch.ones_like(b_o[obj_mask]), reduction="sum")
            / b_o.shape[0],
            "noobj_loss": self.lambda_noobj
            * F.binary_cross_entropy_with_logits(b_o[noobj_mask], torch.zeros_like(b_o[noobj_mask]), reduction="sum")
            / b_o.shape[0],
            "bbox_loss": self.lambda_coords * bbox_loss / b_o.shape[0],
            "clf_loss": self.lambda_class
            * F.binary_cross_entropy_with_logits(b_scores[obj_mask], target_scores[obj_mask], reduction="sum")
            / b_o.shape[0],
        }

    def forward(
        self, x: Tensor, target: list[dict[str, Tensor]] | None = None
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """Perform detection on an image tensor and returns either the loss dictionary in training mode
        or the list of detections in eval mode.

        Args:
            x (torch.Tensor[N, 3, H, W]): input image tensor
            target (list<dict>, optional): each dict must have two keys `boxes` of type torch.Tensor[*, 4]
                and `labels` of type torch.Tensor[*]

        Returns:
            loss dictionary in training mode or list of detections in eval mode

        Raises:
            ValueError: if `target` is not specified in training mode
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
        anchors: Tensor | None = None,
        act_layer: nn.Module | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
        drop_layer: Callable[..., nn.Module] | None = None,
        conv_layer: Callable[..., nn.Module] | None = None,
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
        all_anchors = anchors.reshape(-1, 2)

        self.head1 = nn.Sequential(
            *conv_sequence(
                128, 256, act_layer, norm_layer, None, conv_layer, kernel_size=3, padding=1, bias=(norm_layer is None)
            ),
            *conv_sequence(256, (5 + num_classes) * 3, None, None, None, conv_layer, kernel_size=1, bias=True),
        )

        self.yolo1 = YoloLayer(
            anchors[0],
            num_classes=num_classes,
            scale_xy=1.2,
            all_anchors=all_anchors,
            anchor_mask=torch.arange(3),
        )

        conv = partial(
            conv_sequence, act_layer=act_layer, norm_layer=norm_layer, drop_layer=drop_layer, conv_layer=conv_layer
        )

        self.pre_head2 = nn.Sequential(*conv(128, 256, kernel_size=3, padding=1, stride=2))
        self.head2_1 = nn.Sequential(
            *conv(512, 256, kernel_size=1),
            *conv(256, 512, kernel_size=3, padding=1),
            *conv(512, 256, kernel_size=1),
            *conv(256, 512, kernel_size=3, padding=1),
            *conv(512, 256, kernel_size=1),
        )
        self.head2_2 = nn.Sequential(
            *conv_sequence(
                256, 512, act_layer, norm_layer, None, conv_layer, kernel_size=3, padding=1, bias=(norm_layer is None)
            ),
            *conv_sequence(512, (5 + num_classes) * 3, None, None, None, conv_layer, kernel_size=1, bias=True),
        )

        self.yolo2 = YoloLayer(
            anchors[1],
            num_classes=num_classes,
            scale_xy=1.1,
            all_anchors=all_anchors,
            anchor_mask=torch.arange(3, 6),
        )

        self.pre_head3 = nn.Sequential(*conv(256, 512, kernel_size=3, padding=1, stride=2))
        self.head3 = nn.Sequential(
            *conv(1024, 512, kernel_size=1),
            *conv(512, 1024, kernel_size=3, padding=1),
            *conv(1024, 512, kernel_size=1),
            *conv(512, 1024, kernel_size=3, padding=1),
            *conv(1024, 512, kernel_size=1),
            *conv(512, 1024, kernel_size=3, padding=1),
            *conv_sequence(1024, (5 + num_classes) * 3, None, None, None, conv_layer, kernel_size=1, bias=True),
        )

        self.yolo3 = YoloLayer(
            anchors[2],
            num_classes=num_classes,
            scale_xy=1.05,
            all_anchors=all_anchors,
            anchor_mask=torch.arange(6, 9),
        )
        init_module(self, "leaky_relu")
        # Zero init
        self.head1[-1].weight.data.zero_()
        self.head1[-1].bias.data.zero_()
        self.head2_2[-1].weight.data.zero_()
        self.head2_2[-1].bias.data.zero_()
        self.head3[-1].weight.data.zero_()
        self.head3[-1].bias.data.zero_()
        self.register_load_state_dict_post_hook(self._sync_anchors)

    def _sync_anchors(self, _module: nn.Module, _incompatible_keys: Any) -> None:
        layers = (self.yolo1, self.yolo2, self.yolo3)
        all_anchors = torch.cat([layer.anchors for layer in layers])
        for layer in layers:
            layer.all_anchors = all_anchors

    def forward(
        self, feats: list[Tensor], target: list[dict[str, Tensor]] | None = None
    ) -> list[dict[str, Tensor]] | dict[str, Tensor]:
        o1 = self.head1(feats[0])

        h2 = self.pre_head2(feats[0])
        h2 = torch.cat([h2, feats[1]], dim=1)
        h2 = self.head2_1(h2)
        o2 = self.head2_2(h2)

        h3 = self.pre_head3(h2)
        h3 = torch.cat([h3, feats[2]], dim=1)
        o3 = self.head3(h3)

        if not self.training:
            outputs = [
                layer._format_outputs(output)  # noqa: SLF001
                for layer, output in zip((self.yolo1, self.yolo2, self.yolo3), (o1, o2, o3), strict=True)
            ]
            boxes = torch.cat([output[0].reshape(output[0].shape[0], -1, 4) for output in outputs], dim=1)
            objectness = torch.cat([output[1].flatten(1) for output in outputs], dim=1).sigmoid()
            scores = torch.cat(
                [output[2].reshape(output[2].shape[0], -1, output[2].shape[-1]) for output in outputs], dim=1
            ).sigmoid()
            return _post_process(
                boxes.clamp(0, 1),
                objectness,
                scores,
                self.yolo1.rpn_nms_thresh,
                self.yolo1.box_score_thresh,
            )

        y1 = self.yolo1(o1, target)
        y2 = self.yolo2(o2, target)
        y3 = self.yolo3(o3, target)

        return {k: y1[k] + y2[k] + y3[k] for k in y1}


class YOLOv4(nn.Module):
    def __init__(
        self,
        layout: list[tuple[int, int]],
        num_classes: int = 80,
        in_channels: int = 3,
        stem_channels: int = 32,
        anchors: Tensor | None = None,
        act_layer: nn.Module | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
        drop_layer: Callable[..., nn.Module] | None = None,
        conv_layer: Callable[..., nn.Module] | None = None,
        backbone_norm_layer: Callable[[int], nn.Module] | None = None,
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

    def forward(
        self, x: Tensor, target: list[dict[str, Tensor]] | None = None
    ) -> list[dict[str, Tensor]] | dict[str, Tensor]:
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
    layout: list[tuple[int, int]],
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
    ["YOLOv4: Optimal Speed and Accuracy of Object Detection"](https://arxiv.org/pdf/2004.10934.pdf).

    The implementation combines a CSPDarknet53-Mish backbone with an SPP/PAN neck and three detection scales.
    Training uses the YOLOv4 anchor masks and multi-anchor assignment, binary cross-entropy for objectness and
    classification, and area-weighted Complete IoU for matched boxes.

    Args:
        pretrained: Whether to request pretrained detector parameters (not currently available)
        progress: If True, displays a progress bar of the download to stderr
        pretrained_backbone: If True, backbone parameters will have been pretrained on Imagenette
        kwargs: keyword args of [`YOLOv4`][holocron.models.detection.yolov4.YOLOv4]

    Returns:
        detection module
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
