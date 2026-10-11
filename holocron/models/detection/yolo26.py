# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Original YOLO26 nano implementation; no upstream code or pretrained weights are bundled."""

import copy
import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import batched_nms, box_iou

from .._yolo26 import C3k2, ConvNormAct, YOLO26Backbone, fuse_model

__all__ = ["YOLO26", "yolo26n"]


class YOLO26Neck(nn.Module):
    """Fuse three backbone resolutions through the nano FPN/PAN topology."""

    def __init__(self) -> None:
        super().__init__()
        self.top_middle = C3k2(384, 128)
        self.top_small = C3k2(256, 64)
        self.down_small = ConvNormAct(64, 64, stride=2)
        self.bottom_middle = C3k2(192, 128)
        self.down_middle = ConvNormAct(128, 128, stride=2)
        self.bottom_large = C3k2(384, 256, attention=True)

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        small, middle, large = features
        middle = self.top_middle(torch.cat((F.interpolate(large, size=middle.shape[-2:], mode="nearest"), middle), 1))
        small = self.top_small(torch.cat((F.interpolate(middle, size=small.shape[-2:], mode="nearest"), small), 1))
        middle = self.bottom_middle(torch.cat((self.down_small(small), middle), 1))
        large = self.bottom_large(torch.cat((self.down_middle(middle), large), 1))
        return small, middle, large


class PredictionHead(nn.Module):
    """Separate direct box regression and depthwise classification at each scale."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.boxes = nn.ModuleList()
        self.classes = nn.ModuleList()
        hidden_classes = max(64, min(num_classes, 100))
        for channels, stride in zip((64, 128, 256), (8, 16, 32), strict=True):
            box_branch = nn.Sequential(ConvNormAct(channels, 16), ConvNormAct(16, 16), nn.Conv2d(16, 4, 1))
            class_branch = nn.Sequential(
                ConvNormAct(channels, channels, groups=channels),
                ConvNormAct(channels, hidden_classes, kernel_size=1),
                ConvNormAct(hidden_classes, hidden_classes, groups=hidden_classes),
                ConvNormAct(hidden_classes, hidden_classes, kernel_size=1),
                nn.Conv2d(hidden_classes, num_classes, 1),
            )
            nn.init.constant_(box_branch[-1].bias, 1.0)
            nn.init.constant_(class_branch[-1].bias, math.log(5 / num_classes / (640 / stride) ** 2))
            self.boxes.append(box_branch)
            self.classes.append(class_branch)

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        distances, logits, points, scales = [], [], [], []
        for feature, box_branch, class_branch in zip(features, self.boxes, self.classes, strict=True):
            height, width = feature.shape[-2:]
            distances.append(box_branch(feature).flatten(2).transpose(1, 2))
            logits.append(class_branch(feature).flatten(2).transpose(1, 2))
            y, x = torch.meshgrid(
                torch.arange(height, device=feature.device, dtype=torch.float32),
                torch.arange(width, device=feature.device, dtype=torch.float32),
                indexing="ij",
            )
            one = feature.new_ones((), dtype=torch.float32)
            scale = torch.stack((one / width, one / height))
            points.append((torch.stack((x, y), -1).reshape(-1, 2) + 0.5) * scale)
            scales.append(scale.expand(height * width, 2))
        raw = torch.cat(distances, 1).float()
        points_tensor, scales_tensor = torch.cat(points), torch.cat(scales)
        # Four signed distances, without DFL bins or an upper bound. A numerical
        # guard keeps a temporarily negative predicted width valid for the loss.
        center = points_tensor + 0.5 * (raw[..., 2:] - raw[..., :2]) * scales_tensor
        size = (raw[..., 2:] + raw[..., :2]).clamp_min(1e-4) * scales_tensor
        boxes = torch.cat((center - size / 2, center + size / 2), -1)
        return boxes, torch.cat(logits, 1).float(), points_tensor, scales_tensor


def _paired_ciou(boxes: Tensor, targets: Tensor) -> Tensor:
    """Return differentiable CIoU loss for corresponding normalized boxes.

    Returns:
        one loss value per corresponding pair.
    """
    size, target_size = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-7), targets[:, 2:] - targets[:, :2]
    intersection = (
        torch.minimum(boxes[:, 2:], targets[:, 2:]) - torch.maximum(boxes[:, :2], targets[:, :2])
    ).clamp_min(0)
    area_intersection = intersection.prod(-1)
    iou = area_intersection / (size.prod(-1) + target_size.prod(-1) - area_intersection).clamp_min(1e-7)
    enclosure = torch.maximum(boxes[:, 2:], targets[:, 2:]) - torch.minimum(boxes[:, :2], targets[:, :2])
    center_distance = ((boxes[:, :2] + boxes[:, 2:] - targets[:, :2] - targets[:, 2:]) / 2).square().sum(-1)
    aspect = (4 / math.pi**2) * (
        torch.atan(target_size[:, 0] / target_size[:, 1]) - torch.atan(size[:, 0] / size[:, 1])
    ).square()
    with torch.no_grad():
        alpha = aspect / (1 - iou + aspect).clamp_min(1e-7)
    return 1 - iou + center_distance / enclosure.square().sum(-1).clamp_min(1e-7) + alpha * aspect


@torch.no_grad()
def _assign(
    boxes: Tensor, logits: Tensor, points: Tensor, target: dict[str, Tensor], topk: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Task-aligned matching with per-object top-k and per-point conflict resolution.

    Returns:
        assigned boxes, soft class targets, and the positive point mask.
    """
    assigned_boxes = torch.zeros_like(boxes)
    scores = torch.zeros_like(logits)
    positive = torch.zeros(len(boxes), device=boxes.device, dtype=torch.bool)
    if not len(target["boxes"]):
        return assigned_boxes, scores, positive
    # Match the decoder's calculation dtype without changing caller-owned targets.
    gt, labels = target["boxes"].to(dtype=boxes.dtype), target["labels"]
    overlap = box_iou(gt, boxes).clamp_min(0)
    alignment = logits.sigmoid()[:, labels].T.sqrt() * overlap.pow(6)
    inside = ((points[None] > gt[:, None, :2]) & (points[None] < gt[:, None, 2:])).all(-1)
    # A tiny object can fall between feature-grid points. Give it a candidate,
    # rather than silently dropping its supervision. This is not full STAL.
    nearest = (points[None] - (gt[:, None, :2] + gt[:, None, 2:]) / 2).square().sum(-1).argmin(-1)
    inside[torch.arange(len(gt), device=gt.device), nearest] = True
    ranking = alignment.masked_fill(~inside, -1)
    indices = ranking.topk(min(topk, len(boxes)), dim=1).indices
    selected = torch.zeros_like(inside).scatter_(1, indices, True) & inside
    # Resolve competing objects only among their selected candidates.
    quality, owner = overlap.masked_fill(~selected, -1).max(0)
    positive = quality >= 0
    selected &= F.one_hot(owner, len(gt)).T.bool() & positive[None]
    selected_alignment = alignment * selected
    peak_alignment = selected_alignment.amax(1).clamp_min(1e-12)
    peak_iou = (overlap * selected).amax(1)
    soft_quality = (selected_alignment * (peak_iou / peak_alignment)[:, None]).amax(0)
    # Keep a small learning signal when an assigned prediction has no overlap.
    # The floor is a local training safeguard, not the paper's STAL recipe.
    soft_quality = soft_quality.clamp_min(1e-3)
    assigned_boxes[positive] = gt[owner[positive]]
    scores[positive, labels[owner[positive]]] = soft_quality[positive]
    return assigned_boxes, scores, positive


class YOLO26(nn.Module):
    """YOLO26 nano detector with direct box regression and dual training heads.

    Training accepts normalized ``xyxy`` boxes and zero-based integer labels,
    and returns four scalar losses. Evaluation uses the one-to-one head without
    NMS by default. Inputs must have equal dimensions divisible by 32.

    This is an original architecture implementation, not a reproduction of the
    released weights. The full MuSGD, STAL and Progressive Loss training recipe
    is not included. See the reference guide before comparing COCO results.

    Args:
        num_classes: number of foreground classes (no background class).
        in_channels: number of input channels.
        box_score_thresh: minimum returned class probability.
        max_detections: maximum detections per image.
        nms: use the one-to-many head and class-aware NMS at evaluation.
        nms_thresh: IoU threshold for optional NMS.
        box_weight: multiplier for each head's CIoU loss.
        class_weight: multiplier for each head's classification BCE.
    """

    def __init__(
        self,
        num_classes: int = 80,
        in_channels: int = 3,
        box_score_thresh: float = 0.05,
        max_detections: int = 300,
        nms: bool = False,
        nms_thresh: float = 0.7,
        box_weight: float = 7.5,
        class_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if num_classes < 1 or max_detections < 1:
            raise ValueError("num_classes and max_detections must be positive")
        if not 0 <= box_score_thresh <= 1 or not 0 <= nms_thresh <= 1:
            raise ValueError("Score and NMS thresholds must be in [0, 1]")
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.box_score_thresh = box_score_thresh
        self.max_detections = max_detections
        self.nms = nms
        self.nms_thresh = nms_thresh
        self.box_weight, self.class_weight = box_weight, class_weight
        self.backbone = YOLO26Backbone(in_channels=in_channels)
        self.neck = YOLO26Neck()
        self.one_to_many = PredictionHead(num_classes)
        self.one_to_one = copy.deepcopy(self.one_to_many)
        self.deployed = False

    def _validate_targets(self, targets: list[dict[str, Tensor]], images: Tensor) -> None:
        if len(targets) != len(images):
            raise ValueError("Provide one target dictionary per image")
        for target in targets:
            boxes, labels = target["boxes"], target["labels"]
            if boxes.ndim != 2 or boxes.shape[1] != 4 or labels.shape != (len(boxes),):
                raise ValueError("Targets need boxes shaped (N, 4) and labels shaped (N,)")
            if not boxes.is_floating_point() or labels.dtype != torch.long:
                raise ValueError("Boxes must be floating point and labels must be torch.long")
            if boxes.device != images.device or labels.device != images.device:
                raise ValueError("Images and targets must be on the same device")
            if not torch.isfinite(boxes).all() or (boxes < 0).any() or (boxes > 1).any():
                raise ValueError("Boxes must contain finite normalized coordinates in [0, 1]")
            if (boxes[:, 2:] <= boxes[:, :2]).any() or (labels < 0).any() or (labels >= self.num_classes).any():
                raise ValueError("Boxes need positive area and labels must be valid foreground class indices")

    def _losses(
        self, output: tuple[Tensor, Tensor, Tensor, Tensor], targets: list[dict[str, Tensor]], topk: int
    ) -> tuple[Tensor, Tensor]:
        boxes, logits, points, _ = output
        box_loss, class_loss = boxes.sum() * 0, logits.sum() * 0
        for predicted, scores, target in zip(boxes, logits, targets, strict=True):
            matched, quality, positive = _assign(predicted.detach(), scores.detach(), points, target, topk)
            normalizer = quality.sum().clamp_min(1)
            class_loss += F.binary_cross_entropy_with_logits(scores, quality, reduction="sum") / normalizer
            if positive.any():
                box_loss += (
                    _paired_ciou(predicted[positive], matched[positive]) * quality[positive].sum(-1)
                ).sum() / normalizer
        return self.box_weight * box_loss / len(boxes), self.class_weight * class_loss / len(boxes)

    def forward(
        self, images: Tensor | Sequence[Tensor], target: list[dict[str, Tensor]] | None = None
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        if isinstance(images, (list, tuple)):
            if not images or any(image.shape != images[0].shape for image in images):
                raise ValueError("Provide a non-empty list of equally sized images")
            images = torch.stack(images)
        if images.ndim != 4 or images.shape[0] < 1 or images.shape[1] != self.in_channels:
            raise ValueError("Images must have shape (N, in_channels, H, W) with N > 0")
        if min(images.shape[-2:]) < 32 or any(size % 32 for size in images.shape[-2:]):
            raise ValueError("Image height and width must be at least 32 and divisible by 32")
        if self.training:
            if self.deployed:
                raise RuntimeError("A deployment model cannot train; keep the original training model")
            if target is None:
                raise ValueError("Training requires targets")
            self._validate_targets(target, images)
        else:
            self._inference_head()
        features = self.neck(self.backbone(images))
        if self.training:
            many_box, many_class = self._losses(self.one_to_many(features), target, topk=10)
            detached = tuple(feature.detach() for feature in features)
            one_box, one_class = self._losses(self.one_to_one(detached), target, topk=1)
            return {
                "many_box_loss": many_box,
                "many_class_loss": many_class,
                "one_box_loss": one_box,
                "one_class_loss": one_class,
            }
        boxes, logits, _, _ = self._inference_head()(features)
        return self.post_process(boxes, logits)

    def _inference_head(self) -> nn.Module:
        head = self.one_to_many if self.nms else self.one_to_one
        if isinstance(head, nn.Identity):
            # Identity marks an intentionally removed head, not a bad input type.
            raise RuntimeError(  # noqa: TRY004
                "The selected head was removed during deployment; use the original model to change nms"
            )
        return head

    def post_process(self, boxes: Tensor, logits: Tensor) -> list[dict[str, Tensor]]:
        """Select the top class per point, optionally apply NMS, and bound output size.

        Returns:
            one dictionary with boxes, scores, and labels per image.
        """
        predictions = []
        for raw_boxes, image_logits in zip(boxes, logits, strict=True):
            image_boxes = raw_boxes.clamp(0, 1)
            score_logits, labels = image_logits.max(-1)
            scores = score_logits.sigmoid()
            valid_boxes = (image_boxes[:, 2:] > image_boxes[:, :2]).all(-1)
            selection_scores = scores.masked_fill(~valid_boxes, -torch.inf)
            candidate_limit = 3000 if self.nms else self.max_detections
            selected = selection_scores.topk(min(candidate_limit, scores.shape[0])).indices
            selected = selected[selection_scores[selected] >= self.box_score_thresh]
            if self.nms:
                selected = selected[
                    batched_nms(image_boxes[selected], scores[selected], labels[selected], self.nms_thresh)
                ]
            selected = selected[: self.max_detections]
            predictions.append({"boxes": image_boxes[selected], "scores": scores[selected], "labels": labels[selected]})
        return predictions

    def to_deploy(self) -> "YOLO26":
        """Return an evaluation copy with folded batch norms and only the selected head.

        The original model remains trainable. Save a deployment state dictionary
        and load it into another ``yolo26n(...).eval().to_deploy()`` instance.

        Returns:
            independent model ready for inference.

        Raises:
            ValueError: if called in training mode.
        """
        if self.training:
            raise ValueError("Call eval() before to_deploy()")
        self._inference_head()
        model = copy.deepcopy(self)
        if model.nms:
            model.one_to_one = nn.Identity()
        else:
            model.one_to_many = nn.Identity()
        fuse_model(model)
        model.deployed = True
        return model


def yolo26n(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> YOLO26:
    """Build a randomly initialized YOLO26 nano detector.

    Args:
        pretrained: unavailable; requesting weights raises an error.
        progress: kept for compatibility with other detection factories.
        **kwargs: arguments passed to :class:`YOLO26`.

    Returns:
        nano detector with dual training heads.

    Raises:
        ValueError: if pretrained detector or backbone weights are requested.
    """
    if pretrained or kwargs.pop("pretrained_backbone", False):
        raise ValueError("No Holocron YOLO26 pretrained weights are available")
    return YOLO26(**kwargs)
