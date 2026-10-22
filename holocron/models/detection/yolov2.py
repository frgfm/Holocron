# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops.misc import FrozenBatchNorm2d

from holocron.nn import ConcatDownsample2d
from holocron.nn.init import init_module

from ..classification.darknetv2 import DarknetBodyV2
from ..classification.darknetv2 import default_cfgs as dark_cfgs
from ..utils import conv_sequence, load_pretrained_params
from .yolo import _YOLO

__all__ = ["YOLOv2", "yolov2"]


default_cfgs: dict[str, dict[str, Any]] = {
    "yolov2": {"arch": "YOLOv2", "backbone": dark_cfgs["darknet19"], "url": None},
}


class YOLOv2(_YOLO):
    def __init__(
        self,
        layout: list[tuple[int, int]],
        num_classes: int = 20,
        in_channels: int = 3,
        stem_chanels: int = 32,
        anchors: Tensor | None = None,
        passthrough_ratio: int = 8,
        lambda_obj: float = 1,
        lambda_noobj: float = 0.5,
        lambda_class: float = 1,
        lambda_coords: float = 5,
        rpn_nms_thresh: float = 0.7,
        box_score_thresh: float = 0.05,
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
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if backbone_norm_layer is None:
            backbone_norm_layer = norm_layer

        # Priors computed using K-means
        if anchors is None:
            # cf. https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-voc.cfg#L242
            anchors = (
                torch.tensor([
                    [1.3221, 1.73145],
                    [3.19275, 4.00944],
                    [5.05587, 8.09892],
                    [9.47112, 4.84053],
                    [11.2364, 10.0071],
                ])
                / 13
            )

        self.backbone = DarknetBodyV2(
            layout, in_channels, stem_chanels, True, act_layer, backbone_norm_layer, drop_layer, conv_layer
        )

        self.block5 = nn.Sequential(
            *conv_sequence(
                layout[-1][0],
                layout[-1][0],
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=1,
                bias=(norm_layer is None),
            ),
            *conv_sequence(
                layout[-1][0],
                layout[-1][0],
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=1,
                bias=(norm_layer is None),
            ),
        )

        self.passthrough_layer = nn.Sequential(
            *conv_sequence(
                layout[-2][0],
                layout[-2][0] // passthrough_ratio,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=1,
                bias=(norm_layer is None),
            ),
            ConcatDownsample2d(scale_factor=2),
        )

        self.block6 = nn.Sequential(
            *conv_sequence(
                layout[-1][0] + layout[-2][0] // passthrough_ratio * 2**2,
                layout[-1][0],
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=1,
                bias=(norm_layer is None),
            )
        )

        # Each box has P_objectness, 4 coords, and score for each class
        self.head = nn.Conv2d(layout[-1][0], anchors.shape[0] * (5 + num_classes), 1)

        # Register losses
        self.register_buffer("anchors", anchors)

        init_module(self.block5, "leaky_relu")
        init_module(self.passthrough_layer, "leaky_relu")
        init_module(self.block6, "leaky_relu")
        # Initialize the head like a linear (default Conv2D init is the same as Linear)
        if self.head.bias is not None:
            self.head.bias.data.zero_()

    @property
    def num_anchors(self) -> int:
        return self.anchors.shape[0]

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
        xy = b_coords[..., :2]
        wh = b_coords[..., 2:]
        pred_xyxy = torch.cat((xy - wh / 2, xy + wh / 2), dim=-1).reshape(*b_coords.shape)
        if clamp:
            pred_xyxy.clamp_(0, 1)

        return pred_xyxy

    def _format_outputs(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Formats convolutional layer output

        Args:
            x (torch.Tensor[N, num_anchors * (5 + num_classes), H, W]): output tensor

        Returns:
            torch.Tensor[N, H, W, num_anchors, 4]: relative coordinates in format (x, y, w, h)
            torch.Tensor[N, H, W, num_anchors]: objectness scores
            torch.Tensor[N, H, W, num_anchors, num_classes]: classification scores
        """
        b, _, h, w = x.shape
        # (B, C, H, W) --> (B, H, W, num_anchors, 5 + num_classes)
        x = x.reshape(b, self.num_anchors, 5 + self.num_classes, h, w).permute(0, 3, 4, 1, 2)
        # Classification scores
        b_scores = F.softmax(x[..., -self.num_classes :], dim=-1)

        # Cell offset
        c_x = torch.arange(w, dtype=torch.float, device=x.device)
        c_y = torch.arange(h, dtype=torch.float, device=x.device)
        # Box coordinates
        b_x = (torch.sigmoid(x[..., 0]) + c_x.reshape(1, 1, -1, 1)) / w
        b_y = (torch.sigmoid(x[..., 1]) + c_y.reshape(1, -1, 1, 1)) / h
        self.anchors: Tensor
        b_w = self.anchors[:, 0].reshape(1, 1, 1, -1) * torch.exp(x[..., 2])
        b_h = self.anchors[:, 1].reshape(1, 1, 1, -1) * torch.exp(x[..., 3])
        # (B, H, W, num_anchors, 4)
        b_coords = torch.stack((b_x, b_y, b_w, b_h), dim=4)
        # Objectness
        b_o = torch.sigmoid(x[..., 4])

        return b_coords, b_o, b_scores

    def _forward(self, x: Tensor) -> Tensor:
        out, passthrough = self.backbone(x)
        # Downsample the feature map by stacking adjacent features on the channel dimension
        passthrough = self.passthrough_layer(passthrough)

        out = self.block5(out)
        # Stack the downsampled feature map on the channel dimension
        out = torch.cat((passthrough, out), 1)
        out = self.block6(out)

        return self.head(out)

    def forward(  # ty: ignore[invalid-mutable-override]
        self, x: Tensor | list[Tensor] | tuple[Tensor, ...], target: list[dict[str, Tensor]] | None = None
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

        # (B, H, W, num_anchors)
        b_coords, b_o, b_scores = self._format_outputs(out)

        if self.training:
            # Update losses
            return self._compute_losses(b_coords, b_o, b_scores, target)  # type: ignore[arg-type]

        # (B, H * W * num_anchors)
        b_coords = b_coords.reshape(b_coords.shape[0], -1, 4)
        b_o = b_o.reshape(b_o.shape[0], -1)
        b_scores = b_scores.reshape(b_scores.shape[0], -1, self.num_classes)

        # Stack detections into a list
        return self.post_process(
            b_coords,
            b_o,
            b_scores,
            out.shape[-2:],  # type: ignore[arg-type]
            self.rpn_nms_thresh,
            self.box_score_thresh,
        )


def _yolo(
    arch: str, pretrained: bool, progress: bool, pretrained_backbone: bool, layout: list[tuple[int, int]], **kwargs: Any
) -> YOLOv2:
    if pretrained:
        pretrained_backbone = False

    # Build the model
    model = YOLOv2(layout, **kwargs)
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


def yolov2(pretrained: bool = False, progress: bool = True, pretrained_backbone: bool = True, **kwargs: Any) -> YOLOv2:
    r"""YOLOv2 model from
    ["YOLO9000: Better, Faster, Stronger"](https://pjreddie.com/media/files/papers/YOLO9000.pdf).

    YOLOv2 improves upon YOLO by raising the number of boxes predicted by grid cell (default: 5), introducing
    bounding box priors and predicting class scores for each anchor box in the grid cell.

    For training, YOLOv2 uses the same multi-part loss as YOLO apart from its classification loss:

    $$
    \mathcal{L}_{classification} = \sum\limits_{i=0}^{S^2}  \sum\limits_{j=0}^{B}
    \mathbb{1}_{ij}^{obj} \sum\limits_{c \in classes}
    (p_{ij}(c) - \hat{p}_{ij}(c))^2
    $$

    where:
    - $S$ is size of the output feature map (13 for an input size $(416, 416)$),
    - $B$ is the number of anchor boxes per grid cell (default: 5),
    - $\mathbb{1}_{ij}^{obj}$ equals to 1 if a GT center falls inside the i-th grid cell and among the
    anchor boxes of that cell, has the highest IoU with the j-th box else 0,
    - $p_{ij}(c)$ equals 1 if the assigned ground truth to the j-th anchor box of the i-th cell is classified
    as class $c$,
    - $\hat{p}_{ij}(c)$ is the predicted probability of class $c$ for the j-th anchor box
    in the i-th cell.

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        progress: If True, displays a progress bar of the download to stderr
        pretrained_backbone: If True, backbone parameters will have been pretrained on Imagenette
        kwargs: keyword args of [`YOLOv2`][holocron.models.detection.yolov2.YOLOv2]

    Returns:
        detection module
    """
    if pretrained_backbone:
        kwargs["backbone_norm_layer"] = FrozenBatchNorm2d

    return _yolo(
        "yolov2",
        pretrained,
        progress,
        pretrained_backbone,
        [(64, 0), (128, 1), (256, 1), (512, 2), (1024, 2)],
        **kwargs,
    )
