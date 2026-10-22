# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Independent implementation of the published YOLO26 semantic nano graph.

Architecture specification:
https://github.com/ultralytics/ultralytics/blob/abd16e057bc0fde135c557d95e1fac31413d8575/ultralytics/cfg/models/26/yolo26-sem.yaml
"""

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .._yolo26 import C3k2, ConvNormAct, YOLO26Backbone, fuse_model

__all__ = ["YOLO26Semantic", "yolo26n_sem"]


class YOLO26Semantic(nn.Module):
    """YOLO26 nano model for pixel classification.

    The backbone and top-down neck follow the published nano architecture. A
    stride-eight classifier predicts the main mask. An auxiliary classifier
    supervises stride-sixteen features during training. Both outputs are
    resized to the exact input dimensions for Holocron's segmentation trainer.

    This is an independent implementation. Upstream checkpoints are not
    supported, and no Cityscapes accuracy is claimed for these random weights.

    Args:
        num_classes: number of output classes
        in_channels: number of input image channels
        auxiliary: include the training-only auxiliary classifier

    Raises:
        ValueError: if the number of classes or input channels is not positive
    """

    def __init__(self, num_classes: int = 19, in_channels: int = 3, auxiliary: bool = True) -> None:
        super().__init__()
        if num_classes < 1 or in_channels < 1:
            raise ValueError("num_classes and in_channels must be positive")
        self.num_classes = num_classes
        self.backbone = YOLO26Backbone(in_channels=in_channels)
        self.neck_p4 = C3k2(384, 128, use_c3k=True)
        self.neck_p3 = C3k2(256, 64, use_c3k=True)
        self.classifier = nn.Sequential(ConvNormAct(64, 64), nn.Conv2d(64, num_classes, 1))
        self.aux_classifier = nn.Sequential(ConvNormAct(128, 64), nn.Conv2d(64, num_classes, 1)) if auxiliary else None

    def forward(self, images: Tensor) -> Tensor | dict[str, Tensor]:  # ty: ignore[invalid-mutable-override]
        p3, p4, p5 = self.backbone(images)
        p4 = self.neck_p4(torch.cat((F.interpolate(p5, size=p4.shape[-2:], mode="nearest"), p4), dim=1))
        p3 = self.neck_p3(torch.cat((F.interpolate(p4, size=p3.shape[-2:], mode="nearest"), p3), dim=1))
        logits = F.interpolate(self.classifier(p3), size=images.shape[-2:], mode="bilinear", align_corners=False)
        if self.training and self.aux_classifier is not None:
            aux = F.interpolate(self.aux_classifier(p4), size=images.shape[-2:], mode="bilinear", align_corners=False)
            return {"out": logits, "aux": aux}
        return logits

    def fuse(self) -> "YOLO26Semantic":
        """Fold normalization layers and remove the auxiliary head in place.

        Call ``eval()`` before this method. The fused model is for inference.
        To load its state dictionary, first construct and fuse a model with
        the same class and input channel counts.

        Returns:
            this model, ready for inference
        """
        fuse_model(self)
        self.aux_classifier = None
        return self


def yolo26n_sem(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> YOLO26Semantic:
    """Build the YOLO26 nano semantic segmentation model.

    Args:
        pretrained: request pretrained weights, which are not available
        progress: kept for compatibility with the model factory API
        **kwargs: arguments of :class:`YOLO26Semantic`

    Returns:
        an untrained semantic segmentation model

    Raises:
        ValueError: if pretrained weights are requested
    """
    if pretrained:
        raise ValueError("Pretrained YOLO26 semantic weights are not available in Holocron")
    return YOLO26Semantic(**kwargs)
