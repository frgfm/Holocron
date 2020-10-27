import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.misc import FrozenBatchNorm2d
from typing import Dict, Any, Optional, Callable, Tuple, List, Union

from ..utils import conv_sequence, load_pretrained_params
from .yolo import _YOLO
from ..darknetv2 import DarknetBodyV2, default_cfgs as dark_cfgs
from holocron.nn import ConcatDownsample2d
from holocron.nn.init import init_module


__all__ = ['YOLOv2', 'yolov2']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'yolov2': {'arch': 'YOLOv2', 'backbone': dark_cfgs['darknet19'],
               'url': None},
}


class YOLOv2(_YOLO):

    def __init__(
        self,
        layout: List[Tuple[int, int]],
        num_classes: int = 20,
        in_channels: int = 3,
        stem_chanels: int = 32,
        anchors: Optional[Tensor] = None,
        passthrough_ratio: int = 8,
        lambda_noobj: float = 0.5,
        lambda_coords: float = 5.,
        rpn_nms_thresh: float = 0.7,
        box_score_thresh: float = 0.05,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
        backbone_norm_layer: Optional[Callable[[int], nn.Module]] = None
    ) -> None:

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
    def num_anchors(self) -> int:
        return self.anchors.shape[0]  # type: ignore[index, return-value]

    def _format_outputs(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
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
        b_w = self.anchors[:, 0].view(1, 1, 1, -1) / w * torch.exp(x[..., 2])  # type: ignore[index]
        b_h = self.anchors[:, 1].view(1, 1, 1, -1) / h * torch.exp(x[..., 3])  # type: ignore[index]
        # B * H * W * num_anchors * 4
        b_coords = torch.stack((b_x, b_y, b_w, b_h), dim=4)
        # Objectness
        b_o = torch.sigmoid(x[..., 4])
        # Classification scores
        b_scores = F.softmax(x[..., 5:], dim=-1)

        return b_coords, b_o, b_scores

    def _forward(self, x: Tensor) -> Tensor:

        out, passthrough = self.backbone(x)
        # Downsample the feature map by stacking adjacent features on the channel dimension
        passthrough = self.passthrough_layer(passthrough)

        out = self.block5(out)
        # Stack the downsampled feature map on the channel dimension
        out = torch.cat((passthrough, out), 1)
        out = self.block6(out)

        out = self.head(out)

        return out

    def forward(
        self,
        x: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
        target: Optional[List[Dict[str, Tensor]]] = None
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """Perform detection on an image tensor and returns either the loss dictionary in training mode
        or the list of detections in eval mode.

        Args:
            x (torch.Tensor[N, 3, H, W]): input image tensor
            target (list<dict>, optional): each dict must have two keys `boxes` of type torch.Tensor[-1, 4]
            and `labels` of type torch.Tensor[-1]
        """

        if self.training and target is None:
            raise ValueError("`target` needs to be specified in training mode")

        if isinstance(x, (list, tuple)):
            x = torch.stack(x, dim=0)

        out = self._forward(x)

        # B * H * W * num_anchors
        b_coords, b_o, b_scores = self._format_outputs(out)

        if self.training:
            # Update losses
            return self._compute_losses(b_coords, b_o, b_scores, target)  # type: ignore[arg-type]
        else:
            # B * (H * W * num_anchors)
            b_coords = b_coords.view(b_coords.shape[0], -1, 4)
            b_o = b_o.view(b_o.shape[0], -1)
            b_scores = b_scores.reshape(b_scores.shape[0], -1, self.num_classes)

            # Stack detections into a list
            return self.post_process(b_coords, b_o, b_scores, self.rpn_nms_thresh, self.box_score_thresh)


def _yolo(arch: str, pretrained: bool, progress: bool, pretrained_backbone: bool, **kwargs: Any) -> YOLOv2:

    if pretrained:
        pretrained_backbone = False

    # Build the model
    model = YOLOv2(default_cfgs[arch]['backbone']['layout'], **kwargs)
    # Load backbone pretrained parameters
    if pretrained_backbone:
        load_pretrained_params(model.backbone, default_cfgs[arch]['backbone']['url'], progress,
                               key_replacement=('features.', ''), key_filter='features.')
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'], progress)

    return model


def yolov2(pretrained: bool = False, progress: bool = True, pretrained_backbone: bool = True, **kwargs: Any) -> YOLOv2:
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

    return _yolo('yolov2', pretrained, progress, pretrained_backbone, **kwargs)  # type: ignore[return-value]
