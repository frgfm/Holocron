# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections import OrderedDict
from collections.abc import Callable
from itertools import pairwise
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import get_model, get_model_weights
from torchvision.models._utils import IntermediateLayerGetter  # noqa: PLC2701

from ...nn import GlobalAvgPool2d
from ...nn.init import init_module
from ..classification.rexnet import rexnet1_3x
from ..utils import conv_sequence, load_pretrained_params

__all__ = ["DynamicUNet", "UNet", "unet", "unet2", "unet_rexnet13", "unet_tvresnet34", "unet_tvvgg11"]


default_cfgs: dict[str, dict[str, Any]] = {
    "unet": {"encoder_layout": [64, 128, 256, 512], "url": None},
    "unet2": {"encoder_layout": [64, 128, 256, 512], "backbone_layers": ["0", "1", "2", "3"], "url": None},
    "unet_vgg11": {"backbone_layers": ["1", "4", "9", "14", "19"], "url": None},
    "unet_tvresnet34": {"backbone_layers": ["relu", "layer1", "layer2", "layer3", "layer4"], "url": None},
    "unet_rexnet13": {
        "backbone_layers": ["3", "5", "7", "13", "18"],
        "url": "https://github.com/frgfm/Holocron/releases/download/v0.1.3/unet_rexnet13_256-38315ff3.pth",
    },
}


def down_path(
    in_chan: int,
    out_chan: int,
    downsample: bool = True,
    padding: int = 0,
    act_layer: nn.Module | None = None,
    norm_layer: Callable[[int], nn.Module] | None = None,
    drop_layer: Callable[..., nn.Module] | None = None,
    conv_layer: Callable[..., nn.Module] | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.MaxPool2d(2)] if downsample else []
    layers.extend([
        *conv_sequence(
            in_chan, out_chan, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=padding
        ),
        *conv_sequence(
            out_chan, out_chan, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=padding
        ),
    ])
    return nn.Sequential(*layers)


class UpPath(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        bilinear_upsampling: bool = True,
        padding: int = 0,
        act_layer: nn.Module | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
        drop_layer: Callable[..., nn.Module] | None = None,
        conv_layer: Callable[..., nn.Module] | None = None,
        up_chan: int | None = None,
    ) -> None:
        super().__init__()

        self.padding = padding
        self.bilinear_upsampling = bilinear_upsampling
        self.upsample: nn.Module
        if bilinear_upsampling:
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            up_chan = in_chan // 2 if up_chan is None else up_chan
            self.upsample = nn.ConvTranspose2d(up_chan, out_chan, 2, stride=2)
            in_chan = in_chan - up_chan + out_chan

        self.block = nn.Sequential(
            *conv_sequence(
                in_chan, out_chan, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=padding
            ),
            *conv_sequence(
                out_chan, out_chan, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=padding
            ),
        )

    def forward(self, downfeats: Tensor | list[Tensor], upfeat: Tensor) -> Tensor:
        downfeats = list(downfeats) if isinstance(downfeats, list) else [downfeats]
        # Upsample expansive features
        if self.padding > 0 and self.bilinear_upsampling:
            upfeat_ = F.interpolate(upfeat, size=downfeats[0].shape[-2:], mode="bilinear", align_corners=True)
        else:
            upfeat_ = self.upsample(upfeat)
        if self.padding > 0 and not self.bilinear_upsampling:
            delta_h = downfeats[0].shape[-2] - upfeat_.shape[-2]
            delta_w = downfeats[0].shape[-1] - upfeat_.shape[-1]
            if torch.jit.is_tracing() or torch.compiler.is_compiling() or delta_h != 0 or delta_w != 0:
                upfeat_ = F.pad(upfeat_, (delta_w // 2, delta_w - delta_w // 2, delta_h // 2, delta_h - delta_h // 2))
        # Crop contracting path features
        for idx, downfeat in enumerate(downfeats):
            delta_w = downfeat.shape[-1] - upfeat_.shape[-1]
            delta_h = downfeat.shape[-2] - upfeat_.shape[-2]
            w_slice = slice(delta_w // 2, delta_w // 2 + upfeat_.shape[-1])
            h_slice = slice(delta_h // 2, delta_h // 2 + upfeat_.shape[-2])
            downfeats[idx] = downfeat[..., h_slice, w_slice]
        # Concatenate both feature maps and forward them
        return self.block(torch.cat((*downfeats, upfeat_), dim=1))


class UNetBackbone(nn.Sequential):
    def __init__(
        self,
        layout: list[int],
        in_channels: int = 3,
        num_classes: int = 10,
        act_layer: nn.Module | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
        drop_layer: Callable[..., nn.Module] | None = None,
        conv_layer: Callable[..., nn.Module] | None = None,
        same_padding: bool = True,
    ) -> None:
        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        # Contracting path
        layers: list[nn.Module] = []
        layout_ = [in_channels, *layout]
        pool = False
        for in_chan, out_chan in pairwise(layout_):
            layers.append(
                down_path(in_chan, out_chan, pool, int(same_padding), act_layer, norm_layer, drop_layer, conv_layer)
            )
            pool = True

        super().__init__(
            OrderedDict([
                ("features", nn.Sequential(*layers)),
                ("pool", GlobalAvgPool2d(flatten=True)),
                ("head", nn.Linear(layout[-1], num_classes)),
            ])
        )

        init_module(self, "relu")


class UNet(nn.Module):
    """Implements a U-Net architecture

    Args:
        layout: number of channels after each contracting block
        in_channels: number of channels in the input tensor
        num_classes: number of output classes
        act_layer: activation layer
        norm_layer: normalization layer
        drop_layer: dropout layer
        conv_layer: convolutional layer
        same_padding: enforces same padding in convolutions
        bilinear_upsampling: replaces transposed conv by bilinear interpolation for upsampling
    """

    def __init__(
        self,
        layout: list[int],
        in_channels: int = 3,
        num_classes: int = 10,
        act_layer: nn.Module | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
        drop_layer: Callable[..., nn.Module] | None = None,
        conv_layer: Callable[..., nn.Module] | None = None,
        same_padding: bool = True,
        bilinear_upsampling: bool = True,
    ) -> None:
        super().__init__()

        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        # Contracting path
        self.encoder = nn.ModuleList([])
        layout_ = [in_channels, *layout]
        pool = False
        for in_chan, out_chan in pairwise(layout_):
            self.encoder.append(
                down_path(in_chan, out_chan, pool, int(same_padding), act_layer, norm_layer, drop_layer, conv_layer)
            )
            pool = True

        self.bridge = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            *conv_sequence(
                layout[-1],
                2 * layout[-1],
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=int(same_padding),
            ),
            *conv_sequence(
                2 * layout[-1],
                layout[-1],
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=int(same_padding),
            ),
        )

        # Expansive path
        self.decoder = nn.ModuleList([])
        layout_ = [chan // 2 if bilinear_upsampling else chan for chan in layout[::-1][:-1]] + [layout[0]]
        up_chan = layout[-1]
        for left_chan, out_chan in zip(layout[::-1], layout_, strict=True):
            self.decoder.append(
                UpPath(
                    left_chan + up_chan,
                    out_chan,
                    bilinear_upsampling,
                    int(same_padding),
                    act_layer,
                    norm_layer,
                    drop_layer,
                    conv_layer,
                    up_chan=up_chan,
                )
            )
            up_chan = out_chan

        # Classifier
        self.classifier = nn.Conv2d(layout[0], num_classes, 1)

        init_module(self, "relu")

    def forward(self, x: Tensor) -> Tensor:
        xs: list[Tensor] = []
        # Contracting path
        for encoder in self.encoder:
            xs.append(encoder(xs[-1] if len(xs) > 0 else x))
        x = self.bridge(xs[-1])

        # Expansive path
        for decoder in self.decoder:
            x = decoder(xs.pop(), x)

        # Classifier
        return self.classifier(x)


class UBlock(nn.Module):
    def __init__(
        self,
        left_chan: int,
        up_chan: int,
        out_chan: int,
        padding: int = 0,
        act_layer: nn.Module | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
        drop_layer: Callable[..., nn.Module] | None = None,
        conv_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()

        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        self.upsample = nn.Sequential(
            *conv_sequence(up_chan, up_chan * 2**2, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=1),
            nn.PixelShuffle(upscale_factor=2),
        )

        self.bn = nn.BatchNorm2d(left_chan) if norm_layer is None else norm_layer(left_chan)

        self.block = nn.Sequential(
            act_layer,
            *conv_sequence(
                left_chan + up_chan,
                out_chan,
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                kernel_size=3,
                padding=padding,
            ),
            *conv_sequence(
                out_chan, out_chan, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=padding
            ),
        )

    def forward(self, downfeat: Tensor, upfeat: Tensor) -> Tensor:
        # Upsample expansive features
        upfeat_ = self.upsample(upfeat)

        # Crop upsampled features
        if torch.jit.is_tracing() or downfeat.shape[-2:] != upfeat_.shape[-2:]:
            upfeat_ = F.interpolate(upfeat_, downfeat.shape[-2:], mode="nearest")

        # Concatenate both feature maps and forward them
        return self.block(torch.cat((self.bn(downfeat), upfeat_), dim=1))


class DynamicUNet(nn.Module):
    """Implements a dynamic U-Net architecture

    Args:
        encoder: feature extractor used for encoding
        num_classes: number of output classes
        act_layer: activation layer
        norm_layer: normalization layer
        drop_layer: dropout layer
        conv_layer: convolutional layer
        same_padding: enforces same padding in convolutions
        input_shape: shape of the input tensor
        final_upsampling: whether to upsample final decoder features to the input resolution
    """

    def __init__(
        self,
        encoder: IntermediateLayerGetter,
        num_classes: int = 10,
        act_layer: nn.Module | None = None,
        norm_layer: Callable[[int], nn.Module] | None = None,
        drop_layer: Callable[..., nn.Module] | None = None,
        conv_layer: Callable[..., nn.Module] | None = None,
        same_padding: bool = True,
        input_shape: tuple[int, int, int] | None = None,
        final_upsampling: bool = False,
    ) -> None:
        super().__init__()

        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        self.encoder = encoder
        # Determine all feature map shapes
        training_mode = self.encoder.training
        self.encoder.eval()
        input_shape = (3, 256, 256) if input_shape is None else input_shape
        with torch.no_grad():
            shapes = [v.shape[1:] for v in self.encoder(torch.zeros(1, *input_shape)).values()]
        chans = [s[0] for s in shapes]
        if training_mode:
            self.encoder.train()

        # Middle layers
        self.bridge = nn.Sequential(
            nn.BatchNorm2d(chans[-1]) if norm_layer is None else norm_layer(chans[-1]),
            act_layer,
            *conv_sequence(
                chans[-1], 2 * chans[-1], act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=1
            ),
            *conv_sequence(
                2 * chans[-1], chans[-1], act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=1
            ),
        )

        # Expansive path
        self.decoder = nn.ModuleList([])
        layout = [*chans[::-1][1:], chans[0]]
        for up_chan, out_chan in zip(chans[::-1], layout, strict=True):
            self.decoder.append(
                UBlock(up_chan, up_chan, out_chan, int(same_padding), act_layer, norm_layer, drop_layer, conv_layer)
            )

        # Final upsampling if sizes don't match
        self.upsample: nn.Sequential | None = None
        if final_upsampling:
            self.upsample = nn.Sequential(
                *conv_sequence(chans[0], chans[0] * 2**2, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=1),
                nn.PixelShuffle(upscale_factor=2),
            )

        # Classifier
        self.classifier = nn.Conv2d(chans[0], num_classes, 1)

        for module in (self.bridge, self.decoder, self.upsample, self.classifier):
            if module is not None:
                init_module(module, "relu")

    def forward(self, x: Tensor) -> Tensor:
        input_size = x.shape[-2:]
        # Contracting path
        xs: list[Tensor] = list(self.encoder(x).values())
        x = self.bridge(xs[-1])

        # Expansive path
        for decoder in self.decoder:
            x = decoder(xs.pop(), x)

        if self.upsample is not None:
            x = self.upsample(x)
            if torch.jit.is_tracing() or x.shape[-2:] != input_size:
                x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)

        # Classifier
        return self.classifier(x)


def _unet(arch: str, pretrained: bool, progress: bool, **kwargs: Any) -> UNet:
    # Build the model
    model = UNet(default_cfgs[arch]["encoder_layout"], **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"], progress)

    return model


def unet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNet:
    """U-Net from
    ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf)

    ![UNet explanation](https://github.com/frgfm/Holocron/releases/download/v0.1.3/unet.png)

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of [`UNet`][holocron.models.segmentation.unet.UNet]

    Returns:
        semantic segmentation model
    """
    return _unet("unet", pretrained, progress, **kwargs)


def _dynamic_unet(
    arch: str, backbone: nn.Module, pretrained: bool, progress: bool, num_classes: int = 21, **kwargs: Any
) -> DynamicUNet:
    # Build the encoder
    encoder = IntermediateLayerGetter(
        backbone, {name: str(idx) for idx, name in enumerate(default_cfgs[arch]["backbone_layers"])}
    )
    # Build the model
    model = DynamicUNet(encoder, num_classes=num_classes, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"], progress)

    return model


def unet2(pretrained: bool = False, progress: bool = True, in_channels: int = 3, **kwargs: Any) -> DynamicUNet:
    """Modified version of U-Net from
    ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf)
    that includes a more advanced upscaling block inspired by [fastai](https://docs.fast.ai/vision.models.unet.html#DynamicUnet)

    ![UNet architecture](https://github.com/frgfm/Holocron/releases/download/v0.1.3/unet.png)

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        progress: If True, displays a progress bar of the download to stderr
        in_channels: number of input channels
        kwargs: keyword args of [`DynamicUNet`][holocron.models.segmentation.unet.DynamicUNet]

    Returns:
        semantic segmentation model
    """
    backbone = UNetBackbone(default_cfgs["unet2"]["encoder_layout"], in_channels=in_channels).features
    kwargs["input_shape"] = kwargs.get("input_shape") or (in_channels, 256, 256)

    return _dynamic_unet("unet2", backbone, pretrained, progress, **kwargs)  # ty: ignore[invalid-argument-type]


def unet_tvvgg11(
    pretrained: bool = False, pretrained_backbone: bool = True, progress: bool = True, **kwargs: Any
) -> DynamicUNet:
    """U-Net from
    ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf)
    with a VGG-11 backbone used as encoder, and more advanced upscaling blocks inspired by [fastai](https://docs.fast.ai/vision.models.unet.html#DynamicUnet)

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        pretrained_backbone: If True, the encoder will load pretrained parameters from ImageNet
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of [`DynamicUNet`][holocron.models.segmentation.unet.DynamicUNet]

    Returns:
        semantic segmentation model
    """
    weights = get_model_weights("vgg11").DEFAULT if pretrained_backbone and not pretrained else None  # ty: ignore[unresolved-attribute]
    backbone: nn.Module = get_model("vgg11", weights=weights).features  # ty: ignore[invalid-assignment]

    return _dynamic_unet("unet_vgg11", backbone, pretrained, progress, **kwargs)


def unet_tvresnet34(
    pretrained: bool = False, pretrained_backbone: bool = True, progress: bool = True, **kwargs: Any
) -> DynamicUNet:
    """U-Net from
    ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf)
    with a ResNet-34 backbone used as encoder, and more advanced upscaling blocks inspired by [fastai](https://docs.fast.ai/vision.models.unet.html#DynamicUnet)

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        pretrained_backbone: If True, the encoder will load pretrained parameters from ImageNet
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of [`DynamicUNet`][holocron.models.segmentation.unet.DynamicUNet]

    Returns:
        semantic segmentation model
    """
    weights = get_model_weights("resnet34").DEFAULT if pretrained_backbone and not pretrained else None  # ty: ignore[unresolved-attribute]
    backbone = get_model("resnet34", weights=weights)
    kwargs["final_upsampling"] = kwargs.get("final_upsampling", True)

    return _dynamic_unet("unet_tvresnet34", backbone, pretrained, progress, **kwargs)


def unet_rexnet13(
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    progress: bool = True,
    in_channels: int = 3,
    **kwargs: Any,
) -> DynamicUNet:
    """U-Net from
    ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf)
    with a ReXNet-1.3x backbone used as encoder, and more advanced upscaling blocks inspired by [fastai](https://docs.fast.ai/vision.models.unet.html#DynamicUnet).

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        pretrained_backbone: If True, the encoder will load pretrained parameters from ImageNet
        progress: If True, displays a progress bar of the download to stderr
        in_channels: the number of input channels
        kwargs: keyword args of [`DynamicUNet`][holocron.models.segmentation.unet.DynamicUNet]

    Returns:
        semantic segmentation model
    """
    backbone = rexnet1_3x(pretrained=pretrained_backbone and not pretrained, in_channels=in_channels).features
    kwargs["input_shape"] = kwargs.get("input_shape") or (in_channels, 256, 256)
    kwargs["final_upsampling"] = kwargs.get("final_upsampling", True)
    kwargs["act_layer"] = kwargs.get("act_layer", nn.SiLU(inplace=True))
    # hotfix of https://github.com/pytorch/vision/issues/3802
    backbone[21] = nn.SiLU(inplace=True)

    return _dynamic_unet("unet_rexnet13", backbone, pretrained, progress, **kwargs)  # ty: ignore[invalid-argument-type]
