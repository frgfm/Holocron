# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Shared nano feature extractor for the YOLO26 task models.

The channel counts, stage depths, and connectivity follow the published YOLO26
nano model configuration (depth 0.5, width 0.25). The operations below are native
PyTorch implementations; no Ultralytics dependency or pretrained weights are used.
Reference: the YOLO26 YAML at Ultralytics revision
abd16e057bc0fde135c557d95e1fac31413d8575, path
ultralytics/cfg/models/26/yolo26.yaml.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ConvNormAct(nn.Sequential):
    """A same-padded convolution with batch normalization and optional SiLU.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: odd convolution kernel size.
        stride: convolution stride.
        groups: number of convolution groups.
        activation: whether to apply SiLU after normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03),
            nn.SiLU(inplace=True) if activation else nn.Identity(),
        )


def fuse_model(module: nn.Module) -> None:
    """Fold batch normalization into convolutions in place, for inference.

    Device and dtype are preserved. Calling this twice is harmless. Save an
    unfused checkpoint if training must resume afterwards.

    Args:
        module: model or block in evaluation mode.

    Raises:
        ValueError: if the model or a convolution block is in training mode.
    """
    blocks = [layer for layer in module.modules() if isinstance(layer, ConvNormAct)]
    if module.training or any(layer.training or layer[0].training or layer[1].training for layer in blocks):
        raise ValueError("Call eval() before fusing batch normalization")
    for block in blocks:
        if isinstance(block[1], nn.BatchNorm2d):
            block[0] = nn.utils.fuse_conv_bn_eval(block[0], block[1])
            block[1] = nn.Identity().eval()


class _ResidualBottleneck(nn.Module):
    def __init__(self, channels: int, expansion: float = 0.5) -> None:
        super().__init__()
        hidden = int(channels * expansion)
        self.transform = nn.Sequential(ConvNormAct(channels, hidden), ConvNormAct(hidden, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        return x + self.transform(x)


class _C3k(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = channels // 2
        self.process = nn.Sequential(
            ConvNormAct(channels, hidden, 1),
            _ResidualBottleneck(hidden, 1.0),
            _ResidualBottleneck(hidden, 1.0),
        )
        self.bypass = ConvNormAct(channels, hidden, 1)
        self.combine = ConvNormAct(2 * hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        return self.combine(torch.cat((self.process(x), self.bypass(x)), dim=1))


class _PositionAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.num_heads = max(channels // 64, 1)
        self.value_dim = channels // self.num_heads
        self.key_dim = self.value_dim // 2
        self.qkv = ConvNormAct(channels, channels + 2 * self.num_heads * self.key_dim, 1, activation=False)
        self.position = ConvNormAct(channels, channels, groups=channels, activation=False)
        self.projection = ConvNormAct(channels, channels, 1, activation=False)
        self.feedforward = nn.Sequential(
            ConvNormAct(channels, 2 * channels, 1), ConvNormAct(2 * channels, channels, 1, activation=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        batch, channels, height, width = x.shape
        packed = self.qkv(x).reshape(batch, self.num_heads, 2 * self.key_dim + self.value_dim, height * width)
        query, key, value = packed.split((self.key_dim, self.key_dim, self.value_dim), dim=2)
        attended = F.scaled_dot_product_attention(
            query.transpose(2, 3), key.transpose(2, 3), value.transpose(2, 3)
        ).transpose(2, 3)
        context = attended.reshape(batch, channels, height, width)
        context = context + self.position(value.reshape(batch, channels, height, width))  # noqa: PLR6104
        residual = x + self.projection(context)
        return residual + self.feedforward(residual)


class C3k2(nn.Module):
    """Split features, refine one part, then combine all intermediate features.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        use_c3k: use two nested residual bottlenecks in each refinement block.
        expansion: hidden channels as a fraction of output channels.
        attention: append position attention to a simple bottleneck instead.
        num_blocks: number of refinement blocks after depth scaling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        use_c3k: bool = True,
        expansion: float = 0.5,
        attention: bool = False,
        num_blocks: int = 1,
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        self.split = ConvNormAct(in_channels, 2 * hidden, 1)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if attention:
                block = nn.Sequential(_ResidualBottleneck(hidden), _PositionAttention(hidden))
            else:
                block = _C3k(hidden) if use_c3k else _ResidualBottleneck(hidden)
            self.blocks.append(block)
        self.combine = ConvNormAct((2 + num_blocks) * hidden, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        """Return the fused split and refined feature maps.

        Returns:
            Feature tensor with the requested output channels.
        """
        features = list(self.split(x).chunk(2, dim=1))
        for block in self.blocks:
            features.append(block(features[-1]))
        return self.combine(torch.cat(features, dim=1))


class _PyramidPool(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.reduce = ConvNormAct(channels, channels // 2, 1, activation=False)
        self.pool = nn.MaxPool2d(5, stride=1, padding=2)
        self.combine = ConvNormAct(2 * channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        pooled = [self.reduce(x)]
        for _ in range(3):
            pooled.append(self.pool(pooled[-1]))
        return x + self.combine(torch.cat(pooled, dim=1))


class _PartialAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.split = ConvNormAct(channels, channels, 1)
        self.attention = _PositionAttention(channels // 2)
        self.combine = ConvNormAct(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        bypass, attended = self.split(x).chunk(2, dim=1)
        return self.combine(torch.cat((bypass, self.attention(attended)), dim=1))


class YOLO26Backbone(nn.Module):
    """YOLO26 nano backbone returning stride-8, stride-16, stride-32 features.

    Both the detection and semantic task models use the complete backbone.
    Spatial dimensions are rounded up at each stride-2 convolution.

    Args:
        in_channels: number of image channels.

    Raises:
        ValueError: if the number of image channels is not a positive integer.
    """

    out_channels = (128, 128, 256)

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        if isinstance(in_channels, bool) or not isinstance(in_channels, int) or in_channels < 1:
            raise ValueError("in_channels must be a positive integer")
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, 16, stride=2),
            ConvNormAct(16, 32, stride=2),
            C3k2(32, 64, use_c3k=False, expansion=0.25),
        )
        self.stage3 = nn.Sequential(ConvNormAct(64, 64, stride=2), C3k2(64, 128, use_c3k=False, expansion=0.25))
        self.stage4 = nn.Sequential(ConvNormAct(128, 128, stride=2), C3k2(128, 128))
        self.stage5 = nn.Sequential(
            ConvNormAct(128, 256, stride=2), C3k2(256, 256), _PyramidPool(256), _PartialAttention(256)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # ty: ignore[invalid-mutable-override]
        """Return the three feature maps in increasing stride order.

        Returns:
            Stride-8, stride-16, and stride-32 feature tensors.
        """
        stride8 = self.stage3(self.stem(x))
        stride16 = self.stage4(stride8)
        stride32 = self.stage5(stride16)
        return stride8, stride16, stride32
