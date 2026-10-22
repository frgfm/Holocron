# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""A shared character backbone and compact CNN/BiGRU CTC recognizer."""

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = ["CTCRecognizer", "CharacterBackbone", "CharacterClassifier"]


class CharacterBackbone(nn.Module):
    """Shared grayscale encoder for glyph classification and line recognition.

    Input images have shape ``(N, 1, 32, W)`` with widths divisible by four.
    The output has shape ``(N, 96, 4, W // 4)``. Normalize grayscale images
    to [-1, 1], using +1 for white padding.
    """

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = 1
        for output, pool in [(24, (2, 2)), (48, (2, 2)), (96, (2, 1)), (96, None)]:
            layers.extend([nn.Conv2d(channels, output, 3, padding=1, bias=False), nn.BatchNorm2d(output), nn.ReLU()])
            if pool is not None:
                layers.append(nn.MaxPool2d(pool))
            channels = output
        self.features = nn.Sequential(*layers)

    def forward(self, images: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        if images.ndim != 4 or images.shape[1] != 1 or images.shape[2] != 32:
            raise ValueError("images must have shape (N, 1, 32, W)")
        if images.shape[0] == 0 or images.shape[3] < 4 or images.shape[3] % 4:
            raise ValueError("image batches must be nonempty and widths positive multiples of four")
        return self.features(images)


class CharacterClassifier(nn.Module):
    """Classify glyphs while retaining their vertical baseline position.

    Args:
        num_classes: number of output character classes

    The input contract matches :class:`CharacterBackbone`. ``lengths`` is a
    one-dimensional integer tensor containing each image's unpadded width
    divided by four. Returns logits of shape ``(N, num_classes)``.
    Transfer ``backbone.state_dict()`` directly to :class:`CTCRecognizer`.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.backbone = CharacterBackbone()
        self.head = nn.Linear(384, num_classes)

    def forward(self, images: Tensor, lengths: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        features = self.backbone(images)
        _validate_lengths(lengths, images.shape[0], features.shape[-1])
        mask = torch.arange(features.shape[-1], device=images.device)[None, :] < lengths.to(images.device)[:, None]
        pooled = (features * mask[:, None, None, :]).sum(-1) / lengths.to(images.device)[:, None, None]
        return self.head(pooled.flatten(1))


class CTCRecognizer(nn.Module):
    """Compact CNN/BiGRU recognizer for variable-width text lines.

    Args:
        num_classes: alphabet size excluding the CTC blank

    The input contract matches :class:`CharacterBackbone`. ``lengths`` is a
    one-dimensional integer tensor of unpadded widths divided by four.
    Packed recurrence excludes padding from bidirectional context. Returns
    log probabilities of shape ``(W // 4, N, num_classes + 1)`` with blank at
    index zero. Ignore frames beyond each length during loss and decoding.
    The architecture is experimental; no pretrained weights are downloaded.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.backbone = CharacterBackbone()
        self.projection = nn.Sequential(nn.Linear(384, 128), nn.ReLU())
        self.context = nn.GRU(128, 96, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Linear(192, num_classes + 1)

    def forward(self, images: Tensor, lengths: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        features = self.backbone(images).permute(0, 3, 1, 2).flatten(2)
        _validate_lengths(lengths, images.shape[0], features.shape[1])
        packed = pack_padded_sequence(self.projection(features), lengths.cpu(), batch_first=True, enforce_sorted=False)
        context, _ = self.context(packed)
        context, _ = pad_packed_sequence(context, batch_first=True, total_length=features.shape[1])
        return self.head(context).log_softmax(-1).transpose(0, 1)


def _validate_lengths(lengths: Tensor, batch_size: int, time_steps: int) -> None:
    if lengths.ndim != 1 or lengths.shape[0] != batch_size or lengths.dtype not in {torch.int32, torch.int64}:
        raise ValueError("lengths must be a one-dimensional integer tensor with one value per image")
    if (lengths <= 0).any() or (lengths > time_steps).any():
        raise ValueError("lengths must lie between one and the padded width divided by four")
