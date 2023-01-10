# Copyright (C) 2023, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from enum import Enum
from typing import Any, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from holocron.nn.modules import TransformerEncoderBlock

from ..checkpoints import (
    Checkpoint,
    Dataset,
    Evaluation,
    LoadingMeta,
    Metric,
    PreProcessing,
    TrainingRecipe,
    _handle_legacy_pretrained,
)
from ..presets import IMAGENETTE
from ..utils import _configure_model

__all__ = [
    "ConvPatchEmbed",
    "CCT_2_Checkpoint",
    "cct_2",
    "cct_4",
    "cct_6",
    "cct_7",
]


class ConvPatchEmbed(nn.Module):
    """Compute 2D patch embeddings"""

    def __init__(
        self,
        kernel_size: int,
        stride: int,
        padding: int,
        pool_kernel_size: int = 2,
        pool_stride: int = 2,
        pool_padding: int = 1,
        conv_layers: int = 1,
        input_channels: int = 3,
        output_channels: int = 64,
        in_planes: int = 64,
        activation_fct: Callable[[Any], Any] = nn.ReLU(),
    ) -> None:

        super(ConvPatchEmbed, self).__init__()
        _inner_planes = [in_planes for _ in range(conv_layers - 1)]
        _planes = [input_channels] + _inner_planes + [output_channels]

        self.conv_layers = nn.Sequential(
            *[
                nn.Sequential(  # type: ignore[call-overload]
                    nn.Conv2d(
                        _planes[i],
                        _planes[i + 1],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    ),
                    activation_fct,
                    nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding),
                )
                for i in range(conv_layers)
            ]
        )

        self.apply(self.init_weight)

    def get_num_patches(self, channels: int = 3, img_size: int = 224) -> int:
        return self.forward(torch.zeros(1, channels, img_size, img_size)).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.conv_layers(x)
        return patches.flatten(2, 3).transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class CCT(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: int,
        num_layers: int,
        input_channels: int = 3,
        img_size: int = 224,
        conv_layers: int = 1,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 3,
        pool_kernel_size: int = 3,
        pool_stride: int = 2,
        pool_padding: int = 1,
        dropout: float = 0.1,
        num_classes: int = 10,
        **kwargs: Any,
    ) -> None:
        super(CCT, self).__init__()

        self.patches = ConvPatchEmbed(
            kernel_size,
            stride,
            padding,
            pool_kernel_size,
            pool_stride,
            pool_padding,
            conv_layers,
            input_channels,
            embedding_dim,
            **kwargs,
        )
        self.positions = nn.Parameter(
            torch.zeros(1, self.patches.get_num_patches(input_channels, img_size), embedding_dim), requires_grad=True
        )

        self.encoder = TransformerEncoderBlock(
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=embedding_dim,
            dff=embedding_dim * mlp_ratio,
            dropout=dropout,
            activation_fct=nn.GELU(),
        )
        self.attention_pool = nn.Linear(embedding_dim, 1)
        self.head = nn.Linear(embedding_dim, num_classes)

        # Init all layers
        self.apply(self.init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patches(x)
        patches += self.positions
        features = self.encoder(patches)
        avg_pool = self.attention_pool(features)
        return self.head(torch.matmul(F.softmax(avg_pool, dim=1).transpose(-1, -2), features).squeeze(-2))

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def _cct(
    checkpoint: Union[Checkpoint, None],
    progress: bool,
    **kwargs: Any,
) -> CCT:
    # Build the model
    model = CCT(**kwargs)
    return _configure_model(model, checkpoint, progress=progress)


def _checkpoint(
    arch: str, url: str, acc1: float, acc5: float, sha256: str, size: int, num_params: int, commit: str, train_args: str
) -> Checkpoint:
    return Checkpoint(
        evaluation=Evaluation(
            dataset=Dataset.IMAGENETTE,
            results={Metric.TOP1_ACC: acc1, Metric.TOP5_ACC: acc5},
        ),
        meta=LoadingMeta(
            url=url, sha256=sha256, size=size, num_params=num_params, arch=arch, categories=IMAGENETTE.classes
        ),
        pre_processing=PreProcessing(input_shape=(3, 224, 224), mean=IMAGENETTE.mean, std=IMAGENETTE.std),
        recipe=TrainingRecipe(commit=commit, script="references/classification/train.py", args=train_args),
    )


class CCT_2_Checkpoint(Enum):

    IMAGENETTE = _checkpoint(
        arch="cct_2",
        url="",
        acc1=0.8346,
        acc5=0.9855,
        sha256="",
        size=19797114,
        num_params=619659,
        commit="",
        train_args=(
            "./imagenette2-320/ --arch cct_2 --batch-size 16 --mixup-alpha 0.2 --amp --device 0 --epochs 100"
            " --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 224 --val-resize-size 232"
            " --opt adamw --weight-decay 5e-2"
        ),
    )
    DEFAULT = IMAGENETTE


def cct_2(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> CCT:
    """CCT-2 from
    `"Escaping the Big Data Paradigm with Compact Transformers" <https://arxiv.org/pdf/2104.05704.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model

    .. autoclass:: holocron.models.CCT_2_Checkpoint
        :members:
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        CCT_2_Checkpoint.DEFAULT,  # type: ignore[arg-type]
    )
    return _cct(checkpoint, progress, num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128, **kwargs)


def cct_4(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> CCT:
    """CCT-4 from
    `"Escaping the Big Data Paradigm with Compact Transformers" <https://arxiv.org/pdf/2104.05704.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        None,
    )
    return _cct(checkpoint, progress, num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128, **kwargs)


def cct_6(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> CCT:
    """CCT-6 from
    `"Escaping the Big Data Paradigm with Compact Transformers" <https://arxiv.org/pdf/2104.05704.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        None,
    )
    return _cct(checkpoint, progress, num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256, **kwargs)


def cct_7(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> CCT:
    """CCT-7 from
    `"Escaping the Big Data Paradigm with Compact Transformers" <https://arxiv.org/pdf/2104.05704.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        None,
    )
    return _cct(checkpoint, progress, num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256, **kwargs)


def cct_14(
    pretrained: bool = False,
    checkpoint: Union[Checkpoint, None] = None,
    progress: bool = True,
    **kwargs: Any,
) -> CCT:
    """CCT-14 from
    `"Escaping the Big Data Paradigm with Compact Transformers" <https://arxiv.org/pdf/2104.05704.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNette
        checkpoint: If specified, the model's parameters will be set to the checkpoint's values
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: classification model
    """
    checkpoint = _handle_legacy_pretrained(
        pretrained,
        checkpoint,
        None,
    )
    return _cct(checkpoint, progress, num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384, **kwargs)
