# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import torch
from huggingface_hub import DryRunFileInfo, hf_hub_download
from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.transforms.functional import InterpolationMode

from holocron import models
from holocron.nn import BlurPool2d

from .checkpoints import Checkpoint, Dataset, Evaluation, LoadingMeta, Metric, PreProcessing, TrainingRecipe
from .presets import IMAGENET, IMAGENETTE

__all__ = ["conv_sequence", "fuse_conv_bn", "load_pretrained_params", "model_from_hf_hub"]

M = TypeVar("M", bound=nn.Module)
logger = logging.getLogger(__name__)


def conv_sequence(
    in_channels: int,
    out_channels: int,
    act_layer: nn.Module | None = None,
    norm_layer: Callable[[int], nn.Module] | None = None,
    drop_layer: Callable[..., nn.Module] | None = None,
    conv_layer: Callable[..., nn.Module] | None = None,
    bn_channels: int | None = None,
    attention_layer: Callable[[int], nn.Module] | None = None,
    blurpool: bool = False,
    **kwargs: Any,
) -> list[nn.Module]:
    """Builds a sequence of convolutional layers.

    from torch import nn
    from holocron.models.utils import conv_sequence
    layers = conv_sequence(3, 32, nn.ReLU(), norm_layer=nn.Batchnorm2d(32), kernel_size=3, blurpool=True)

    Args:
        in_channels: number of channels of the input tensor
        out_channels: number of convolutional filters
        act_layer: the non linearity layer to apply
        norm_layer: the normalization layer
        drop_layer: the dropout-like layer for regularization
        conv_layer: if specified, replaces `torch.nn.Conv2d` as the convolutional layer
        bn_channels: if specified, replaces the normalization channels
        attention_layer: the self attention layer
        blurpool: whether blur pooling should be applied
        kwargs: the keyword arguments of the conv_layer

    Returns:
        a list of layers
    """
    if conv_layer is None:
        conv_layer = nn.Conv2d
    if bn_channels is None:
        bn_channels = out_channels

    conv_stride = kwargs.get("stride", 1)
    if blurpool and conv_stride > 1:
        kwargs["stride"] = 1

    # Avoid bias if there is batch normalization
    kwargs["bias"] = kwargs.get("bias", norm_layer is None)

    conv_seq = [conv_layer(in_channels, out_channels, **kwargs)]

    if callable(norm_layer):
        conv_seq.append(norm_layer(bn_channels))
    if callable(act_layer):
        conv_seq.append(act_layer)
    if blurpool and conv_stride > 1:
        conv_seq.append(BlurPool2d(bn_channels, stride=conv_stride))
    if callable(attention_layer):
        conv_seq.append(attention_layer(bn_channels))
    if callable(drop_layer):
        conv_seq.append(drop_layer(inplace=True))

    return conv_seq


def load_pretrained_params(
    model: nn.Module,
    url: str | None = None,
    progress: bool = True,
    key_replacement: tuple[str, str] | None = None,
    key_filter: str | None = None,
) -> None:
    """Loads a checkpoint on a model given its URL.

    Args:
        model: PyTorch model
        url: the URL of the checkpoint to download
        progress: whether a progress br should be displayed when downloading the checkpoint
        key_replacement: a mapping to replace keys in the checkpoint before loading them
        key_filter: prefix of the checkpoint keys to be loaded
    """
    if url is None:
        logger.warning("Invalid model URL, using default initialization.")
    else:
        state_dict = load_state_dict_from_url(url, progress=progress, map_location="cpu", weights_only=True)
        if isinstance(key_filter, str):
            state_dict = {k: v for k, v in state_dict.items() if k.startswith(key_filter)}
        if isinstance(key_replacement, tuple):
            state_dict = {k.replace(*key_replacement): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse convolution and batch normalization layers into a convolution with bias.

    from torch import nn
    from holocron.utils import fuse_conv_bn
    fuse_conv_bn(nn.Conv2d(3, 32, 3), nn.Batchnorm2d(32))

    Args:
        conv: the convolutional layer
        bn: the batch normalization layer

    Returns:
        the fused kernel and bias of the new convolution

    Raises:
        AssertionError: if the number of output channels is not the same for both `conv` and `bn`
    """
    # Check compatibility of both layers
    if bn.bias.data.shape[0] != conv.weight.data.shape[0]:
        raise AssertionError("expected same number of output channels for both `conv` and `bn`")

    scale_factor = bn.weight.data / torch.sqrt(bn.running_var + bn.eps)  # type: ignore[operator, arg-type]

    # Compute the new bias
    fused_bias = bn.bias.data - scale_factor * bn.running_mean
    if conv.bias is not None:
        logger.warning("convolution layers placed before batch normalization should not have a bias.")
        fused_bias += scale_factor * conv.bias.data
    # Scale the kernel
    fused_kernel = scale_factor.view(-1, 1, 1, 1) * conv.weight.data

    return fused_kernel, fused_bias


def model_from_hf_hub(repo_id: str, **kwargs: Any) -> nn.Module:
    """Instantiate & load a pretrained model from HF hub.

    from holocron.models.utils import model_from_hf_hub
    model = model_from_hf_hub("frgfm/rexnet1_0x")

    Args:
        repo_id: HuggingFace model hub repo
        kwargs: kwargs of `hf_hub_download`

    Returns:
        Model loaded with the checkpoint
    """
    # Pin the config and checkpoint to the same immutable revision.
    requested_revision = kwargs.pop("revision", None)
    info = cast(
        DryRunFileInfo,
        hf_hub_download(repo_id, filename="config.json", revision=requested_revision, **{**kwargs, "dry_run": True}),
    )
    kwargs.pop("dry_run", None)

    # Get the config
    with Path(hf_hub_download(repo_id, filename="config.json", revision=info.commit_hash, **kwargs)).open("rb") as f:
        cfg = json.load(f)

    model = models.__dict__[cfg["arch"]](num_classes=len(cfg["classes"]), pretrained=False)
    # Patch the config
    if model.default_cfg is None:
        model.default_cfg = cfg
    elif isinstance(model.default_cfg, Checkpoint):
        model.default_cfg = _checkpoint_from_hub_config(cfg)
    else:
        model.default_cfg.update(cfg)

    # Load the checkpoint
    state_dict = torch.load(
        hf_hub_download(repo_id, filename="pytorch_model.bin", revision=info.commit_hash, **kwargs),
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    return model


def _configure_model(
    model: M,
    checkpoint: Checkpoint | None,
    **kwargs: Any,
) -> M:
    model.default_cfg = checkpoint  # type: ignore[assignment]
    # Load pretrained parameters
    if isinstance(checkpoint, Checkpoint):
        load_pretrained_params(model, checkpoint.meta.url, **kwargs)

    return model


def _checkpoint_from_hub_config(hub_config: dict[str, Any]) -> Checkpoint:
    return Checkpoint(
        evaluation=Evaluation(
            dataset=Dataset.IMAGENETTE,
            results={},
        ),
        meta=LoadingMeta(
            url="N/A", sha256="N/A", size=0, num_params=0, arch=hub_config["arch"], categories=hub_config["classes"]
        ),
        pre_processing=PreProcessing(
            input_shape=hub_config["input_shape"], mean=hub_config["mean"], std=hub_config["std"]
        ),
        recipe=TrainingRecipe(commit=None, script="references/classification/train.py", args=None),
    )


_IMAGENETTE_VAL_RESIZE = 232


def _imagenette_recipe(arch: str, batch_size: int = 64) -> str:
    """Return the fixed Imagenette training arguments for published v0.2.1 checkpoints.

    Returns:
        Historical command-line arguments, including accumulation for batches of 32.
    """
    batch_args = f"{batch_size} --grad-acc 2" if batch_size == 32 else str(batch_size)
    return (
        f"./imagenette2-320/ --arch {arch} --batch-size {batch_args} --mixup-alpha 0.2 --amp"
        " --device 0 --epochs 100 --lr 1e-3 --label-smoothing 0.1 --random-erase 0.1 --train-crop-size 176"
        f" --val-resize-size {_IMAGENETTE_VAL_RESIZE} --opt adamw --weight-decay 5e-2"
    )


def _checkpoint(
    arch: str,
    url: str,
    acc1: float,
    acc5: float,
    sha256: str,
    size: int,
    num_params: int,
    commit: str | None = None,
    train_args: str | None = None,
    dataset: Dataset = Dataset.IMAGENETTE,
) -> Checkpoint:
    preset = IMAGENETTE if dataset == Dataset.IMAGENETTE else IMAGENET
    return Checkpoint(
        evaluation=Evaluation(
            dataset=dataset,
            results={Metric.TOP1_ACC: acc1, Metric.TOP5_ACC: acc5},
        ),
        meta=LoadingMeta(
            url=url, sha256=sha256, size=size, num_params=num_params, arch=arch, categories=preset.classes
        ),
        # ImageNet-1K weights are timm ports, evaluated with bicubic resizing and crop_pct 0.875
        pre_processing=PreProcessing((3, 224, 224), preset.mean, preset.std, resize_size=_IMAGENETTE_VAL_RESIZE)
        if dataset == Dataset.IMAGENETTE
        else PreProcessing((3, 224, 224), preset.mean, preset.std, InterpolationMode.BICUBIC, resize_size=256),
        recipe=TrainingRecipe(commit=commit, script="references/classification/train.py", args=train_args),
    )
