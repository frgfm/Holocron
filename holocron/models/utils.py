# Copyright (C) 2019-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import json
import logging
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torch.hub import load_state_dict_from_url

from holocron import models
from holocron.nn import BlurPool2d

__all__ = ["conv_sequence", "load_pretrained_params", "fuse_conv_bn", "model_from_hf_hub"]


def conv_sequence(
    in_channels: int,
    out_channels: int,
    act_layer: Optional[nn.Module] = None,
    norm_layer: Optional[Callable[[int], nn.Module]] = None,
    drop_layer: Optional[Callable[..., nn.Module]] = None,
    conv_layer: Optional[Callable[..., nn.Module]] = None,
    bn_channels: Optional[int] = None,
    attention_layer: Optional[Callable[[int], nn.Module]] = None,
    blurpool: bool = False,
    **kwargs: Any,
) -> List[nn.Module]:

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
    url: Optional[str] = None,
    progress: bool = True,
    key_replacement: Optional[Tuple[str, str]] = None,
    key_filter: Optional[str] = None,
) -> None:

    if url is None:
        logging.warning("Invalid model URL, using default initialization.")
    else:
        state_dict = load_state_dict_from_url(url, progress=progress, map_location="cpu")  # type: ignore[arg-type]
        if isinstance(key_filter, str):
            state_dict = {k: v for k, v in state_dict.items() if k.startswith(key_filter)}
        if isinstance(key_replacement, tuple):
            state_dict = {k.replace(*key_replacement): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse convolution and batch normalization layers into a convolution with bias

    Args:
        conv: the convolutional layer
        bn: the batch normalization layer
    Returns:
        the fused kernel and bias of the new convolution
    """

    # Check compatibility of both layers
    if bn.bias.data.shape[0] != conv.weight.data.shape[0]:
        raise AssertionError("expected same number of output channels for both `conv` and `bn`")

    scale_factor = bn.weight.data / torch.sqrt(bn.running_var + bn.eps)  # type: ignore[operator, arg-type]

    # Compute the new bias
    fused_bias = bn.bias.data - scale_factor * bn.running_mean
    if conv.bias is not None:
        logging.warning("convolution layers placed before batch normalization should not have a bias.")
        fused_bias += scale_factor * conv.bias.data
    # Scale the kernel
    fused_kernel = scale_factor.view(-1, 1, 1, 1) * conv.weight.data

    return fused_kernel, fused_bias


def model_from_hf_hub(repo_id: str, **kwargs: Any) -> nn.Module:
    """Instantiate & load a pretrained model from HF hub.

    Args:
        repo_id: HuggingFace model hub repo
        kwargs: kwargs of `hf_hub_download`
    Returns:
        Model loaded with the checkpoint
    """

    # Get the config
    with open(hf_hub_download(repo_id, filename="config.json", **kwargs), "rb") as f:
        cfg = json.load(f)

    model = models.__dict__[cfg["arch"]](num_classes=len(cfg["classes"]), pretrained=False)
    # Patch the config
    model.default_cfg.update(cfg)

    # Load the checkpoint
    state_dict = torch.load(hf_hub_download(repo_id, filename="pytorch_model.bin", **kwargs), map_location="cpu")
    model.load_state_dict(state_dict)

    return model
