# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import SqueezeExcitation

from holocron.nn import GlobalAvgPool2d

from ..checkpoints import Checkpoint, _handle_legacy_pretrained
from ..utils import _configure_model, fuse_conv_bn

__all__ = ["RepViT", "repvit_m0_9", "repvit_m1_0", "repvit_m1_1"]


def _make_divisible(value: float, divisor: int = 8) -> int:
    # Match timm's SE reduction with round_limit=0 in the official implementation.
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


class _ConvNorm(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bn_weight_init: float = 1.0,
    ) -> None:
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        norm = nn.BatchNorm2d(out_channels)
        super().__init__(conv, norm)

        if norm.weight is not None:
            nn.init.constant_(norm.weight, bn_weight_init)
        if norm.bias is not None:
            nn.init.zeros_(norm.bias)

    @torch.no_grad()
    def reparametrize(self) -> nn.Conv2d:
        conv, norm = cast(nn.Conv2d, self[0]), cast(nn.BatchNorm2d, self[1])
        kernel, bias = fuse_conv_bn(conv, norm)
        fused = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bias=True,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        fused.weight.copy_(kernel)
        cast(Tensor, fused.bias).copy_(bias)
        return fused.train(self.training)


class _BatchNormLinear(nn.Sequential):
    def __init__(self, in_features: int, out_features: int) -> None:
        norm = nn.BatchNorm1d(in_features)
        linear = nn.Linear(in_features, out_features)
        super().__init__(norm, linear)

        nn.init.trunc_normal_(linear.weight, std=0.02)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)

    @torch.no_grad()
    def reparametrize(self) -> nn.Linear:
        norm, linear = cast(nn.BatchNorm1d, self[0]), cast(nn.Linear, self[1])
        norm_weight, norm_bias = cast(Tensor, norm.weight), cast(Tensor, norm.bias)
        running_mean, running_var = cast(Tensor, norm.running_mean), cast(Tensor, norm.running_var)
        scale = norm_weight / torch.sqrt(running_var + norm.eps)
        shift = norm_bias - running_mean * scale
        fused = nn.Linear(
            linear.in_features,
            linear.out_features,
            bias=True,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        fused.weight.copy_(linear.weight * scale.unsqueeze(0))
        fused_bias = cast(Tensor, fused.bias)
        fused_bias.copy_(linear.weight @ shift)
        if linear.bias is not None:
            fused_bias.add_(linear.bias)
        return fused.train(self.training)


class _Residual(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, x: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        return x + self.block(x)


class _RepVGGDW(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv3 = _ConvNorm(channels, channels, 3, padding=1, groups=channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, groups=channels)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        return self.norm(self.conv3(x) + self.conv1(x) + x)

    @torch.no_grad()
    def reparametrize(self) -> nn.Conv2d:
        conv3 = self.conv3.reparametrize()
        conv3_bias, conv1_bias = cast(Tensor, conv3.bias), cast(Tensor, self.conv1.bias)

        conv3.weight.add_(F.pad(self.conv1.weight, [1, 1, 1, 1]))
        conv3.weight.add_(F.pad(torch.ones_like(self.conv1.weight), [1, 1, 1, 1]))
        conv3_bias.add_(conv1_bias)

        norm_weight, norm_bias = cast(Tensor, self.norm.weight), cast(Tensor, self.norm.bias)
        running_mean, running_var = cast(Tensor, self.norm.running_mean), cast(Tensor, self.norm.running_var)
        scale = norm_weight / torch.sqrt(running_var + self.norm.eps)
        conv3.weight.mul_(scale.view(-1, 1, 1, 1))
        conv3_bias.sub_(running_mean).mul_(scale).add_(norm_bias)
        return conv3.train(self.training)


class _RepViTBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, use_se: bool) -> None:
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError("`stride` is expected to be 1 or 2")
        if stride == 1 and in_channels != out_channels:
            raise ValueError("stride-1 blocks require matching input and output channels")

        attention = SqueezeExcitation(in_channels, _make_divisible(in_channels / 4)) if use_se else nn.Identity()
        if stride == 1:
            self.token_mixer = nn.Sequential(_RepVGGDW(in_channels), attention)
        else:
            self.token_mixer = nn.Sequential(
                _ConvNorm(in_channels, in_channels, 3, stride=2, padding=1, groups=in_channels),
                attention,
                _ConvNorm(in_channels, out_channels),
            )

        self.channel_mixer = _Residual(
            nn.Sequential(
                _ConvNorm(out_channels, 2 * out_channels),
                nn.GELU(),
                _ConvNorm(2 * out_channels, out_channels, bn_weight_init=0),
            )
        )

    def forward(self, x: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        return self.channel_mixer(self.token_mixer(x))

    def reparametrize(self) -> None:
        token_mixer = self.token_mixer
        first = token_mixer[0]
        if isinstance(first, (_ConvNorm, _RepVGGDW)):
            token_mixer[0] = first.reparametrize()
        if isinstance(token_mixer[-1], _ConvNorm):
            token_mixer[-1] = token_mixer[-1].reparametrize()

        channel_mixer = cast(nn.Sequential, self.channel_mixer.block)
        channel_mixer[0] = cast(_ConvNorm, channel_mixer[0]).reparametrize()
        channel_mixer[-1] = cast(_ConvNorm, channel_mixer[-1]).reparametrize()


class RepViT(nn.Sequential):
    """Implements RepViT as described in
    ["RepViT: Revisiting Mobile CNN From ViT Perspective"](https://arxiv.org/abs/2307.09283).

    Args:
        channels: number of output channels in each stage
        num_blocks: number of blocks in each stage
        num_classes: number of output classes
        in_channels: number of input channels
    """

    def __init__(
        self,
        channels: list[int],
        num_blocks: list[int],
        num_classes: int = 10,
        in_channels: int = 3,
    ) -> None:
        if len(channels) != 4 or len(num_blocks) != 4:
            raise ValueError("`channels` and `num_blocks` are expected to contain four stages")

        patch_embed = nn.Sequential(
            _ConvNorm(in_channels, channels[0] // 2, 3, stride=2, padding=1),
            nn.GELU(),
            _ConvNorm(channels[0] // 2, channels[0], 3, stride=2, padding=1),
        )
        stages: list[nn.Sequential] = []
        in_planes = channels[0]
        for stage_idx, (out_planes, depth) in enumerate(zip(channels, num_blocks, strict=True)):
            blocks: list[nn.Module] = []
            for block_idx in range(depth):
                stride = 2 if stage_idx > 0 and block_idx == 0 else 1
                # Official configs: SE on the first block, then alternating blocks except stage ends.
                use_se = block_idx == 0 if stage_idx == 0 else block_idx % 2 == 1 and block_idx < depth - 1
                blocks.append(_RepViTBlock(in_planes, out_planes, stride, use_se))
                in_planes = out_planes
            stages.append(nn.Sequential(*blocks))

        super().__init__(
            OrderedDict([
                ("features", nn.Sequential(patch_embed, *stages)),
                ("pool", GlobalAvgPool2d(flatten=True)),
                ("head", _BatchNormLinear(channels[-1], num_classes)),
            ])
        )

    def load_official_state_dict(self, state_dict: Mapping[str, Tensor], *, include_head: bool = True) -> None:
        """Import an unfused checkpoint from the authors' RepViT implementation.

        Pass the checkpoint's ``model`` entry, loaded with ``weights_only=True``.
        Distilled checkpoints have two classifiers. Their evaluation outputs are
        averaged into one classifier here. This preserves evaluation predictions,
        but does not reproduce the authors' two-head distillation training recipe.
        Set ``include_head=False`` to keep a new classifier for a different task.

        Args:
            state_dict: unfused state dictionary from ``THU-MIG/RepViT``
            include_head: whether to import the ImageNet classifier as well as the backbone

        Raises:
            ValueError: if the model is fused or the checkpoint keys or shapes do not match
        """
        if not isinstance(self.head, _BatchNormLinear):
            raise ValueError("load the official checkpoint before reparametrizing RepViT")  # noqa: TRY004

        block_locations = [
            f"features.{stage_idx}.{block_idx}"
            for stage_idx, stage in enumerate(self.features[1:], 1)
            for block_idx in range(len(stage))
        ]
        converted: dict[str, Tensor] = {}
        classifier_state: dict[str, Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("classifier."):
                classifier_state[key] = value
                continue
            parts = key.split(".", 2)
            if len(parts) != 3 or parts[0] != "features" or not parts[1].isdigit():
                raise ValueError(f"unexpected official checkpoint key: {key}")
            block_idx = int(parts[1])
            if block_idx > len(block_locations):
                raise ValueError("official checkpoint has a different number of RepViT blocks")
            prefix = "features.0" if block_idx == 0 else block_locations[block_idx - 1]
            suffix = parts[2]
            suffix = suffix.replace("token_mixer.0.conv.", "token_mixer.0.conv3.")
            suffix = suffix.replace("token_mixer.0.bn.", "token_mixer.0.norm.")
            suffix = suffix.replace("channel_mixer.m.", "channel_mixer.block.")
            suffix = suffix.replace(".c.", ".0.").replace(".bn.", ".1.")
            converted[f"{prefix}.{suffix}"] = value

        # Determine which token mixers are downsampling Conv-BN blocks.
        for prefix in block_locations:
            norm_prefix = f"{prefix}.token_mixer.0.norm."
            if f"{prefix}.token_mixer.0.0.weight" in converted:
                for key in tuple(converted):
                    if key.startswith(norm_prefix):
                        converted[key.replace(norm_prefix, f"{prefix}.token_mixer.0.1.")] = converted.pop(key)

        own_state = self.state_dict()
        if include_head:
            converted.update(self._convert_official_classifier(classifier_state))
        else:
            converted.update({key: value for key, value in own_state.items() if key.startswith("head.")})
        if converted.keys() != own_state.keys() or any(
            converted[key].shape != value.shape for key, value in own_state.items() if key in converted
        ):
            raise ValueError("official checkpoint does not match this RepViT architecture or class count")
        self.load_state_dict(converted)

    def _convert_official_classifier(self, state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
        prefixes = ["classifier.classifier."]
        if any(key.startswith("classifier.classifier_dist.") for key in state_dict):
            prefixes.append("classifier.classifier_dist.")
        required = {
            f"{prefix}{key}"
            for prefix in prefixes
            for key in (
                "bn.weight",
                "bn.bias",
                "bn.running_mean",
                "bn.running_var",
                "bn.num_batches_tracked",
                "l.weight",
                "l.bias",
            )
        }
        if state_dict.keys() != required:
            raise ValueError("official checkpoint classifier is incomplete or has unexpected keys")
        head = cast(_BatchNormLinear, self.head)
        norm = cast(nn.BatchNorm1d, head[0])
        weights, biases = [], []
        for prefix in prefixes:
            scale = state_dict[f"{prefix}bn.weight"] / (state_dict[f"{prefix}bn.running_var"] + norm.eps).sqrt()
            shift = state_dict[f"{prefix}bn.bias"] - state_dict[f"{prefix}bn.running_mean"] * scale
            weight = state_dict[f"{prefix}l.weight"]
            weights.append(weight * scale.unsqueeze(0))
            biases.append(weight @ shift + state_dict[f"{prefix}l.bias"])
        # An identity BN keeps the standard trainable head and exactly represents
        # the average of independently normalized classifier logits in eval mode.
        return {
            "head.0.weight": torch.ones_like(state_dict[f"{prefixes[0]}bn.weight"]),
            "head.0.bias": torch.zeros_like(state_dict[f"{prefixes[0]}bn.bias"]),
            "head.0.running_mean": torch.zeros_like(state_dict[f"{prefixes[0]}bn.running_mean"]),
            "head.0.running_var": torch.full_like(state_dict[f"{prefixes[0]}bn.running_var"], 1 - norm.eps),
            "head.0.num_batches_tracked": torch.zeros_like(state_dict[f"{prefixes[0]}bn.num_batches_tracked"]),
            "head.1.weight": torch.stack(weights).mean(0),
            "head.1.bias": torch.stack(biases).mean(0),
        }

    def reparametrize(self) -> None:
        """Fuse training-time branches and batch-normalization layers in place for deployment.

        Call `eval()` before conversion. Repeated calls leave the deployment model unchanged.

        Raises:
            ValueError: if the model or any batch-normalization layer is in training mode
        """
        if self.training or any(
            mod.training for mod in self.modules() if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d))
        ):
            raise ValueError("call eval() before reparametrizing RepViT")
        self.features: nn.Sequential
        patch_embed = cast(nn.Sequential, self.features[0])
        if not isinstance(patch_embed[0], _ConvNorm):
            return
        patch_embed[0] = cast(_ConvNorm, patch_embed[0]).reparametrize()
        patch_embed[-1] = cast(_ConvNorm, patch_embed[-1]).reparametrize()
        for stage in self.features[1:]:
            for block in cast(nn.Sequential, stage):
                cast(_RepViTBlock, block).reparametrize()
        self.head = cast(_BatchNormLinear, self.head).reparametrize()


def _repvit(
    checkpoint: Checkpoint | None,
    progress: bool,
    channels: list[int],
    num_blocks: list[int],
    **kwargs: Any,
) -> RepViT:
    model = RepViT(channels, num_blocks, **kwargs)
    return _configure_model(model, checkpoint, progress=progress)


def repvit_m0_9(
    pretrained: bool = False,
    checkpoint: Checkpoint | None = None,
    progress: bool = True,
    **kwargs: Any,
) -> RepViT:
    """RepViT-M0.9 model.

    Args:
        pretrained: If True, loads the default checkpoint when one is available
        checkpoint: If specified, sets the model parameters to the checkpoint values
        progress: If True, displays a download progress bar
        kwargs: keyword arguments of [`RepViT`][holocron.models.classification.repvit.RepViT]

    Returns:
        A RepViT-M0.9 model
    """
    checkpoint = _handle_legacy_pretrained(pretrained, checkpoint, None)
    return _repvit(checkpoint, progress, [48, 96, 192, 384], [3, 4, 16, 3], **kwargs)


def repvit_m1_0(
    pretrained: bool = False,
    checkpoint: Checkpoint | None = None,
    progress: bool = True,
    **kwargs: Any,
) -> RepViT:
    """RepViT-M1.0 model.

    Args:
        pretrained: If True, loads the default checkpoint when one is available
        checkpoint: If specified, sets the model parameters to the checkpoint values
        progress: If True, displays a download progress bar
        kwargs: keyword arguments of [`RepViT`][holocron.models.classification.repvit.RepViT]

    Returns:
        A RepViT-M1.0 model
    """
    checkpoint = _handle_legacy_pretrained(pretrained, checkpoint, None)
    return _repvit(checkpoint, progress, [56, 112, 224, 448], [3, 4, 16, 3], **kwargs)


def repvit_m1_1(
    pretrained: bool = False,
    checkpoint: Checkpoint | None = None,
    progress: bool = True,
    **kwargs: Any,
) -> RepViT:
    """RepViT-M1.1 model.

    Args:
        pretrained: If True, loads the default checkpoint when one is available
        checkpoint: If specified, sets the model parameters to the checkpoint values
        progress: If True, displays a download progress bar
        kwargs: keyword arguments of [`RepViT`][holocron.models.classification.repvit.RepViT]

    Returns:
        A RepViT-M1.1 model
    """
    checkpoint = _handle_legacy_pretrained(pretrained, checkpoint, None)
    return _repvit(checkpoint, progress, [64, 128, 256, 512], [3, 4, 14, 3], **kwargs)
