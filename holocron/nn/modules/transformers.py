# Copyright (C) 2023, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import math
from typing import Any, Callable, Optional, Tuple

import torch
from torch import nn

__all__ = ["TransformerEncoderBlock"]


def scaled_dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Scaled Dot-Product Attention from `"Attention is All You Need" <https://arxiv.org/pdf/1706.03762.pdf>`_.

    ..math::
        Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V

    Args:
        query (torch.Tensor): query tensor
        key (torch.Tensor): key tensor
        value (torch.Tensor): value tensor
        mask (torch.Tensor): optional mask
    """

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    if mask is not None:
        # NOTE: to ensure the ONNX compatibility, masked_fill works only with int equal condition
        scores = scores.masked_fill(mask == 0, float("-inf"))
    p_attn = torch.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer from `"Attention is All You Need" <https://arxiv.org/pdf/1706.03762.pdf>`_.

    Args:
        num_heads (int): number of attention heads
        d_model (int): dimension of the model
        dropout (float): dropout rate
    """

    def __init__(self, num_heads: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None) -> torch.Tensor:
        batch_size = query.size(0)

        # linear projections of Q, K, V
        query, key, value = [
            linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linear_layers, (query, key, value))
        ]

        # apply attention on all the projected vectors in batch
        x, attn = scaled_dot_product_attention(query, key, value, mask=mask)

        # Concat attention heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.output_linear(x)


class PositionwiseFeedForward(nn.Sequential):
    """Position-wise Feed-Forward Network from `"Attention is All You Need" <https://arxiv.org/pdf/1706.03762.pdf>`_.

    Args:
        d_model (int): dimension of the model
        ffd (int): hidden dimension of the feedforward network
        dropout (float): dropout rate
        activation_fct (nn.Module): activation function
    """

    def __init__(
        self, d_model: int, ffd: int, dropout: float = 0.1, activation_fct: Callable[[Any], Any] = nn.ReLU()
    ) -> None:
        super().__init__(  # type: ignore[call-overload]
            nn.Linear(d_model, ffd),
            activation_fct,
            nn.Dropout(p=dropout),
            nn.Linear(ffd, d_model),
            nn.Dropout(p=dropout),
        )


class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder Block

    Args:
        num_layers (int): number of layers
        num_heads (int): number of attention heads
        d_model (int): dimension of the model
        dff (int): hidden dimension of the feedforward network
        dropout (float): dropout rate
        activation_fct (nn.Module): activation function
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        dff: int,
        dropout: float,
        activation_fct: Callable[[Any], Any] = nn.ReLU(),
    ) -> None:
        super().__init__()

        self.num_layers = num_layers

        self.layer_norm_input = nn.LayerNorm(d_model, eps=1e-5)
        self.layer_norm_attention = nn.LayerNorm(d_model, eps=1e-5)
        self.layer_norm_output = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout = nn.Dropout(dropout)

        self.attention = nn.ModuleList(
            [MultiHeadAttention(num_heads, d_model, dropout) for _ in range(self.num_layers)]
        )
        self.position_feed_forward = nn.ModuleList(
            [PositionwiseFeedForward(d_model, dff, dropout, activation_fct) for _ in range(self.num_layers)]
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        output = x

        for i in range(self.num_layers):
            normed_output = self.layer_norm_input(output)
            output = output + self.dropout(self.attention[i](normed_output, normed_output, normed_output, mask))
            normed_output = self.layer_norm_attention(output)
            output = output + self.dropout(self.position_feed_forward[i](normed_output))

        # (batch_size, seq_len, d_model)
        return self.layer_norm_output(output)
