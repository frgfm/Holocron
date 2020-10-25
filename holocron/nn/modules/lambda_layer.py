import torch
from torch import nn, einsum
import torch.nn.functional as F
from typing import Optional

__all__ = ['LambdaLayer']


class LambdaLayer(nn.Module):
    """Lambda layer from `"LambdaNetworks: Modeling long-range interactions without attention"
    <https://openreview.net/pdf?id=xTJEN-ggl1b>`_. The implementation was adapted from `lucidrains'
    <https://github.com/lucidrains/lambda-networks/blob/main/lambda_networks/lambda_networks.py>`.

    Args:
        in_channels (int): input channels
        out_channels (int, optional): output channels
        dim_k (int): key dimension
        n (int, optional): number of input pixels
        r (int, optional): receptive field for relative positional encoding
        num_heads (int, optional): number of attention heads
        dim_u (int, optional): intra-depth dimension
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim_k: int,
        n: Optional[int] = None,
        r: Optional[int] = None,
        num_heads: int = 4,
        dim_u: int = 1
    ) -> None:
        super().__init__()
        self.u = dim_u
        self.num_heads = num_heads

        if out_channels % num_heads != 0:
            raise AssertionError('values dimension must be divisible by number of heads for multi-head query')
        dim_v = out_channels // num_heads

        # Project input and context to get queries, keys & values
        self.to_q = nn.Conv2d(in_channels, dim_k * num_heads, 1, bias=False)
        self.to_k = nn.Conv2d(in_channels, dim_k * dim_u, 1, bias=False)
        self.to_v = nn.Conv2d(in_channels, dim_v * dim_u, 1, bias=False)

        self.norm_q = nn.BatchNorm2d(dim_k * num_heads)
        self.norm_v = nn.BatchNorm2d(dim_v * dim_u)

        self.local_contexts = r is not None
        if r is not None:
            if r % 2 != 1:
                raise AssertionError('Receptive kernel size should be odd')
            self.padding = r // 2
            self.R = nn.Parameter(torch.randn(dim_k, dim_u, 1, r, r))
        else:
            if n is None:
                raise AssertionError('You must specify the total sequence length (h x w)')
            self.pos_emb = nn.Parameter(torch.randn(n, n, dim_k, dim_u))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # Project inputs & context to retrieve queries, keys and values
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Normalize queries & values
        q = self.norm_q(q)
        v = self.norm_v(v)

        # B x (num_heads * dim_k) * H * W -> B x num_heads x dim_k x (H * W)
        q = q.reshape(b, self.num_heads, -1, h * w)
        # B x (dim_k * dim_u) * H * W -> B x dim_u x dim_k x (H * W)
        k = k.reshape(b, -1, self.u, h * w).permute(0, 2, 1, 3)
        # B x (dim_v * dim_u) * H * W -> B x dim_u x dim_v x (H * W)
        v = v.reshape(b, -1, self.u, h * w).permute(0, 2, 1, 3)

        # Normalized keys
        k = k.softmax(dim=-1)

        # Content function
        λc = einsum('b u k m, b u v m -> b k v', k, v)
        Yc = einsum('b h k n, b k v -> b n h v', q, λc)

        # Position function
        if self.local_contexts:
            # B x dim_u x dim_v x (H * W) -> B x dim_u x dim_v x H x W
            v = v.reshape(b, self.u, v.shape[2], h, w)
            λp = F.conv3d(v, self.R, padding=(0, self.padding, self.padding))
            Yp = einsum('b h k n, b k v n -> b n h v', q, λp.flatten(3))
        else:
            λp = einsum('n m k u, b u v m -> b n k v', self.pos_emb, v)
            Yp = einsum('b h k n, b n k v -> b n h v', q, λp)

        Y = Yc + Yp
        # B x (H * W) x num_heads x dim_v -> B x (num_heads * dim_v) x H x W
        out = Y.permute(0, 2, 3, 1).reshape(b, self.num_heads * v.shape[2], h, w)
        return out
