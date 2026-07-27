"""
Stage-0 encoder, ViG (Vision GNN) path -- stretch/experimental swap-in.

Patches are graph nodes; instead of dense self-attention, each block builds
a k-NN graph over patch embeddings (in feature space, rebuilt every block/
forward pass) and aggregates neighbor information with Max-Relative graph
convolution:

    x_i' = Linear( concat( x_i, max_{j in N(i)} (x_j - x_i) ) )

No off-the-shelf ultrasound-domain ViG implementation exists to adapt from,
so this is written from the published Vision GNN formulation directly --
treat it as the higher-risk path relative to encoders/vit.py, and keep the
two behind the same build_encoder() factory so falling back to ViT is a
one-line config change, not a code change.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .base import VisionEncoder


def _knn_graph(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    x: (B, N, D) patch embeddings.
    Returns idx: (B, N, k) -- for each node, indices of its k nearest
    neighbors in feature space, excluding itself.
    """
    B, N, _D = x.shape
    k = min(k, N - 1)  # can't have more neighbors than other nodes exist
    dist = torch.cdist(x, x)  # (B, N, N)
    diag_mask = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
    dist = dist.masked_fill(diag_mask, float("inf"))
    idx = dist.topk(k, largest=False).indices  # (B, N, k)
    return idx


class MRGraphConv(nn.Module):
    """Max-Relative graph convolution block (see module docstring for the math)."""

    def __init__(self, dim: int, k: int):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(2 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape
        idx = _knn_graph(x, self.k)  # (B, N, k)
        k_eff = idx.shape[-1]

        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, k_eff)
        x_j = x[batch_idx, idx]                       # (B, N, k_eff, D) -- neighbor feats
        x_i = x.unsqueeze(2).expand(B, N, k_eff, D)     # (B, N, k_eff, D) -- self, broadcast

        max_relative = (x_j - x_i).max(dim=2).values     # (B, N, D)
        out = self.proj(torch.cat([x, max_relative], dim=-1))  # (B, N, D)
        return out


class ViGBlock(nn.Module):
    def __init__(self, dim: int, k: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.graph_conv = MRGraphConv(dim, k)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.graph_conv(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ViGEncoder(VisionEncoder):
    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 128,
        depth: int = 4,
        k: int = 9,
    ):
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by "
            f"patch_size ({patch_size})"
        )
        self.out_dim = embed_dim
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [ViGBlock(embed_dim, k=k) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, C, H, W)
        x = self.patch_embed(frames)          # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)        # (B, N, D)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x.mean(dim=1)                      # (B, D) -- global avg pool, no cls token
