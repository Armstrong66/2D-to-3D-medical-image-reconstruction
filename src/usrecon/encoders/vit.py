"""
Stage-0 encoder, ViT path (default/safety-net).

Uses nn.TransformerEncoderLayer for the attention blocks rather than a
hand-rolled implementation -- fewer places for a subtle bug to hide, which
matters here since this code is written and reviewed without the ability to
execute it against real tensors before handoff.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from usrecon.encoders.base import VisionEncoder


class ViTEncoder(VisionEncoder):
    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by "
            f"patch_size ({patch_size})"
        )
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.out_dim = embed_dim

        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm, matches standard ViT block ordering
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, C, H, W)
        B = frames.shape[0]
        x = self.patch_embed(frames)               # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)            # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)       # (B, 1, D)
        x = torch.cat([cls, x], dim=1)               # (B, N+1, D)
        x = x + self.pos_embed
        x = self.blocks(x)                            # (B, N+1, D)
        x = self.norm(x)
        return x[:, 0]                                  # (B, D) -- CLS token
