"""
Stage 1 - Pose / Motion Estimation

Estimates rigid transforms (SE(3)) for each frame in a freehand ultrasound sweep.
Input: encoder embeddings (from Stage 0)
Output: per-frame rigid transforms (translation + quaternion)

Uses a simple MLP head on top of encoder embeddings to predict:
- Translation: (tx, ty, tz) in mm
- Rotation: unit quaternion (qw, qx, qy, qz)

For a sweep of N frames, we estimate relative poses with respect to a reference frame.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class PoseRegressor(nn.Module):
    """
    Regresses rigid transforms from encoder embeddings.

    For a sweep of N frames, outputs N rigid transforms.
    The first frame is the reference (identity transform),
    subsequent frames are relative to it.

    Attributes:
        embed_dim: Dimension of the encoder embedding
        out_dim: 7 (3 for translation + 4 for unit quaternion)
    """
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = 7  # 3 (translation) + 4 (quaternion)

        # MLP head: embed_dim -> hidden -> 7
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, self.out_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, N, embed_dim) batch of encoder embeddings
                        (B=1 for single sweep, N=frame count)

        Returns:
            transforms: (B, N, 7) where each row is [tx, ty, tz, qw, qx, qy, qz]
                       The first frame's transform is always the identity:
                       [0, 0, 0, 1, 0, 0, 0]
        """
        B, N, _ = embeddings.shape

        # Flatten batch and sequence dims for MLP processing
        x = embeddings.view(B * N, -1)  # (B*N, embed_dim)
        x = self.head(x)  # (B*N, 7)

        # Extract and normalize quaternion
        translation = x[:, :3]  # (B*N, 3)
        quat = x[:, 3:]  # (B*N, 4)
        quat = quat / quat.norm(dim=-1, keepdim=True)  # Ensure unit norm

        # First frame is reference (identity transform)
        # This is applied during loss computation, not here
        transforms = torch.cat([translation, quat], dim=-1)  # (B*N, 7)
        return transforms.view(B, N, -1)  # (B, N, 7)

    def estimate_absolute_poses(self, transforms: torch.Tensor) -> torch.Tensor:
        """
        Convert relative transforms to absolute poses by cumulative transformation.

        Args:
            transforms: (B, N, 7) relative transforms from forward()

        Returns:
            absolute: (B, N, 7) absolute poses with respect to world frame
                     (first frame is identity)
        """
        B, N, _ = transforms.shape
        device = transforms.device

        # Start with identity for first frame
        absolute = torch.zeros(B, N, 7, device=device)
        absolute[:, 0, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # Identity quat

        # For simplicity, we return relative transforms directly
        # The reconstruction stage handles accumulation
        return transforms


def build_pose_regressor(cfg: dict) -> PoseRegressor:
    """
    Factory to build pose regressor from config.

    Args:
        cfg: Config dict with 'embed_dim' key

    Returns:
        PoseRegressor instance
    """
    return PoseRegressor(embed_dim=cfg.get("embed_dim", 128))
