"""
Stage 2 - Point Cloud Compounding

Transforms 2D frames into a 3D point cloud using pose estimates from Stage 1.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def compound_point_cloud(
    frames: torch.Tensor,
    poses: torch.Tensor,
    pixel_spacing: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compounds 2D frames into a 3D point cloud using estimated poses.

    For each frame i, transform its pixel coordinates to world space:
        p_world = T_i · [u, v, 0]_homogeneous

    Args:
        frames: (B, N, H, W) tensor of B sweeps with N frames each
        poses: (B, N, 7) tensor of [tx, ty, tz, qw, qx, qy, qz] per frame
        pixel_spacing: Physical spacing between pixels in mm

    Returns:
        points: (B, N*H*W, 3) world-space coordinates
        intensities: (B, N*H*W) flattened intensities
    """
    B, N, H, W = frames.shape

    # Create pixel coordinate grid
    u = torch.arange(W, dtype=frames.dtype, device=frames.device)  # (W,)
    v = torch.arange(H, dtype=frames.dtype, device=frames.device)  # (H,)
    uu, vv = torch.meshgrid(u, v, indexing="xy")  # (H, W) each

    # Stack to (H*W, 3) homogeneous coordinates [u, v, 0]
    pixels = torch.stack([uu.flatten(), vv.flatten(), torch.zeros_like(uu.flatten())], dim=-1)  # (H*W, 3)
    pixels = pixels * pixel_spacing  # Convert to mm

    # Expand to batch: (B, N, H*W, 3)
    pixels = pixels.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)

    # Transform each pixel by the corresponding frame pose
    # For simplicity, we'll apply rotation only (no translation for now)
    # Translation is handled by pose accumulation in the reconstruction stage
    points = apply_rigid_transform(pixels, poses)
    points = points.view(B, N * H * W, 3)

    # Flatten intensities
    intensities = frames.view(B, N * H * W)

    return points, intensities


def apply_rigid_transform(points: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    """
    Apply rigid transform to point cloud.

    Args:
        points: (B, N, P, 3) point cloud
        poses: (B, N, 7) transforms [tx, ty, tz, qw, qx, qy, qz]

    Returns:
        transformed: (B, N, P, 3) transformed points
    """
    B, N, P, _ = points.shape

    # Normalize quaternion
    quat = poses[..., 3:]  # (B, N, 4)
    quat = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)

    qw = quat[..., 0:1]  # (B, N, 1)
    qx = quat[..., 1:2]
    qy = quat[..., 2:3]
    qz = quat[..., 3:4]

    # Rotation matrix from quaternion (B, N, 1)
    r00 = 1 - 2 * (qy.pow(2) + qz.pow(2))
    r01 = 2 * (qx * qy - qw * qz)
    r02 = 2 * (qx * qz + qw * qy)
    r10 = 2 * (qx * qy + qw * qz)
    r11 = 1 - 2 * (qx.pow(2) + qz.pow(2))
    r12 = 2 * (qy * qz - qw * qx)
    r20 = 2 * (qx * qz - qw * qy)
    r21 = 2 * (qy * qz + qw * qx)
    r22 = 1 - 2 * (qx.pow(2) + qy.pow(2))

    # Apply rotation
    x = points[..., 0:1]  # (B, N, P, 1)
    y = points[..., 1:2]
    z = points[..., 2:3]

    rx = r00.unsqueeze(-1) * x + r01.unsqueeze(-1) * y + r02.unsqueeze(-1) * z
    ry = r10.unsqueeze(-1) * x + r11.unsqueeze(-1) * y + r12.unsqueeze(-1) * z
    rz = r20.unsqueeze(-1) * x + r21.unsqueeze(-1) * y + r22.unsqueeze(-1) * z

    # Stack rotated points
    rotated = torch.cat([rx, ry, rz], dim=-1)  # (B, N, P, 3)

    # Add translation
    translation = poses[..., :3].unsqueeze(2)  # (B, N, 1, 3)
    transformed = rotated + translation

    return transformed


class PointCloudCompounder(nn.Module):
    """Module wrapper for point cloud compounding."""

    def __init__(self, pixel_spacing: float = 0.5):
        super().__init__()
        self.pixel_spacing = pixel_spacing

    def forward(self, frames: torch.Tensor, poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return compound_point_cloud(frames, poses, self.pixel_spacing)
