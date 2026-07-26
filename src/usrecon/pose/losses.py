"""
Pose estimation losses for Stage 1.

Implements the point-based distance loss described in the project docs:
    L_pose = Σ_k ‖ T_pred·p_k − T_gt·p_k ‖²

The loss is computed in closed form using quaternions for rotation.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def transform_point_cloud(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """
    Apply rigid transform to a point cloud.

    Args:
        points: (B, N_pts, 3) 3D point coordinates
        transform: (B, 7) or (B, N, 7) rigid transform [tx, ty, tz, qw, qx, qy, qz]

    Returns:
        transformed: (B, N_pts, 3) transformed point cloud
    """
    B = points.shape[0]

    # Handle batched transforms
    if transform.ndim == 2:
        transform = transform.unsqueeze(1)  # (B, 1, 7)
        transform = transform.expand(-1, points.shape[1], -1)  # (B, N_pts, 7)

    # Extract translation and quaternion
    translation = transform[..., :3]  # (B, N, 3)
    quat = transform[..., 3:]  # (B, N, 4)

    # Normalize quaternion
    quat = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)

    # Quaternion-based rotation: v' = q * v * q^-1
    # For unit quaternion, q^-1 = [qw, -qx, -qy, -qz]
    qw, qx, qy, qz = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # v × q (cross product)
    vx, vy, vz = points[..., 0], points[..., 1], points[..., 2]

    # q * v (quaternion multiplication)
    qv_w = -qx * vx - qy * vy - qz * vz
    qv_x = qw * vx + qy * vz - qz * vy
    qv_y = qw * vy + qz * vx - qx * vz
    qv_z = qw * vz + qx * vy - qy * vx

    # (q * v) * q^-1
    # q^-1 = [qw, -qx, -qy, -qz]
    res_x = qv_w * (-qx) + qv_x * qw + qv_y * (-qz) - qv_z * (-qy)
    res_y = qv_w * (-qy) - qv_x * (-qz) + qv_y * qw + qv_z * (-qx)
    res_z = qv_w * (-qz) + qv_x * (-qy) - qv_y * (-qx) + qv_z * qw

    rotated = torch.stack([res_x, res_y, res_z], dim=-1)

    # Apply translation
    transformed = rotated + translation

    return transformed


def pose_point_loss(
    pred_transforms: torch.Tensor,
    gt_transforms: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """
    Point-based distance loss for pose estimation.

    L_pose = Σ_k ‖ T_pred·p_k − T_gt·p_k ‖²

    Args:
        pred_transforms: (B, N, 7) predicted transforms
        gt_transforms: (B, N, 7) ground truth transforms
        points: (B, N_pts, 3) 3D points to transform

    Returns:
        loss: scalar loss value
    """
    pred_points = transform_point_cloud(points, pred_transforms)
    gt_points = transform_point_cloud(points, gt_transforms)

    # Mean squared error between transformed points
    loss = F.mse_loss(pred_points, gt_points)

    return loss


def pose_smoothness_loss(transforms: torch.Tensor) -> torch.Tensor:
    """
    Smoothness loss to encourage motion continuity between frames.

    L_smooth = Σ_i ‖T_i - T_{i-1}‖²

    Args:
        transforms: (B, N, 7) transforms

    Returns:
        loss: scalar smoothness penalty
    """
    if transforms.shape[1] < 2:
        return torch.tensor(0.0, device=transforms.device)

    # Compute differences between consecutive transforms
    diff = transforms[:, 1:] - transforms[:, :-1]  # (B, N-1, 7)

    # MSE on the difference
    loss = diff.pow(2).mean()

    return loss


def pose_loss(
    pred_transforms: torch.Tensor,
    gt_transforms: torch.Tensor,
    points: torch.Tensor,
    lambda_smooth: float = 0.01,
) -> dict:
    """
    Complete pose loss with regularization.

    L = L_pose + λ_smooth * L_smooth

    Args:
        pred_transforms: (B, N, 7) predicted transforms
        gt_transforms: (B, N, 7) ground truth transforms
        points: (B, N_pts, 3) 3D points for point-based loss
        lambda_smooth: weight for smoothness regularization

    Returns:
        dict with 'total', 'point_loss', and 'smoothness_loss'
    """
    point_loss = pose_point_loss(pred_transforms, gt_transforms, points)
    smoothness_loss = pose_smoothness_loss(pred_transforms)

    total = point_loss + lambda_smooth * smoothness_loss

    return {
        "total": total,
        "point_loss": point_loss,
        "smoothness_loss": smoothness_loss,
    }
