"""
Tiny synthetic data generators -- shape/dtype-matched stand-ins for real
ultrasound frames, used only for smoke tests. No real data, no download,
CPU-only, runs in well under a second. Keep these signatures aligned with
the real Dataset classes in data/datasets.py once those exist, so a smoke
test failure reflects a real interface mismatch and not a fixture drift.
"""
from __future__ import annotations
import torch


def make_synthetic_frames(
    batch_size: int = 4,
    channels: int = 1,
    height: int = 128,
    width: int = 128,
    seed: int = 0,
) -> torch.Tensor:
    """
    Fake B-mode-ultrasound-shaped frame batch, normalized to roughly [0, 1]
    with the kind of speckle-ish spatial structure real B-mode has (not
    pure white noise) -- enough to exercise conv/patch-embed shapes and
    catch anything that assumes RGB channels or a fixed resolution.
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(
        (batch_size, channels, height, width), generator=g
    )
    # cheap low-pass to avoid pure iid noise (closer to real speckle texture,
    # though this is *not* a physically accurate ultrasound simulator)
    kernel = torch.ones((channels, 1, 5, 5)) / 25.0
    smoothed = torch.nn.functional.conv2d(
        base, kernel, padding=2, groups=channels
    )
    return smoothed.clamp(0.0, 1.0)


def make_synthetic_pose_pair(batch_size: int = 4, seed: int = 0):
    """
    Fake ground-truth rigid transform between two consecutive frames:
    translation (B, 3) + quaternion (B, 4), unit-normalized.
    """
    g = torch.Generator().manual_seed(seed)
    t = torch.randn((batch_size, 3), generator=g) * 0.01  # small motion, mm-scale
    q = torch.randn((batch_size, 4), generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    return t, q


def make_synthetic_sweep(
    num_frames: int = 4,
    batch_size: int = 1,
    channels: int = 1,
    height: int = 128,
    width: int = 128,
    seed: int = 0,
):
    """
    Fake ultrasound sweep: multiple frames with incremental pose changes.

    Args:
        num_frames: Number of frames in the sweep
        batch_size: Batch size (usually 1 for sweeps)
        channels, height, width: Frame dimensions
        seed: Random seed

    Returns:
        frames: (B, N, C, H, W) tensor of frames
        poses: (B, N, 7) tensor of [tx, ty, tz, qw, qx, qy, qz]
    """
    g = torch.Generator().manual_seed(seed)
    g_state = g.get_state()

    # Generate base frames
    g.set_state(g_state)
    base = torch.rand(
        (batch_size, num_frames, channels, height, width), generator=g
    )

    # Apply low-pass filtering for each frame
    kernel = torch.ones((channels, 1, 5, 5)) / 25.0
    frames_list = []
    for i in range(num_frames):
        frame = base[:, i]
        smoothed = torch.nn.functional.conv2d(
            frame, kernel, padding=2, groups=channels
        )
        frames_list.append(smoothed.clamp(0.0, 1.0))

    frames = torch.stack(frames_list, dim=1)  # (B, N, C, H, W)

    # Generate incremental poses (small motions between frames)
    g.set_state(g_state)
    translations = torch.randn((batch_size, num_frames, 3), generator=g) * 0.005
    quaternions = torch.randn((batch_size, num_frames, 4), generator=g)
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)

    poses = torch.cat([translations, quaternions], dim=-1)  # (B, N, 7)

    return frames, poses
