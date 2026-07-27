"""
Stage-1 smoke tests: pose regression shape, no-NaN, gradient flow.

Tests:
- Output shape correctness
- No NaN values in outputs/losses
- Gradient flows to pose regressor
- Loss values are reasonable
"""
import pytest
import torch

from usrecon.data.synthetic import make_synthetic_sweep
from usrecon.encoders import build_encoder
from usrecon.pose import build_pose_regressor, pose_loss


@pytest.fixture
def encoder_for_pose():
    """Small encoder for smoke testing pose regression."""
    return build_encoder({
        "type": "vit",
        "image_size": 64,  # smaller for faster tests
        "patch_size": 8,
        "in_chans": 1,
        "embed_dim": 64,
        "depth": 2,
        "num_heads": 2,
    })


@pytest.fixture
def pose_regressor_for_test():
    """Small pose regressor for smoke testing."""
    return build_pose_regressor({"embed_dim": 64})


def test_pose_regressor_output_shape(encoder_for_pose, pose_regressor_for_test):
    """Output should be (B, N, 7) for B batches of N frames."""
    # Encode synthetic frames
    frames, _ = make_synthetic_sweep(
        num_frames=4, batch_size=2, channels=1, height=64, width=64, seed=0
    )
    B, N, C, H, W = frames.shape

    encoder = encoder_for_pose
    encoder.eval()

    with torch.no_grad():
        # Process each frame and stack embeddings
        embeddings_list = []
        for i in range(N):
            frame = frames[:, i]  # (B, C, H, W)
            embed = encoder(frame)  # (B, D)
            embeddings_list.append(embed)
        embeddings = torch.stack(embeddings_list, dim=1)  # (B, N, D)

    # Predict poses
    regressor = pose_regressor_for_test
    pred = regressor(embeddings)

    assert pred.shape == (B, N, 7), (
        f"Expected {(B, N, 7)}, got {tuple(pred.shape)}"
    )


def test_pose_regressor_no_nans(encoder_for_pose, pose_regressor_for_test):
    """Output should have no NaN or inf values."""
    frames, _ = make_synthetic_sweep(num_frames=3, height=64, width=64, seed=0)
    B, N, C, H, W = frames.shape

    encoder = encoder_for_pose
    encoder.eval()

    with torch.no_grad():
        embeddings_list = [encoder(frames[:, i]) for i in range(N)]
        embeddings = torch.stack(embeddings_list, dim=1)

    regressor = pose_regressor_for_test
    pred = regressor(embeddings)

    assert torch.isfinite(pred).all(), "Pose output contains NaN or inf"


def test_pose_loss_computes_correctly(encoder_for_pose, pose_regressor_for_test):
    """Loss should be a valid scalar and gradients should flow."""
    frames, gt_poses = make_synthetic_sweep(num_frames=3, height=64, width=64, seed=0)
    B, N, C, H, W = frames.shape

    encoder = encoder_for_pose
    encoder.eval()

    with torch.no_grad():
        embeddings_list = [encoder(frames[:, i]) for i in range(N)]
        embeddings = torch.stack(embeddings_list, dim=1)

    regressor = pose_regressor_for_test
    pred = regressor(embeddings)

    # Generate points for loss
    N_pts = 32
    points = torch.randn(B, N_pts, 3) * 0.1

    losses = pose_loss(pred, gt_poses, points)

    # Check loss is scalar
    assert losses["total"].shape == torch.Size([]), "Loss should be scalar"

    # Check no NaN
    assert torch.isfinite(losses["total"]).all(), "Loss contains NaN"

    # Check gradient flows
    regressor.zero_grad()
    losses["total"].backward()

    has_grad = any(p.grad is not None and torch.isfinite(p.grad).any()
                   for p in regressor.parameters() if p.requires_grad)
    assert has_grad, "No gradient flowed to regressor parameters"


def test_pose_loss_values_reasonable(encoder_for_pose, pose_regressor_for_test):
    """Loss should start large and decrease with training."""
    frames, gt_poses = make_synthetic_sweep(num_frames=3, height=64, width=64, seed=0)
    B, N, C, H, W = frames.shape

    encoder = encoder_for_pose
    encoder.eval()

    with torch.no_grad():
        embeddings_list = [encoder(frames[:, i]) for i in range(N)]
        embeddings = torch.stack(embeddings_list, dim=1)

    regressor = pose_regressor_for_test

    # Initial loss
    N_pts = 32
    points = torch.randn(B, N_pts, 3) * 0.1

    optimizer = torch.optim.Adam(regressor.parameters(), lr=0.01)

    losses = []
    for _ in range(5):
        pred = regressor(embeddings)
        loss_dict = pose_loss(pred, gt_poses, points)
        losses.append(loss_dict["total"].item())

        optimizer.zero_grad()
        loss_dict["total"].backward()
        optimizer.step()

    # Loss should decrease (initial guess is random)
    assert losses[-1] < losses[0] * 1.5, (
        f"Loss should decrease: {losses}"
    )


def test_pose_regressor_trainable_params(encoder_for_pose, pose_regressor_for_test):
    """All params should be trainable."""
    regressor = pose_regressor_for_test

    num_trainable = sum(p.numel() for p in regressor.parameters() if p.requires_grad)
    assert num_trainable > 0, "No trainable parameters in pose regressor"
