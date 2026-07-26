"""Shared synthetic fixtures for all stage smoke tests."""
import pytest


@pytest.fixture
def synthetic_frames():
    from usrecon.data.synthetic import make_synthetic_frames
    return make_synthetic_frames(batch_size=4, channels=1, height=128, width=128, seed=0)


@pytest.fixture
def base_encoder_cfg():
    return {
        "image_size": 128,
        "patch_size": 16,
        "in_chans": 1,
        "embed_dim": 128,
        "depth": 2,   # shallow on purpose -- smoke tests check correctness, not capacity
        "num_heads": 4,
        "k": 9,
    }
