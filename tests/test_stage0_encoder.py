"""
Stage-0 smoke tests: shape correctness, no-NaN, and gradient flow, for both
the ViT and ViG encoder paths, run through the same build_encoder() factory
they'll actually be used through downstream.
"""
import pytest
import torch

from usrecon.encoders import build_encoder


@pytest.mark.parametrize("encoder_type", ["vit", "vig"])
def test_encoder_output_shape(encoder_type, synthetic_frames, base_encoder_cfg):
    cfg = {**base_encoder_cfg, "type": encoder_type}
    encoder = build_encoder(cfg)
    out = encoder(synthetic_frames)
    B = synthetic_frames.shape[0]
    assert out.shape == (B, cfg["embed_dim"]), (
        f"{encoder_type}: expected {(B, cfg['embed_dim'])}, got {tuple(out.shape)}"
    )


@pytest.mark.parametrize("encoder_type", ["vit", "vig"])
def test_encoder_output_has_no_nans(encoder_type, synthetic_frames, base_encoder_cfg):
    cfg = {**base_encoder_cfg, "type": encoder_type}
    encoder = build_encoder(cfg)
    out = encoder(synthetic_frames)
    assert torch.isfinite(out).all(), f"{encoder_type}: non-finite values in output"


@pytest.mark.parametrize("encoder_type", ["vit", "vig"])
def test_encoder_gradient_flow(encoder_type, synthetic_frames, base_encoder_cfg):
    cfg = {**base_encoder_cfg, "type": encoder_type}
    encoder = build_encoder(cfg)
    out = encoder(synthetic_frames)
    loss = out.pow(2).mean()
    loss.backward()
    missing = [
        name for name, p in encoder.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not missing, f"{encoder_type}: no gradient reached params: {missing}"


def test_encoders_share_interface(synthetic_frames, base_encoder_cfg):
    """Swapping vit<->vig via config alone must not change the output shape."""
    vit = build_encoder({**base_encoder_cfg, "type": "vit"})
    vig = build_encoder({**base_encoder_cfg, "type": "vig"})
    assert vit(synthetic_frames).shape == vig(synthetic_frames).shape


def test_unknown_encoder_type_raises(base_encoder_cfg):
    with pytest.raises(ValueError):
        build_encoder({**base_encoder_cfg, "type": "not_a_real_encoder"})
