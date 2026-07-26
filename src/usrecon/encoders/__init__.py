from usrecon.encoders.base import VisionEncoder
from usrecon.encoders.vit import ViTEncoder
from usrecon.encoders.vig import ViGEncoder


def build_encoder(cfg: dict) -> VisionEncoder:
    """
    Single factory so Stage 1+ code never imports ViTEncoder/ViGEncoder
    directly -- swapping encoders is a config change, not a code change.

    cfg example:
        {"type": "vit", "image_size": 128, "patch_size": 16,
         "embed_dim": 128, "depth": 4, "num_heads": 4}
        {"type": "vig", "image_size": 128, "patch_size": 16,
         "embed_dim": 128, "depth": 4, "k": 9}
    """
    etype = cfg["type"].lower()
    if etype == "vit":
        return ViTEncoder(
            image_size=cfg.get("image_size", 128),
            patch_size=cfg.get("patch_size", 16),
            in_chans=cfg.get("in_chans", 1),
            embed_dim=cfg.get("embed_dim", 128),
            depth=cfg.get("depth", 4),
            num_heads=cfg.get("num_heads", 4),
        )
    if etype == "vig":
        return ViGEncoder(
            image_size=cfg.get("image_size", 128),
            patch_size=cfg.get("patch_size", 16),
            in_chans=cfg.get("in_chans", 1),
            embed_dim=cfg.get("embed_dim", 128),
            depth=cfg.get("depth", 4),
            k=cfg.get("k", 9),
        )
    raise ValueError(f"Unknown encoder type: {cfg['type']!r} (expected 'vit' or 'vig')")
