"""
Single CLI entrypoint for running one pipeline stage at a time.

    python -m usrecon.pipeline.run_stage --stage stage0_encoder --config src/usrecon/config/default.yaml

Each stage function is responsible for: loading its config, running the
work, saving required plots (see README.md Sec.7), and reporting through
`stage_run()` so a manifest is always written on both success and failure.

Stages are gated: each stage must pass smoke tests before proceeding to the
next (see CLAUDE.md Sec.5).
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

import yaml
import torch

from usrecon.utils.checkpoint import stage_run, save_checkpoint, load_checkpoint
from usrecon.utils.device import resolve_device
from usrecon.utils.seed import set_seed
from usrecon.utils.viz import plot_frame_grid, plot_loss_curve
from usrecon.encoders import build_encoder
from usrecon.data.synthetic import make_synthetic_frames, make_synthetic_pose_pair, make_synthetic_sweep
from usrecon.pose import build_pose_regressor, pose_loss

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_stage0_encoder(cfg: dict, use_synthetic: bool = True) -> None:
    set_seed(cfg.get("seed", 0))
    device_plan = resolve_device(**cfg["device"])
    logger.info("Device plan: %s", device_plan)

    with stage_run("stage0_encoder", config=cfg) as ctx:
        encoder = build_encoder(cfg["encoder"])

        if use_synthetic:
            # Dev/logic-check path: no real data required.
            frames = make_synthetic_frames(
                batch_size=cfg["training"]["batch_size"],
                channels=cfg["data"]["channels"],
                height=cfg["data"]["image_size"],
                width=cfg["data"]["image_size"],
                seed=cfg.get("seed", 0),
            )
        else:
            raise NotImplementedError(
                "Real-data loading goes through usrecon.data.datasets, "
                "not implemented yet -- run with use_synthetic=True until "
                "that stage is built."
            )

        embeddings = encoder(frames)
        ctx["output_shape"] = tuple(embeddings.shape)
        ctx["encoder_type"] = cfg["encoder"]["type"]

        fig_path = plot_frame_grid(frames.detach().cpu().numpy(), stage="stage0_encoder")
        ctx["sample_figure"] = str(fig_path)

        save_checkpoint("stage0_encoder", "encoder_init", encoder.state_dict())
        logger.info(
            "stage0_encoder OK: %s embeddings -> %s",
            cfg["encoder"]["type"], tuple(embeddings.shape),
        )


def run_stage1_pose(cfg: dict, use_synthetic: bool = True) -> None:
    set_seed(cfg.get("seed", 0))
    device_plan = resolve_device(**cfg["device"])
    logger.info("Device plan: %s", device_plan)

    with stage_run("stage1_pose", config=cfg) as ctx:
        # Load encoder from checkpoint (pretrained in stage0)
        encoder = build_encoder(cfg["encoder"])
        encoder_path = Path(cfg["training"].get("encoder_checkpoint", ""))
        if encoder_path.exists():
            encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        else:
            # If no checkpoint, use encoder from config (smoke test mode)
            logger.info("No encoder checkpoint found, using fresh encoder for smoke test")

        # Freeze encoder for pose estimation
        for param in encoder.parameters():
            param.requires_grad = False

        # Build pose regressor
        encoder_out_dim = cfg["encoder"]["embed_dim"]
        pose_regressor = build_pose_regressor({"embed_dim": encoder_out_dim})

        if use_synthetic:
            # Generate synthetic frames
            frames = make_synthetic_frames(
                batch_size=cfg["training"]["batch_size"],
                channels=cfg["data"]["channels"],
                height=cfg["data"]["image_size"],
                width=cfg["data"]["image_size"],
                seed=cfg.get("seed", 0),
            )
            B, C, H, W = frames.shape

            # Generate synthetic poses
            t_gt, q_gt = make_synthetic_pose_pair(B, seed=cfg.get("seed", 0) + 1)

            # Encode frames
            with torch.no_grad():
                embeddings = encoder(frames)

            # For synthetic test, treat each batch item as a single-frame sequence
            # embeddings: (B, D) -> (B, 1, D)
            if embeddings.ndim == 2:
                embeddings = embeddings.unsqueeze(1)  # (B, 1, D)

            # Predict poses
            pred_transforms = pose_regressor(embeddings)  # (B, 1, 7)

            # Convert GT to transform format
            gt_transforms = torch.cat([t_gt, q_gt], dim=-1).unsqueeze(1)  # (B, 1, 7)

            # Compute loss
            # Generate random points for point-based loss
            N_pts = cfg["training"].get("num_points", 64)
            points = torch.randn(B, N_pts, 3, device=frames.device) * 0.1

            losses = pose_loss(pred_transforms, gt_transforms, points)

            # Train one step for smoke test
            optimizer = torch.optim.Adam(pose_regressor.parameters(), lr=cfg["training"]["lr"])
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            ctx["loss_total"] = losses["total"].item()
            ctx["loss_point"] = losses["point_loss"].item()
            ctx["loss_smooth"] = losses["smoothness_loss"].item()
            ctx["pred_shape"] = tuple(pred_transforms.shape)
        else:
            raise NotImplementedError(
                "Real-data pose estimation not implemented yet."
            )

        # Save checkpoint and figure
        save_checkpoint("stage1_pose", "pose_regressor", pose_regressor.state_dict())
        ctx["output_shape"] = (B, 1, 7)  # (batch, frames, 7)

        logger.info(
            "stage1_pose OK: loss_total=%.4f, pred_shape=%s",
            ctx["loss_total"], tuple(ctx["output_shape"]),
        )


_STAGES = {
    "stage0_encoder": run_stage0_encoder,
    "stage1_pose": run_stage1_pose,
}


def main():
    parser = argparse.ArgumentParser(description="Run one usrecon pipeline stage.")
    parser.add_argument("--stage", required=True, choices=sorted(_STAGES.keys()))
    parser.add_argument("--config", default="src/usrecon/config/default.yaml")
    parser.add_argument(
        "--real-data", action="store_true",
        help="Use real data instead of synthetic (only valid once a stage supports it).",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    fn = _STAGES[args.stage]
    fn(cfg, use_synthetic=not args.real_data)


if __name__ == "__main__":
    main()
