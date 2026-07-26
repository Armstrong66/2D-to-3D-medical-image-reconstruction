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
import torch.nn.functional as F

from ..utils.checkpoint import stage_run, save_checkpoint, load_checkpoint
from ..utils.device import resolve_device
from ..utils.seed import set_seed
from ..utils.viz import plot_frame_grid, plot_loss_curve
from ..encoders import build_encoder
from ..data.synthetic import make_synthetic_frames, make_synthetic_pose_pair, make_synthetic_sweep
from ..pose import build_pose_regressor, pose_loss
from ..reconstruction import (
    compound_point_cloud,
    ImplicitFieldRegressor,
    build_positional_encoder,
    FOURIER,
)

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


def run_stage2_compounding(cfg: dict, use_synthetic: bool = True) -> None:
    set_seed(cfg.get("seed", 0))
    device_plan = resolve_device(**cfg["device"])
    logger.info("Device plan: %s", device_plan)

    with stage_run("stage2_compounding", config=cfg) as ctx:
        # Load encoder and pose regressor from checkpoints
        encoder = build_encoder(cfg["encoder"])
        encoder_path = Path(cfg["training"].get("encoder_checkpoint", ""))
        if encoder_path.exists():
            encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))

        pose_regressor = build_pose_regressor({"embed_dim": cfg["encoder"]["embed_dim"]})
        pose_path = Path(cfg["training"].get("pose_checkpoint", ""))
        if pose_path.exists():
            pose_regressor.load_state_dict(torch.load(pose_path, map_location="cpu"))

        encoder.eval()
        pose_regressor.eval()

        if use_synthetic:
            # Generate synthetic sweep
            frames, poses = make_synthetic_sweep(
                num_frames=cfg["training"].get("num_frames", 4),
                batch_size=cfg["training"]["batch_size"],
                channels=cfg["data"]["channels"],
                height=cfg["data"]["image_size"],
                width=cfg["data"]["image_size"],
                seed=cfg.get("seed", 0),
            )
            B, N, C, H, W = frames.shape

            # Encode frames
            with torch.no_grad():
                embeddings_list = [encoder(frames[:, i]) for i in range(N)]
                embeddings = torch.stack(embeddings_list, dim=1)  # (B, N, D)

            # Predict poses
            pred_transforms = pose_regressor(embeddings)  # (B, N, 7)

            # Compound to point cloud
            pixel_spacing = cfg["data"].get("pixel_spacing", 0.5)
            points, intensities = compound_point_cloud(
                frames.view(B, N, H, W), pred_transforms, pixel_spacing
            )

            ctx["points_shape"] = tuple(points.shape)
            ctx["intensities_shape"] = tuple(intensities.shape)
            ctx["num_points"] = points.shape[1]
        else:
            raise NotImplementedError("Real-data compounding not implemented yet.")

        save_checkpoint("stage2_compounding", "point_cloud", {"points": points, "intensities": intensities})
        logger.info(
            "stage2_compounding OK: %d points, shape=%s",
            ctx["num_points"], tuple(ctx["points_shape"]),
        )


def run_stage3_implicit_field(cfg: dict, use_synthetic: bool = True) -> None:
    set_seed(cfg.get("seed", 0))
    device_plan = resolve_device(**cfg["device"])
    logger.info("Device plan: %s", device_plan)

    with stage_run("stage3_implicit_field", config=cfg) as ctx:
        # Load data from checkpoint
        checkpoint_path = Path(cfg["training"].get("checkpoint_path", ""))
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            points = ckpt.get("points")
            intensities = ckpt.get("intensities")
        else:
            # Generate synthetic data
            frames, _ = make_synthetic_sweep(
                num_frames=cfg["training"].get("num_frames", 4),
                batch_size=cfg["training"]["batch_size"],
                channels=cfg["data"]["channels"],
                height=cfg["data"]["image_size"],
                width=cfg["data"]["image_size"],
                seed=cfg.get("seed", 0),
            )
            B, N, C, H, W = frames.shape
            points, intensities = compound_point_cloud(
                frames.view(B, N, H, W),
                torch.zeros(B, N, 7),  # identity poses for synthetic
                cfg["data"].get("pixel_spacing", 0.5),
            )

        # Build implicit field
        implicit_field = ImplicitFieldRegressor(
            dim_in=3,
            hidden_dim=cfg["reconstruction"].get("implicit_hidden_dim", 128),
            num_layers=cfg["reconstruction"].get("implicit_num_layers", 4),
            positional_encoding=cfg["reconstruction"].get("positional_encoding", "fourier"),
            pe_num_freqs=cfg["reconstruction"].get("pe_num_freqs", 6),
        )

        # Train
        lr = cfg["training"].get("lr", 0.0003)
        optimizer = torch.optim.Adam(implicit_field.parameters(), lr=lr)

        num_epochs = cfg["training"].get("epochs", 10)
        losses = []

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            pred_intensities = implicit_field(points)  # (B, N_pts, 1)
            loss = torch.nn.functional.mse_loss(pred_intensities, intensities.unsqueeze(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        ctx["final_loss"] = losses[-1]
        ctx["losses"] = losses
        ctx["points_shape"] = tuple(points.shape)

        # Save checkpoint
        save_checkpoint("stage3_implicit_field", "implicit_model", implicit_field.state_dict())
        fig_path = plot_loss_curve(losses, stage="stage3_implicit_field")
        ctx["loss_figure"] = str(fig_path)

        logger.info(
            "stage3_implicit_field OK: final_loss=%.6f, points=%d",
            ctx["final_loss"], points.shape[1],
        )


def run_stage4_render(cfg: dict, use_synthetic: bool = True) -> None:
    set_seed(cfg.get("seed", 0))
    device_plan = resolve_device(**cfg["device"])
    logger.info("Device plan: %s", device_plan)

    with stage_run("stage4_render", config=cfg) as ctx:
        # Load implicit field from checkpoint
        implicit_field = ImplicitFieldRegressor(
            dim_in=3,
            hidden_dim=cfg["reconstruction"].get("implicit_hidden_dim", 128),
            num_layers=cfg["reconstruction"].get("implicit_num_layers", 4),
            positional_encoding=cfg["reconstruction"].get("positional_encoding", "fourier"),
            pe_num_freqs=cfg["reconstruction"].get("pe_num_freqs", 6),
        )

        checkpoint_path = Path(cfg["training"].get("implicit_checkpoint", ""))
        if checkpoint_path.exists():
            implicit_field.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        else:
            raise FileNotFoundError("No implicit field checkpoint found. Run stage3 first.")

        implicit_field.eval()

        if use_synthetic:
            # Create a regular 3D grid for rendering
            grid_size = cfg["reconstruction"].get("grid_size", 32)
            x = torch.linspace(-1, 1, grid_size)
            y = torch.linspace(-1, 1, grid_size)
            z = torch.linspace(-1, 1, grid_size)
            xx, yy, zz = torch.meshgrid(x, y, z, indexing="xyz")
            points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1).unsqueeze(0)  # (1, N, 3)

            with torch.no_grad():
                intensities = implicit_field(points)  # (1, N, 1)
                intensities = intensities.view(grid_size, grid_size, grid_size)

            ctx["voxel_shape"] = (grid_size, grid_size, grid_size)
            ctx["intensities_min"] = intensities.min().item()
            ctx["intensities_max"] = intensities.max().item()
        else:
            raise NotImplementedError("Real-data rendering not implemented yet.")

        logger.info(
            "stage4_render OK: voxel_shape=%s, range=[%.4f, %.4f]",
            ctx["voxel_shape"], ctx["intensities_min"], ctx["intensities_max"],
        )


_STAGES = {
    "stage0_encoder": run_stage0_encoder,
    "stage1_pose": run_stage1_pose,
    "stage2_compounding": run_stage2_compounding,
    "stage3_implicit_field": run_stage3_implicit_field,
    "stage4_render": run_stage4_render,
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
