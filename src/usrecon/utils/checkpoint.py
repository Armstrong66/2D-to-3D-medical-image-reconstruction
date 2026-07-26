"""
Per-stage run manifests and checkpoint I/O.

Every stage run writes outputs/<stage>/manifest.json describing what
happened -- this is the first thing to check when something breaks, and
what makes "send me the traceback" fast instead of a re-run-and-guess loop.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import traceback

from ..paths import OUTPUT_DIR


def stage_dir(stage: str) -> Path:
    d = OUTPUT_DIR / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path(stage: str) -> Path:
    return stage_dir(stage) / "manifest.json"


def write_manifest(stage: str, status: str, **extra) -> None:
    payload = {
        "stage": stage,
        "status": status,  # "pass" or "fail"
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    with open(_manifest_path(stage), "w") as f:
        json.dump(payload, f, indent=2, default=str)


def read_manifest(stage: str) -> dict | None:
    path = _manifest_path(stage)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@contextmanager
def stage_run(stage: str, config: dict | None = None):
    """
    Wrap a stage's execution:

        with stage_run("stage0_encoder", config=cfg) as ctx:
            output = run_the_stage(...)
            ctx["output_shape"] = tuple(output.shape)

    On success, writes a "pass" manifest with whatever keys were added to
    `ctx`. On any exception, writes a "fail" manifest with the full
    traceback and config, then re-raises -- it never swallows the error.
    """
    ctx: dict = {}
    try:
        yield ctx
    except Exception:
        write_manifest(
            stage, status="fail",
            config=config, error=traceback.format_exc(), **ctx,
        )
        raise
    else:
        write_manifest(stage, status="pass", config=config, **ctx)


def save_checkpoint(stage: str, name: str, state: dict) -> Path:
    import torch
    path = stage_dir(stage) / f"{name}.pt"
    torch.save(state, path)
    return path


def load_checkpoint(stage: str, name: str) -> dict:
    import torch
    path = stage_dir(stage) / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path} -- run stage '{stage}' first "
            f"(or pass --force to re-run it)."
        )
    return torch.load(path, map_location="cpu")
