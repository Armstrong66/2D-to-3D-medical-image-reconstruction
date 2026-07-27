"""
Project-root and standard-directory resolution.

Never hardcode absolute paths anywhere else in the codebase -- import
PROJECT_ROOT / DATA_DIR / OUTPUT_DIR from here instead. This resolves
identically whether the repo lives under /home/<user>/..., /kaggle/working/...,
or a workstation path, because it locates itself relative to a marker file
rather than a fixed string.
"""
from __future__ import annotations
import os
from pathlib import Path


def find_project_root(marker: str = "pyproject.toml") -> Path:
    """Walk up from this file until a directory containing `marker` is found."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(
        f"Could not locate project root: no '{marker}' found in any parent "
        f"of {here}. Are you running from inside a checkout of the repo?"
    )


def _running_on_kaggle() -> bool:
    """Check if running on Kaggle by environment variable or path."""
    return (
        "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or Path("/kaggle/input").exists()
        or Path("/kaggle/working").exists()
    )


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
CONFIG_DIR = PROJECT_ROOT / "src" / "usrecon" / "config"

# SCRATCH_DIR is for large, non-persistent downloads (raw dataset archives)
# that should NEVER land under /kaggle/working, since that directory is
# capped at 20GB and counts as saved notebook output. On Kaggle this
# resolves to /kaggle/tmp -- outside /kaggle/working, not saved when the
# session ends, but with a much larger (~60GB) quota. Off Kaggle it falls
# back to DATA_DIR, since there's no equivalent persistence distinction.
SCRATCH_DIR = Path("/kaggle/tmp") if _running_on_kaggle() else DATA_DIR

for _d in (DATA_DIR, OUTPUT_DIR, FIGURES_DIR, SCRATCH_DIR):
    _d.mkdir(parents=True, exist_ok=True)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "OUTPUT_DIR",
    "FIGURES_DIR",
    "CONFIG_DIR",
    "SCRATCH_DIR",
]
