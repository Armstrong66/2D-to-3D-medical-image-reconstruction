"""
Project-root and standard-directory resolution.

Never hardcode absolute paths anywhere else in the codebase -- import
PROJECT_ROOT / DATA_DIR / OUTPUT_DIR from here instead. This resolves
identically whether the repo lives under /home/<user>/..., /kaggle/working/...,
or a workstation path, because it locates itself relative to a marker file
rather than a fixed string.
"""
from __future__ import annotations
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


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
CONFIG_DIR = PROJECT_ROOT / "src" / "usrecon" / "config"

for _d in (DATA_DIR, OUTPUT_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
