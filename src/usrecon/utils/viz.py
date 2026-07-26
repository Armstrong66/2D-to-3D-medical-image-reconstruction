"""
All plotting lives here so every stage saves figures the same way, to the
same place, with the same naming convention: outputs/figures/<stage>/*.png

Import matplotlib lazily inside functions (not at module load) so importing
`usrecon.utils.viz` never fails in an environment without a display backend
or without matplotlib installed at all (e.g. a pure logic-check pass).
"""
from __future__ import annotations
from pathlib import Path

from ..paths import FIGURES_DIR


def _fig_path(stage: str, name: str) -> Path:
    d = FIGURES_DIR / stage
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.png"


def plot_frame_grid(frames, stage: str, name: str = "sample_frames", ncols: int = 4):
    """
    frames: array-like, shape (N, H, W) or (N, 1, H, W), values in [0, 1] or [0, 255].
    Saves a grid of up to N sample frames.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    frames = np.asarray(frames)
    if frames.ndim == 4:  # (N, C, H, W) -> squeeze channel
        frames = frames[:, 0]

    n = frames.shape[0]
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)
    for i in range(nrows * ncols):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            ax.imshow(frames[i], cmap="gray")
            ax.set_title(f"frame {i}", fontsize=8)
    fig.tight_layout()
    out = _fig_path(stage, name)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_before_after(before, after, stage: str, name: str = "augmentation"):
    """Single before/after pair, e.g. raw frame vs. augmented frame."""
    import matplotlib.pyplot as plt
    import numpy as np

    before, after = np.asarray(before), np.asarray(after)
    if before.ndim == 3:
        before = before[0]
    if after.ndim == 3:
        after = after[0]

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(before, cmap="gray"); axes[0].set_title("before"); axes[0].axis("off")
    axes[1].imshow(after, cmap="gray"); axes[1].set_title("after"); axes[1].axis("off")
    fig.tight_layout()
    out = _fig_path(stage, name)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_loss_curve(losses, stage: str, name: str = "loss_curve"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(losses)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"{stage} training loss")
    fig.tight_layout()
    out = _fig_path(stage, name)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def show_latest(stage: str):
    """Kaggle/Jupyter convenience: inline-display every figure saved for a stage."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    d = FIGURES_DIR / stage
    if not d.exists():
        print(f"No figures yet for stage '{stage}' (looked in {d})")
        return
    for path in sorted(d.glob("*.png")):
        img = mpimg.imread(path)
        plt.figure(figsize=(6, 4))
        plt.imshow(img)
        plt.axis("off")
        plt.title(path.name)
        plt.show()
