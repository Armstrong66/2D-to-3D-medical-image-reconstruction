# Real-data loader + official 4-metric evaluation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load real TUS-REC HDF5 sweeps and score predicted probe motion with the challenge's official 4 reconstruction-error metrics, all smoke-testable on CPU here and runnable on Kaggle.

**Architecture:** A pure-torch geometry module (euler-ZYX, calibration, transform accumulation, point→mm) is the tested foundation. On top of it sit a `TUSRECScanDataset` reading the real `frames`/`tforms` HDF5 schema, an `eval/metrics.py` computing {Global,Local}×{all-pixel,landmark} errors in mm, and a `stage_eval` runner that produces the Day-1 checkpoint numbers + figures. Synthetic `.h5` fixtures exercise the real code paths without any real data.

**Tech Stack:** Python 3.11+, PyTorch (CPU here / CUDA on Kaggle), h5py, numpy, matplotlib, pytest.

## Global Constraints

- Package is `usrecon`, `src/`-layout, editable-installed (`pip install -e .`). Import as `from usrecon...`.
- No `pytorch3d`, no `scipy` — transform + metric math is pure torch. Only new dependency is `h5py`.
- Real HDF5 schema is fixed: per-scan file `<subject>/<scan>.h5` with datasets `frames` (N,H,W) and `tforms` (N,4,4). Calibration and landmarks are in separate files. Never feed `tforms` to the network (trackerless).
- Our Stage-1 pose output format is the 7-vector `[tx,ty,tz,qw,qx,qy,qz]`; the official convention is the 6-vector `[rx,ry,rz,tx,ty,tz]` with euler order **ZYX**. Bridge, don't rewrite Stage 1.
- Every stage writes `outputs/<stage>/manifest.json` (pass/fail + traceback) via the existing `stage_run` context manager; figures go to `outputs/figures/<stage>/`.
- Paths resolve via `usrecon.paths` / `download.get_dataset` — never hardcode absolute paths.
- TDD: failing test first, minimal impl, green, commit. All tests CPU-only and fast.
- The correctness fence for the geometry/metric conventions: **identity prediction → ~0 mm error**, and a **known transform → analytically-known distance**.

---

### Task 1: Pure-torch geometry primitives

**Files:**
- Create: `src/usrecon/geometry/__init__.py`
- Create: `src/usrecon/geometry/transforms.py`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Consumes: nothing (leaf module, torch only).
- Produces:
  - `euler_zyx_to_matrix(params: Tensor[...,6]) -> Tensor[...,4,4]` — params `[rx,ry,rz,tx,ty,tz]`, rotation from euler ZYX (intrinsic Z then Y then X), translation in last column.
  - `matrix_to_euler_zyx(T: Tensor[...,4,4]) -> Tensor[...,6]` — inverse of the above.
  - `quat_trans_to_matrix(p7: Tensor[...,7]) -> Tensor[...,4,4]` — p7 `[tx,ty,tz,qw,qx,qy,qz]`, quaternion assumed (or forced) unit-norm.
  - `invert_transform(T: Tensor[...,4,4]) -> Tensor[...,4,4]`.
  - `apply_calibration(rel_tool: Tensor[...,4,4], calib: Tensor[4,4]) -> Tensor[...,4,4]` = `calib⁻¹ · rel_tool · calib`.
  - `relative_from_absolute(tf: Tensor[N,4,4]) -> Tensor[N,4,4]` — `rel[0]=I`, `rel[k]=inv(tf[k-1])·tf[k]` (maps frame k into frame k-1).
  - `accumulate_global(rel: Tensor[N,4,4]) -> Tensor[N,4,4]` — `g[0]=I`, `g[k]=g[k-1]·rel[k]` (frame k → frame 0).
  - `accumulate_local(rel: Tensor[N,4,4], window: int = 1) -> Tensor[N,4,4]` — `l[k]=rel[k-window+1]·…·rel[k]` (frame k → frame k-window); for `window=1`, `l[k]=rel[k]`.
  - `points_pixel_to_world_mm(tforms: Tensor[...,4,4], pts_pixel: Tensor[4,P], pixel_to_mm: Tensor[4,4]) -> Tensor[...,3,P]` — `(tforms · pixel_to_mm · pts_pixel)[...,:3,:]`; `pts_pixel` is homogeneous `[u,v,0,1]ᵀ` columns.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_geometry.py
import torch
from usrecon.geometry.transforms import (
    euler_zyx_to_matrix, matrix_to_euler_zyx, quat_trans_to_matrix,
    invert_transform, apply_calibration, relative_from_absolute,
    accumulate_global, accumulate_local, points_pixel_to_world_mm,
)

def test_euler_identity_is_identity():
    T = euler_zyx_to_matrix(torch.zeros(6))
    assert torch.allclose(T, torch.eye(4), atol=1e-6)

def test_euler_roundtrip():
    p = torch.tensor([0.3, -0.2, 0.1, 5.0, -3.0, 2.0])
    T = euler_zyx_to_matrix(p)
    p2 = matrix_to_euler_zyx(T)
    assert torch.allclose(euler_zyx_to_matrix(p2), T, atol=1e-5)

def test_last_row_and_translation():
    p = torch.tensor([0.1, 0.2, 0.3, 7.0, 8.0, 9.0])
    T = euler_zyx_to_matrix(p)
    assert torch.allclose(T[3], torch.tensor([0., 0., 0., 1.]), atol=1e-6)
    assert torch.allclose(T[:3, 3], torch.tensor([7., 8., 9.]), atol=1e-6)

def test_rotation_is_orthonormal():
    p = torch.tensor([0.4, 0.5, -0.6, 0., 0., 0.])
    R = euler_zyx_to_matrix(p)[:3, :3]
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
    assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-5)

def test_quat_trans_identity():
    p7 = torch.tensor([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])  # unit quat = no rotation
    T = quat_trans_to_matrix(p7)
    assert torch.allclose(T[:3, :3], torch.eye(3), atol=1e-6)
    assert torch.allclose(T[:3, 3], torch.tensor([1., 2., 3.]), atol=1e-6)

def test_invert_transform():
    T = euler_zyx_to_matrix(torch.tensor([0.2, 0.1, -0.3, 1.0, 2.0, 3.0]))
    assert torch.allclose(invert_transform(T) @ T, torch.eye(4), atol=1e-5)

def test_apply_calibration_identity_calib():
    rel = euler_zyx_to_matrix(torch.tensor([0.1, 0., 0., 1., 0., 0.]))
    assert torch.allclose(apply_calibration(rel, torch.eye(4)), rel, atol=1e-6)

def test_relative_then_global_recovers_absolute():
    # tf: absolute frame->world; rel = per-step; global should map frame k back to frame 0
    torch.manual_seed(0)
    tf = torch.stack([euler_zyx_to_matrix(torch.randn(6) * 0.1) for _ in range(5)])
    rel = relative_from_absolute(tf)
    g = accumulate_global(rel)
    # g[k] == inv(tf[0]) @ tf[k]
    expected = torch.stack([invert_transform(tf[0]) @ tf[k] for k in range(5)])
    assert torch.allclose(g, expected, atol=1e-4)

def test_accumulate_local_window1_is_rel():
    torch.manual_seed(1)
    rel = torch.stack([euler_zyx_to_matrix(torch.randn(6) * 0.1) for _ in range(4)])
    assert torch.allclose(accumulate_local(rel, window=1), rel, atol=1e-6)

def test_points_pixel_to_world_mm_identity():
    pts = torch.tensor([[0., 1., 2.], [0., 0., 1.], [0., 0., 0.], [1., 1., 1.]])  # (4,3)
    T = torch.eye(4).unsqueeze(0)  # (1,4,4)
    out = points_pixel_to_world_mm(T, pts, torch.eye(4))  # (1,3,3)
    assert torch.allclose(out[0], pts[:3], atol=1e-6)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_geometry.py -q`
Expected: FAIL / ImportError (module not created yet).

- [ ] **Step 3: Implement `transforms.py`**

```python
# src/usrecon/geometry/transforms.py
"""Pure-torch SE(3) primitives for TUS-REC pose accumulation and scoring.

No pytorch3d. Conventions match the official baseline:
- euler order ZYX (intrinsic), params [rx, ry, rz, tx, ty, tz]
- relative transform maps frame k into frame k-1
- point pipeline: pixel (homogeneous) -> mm -> transform -> world
"""
from __future__ import annotations
import torch
from torch import Tensor


def _rot_z(a: Tensor) -> Tensor:
    c, s = torch.cos(a), torch.sin(a)
    o, z = torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([c, -s, z, s, c, z, z, z, o], dim=-1).reshape(*a.shape, 3, 3)


def _rot_y(a: Tensor) -> Tensor:
    c, s = torch.cos(a), torch.sin(a)
    o, z = torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([c, z, s, z, o, z, -s, z, c], dim=-1).reshape(*a.shape, 3, 3)


def _rot_x(a: Tensor) -> Tensor:
    c, s = torch.cos(a), torch.sin(a)
    o, z = torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([o, z, z, z, c, -s, z, s, c], dim=-1).reshape(*a.shape, 3, 3)


def euler_zyx_to_matrix(params: Tensor) -> Tensor:
    """params: (...,6) [rx,ry,rz,tx,ty,tz] -> (...,4,4). Rotation R = Rz@Ry@Rx."""
    rx, ry, rz = params[..., 0], params[..., 1], params[..., 2]
    t = params[..., 3:6]
    R = _rot_z(rz) @ _rot_y(ry) @ _rot_x(rx)               # (...,3,3)
    T = torch.zeros(*params.shape[:-1], 4, 4, dtype=params.dtype, device=params.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def matrix_to_euler_zyx(T: Tensor) -> Tensor:
    """Inverse of euler_zyx_to_matrix for R = Rz@Ry@Rx (no gimbal-lock handling)."""
    R = T[..., :3, :3]
    ry = torch.asin(torch.clamp(R[..., 0, 2], -1.0, 1.0))
    rz = torch.atan2(-R[..., 0, 1], R[..., 0, 0])
    rx = torch.atan2(-R[..., 1, 2], R[..., 2, 2])
    t = T[..., :3, 3]
    return torch.cat([rx.unsqueeze(-1), ry.unsqueeze(-1), rz.unsqueeze(-1), t], dim=-1)


def quat_trans_to_matrix(p7: Tensor) -> Tensor:
    """p7: (...,7) [tx,ty,tz,qw,qx,qy,qz] -> (...,4,4). Quaternion forced unit-norm."""
    t = p7[..., 0:3]
    q = p7[..., 3:7]
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*p7.shape[:-1], 3, 3)
    T = torch.zeros(*p7.shape[:-1], 4, 4, dtype=p7.dtype, device=p7.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def invert_transform(T: Tensor) -> Tensor:
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    out = torch.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    out[..., 3, 3] = 1.0
    return out


def apply_calibration(rel_tool: Tensor, calib: Tensor) -> Tensor:
    calib_inv = invert_transform(calib)
    return calib_inv @ rel_tool @ calib


def relative_from_absolute(tf: Tensor) -> Tensor:
    """tf: (N,4,4) absolute frame->world -> (N,4,4) rel[k]=inv(tf[k-1])@tf[k], rel[0]=I."""
    N = tf.shape[0]
    rel = torch.empty_like(tf)
    rel[0] = torch.eye(4, dtype=tf.dtype, device=tf.device)
    if N > 1:
        rel[1:] = invert_transform(tf[:-1]) @ tf[1:]
    return rel


def accumulate_global(rel: Tensor) -> Tensor:
    """rel: (N,4,4) -> (N,4,4) g[k]=g[k-1]@rel[k], g[0]=I (frame k -> frame 0)."""
    N = rel.shape[0]
    g = torch.empty_like(rel)
    g[0] = torch.eye(4, dtype=rel.dtype, device=rel.device)
    for k in range(1, N):
        g[k] = g[k - 1] @ rel[k]
    return g


def accumulate_local(rel: Tensor, window: int = 1) -> Tensor:
    """rel: (N,4,4) -> (N,4,4) l[k]=rel[k-window+1]@...@rel[k] (frame k -> frame k-window)."""
    N = rel.shape[0]
    out = torch.empty_like(rel)
    for k in range(N):
        acc = torch.eye(4, dtype=rel.dtype, device=rel.device)
        for j in range(max(1, k - window + 1), k + 1):
            acc = acc @ rel[j]
        out[k] = acc
    return out


def points_pixel_to_world_mm(tforms: Tensor, pts_pixel: Tensor, pixel_to_mm: Tensor) -> Tensor:
    """tforms:(...,4,4), pts_pixel:(4,P) homogeneous, pixel_to_mm:(4,4) -> (...,3,P)."""
    pts_mm = pixel_to_mm @ pts_pixel                 # (4,P)
    world = tforms @ pts_mm                           # (...,4,P)
    return world[..., :3, :]
```

- [ ] **Step 4: Run tests, verify green**

Run: `python -m pytest tests/test_geometry.py -q`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add src/usrecon/geometry/ tests/test_geometry.py
git commit -m "feat(geometry): pure-torch SE(3) primitives for TUS-REC pose math"
```

---

### Task 2: Synthetic TUS-REC HDF5 fixtures + h5py dependency

**Files:**
- Modify: `src/usrecon/data/synthetic.py` (append; keep existing functions)
- Modify: `requirements.txt` (add `h5py`)
- Test: `tests/test_synthetic_tus_rec.py`

**Interfaces:**
- Consumes: nothing (torch, h5py, numpy).
- Produces:
  - `write_synthetic_tus_rec(root: Path, num_subjects=2, scans_per_subject=2, num_frames=6, H=40, W=32, with_landmarks=True, seed=0) -> Path` — creates `root/<subject>/<scan>.h5` (datasets `frames` uint8 (N,H,W), `tforms` float64 (N,4,4)), a calibration file `root/calib_matrix.csv` (a 4×4 pixel→mm matrix, plus tool calib as identity), and `root/landmark_%03d.h5` keyed by scan filename stem when `with_landmarks`. Returns `root`.

- [ ] **Step 1: Add `h5py` to requirements**

Edit `requirements.txt`, under the "required now" block add:
```
h5py>=3.10
```

- [ ] **Step 2: Write failing test**

```python
# tests/test_synthetic_tus_rec.py
import h5py, numpy as np
from usrecon.data.synthetic import write_synthetic_tus_rec

def test_writes_expected_tree(tmp_path):
    root = write_synthetic_tus_rec(tmp_path, num_subjects=2, scans_per_subject=2,
                                   num_frames=5, H=40, W=32, with_landmarks=True, seed=0)
    subjects = sorted(p for p in root.iterdir() if p.is_dir())
    assert len(subjects) == 2
    scans = sorted(subjects[0].glob("*.h5"))
    assert len(scans) == 2
    with h5py.File(scans[0], "r") as f:
        assert f["frames"].shape == (5, 40, 32)
        assert f["frames"].dtype == np.uint8
        assert f["tforms"].shape == (5, 4, 4)
    assert (root / "calib_matrix.csv").exists()
    assert len(list(root.glob("landmark_*.h5"))) >= 1
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest tests/test_synthetic_tus_rec.py -q`
Expected: FAIL (function not defined).

- [ ] **Step 4: Implement `write_synthetic_tus_rec`**

Append to `src/usrecon/data/synthetic.py`:
```python
from pathlib import Path
import numpy as np
import h5py


def write_synthetic_tus_rec(root, num_subjects=2, scans_per_subject=2, num_frames=6,
                            H=40, W=32, with_landmarks=True, seed=0):
    """Write a real-code-path TUS-REC tree under `root` for CPU smoke tests.

    Layout:
      root/subjectXXX/scanYY.h5   datasets: frames (N,H,W) uint8, tforms (N,4,4) f8
      root/calib_matrix.csv       4x4 pixel->mm calibration
      root/landmark_%03d.h5       per-subject, keyed by scan stem (if with_landmarks)
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    # simple diagonal pixel->mm calibration (0.5 mm/pixel), homogeneous
    calib = np.eye(4)
    calib[0, 0] = calib[1, 1] = 0.5
    np.savetxt(root / "calib_matrix.csv", calib, delimiter=",")

    for s in range(num_subjects):
        subj = root / f"subject{s:03d}"
        subj.mkdir(parents=True, exist_ok=True)
        for c in range(scans_per_subject):
            frames = rng.integers(0, 256, size=(num_frames, H, W), dtype=np.uint8)
            # small incremental rigid motion -> plausible tforms
            tforms = np.zeros((num_frames, 4, 4))
            cur = np.eye(4)
            for i in range(num_frames):
                step = np.eye(4)
                step[:3, 3] = rng.normal(0, 0.3, size=3)
                cur = cur @ step
                tforms[i] = cur
            with h5py.File(subj / f"scan{c:02d}.h5", "w") as f:
                f.create_dataset("frames", data=frames)
                f.create_dataset("tforms", data=tforms)
        if with_landmarks:
            with h5py.File(root / f"landmark_{s:03d}.h5", "w") as f:
                for c in range(scans_per_subject):
                    # a few landmarks: rows [frame_idx, u, v]
                    lm = np.stack([
                        rng.integers(0, num_frames, size=4),
                        rng.integers(0, W, size=4),
                        rng.integers(0, H, size=4),
                    ], axis=1).astype(np.float64)
                    f.create_dataset(f"scan{c:02d}", data=lm)
    return root
```

- [ ] **Step 5: Run tests, verify green**

Run: `python -m pytest tests/test_synthetic_tus_rec.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/usrecon/data/synthetic.py tests/test_synthetic_tus_rec.py requirements.txt
git commit -m "feat(data): synthetic TUS-REC HDF5 fixtures + h5py dep"
```

---

### Task 3: `TUSRECScanDataset`

**Files:**
- Create: `src/usrecon/data/datasets.py`
- Modify: `src/usrecon/data/__init__.py` (export `TUSRECScanDataset`)
- Test: `tests/test_data_tus_rec.py`

**Interfaces:**
- Consumes: `write_synthetic_tus_rec` (Task 2); `usrecon.data.preprocess.normalize`.
- Produces:
  - `class TUSRECScanDataset(root: Path, mode: str = "train", num_samples: int = 4, image_size: int = 128, subjects: list[str] | None = None)`; attributes `self.calib: dict[str, Tensor]` (keys `"pixel_to_mm"`, `"image_mm_to_tool"`, `"tool_to_image_mm"`), `self.scan_ids: list[str]`.
  - `__len__`, `__getitem__(idx) -> dict{ "frames": Tensor(n,1,image_size,image_size) f32, "tforms": Tensor(n,4,4) f32, "scan_id": str, "frame_indices": Tensor(n,) long }`. `mode="train"` samples a contiguous sub-sequence of `num_samples`; `mode="eval"` returns the full scan.
  - `load_landmarks(scan_id: str) -> dict{"coords_pixel": Tensor(L,3), } | None`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_data_tus_rec.py
import torch
from usrecon.data.datasets import TUSRECScanDataset
from usrecon.data.synthetic import write_synthetic_tus_rec

def _ds(tmp_path, **kw):
    root = write_synthetic_tus_rec(tmp_path, num_subjects=2, scans_per_subject=2,
                                   num_frames=6, H=40, W=32, seed=0)
    return TUSRECScanDataset(root=root, image_size=16, **kw)

def test_eval_mode_returns_full_scan(tmp_path):
    ds = _ds(tmp_path, mode="eval")
    assert len(ds) == 4  # 2 subjects x 2 scans
    item = ds[0]
    assert item["frames"].shape == (6, 1, 16, 16)
    assert item["frames"].dtype == torch.float32
    assert item["tforms"].shape == (6, 4, 4)
    assert item["frame_indices"].shape == (6,)

def test_train_mode_samples_subsequence(tmp_path):
    ds = _ds(tmp_path, mode="train", num_samples=3)
    item = ds[0]
    assert item["frames"].shape == (3, 1, 16, 16)
    # contiguous indices
    idx = item["frame_indices"]
    assert torch.all(idx[1:] - idx[:-1] == 1)

def test_calibration_loaded(tmp_path):
    ds = _ds(tmp_path, mode="eval")
    assert ds.calib["pixel_to_mm"].shape == (4, 4)
    assert torch.allclose(ds.calib["pixel_to_mm"][0, 0], torch.tensor(0.5))

def test_landmarks_available(tmp_path):
    ds = _ds(tmp_path, mode="eval")
    lm = ds.load_landmarks(ds.scan_ids[0])
    assert lm is not None and lm["coords_pixel"].shape[1] == 3

def test_subjects_filter(tmp_path):
    root = write_synthetic_tus_rec(tmp_path, num_subjects=2, scans_per_subject=2,
                                   num_frames=6, H=40, W=32, seed=0)
    ds = TUSRECScanDataset(root=root, mode="eval", image_size=16, subjects=["subject000"])
    assert len(ds) == 2
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_data_tus_rec.py -q`
Expected: FAIL (module not created).

- [ ] **Step 3: Implement `datasets.py`**

```python
# src/usrecon/data/datasets.py
"""Real TUS-REC HDF5 loader. Reads only `frames` and `tforms`; calibration and
landmarks come from separate files (see data schema memory / spec)."""
from __future__ import annotations
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .preprocess import normalize


class TUSRECScanDataset(Dataset):
    def __init__(self, root, mode="train", num_samples=4, image_size=128, subjects=None):
        self.root = Path(root)
        self.mode = mode
        self.num_samples = num_samples
        self.image_size = image_size
        subj_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        if subjects is not None:
            subj_dirs = [p for p in subj_dirs if p.name in set(subjects)]
        self._scans = []                       # list of (subject_name, scan_path)
        for sd in subj_dirs:
            for scan in sorted(sd.glob("*.h5")):
                self._scans.append((sd.name, scan))
        self.scan_ids = [f"{s}/{p.stem}" for s, p in self._scans]
        self.calib = self._read_calibration()

    def __len__(self):
        return len(self._scans)

    def _read_calibration(self):
        csv = self.root / "calib_matrix.csv"
        pix2mm = (torch.from_numpy(np.loadtxt(csv, delimiter=",")).float()
                  if csv.exists() else torch.eye(4))
        return {"pixel_to_mm": pix2mm,
                "image_mm_to_tool": torch.eye(4),
                "tool_to_image_mm": torch.eye(4)}

    def _frame_sampler(self, n_total):
        if self.mode == "eval" or self.num_samples >= n_total:
            return list(range(n_total))
        n0 = random.randint(0, n_total - self.num_samples)
        return list(range(n0, n0 + self.num_samples))

    def _prep_frames(self, frames_np, idx):
        f = torch.from_numpy(frames_np[idx].astype(np.float32))      # (n,H,W)
        f = f.unsqueeze(1)                                           # (n,1,H,W)
        f = F.interpolate(f, size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False)
        return normalize(f)

    def __getitem__(self, idx):
        subj, scan_path = self._scans[idx]
        with h5py.File(scan_path, "r") as f:
            frames_np = f["frames"][()]
            tforms_np = f["tforms"][()]
        sel = self._frame_sampler(len(frames_np))
        frames = self._prep_frames(frames_np, sel)
        tforms = torch.from_numpy(tforms_np[sel].astype(np.float32))
        return {"frames": frames, "tforms": tforms,
                "scan_id": self.scan_ids[idx],
                "frame_indices": torch.tensor(sel, dtype=torch.long)}

    def load_landmarks(self, scan_id):
        subj, scan_stem = scan_id.split("/")
        # subject index from name "subjectNNN"
        try:
            sidx = int(subj.replace("subject", ""))
        except ValueError:
            return None
        lm_file = self.root / f"landmark_{sidx:03d}.h5"
        if not lm_file.exists():
            return None
        with h5py.File(lm_file, "r") as f:
            if scan_stem not in f:
                return None
            coords = torch.from_numpy(f[scan_stem][()].astype(np.float32))
        return {"coords_pixel": coords}
```

Update `src/usrecon/data/__init__.py` to add:
```python
from .datasets import TUSRECScanDataset  # noqa: F401
```

- [ ] **Step 4: Run tests, verify green**

Run: `python -m pytest tests/test_data_tus_rec.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/usrecon/data/datasets.py src/usrecon/data/__init__.py tests/test_data_tus_rec.py
git commit -m "feat(data): TUSRECScanDataset real HDF5 loader (frames/tforms/calib/landmarks)"
```

---

### Task 4: Official 4-metric evaluator

**Files:**
- Create: `src/usrecon/eval/__init__.py`
- Create: `src/usrecon/eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

**Interfaces:**
- Consumes: `usrecon.geometry.transforms` (Task 1).
- Produces:
  - `frame_corner_points(H: int, W: int, stride: int = 1) -> Tensor[4,P]` — homogeneous `[u,v,0,1]` pixel columns on a subsampled grid.
  - `cal_dist(label: Tensor[3,P], pred: Tensor[3,P], mode: str) -> float` — `mode="all"`: mean over points of per-point L2; `mode="landmark"`: mean over the 3 coord dims of per-dim L2 (mirrors baseline).
  - `reconstruction_errors(pred_rel: Tensor[N,4,4], gt_rel: Tensor[N,4,4], calib: dict, image_size: int, landmarks: Tensor[L,3] | None = None, pixel_stride: int = 8) -> dict{"global_all","local_all","global_landmark","local_landmark"}` — values in mm; landmark keys are `None` when `landmarks is None`. Both pred and gt go through the identical calib+accumulate+point pipeline, so `pred_rel==gt_rel` ⇒ 0.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_eval_metrics.py
import torch
from usrecon.geometry.transforms import euler_zyx_to_matrix, relative_from_absolute
from usrecon.eval.metrics import (frame_corner_points, cal_dist, reconstruction_errors)

def _calib():
    p = torch.eye(4); p[0, 0] = p[1, 1] = 0.5
    return {"pixel_to_mm": p, "image_mm_to_tool": torch.eye(4), "tool_to_image_mm": torch.eye(4)}

def test_corner_points_shape():
    pts = frame_corner_points(16, 16, stride=8)
    assert pts.shape[0] == 4
    assert torch.allclose(pts[3], torch.ones(pts.shape[1]))

def test_cal_dist_zero_when_equal():
    a = torch.randn(3, 20)
    assert cal_dist(a, a, "all") < 1e-6
    assert cal_dist(a, a, "landmark") < 1e-6

def test_identity_prediction_zero_error():
    torch.manual_seed(0)
    tf = torch.stack([euler_zyx_to_matrix(torch.randn(6) * 0.1) for _ in range(5)])
    rel = relative_from_absolute(tf)
    lm = torch.tensor([[0., 3., 4.], [2., 5., 6.]])
    out = reconstruction_errors(rel, rel, _calib(), image_size=16, landmarks=lm, pixel_stride=8)
    for k in out:
        assert out[k] is not None and out[k] < 1e-4

def test_known_translation_distance():
    # gt = identity everywhere; pred = pure +10mm x-translation at frame 1
    N = 2
    gt = torch.stack([torch.eye(4), torch.eye(4)])
    pred = torch.stack([torch.eye(4), euler_zyx_to_matrix(torch.tensor([0., 0., 0., 10., 0., 0.]))])
    out = reconstruction_errors(pred, gt, _calib(), image_size=16, landmarks=None, pixel_stride=8)
    # global all-pixel error at frame 1 is 10mm for every point; frame 0 is 0 -> mean 5mm
    assert abs(out["global_all"] - 5.0) < 1e-3
    assert out["global_landmark"] is None

def test_landmark_none_keys_none():
    rel = torch.stack([torch.eye(4), torch.eye(4)])
    out = reconstruction_errors(rel, rel, _calib(), image_size=16, landmarks=None)
    assert out["global_landmark"] is None and out["local_landmark"] is None
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_eval_metrics.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `metrics.py`**

```python
# src/usrecon/eval/metrics.py
"""Official TUS-REC reconstruction-error metrics, pure torch.

Four numbers in mm: {Global, Local} x {all-pixel, landmark}. Global accumulates
transforms back to frame 0 (drift shows); Local uses a 1-frame window. Both the
prediction and the ground truth flow through the identical pipeline, so an exact
prediction scores 0 regardless of the calibration.
"""
from __future__ import annotations
import torch
from torch import Tensor

from ..geometry.transforms import (
    apply_calibration, accumulate_global, accumulate_local,
    points_pixel_to_world_mm,
)


def frame_corner_points(H: int, W: int, stride: int = 1) -> Tensor:
    us = torch.arange(0, W, stride, dtype=torch.float32)
    vs = torch.arange(0, H, stride, dtype=torch.float32)
    uu, vv = torch.meshgrid(us, vs, indexing="xy")
    u = uu.reshape(-1); v = vv.reshape(-1)
    return torch.stack([u, v, torch.zeros_like(u), torch.ones_like(u)], dim=0)  # (4,P)


def cal_dist(label: Tensor, pred: Tensor, mode: str) -> float:
    diff = label - pred                       # (3,P)
    if mode == "all":
        return torch.sqrt((diff ** 2).sum(dim=0)).mean().item()
    elif mode == "landmark":
        return torch.sqrt((diff ** 2).sum(dim=1)).mean().item()
    raise ValueError(mode)


def _world_points(rel, calib, pts_pixel):
    rel_img = apply_calibration(rel, calib["image_mm_to_tool"])   # identity calib for synthetic
    g = accumulate_global(rel_img)
    l = accumulate_local(rel_img, window=1)
    pix2mm = calib["pixel_to_mm"]
    gw = points_pixel_to_world_mm(g, pts_pixel, pix2mm)           # (N,3,P)
    lw = points_pixel_to_world_mm(l, pts_pixel, pix2mm)
    return gw, lw


def reconstruction_errors(pred_rel, gt_rel, calib, image_size, landmarks=None, pixel_stride=8):
    pts = frame_corner_points(image_size, image_size, stride=pixel_stride)   # (4,P)
    g_pred, l_pred = _world_points(pred_rel, calib, pts)
    g_gt, l_gt = _world_points(gt_rel, calib, pts)
    # all-pixel: average per-frame all-point distance across frames
    def _mean_all(a, b):
        return sum(cal_dist(a[i], b[i], "all") for i in range(a.shape[0])) / a.shape[0]
    out = {"global_all": _mean_all(g_gt, g_pred),
           "local_all": _mean_all(l_gt, l_pred),
           "global_landmark": None, "local_landmark": None}
    if landmarks is not None:
        lm_pix = torch.stack([landmarks[:, 1], landmarks[:, 2],
                              torch.zeros(landmarks.shape[0]),
                              torch.ones(landmarks.shape[0])], dim=0)  # (4,L)
        # use frame 0's accumulated transform (landmarks defined in a frame's pixels)
        glm_pred = points_pixel_to_world_mm(accumulate_global(
            apply_calibration(pred_rel, calib["image_mm_to_tool"])), lm_pix, calib["pixel_to_mm"])
        glm_gt = points_pixel_to_world_mm(accumulate_global(
            apply_calibration(gt_rel, calib["image_mm_to_tool"])), lm_pix, calib["pixel_to_mm"])
        out["global_landmark"] = sum(cal_dist(glm_gt[i], glm_pred[i], "landmark")
                                     for i in range(glm_gt.shape[0])) / glm_gt.shape[0]
        llm_pred = points_pixel_to_world_mm(accumulate_local(
            apply_calibration(pred_rel, calib["image_mm_to_tool"]), 1), lm_pix, calib["pixel_to_mm"])
        llm_gt = points_pixel_to_world_mm(accumulate_local(
            apply_calibration(gt_rel, calib["image_mm_to_tool"]), 1), lm_pix, calib["pixel_to_mm"])
        out["local_landmark"] = sum(cal_dist(llm_gt[i], llm_pred[i], "landmark")
                                    for i in range(llm_gt.shape[0])) / llm_gt.shape[0]
    return out
```

- [ ] **Step 4: Run tests, verify green**

Run: `python -m pytest tests/test_eval_metrics.py -q`
Expected: PASS (5 tests). If `test_known_translation_distance` is off, verify the corner-point pipeline applies the +10mm translation as a rigid world offset (each point shifts 10mm ⇒ per-point distance 10mm at frame 1, 0 at frame 0, mean 5mm).

- [ ] **Step 5: Commit**

```bash
git add src/usrecon/eval/ tests/test_eval_metrics.py
git commit -m "feat(eval): official 4-metric reconstruction error (pure torch)"
```

---

### Task 5: `--real-data` wiring + `stage_eval` runner + figures

**Files:**
- Modify: `src/usrecon/pipeline/run_stage.py` (add `run_stage_eval`, register in `_STAGES`; add real-data branch to `run_stage0_encoder`)
- Modify: `src/usrecon/utils/viz.py` (add `plot_error_histogram`, `plot_drift_vs_length`, `plot_trajectory_2d`)
- Modify: `src/usrecon/config/default.yaml` (add `data.root` + `eval` block)
- Test: `tests/test_stage_eval.py`

**Interfaces:**
- Consumes: `TUSRECScanDataset` (Task 3), `reconstruction_errors` (Task 4), `geometry.quat_trans_to_matrix` + `relative_from_absolute` (Task 1), existing `build_encoder`, `build_pose_regressor`, `stage_run`.
- Produces:
  - `run_stage_eval(cfg: dict, use_synthetic: bool = True) -> None` — builds/loads encoder + pose regressor, iterates a `mode="eval"` dataset (synthetic tree when `use_synthetic`), converts predicted 7-vectors → relative 4×4 via `quat_trans_to_matrix`, derives GT rel via `relative_from_absolute(item["tforms"])`, calls `reconstruction_errors` per scan, aggregates mean, writes manifest fields + 3 figures.
  - viz: `plot_error_histogram(errors: list[float], stage: str) -> Path`, `plot_drift_vs_length(drift: list[float], stage: str) -> Path`, `plot_trajectory_2d(pred_xy, gt_xy, stage: str) -> Path`.

- [ ] **Step 1: Add config keys**

In `src/usrecon/config/default.yaml`, under `data:` add:
```yaml
  root: ""            # real-data root; empty -> synthetic tree for smoke test
```
Add a top-level block:
```yaml
eval:
  pixel_stride: 8
  num_samples: 6      # frames per scan when building synthetic eval tree
```

- [ ] **Step 2: Write failing test**

```python
# tests/test_stage_eval.py
import json
from usrecon.pipeline.run_stage import run_stage_eval
from usrecon.paths import OUTPUT_DIR

def _cfg():
    import yaml
    from usrecon.paths import CONFIG_DIR
    with open(CONFIG_DIR / "default.yaml") as f:
        return yaml.safe_load(f)

def test_stage_eval_writes_metrics_manifest(tmp_path):
    cfg = _cfg()
    cfg["data"]["image_size"] = 16
    cfg["encoder"]["image_size"] = 16
    run_stage_eval(cfg, use_synthetic=True)
    manifest = OUTPUT_DIR / "stage_eval" / "manifest.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text())
    assert data["status"] == "pass"
    assert "global_all" in data and data["global_all"] is not None
```

- [ ] **Step 3: Run to verify fail**

Run: `python -m pytest tests/test_stage_eval.py -q`
Expected: FAIL (`run_stage_eval` undefined).

- [ ] **Step 4: Implement viz helpers**

Append to `src/usrecon/utils/viz.py` (follow the existing `plot_*`/save pattern — resolve dir via the module's existing figures-path helper):
```python
def plot_error_histogram(errors, stage):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.hist(errors, bins=min(20, max(3, len(errors))))
    ax.set_xlabel("per-scan global all-pixel error (mm)"); ax.set_ylabel("count")
    ax.set_title(f"{stage}: reconstruction error")
    return _save_fig(fig, stage, "error_histogram")

def plot_drift_vs_length(drift, stage):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(range(1, len(drift) + 1), drift, marker="o")
    ax.set_xlabel("frame index"); ax.set_ylabel("cumulative drift (mm)")
    ax.set_title(f"{stage}: drift vs sequence length")
    return _save_fig(fig, stage, "drift_vs_length")

def plot_trajectory_2d(pred_xy, gt_xy, stage):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], "-o", label="GT")
    ax.plot(pred_xy[:, 0], pred_xy[:, 1], "-x", label="pred")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.legend()
    ax.set_title(f"{stage}: probe trajectory")
    return _save_fig(fig, stage, "trajectory_2d")
```
If `_save_fig` does not already exist in `viz.py`, reuse whatever the existing `plot_loss_curve`/`plot_frame_grid` functions call to persist figures (match that helper's signature exactly rather than inventing a new one).

- [ ] **Step 5: Implement `run_stage_eval`**

Add to `src/usrecon/pipeline/run_stage.py` (imports at top: `from ..data.datasets import TUSRECScanDataset`, `from ..data.synthetic import write_synthetic_tus_rec`, `from ..eval.metrics import reconstruction_errors`, `from ..geometry.transforms import quat_trans_to_matrix, relative_from_absolute`, `from ..utils.viz import plot_error_histogram, plot_drift_vs_length, plot_trajectory_2d`, `import tempfile`, `import numpy as np`):
```python
def run_stage_eval(cfg, use_synthetic=True):
    set_seed(cfg.get("seed", 0))
    resolve_device(**cfg["device"])
    with stage_run("stage_eval", config=cfg) as ctx:
        encoder = build_encoder(cfg["encoder"])
        for p in encoder.parameters():
            p.requires_grad = False
        pose_regressor = build_pose_regressor({"embed_dim": cfg["encoder"]["embed_dim"]})
        encoder.eval(); pose_regressor.eval()

        if use_synthetic:
            tmp = Path(tempfile.mkdtemp())
            root = write_synthetic_tus_rec(tmp, num_subjects=2, scans_per_subject=2,
                                           num_frames=cfg["eval"]["num_samples"],
                                           H=40, W=32, seed=cfg.get("seed", 0))
        else:
            root = Path(cfg["data"]["root"])
        ds = TUSRECScanDataset(root=root, mode="eval",
                               image_size=cfg["data"]["image_size"])

        scan_errors = []
        first_traj = None
        with torch.no_grad():
            for i in range(len(ds)):
                item = ds[i]
                frames = item["frames"]                        # (n,1,H,W)
                emb = encoder(frames).unsqueeze(0)             # (1,n,D)
                pred7 = pose_regressor(emb)[0]                 # (n,7)
                pred_rel = quat_trans_to_matrix(pred7)         # (n,4,4)
                gt_rel = relative_from_absolute(item["tforms"])
                lm = ds.load_landmarks(item["scan_id"])
                landmarks = lm["coords_pixel"] if lm else None
                m = reconstruction_errors(pred_rel, gt_rel, ds.calib,
                                          image_size=cfg["data"]["image_size"],
                                          landmarks=landmarks,
                                          pixel_stride=cfg["eval"]["pixel_stride"])
                scan_errors.append(m)
                if first_traj is None:
                    from ..geometry.transforms import accumulate_global
                    g_pred = accumulate_global(pred_rel)[:, :2, 3].cpu().numpy()
                    g_gt = accumulate_global(gt_rel)[:, :2, 3].cpu().numpy()
                    first_traj = (g_pred, g_gt)

        def _agg(key):
            vals = [e[key] for e in scan_errors if e[key] is not None]
            return float(np.mean(vals)) if vals else None
        for key in ("global_all", "local_all", "global_landmark", "local_landmark"):
            ctx[key] = _agg(key)
        ctx["num_scans"] = len(ds)

        ctx["error_hist_fig"] = str(plot_error_histogram(
            [e["global_all"] for e in scan_errors], stage="stage_eval"))
        gp, gg = first_traj
        drift = np.linalg.norm(gp - gg, axis=1).tolist()
        ctx["drift_fig"] = str(plot_drift_vs_length(drift, stage="stage_eval"))
        ctx["trajectory_fig"] = str(plot_trajectory_2d(gp, gg, stage="stage_eval"))

        logger.info("stage_eval OK: global_all=%.3fmm local_all=%.3fmm over %d scans",
                    ctx["global_all"], ctx["local_all"], len(ds))
```
Register in `_STAGES`: add `"stage_eval": run_stage_eval,`.

- [ ] **Step 6: Add real-data branch to Stage 0 (unblock `--real-data`)**

In `run_stage0_encoder`, replace the `else: raise NotImplementedError(...)` block with:
```python
        else:
            from ..data.datasets import TUSRECScanDataset
            ds = TUSRECScanDataset(root=Path(cfg["data"]["root"]), mode="train",
                                   num_samples=cfg["training"]["batch_size"],
                                   image_size=cfg["data"]["image_size"])
            frames = ds[0]["frames"]           # (n,1,H,W)
```

- [ ] **Step 7: Run tests, verify green**

Run: `python -m pytest tests/test_stage_eval.py -q`
Expected: PASS. Then run `python -m usrecon.pipeline.run_stage --stage stage_eval` and confirm it logs `stage_eval OK` and writes `outputs/stage_eval/manifest.json` + three figures.

- [ ] **Step 8: Commit**

```bash
git add src/usrecon/pipeline/run_stage.py src/usrecon/utils/viz.py src/usrecon/config/default.yaml tests/test_stage_eval.py
git commit -m "feat(eval): stage_eval runner with 4 metrics + drift/trajectory figures"
```

---

### Task 6: Full-suite green + smoke-test script confirmation

**Files:**
- Modify: `scripts/smoke_test.sh` (only if it does not already run `pytest tests/`)
- Test: entire `tests/` suite.

- [ ] **Step 1: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: all tests pass — the pre-existing 13 (stage0/stage1) plus the new geometry/synthetic/data/metrics/stage_eval tests.

- [ ] **Step 2: Run every stage runner on synthetic data**

Run:
```bash
for s in stage0_encoder stage1_pose stage2_compounding stage3_implicit_field stage4_render stage_eval; do python -m usrecon.pipeline.run_stage --stage $s; done
```
Expected: each logs `... OK`; `outputs/stage_eval/manifest.json` shows `status: pass` with the 4 metric keys.

- [ ] **Step 3: Confirm smoke script**

Ensure `scripts/smoke_test.sh` runs `pytest tests/ -x -q`. If it already does, no change.

- [ ] **Step 4: Commit any script change**

```bash
git add scripts/smoke_test.sh
git commit -m "chore: ensure smoke test covers loader + eval metrics"
```

---

## Handoff to execution environment (Kaggle) — after the plan is green here

Not a code task; the checkpoint gate per README §5:
1. Set `data.root` (or `data.dataset_name`) to the attached TUS-REC dataset and run `run_stage --stage stage_eval --real-data`.
2. Confirm the 4 metrics are finite and in a plausible mm range; review the 3 figures.
3. One-time cross-check that the all-pixel magnitude tracks the baseline's own reported numbers (catches any residual convention mismatch in `geometry/transforms.py`).

---

## Addendum — 2026-07-28 (calibration format + decisions)

- **Task 1 (geometry) must load the real `calib_matrix.csv`**: two labelled 4x4
  blocks — `S = scaling_from_pixel_to_mm` (anisotropic: sx=0.229389, sy=0.220980
  mm/px) and `C = image->tracking-tool` (real rotation + ~117 mm offset). Add a
  `Calibration` loader and a `pixel_to_tool = C @ S` matrix; point placement is
  `p_world = T_i . C . S . [u,v,0,1]^T`. Use these real matrices in the calib
  tests (not a synthetic diagonal).
- **`reconstruction/compounding.py` calibration fix is folded into Task 1**
  (deferred, not a standalone patch): today it uses scalar spacing + no C and is
  only exercised by the synthetic smoke path. Build calibration once in geometry;
  have compounding + real-data stages call it.
- **Metrics: keep the full 4-metric set** (landmark errors included), degrading to
  `None`/`n/a` when landmark files are absent — never a silent zero.
- **Reference:** `notebooks/kaggle_real_data_train_eval.ipynb` already implements
  calibrated `C @ S` placement and the full 4-metric eval; match its conventions.
