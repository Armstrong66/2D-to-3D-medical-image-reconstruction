# Design: Real-data loader + official 4-metric evaluation

**Date:** 2026-07-27
**Status:** Approved (design phase)
**Milestone:** Day-1 must-hit — reproduce the TUS-REC baseline on real held-out
scans with the official evaluation metrics (see `CLAUDE.md` §5).

---

## 1. Problem & goal

The synthetic pipeline (Stages 0–4) already runs end-to-end on CPU. What blocks
the Day-1 checkpoint is that **no stage can consume real data** —
`usrecon.data.datasets` does not exist and every `--real-data` path raises
`NotImplementedError` — and there is **no official evaluation metric** wired in.

Build two things, together:

1. `usrecon.data.datasets.TUSRECScanDataset` — a torch `Dataset` reading the
   real TUS-REC HDF5 schema, environment-aware (Kaggle-input-aware) via the
   existing `download.get_dataset`.
2. The **full official 4-metric** reconstruction-error evaluator:
   {Global, Local} × {all-pixel, landmark}, in millimetres.

Both must be **smoke-testable on CPU here** (synthetic `.h5` fixtures that
exercise the real code paths) and **runnable on Kaggle** to produce the
checkpoint numbers.

**Chosen approach:** *A — Faithful adapter layer.* Reimplement the official
transform + metric math in **pure torch** (no `pytorch3d`), own our loader and
metrics inside the `usrecon` package, and add an adapter mapping our Stage-1
output into the official convention. Rationale: single new dependency (`h5py`),
clean package boundaries, auditable pure-torch math, convention risk fenced off
by tests + a one-time Kaggle cross-check. (Rejected: B — vendoring the
baseline's utils, which drags in `pytorch3d` and their data-layout assumptions;
C — all-pixel-only, ruled out by the requirement for the full 4-metric set.)

---

## 2. Ground truth — the real TUS-REC schema

Confirmed from the official baseline
(`github.com/QiLi111/TUS-REC2025-Challenge_baseline`:
`utils/loader.py`, `utils/transform.py`, `generate_DDF.py`, `utils/metrics.py`).
The `data/download.py` "images/ + poses/" layout is a **placeholder guess**, not
the real format.

- **Per-scan HDF5** `<subject>/<scan>.h5` with exactly two datasets:
  - `frames` — `(N, H, W)` B-mode frames.
  - `tforms` — `(N, 4, 4)` ground-truth tracker transforms (tool→world). GT /
    eval only — **never fed to the network** (trackerless constraint).
  - Each subject folder holds multiple scan `.h5` files (baseline sorts
    filenames; ~2 scans/subject).
- **Frame sampling** (`frame_sampler`): random contiguous start `n0`, take
  `num_samples` frames.
- **Calibration + landmarks are separate files** — the frame loader does not
  read them:
  - Calibration (`opt.FILENAME_CALIB`) → `tform_image_pixel_to_mm` (pixel→mm
    scale), `tform_image_mm_to_tool` / `tform_tool_to_image_mm` (calib + inverse).
  - Landmarks: `landmark_%03d.h5`, keyed by `scan_name` → landmark pixel coords
    per frame.
- **Transform math**:
  - Params `[rx, ry, rz, tx, ty, tz]` → 4×4 via euler **ZYX** + translation column.
  - Relative img1→img0 in tool frame: `calib⁻¹ · T(tool1→tool0) · calib`.
  - Accumulate: `T(img2→img0) = T(img1→img0) · T(img2→img1)` (matmul chain).
  - Point→world mm: `T · tform_image_pixel_to_mm · image_points` → xyz.
- **Metric core** (`cal_dist`): Euclidean distance in mm, aggregated two ways
  (`all` vs `landmark`), assembled into the four numbers above. Global =
  transforms accumulated back to frame 0 (drift shows here); Local = short
  window relative to the previous frame(s).

---

## 3. Module layout

```
src/usrecon/data/datasets.py        NEW   TUSRECScanDataset + private h5 readers
src/usrecon/data/synthetic.py       EDIT  + write_synthetic_tus_rec(dir)
src/usrecon/geometry/__init__.py    NEW
src/usrecon/geometry/transforms.py  NEW   pure-torch transform math
src/usrecon/eval/__init__.py        NEW
src/usrecon/eval/metrics.py         NEW   reconstruction_errors() -> 4 mm numbers
src/usrecon/pipeline/run_stage.py   EDIT  --real-data branch (stage0/1/2) + stage_eval
src/usrecon/utils/viz.py            EDIT  error histogram, drift-vs-length, trajectory
tests/test_data_tus_rec.py          NEW
tests/test_geometry.py              NEW
tests/test_eval_metrics.py          NEW
requirements.txt                    EDIT  add h5py
src/usrecon/config/default.yaml     EDIT  add data paths + eval block
```

Boundaries follow the existing "one responsibility per module" standard:
low-level h5 I/O stays private inside `datasets.py`; the convention-sensitive
matrix math is isolated in `geometry/transforms.py` (its own test module);
metrics depend only on `geometry` + numpy/torch, not on any I/O.

---

## 4. Component specs

### 4.1 `data/datasets.py` — `TUSRECScanDataset`

- **Construction:** `TUSRECScanDataset(dataset_name, mode, num_samples,
  image_size, subjects=None)`. Root via `download.get_dataset(dataset_name)`
  (returns `/kaggle/input/...` when attached, else local cache). Enumerates
  `<subject>/<scan>.h5`; `subjects` optionally restricts to a held-out split.
- **Reads only** `frames` and `tforms`. Frames → float32, normalized via
  existing `preprocess.normalize`, resized to `image_size`, channel dim added →
  `(n, 1, H, W)`. `tforms` → `(n, 4, 4)` float32.
- **Modes:**
  - `mode="train"` — one item = a contiguous sub-sequence of `num_samples`
    frames (baseline `frame_sampler`).
  - `mode="eval"` — one item = a full scan.
- **Returns** `dict{ frames, tforms, scan_id, frame_indices }`.
- **Calibration** loaded once at construction → `self.calib` (a small dict of the
  named 4×4 matrices). Not per item.
- **Landmarks:** `self.load_landmarks(scan_id)` → `{coords_pixel, frame_indices}`
  when `landmark_*.h5` exists for that scan, else `None`.
- Private helpers `_read_scan`, `_read_calibration`, `_read_landmarks` isolate
  all h5py access.

### 4.2 `geometry/transforms.py` (pure torch)

- `euler_zyx_to_matrix(p6) -> (..., 4, 4)` and inverse `matrix_to_euler_zyx`.
- `quat_trans_to_matrix(p7) -> (..., 4, 4)` — bridges our Stage-1 output
  (`[tx,ty,tz,qw,qx,qy,qz]`) into the 4×4 convention.
- `apply_calibration(rel_tool, calib) -> calib⁻¹ · rel_tool · calib`.
- `accumulate_global(rel) -> (N, 4, 4)` (chain back to frame 0).
- `accumulate_local(rel, window) -> (N, 4, 4)`.
- `points_pixel_to_world_mm(tforms, pts_pixel, pixel_to_mm) -> (..., 3, P)`.
- No `pytorch3d`; euler-ZYX implemented from first principles and covered by a
  round-trip test.

### 4.3 `eval/metrics.py`

- `reconstruction_errors(pred_rel, gt_rel, calib, image_size, landmarks=None,
  pixel_stride=...) -> dict`:
  `{ "global_all", "local_all", "global_landmark", "local_landmark" }` in mm.
- all-pixel path uses the frame's pixel grid, subsampled by `pixel_stride` for
  speed; landmark path only when `landmarks` is provided — otherwise those two
  keys are `None` and the absence is **logged, never silently zeroed**.
- Internally: build GT and predicted world-mm point sets via `geometry`, then
  Euclidean distance (`all` vs `landmark` aggregation) matching `cal_dist`.

### 4.4 `stage_eval` runner (in `run_stage.py`)

Loads the frozen pose model, runs `TUSRECScanDataset(mode="eval")` over
held-out val scans, computes the 4 metrics per scan + aggregate, writes
`outputs/stage_eval/manifest.json` and figures. This is the demoable Day-1
checkpoint artifact.

### 4.5 Required figures (README §7)

- Predicted-vs-GT transform error histogram.
- Cumulative drift vs. sequence length.
- 2D projection of the estimated probe trajectory (predicted vs GT).

---

## 5. Verification plan (phase-gated, README §5)

**Dev / CPU (here):**
- `write_synthetic_tus_rec(dir)` builds a real `<subject>/<scan>.h5` tree +
  calibration file + `landmark_*.h5`, so loader, metrics, and `stage_eval` run
  through the **actual** code paths — no mocking of h5py.
- Correctness tests (the convention fence):
  - **identity predictions → ~0 mm** error on all four metrics;
  - a **known fixed transform → analytically-known distance**;
  - param↔matrix and euler round-trip within tolerance.
- `pytest tests/` (incl. new modules) green; existing 13 tests stay green.

**Execution / Kaggle (real subset):**
- Run `stage_eval` on a few val scans; confirm the 4 numbers are finite and in a
  plausible mm range.
- One-time cross-check that our all-pixel metric tracks the baseline's own
  reported magnitude (catches any residual convention mismatch).

Do not proceed to wiring real training beyond the `--real-data` branch until the
Kaggle subset run's numbers are reviewed.

---

## 6. Scope guardrails (YAGNI)

Out of scope for this pass: DDF file export, challenge submission packaging, any
training-loop changes beyond the `--real-data` branch, and multi-GPU concerns
(the existing `device.py` already handles that). Just: loader + metrics +
`geometry` + an eval runner that produces the checkpoint numbers.

---

## 7. Dependencies

- Add `h5py` to `requirements.txt` (required now).
- No `scipy`, no `pytorch3d` — the metric/transform math is pure torch.

---

## 8. Open risks

- **Matrix-convention mismatch** (calib pre/post-multiply order, euler order) is
  the main risk; it is isolated to `geometry/transforms.py` and fenced by the
  identity + known-transform tests here and the baseline cross-check on Kaggle.
- **Landmark file availability** varies by scan; metrics degrade gracefully to
  the two all-pixel numbers with an explicit log, never a silent zero.

---

## Addendum — 2026-07-28 (real calibration file + decisions)

The official `calib_matrix.csv` is now in hand. It is **two labelled 4x4 blocks**:

- `scaling_from_pixel_to_mm` = **S**, anisotropic: `sx=0.229389`, `sy=0.220980`
  mm/px (NOT the isotropic `0.5` placeholder).
- `spatial_calibration_from_image_coordinate_system_to_tracking_tool_coordinate_system`
  = **C**, a real rotation + **~117 mm** translation offset (probe sensor-to-
  imaging-plane lever arm).

Correct point placement is `p_world = T_i . C . S . [u,v,0,1]^T`. `geometry/
transforms.py` (Task 1) must expose a `Calibration` loader for this two-block CSV
and a `pixel_to_tool = C @ S` matrix; `apply_calibration` uses it. The package's
`reconstruction/compounding.py` is currently uncalibrated (scalar spacing, no C)
and is **exercised only by the synthetic smoke path** — its fix is **deferred and
folded into the Task-1 geometry work** (build calibration once in geometry; have
compounding + real-data stages call it), rather than a standalone patch.

**Decisions (2026-07-28):**
- **Package calibration fix:** deferred into the plan's geometry task (above).
- **Metrics:** keep the **full 4-metric** set. Landmark errors degrade gracefully
  to `None`/`n/a` when landmark files are absent (never a silent zero).

**Reference implementation:** the Kaggle notebook
`notebooks/kaggle_real_data_train_eval.ipynb` already implements the calibrated
`C @ S` placement and the full 4-metric evaluation (landmark path guarded by
`LANDMARK_DIR`). The package modules should match its conventions when built.
