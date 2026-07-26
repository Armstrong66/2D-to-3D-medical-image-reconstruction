# Trackerless Freehand Ultrasound 2D→3D Reconstruction

ACVSS26 hackathon project. See **`CLAUDE.md`** for the pipeline design,
math, repo/dataset map, and the 5-day milestone schedule — this file
covers engineering setup, repo scaffolding, execution environments, and the
development workflow.

---

## 0. Execution model — read this first

This repo is developed under a split-environment constraint:

- **Dev/authoring environment** (wherever code is written/reviewed, e.g. an
  agent without GPU access or internet access to the real datasets) can only
  verify code **logically and syntactically** — shapes, dtypes, control
  flow, gradient flow — using tiny **synthetic** stand-ins for real data. It
  cannot download TUS-REC/BUSI or run real training.
- **Execution environment** (Kaggle notebook or the remote RTX workstation)
  is where real data, real training, and GPU-bound work happens.

The workflow is therefore: **write a stage → smoke-test it on synthetic data
in the dev environment → hand off to the execution environment → the person
running it reports back outputs, plots, and any tracebacks → fix and
proceed to the next stage.** Never write two unverified stages in a row —
each stage gates the next (see §5).

---

## 1. Repo structure

```
.
├── README.md
├── CLAUDE.md
├── pyproject.toml
├── environment.yml
├── .gitignore
├── src/
│   └── usrecon/                  # installable package (pip install -e .)
│       ├── __init__.py
│       ├── paths.py              # root-path resolution (no hardcoding)
│       ├── data/
│       │   ├── download.py       # dataset fetch, Kaggle-aware
│       │   ├── datasets.py       # torch Dataset classes
│       │   ├── synthetic.py      # tiny fake-data generators for smoke tests
│       │   └── transforms.py
│       ├── encoders/             # Stage 0 — swappable ViT / ViG
│       │   ├── base.py
│       │   ├── vit.py
│       │   └── vig.py
│       ├── pose/                 # Stage 1
│       │   ├── regression.py
│       │   └── losses.py
│       ├── reconstruction/       # Stage 2–3
│       │   ├── compounding.py
│       │   ├── implicit_field.py
│       │   └── positional_encoding.py
│       ├── segmentation/         # Stage 4b (pluggable)
│       │   └── seg2d.py
│       ├── pipeline/
│       │   └── run_stage.py      # CLI entrypoint, one stage at a time
│       ├── utils/
│       │   ├── device.py         # GPU detection, adaptive parallel strategy
│       │   ├── checkpoint.py     # save/load + run manifest (pass/fail log)
│       │   ├── viz.py            # all plotting lives here
│       │   └── seed.py
│       └── config/
│           └── default.yaml
├── scripts/
│   └── smoke_test.sh             # run all synthetic-data smoke tests, CPU-only
├── tests/
│   ├── conftest.py               # synthetic fixtures shared across tests
│   ├── test_stage0_encoder.py
│   ├── test_stage1_pose.py
│   ├── test_stage2_compounding.py
│   ├── test_stage3_implicit_field.py
│   └── test_stage4_segmentation.py
├── notebooks/
│   └── kaggle_run.ipynb          # thin runner: clone → install → run → plot
├── data/                         # gitignored — downloaded/cached datasets
└── outputs/                      # gitignored — checkpoints, figures, logs, manifests
```

**Package name:** `usrecon` (placeholder — rename freely; keep it short since
it's imported directly in the Kaggle notebook).

---

## 2. Setup

### Conda (local / workstation)
```bash
conda env create -f environment.yml
conda activate usrecon
pip install -e .
```

### Kaggle notebook
The notebook only needs three cells (see `notebooks/kaggle_run.ipynb`):
```python
!git clone https://github.com/<you>/<repo>.git
%cd <repo>
!pip install -e . -q
```
```python
from usrecon.pipeline.run_stage import run
run(stage="stage0_encoder", config="config/default.yaml")
```
```python
# inline plots are written to outputs/figures/ and also displayed
from usrecon.utils.viz import show_latest
show_latest(stage="stage0_encoder")
```
No manual data placement needed — `data/download.py` is Kaggle-aware (§6).

### `environment.yml` (contents to create alongside this README)
Pin: `python=3.11`, `pytorch`, `torchvision`, `pytorch-cuda` (workstation) /
CPU build (dev env), `numpy`, `scipy`, `matplotlib`, `einops`, `pyyaml`,
`tqdm`, `pytest`, `SimpleITK` (medical I/O if needed), `kaggle` (CLI/API).

### `pyproject.toml`
`src/`-layout, `setuptools` backend, editable-installable so the Kaggle
notebook's `pip install -e .` and any future `pip install <repo-url>` both
work without path hacks.

---

## 3. Root-path resolution — no hardcoding

`src/usrecon/paths.py` finds the project root by walking up from the current
file until it finds a marker (`pyproject.toml` or `.git`), then derives
everything else relative to it:

```python
from pathlib import Path

def find_project_root(marker: str = "pyproject.toml") -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not locate project root (looked for {marker})")

PROJECT_ROOT = find_project_root()
DATA_DIR    = PROJECT_ROOT / "data"
OUTPUT_DIR  = PROJECT_ROOT / "outputs"
```

This resolves identically whether the repo lives at `/home/claude/...`,
`/kaggle/working/<repo>`, or a workstation path — nothing in the codebase
should ever construct an absolute path by hand. On Kaggle specifically,
`data/download.py` additionally checks `/kaggle/input/` before falling back
to `DATA_DIR` (§6).

---

## 4. Coding standards

- **Modular, one responsibility per module** — each stage (`encoders/`,
  `pose/`, `reconstruction/`, `segmentation/`) exposes a single documented
  entrypoint function; internal helpers stay private.
- **No repeated logic** — shared math (positional encoding, point-transform
  application, loss reductions) lives once in `reconstruction/` or `utils/`
  and is imported everywhere it's needed, never copy-pasted between stages.
- **Every stage is independently runnable and cacheable**: `run_stage.py
  --stage <name>` reads the previous stage's cached output from
  `outputs/<stage>/`, never requires re-running upstream stages unless their
  cache is missing or `--force` is passed.
- **Every stage writes a run manifest** (`outputs/<stage>/manifest.json`):
  status (`pass`/`fail`), timestamp, config hash, output shapes, and — on
  failure — the full traceback. This is what makes "send me the traceback"
  fast: the manifest is the first thing to check.
- **Type hints + docstrings** on every public function (shapes in the
  docstring, e.g. `frames: (N, H, W) uint8`) — this is the actual
  publication-readiness bar, not decoration.
- **Config-driven, not flag-sprawl**: stage hyperparameters live in
  `config/default.yaml`; CLI flags override config keys, they don't
  duplicate them.

---

## 5. Development workflow — phase gating

For each stage, in order:

1. Implement the stage against the interface defined in `CLAUDE.md` §4/§5.
2. Add/extend a synthetic fixture in `data/synthetic.py` that mimics the
   real tensor shapes and dtypes (e.g. fake `(N, 1, H, W)` frame batches,
   fake `SE(3)` ground-truth transforms) — no real data involved.
3. Run `scripts/smoke_test.sh` (CPU-only, seconds not minutes) — this
   catches shape mismatches, dtype errors, NaN propagation, and broken
   gradient flow before anything touches real data or a GPU.
4. Only once the smoke test passes: hand off to the execution environment
   (Kaggle/workstation) to run the stage on a **real data subset** (not the
   full corpus) and produce its required plots (§7).
5. The subset run's outputs/plots/tracebacks get reported back. Fix if
   needed, re-run the smoke test, then proceed to the next stage.

This is the "confirm outputs before proceeding" loop — downstream stages
are not written against assumed upstream behavior, they're written against
*observed* upstream output.

---

## 6. Data — sources and auto-download

`src/usrecon/data/download.py` handles all fetching, environment-aware:

```python
def get_dataset(name: str) -> Path:
    if _running_on_kaggle() and _kaggle_input_has(name):
        return Path(f"/kaggle/input/{name}")          # already attached, no download
    cache = DATA_DIR / name
    if cache.exists():
        return cache                                    # already cached locally
    _download(name, dest=cache)                          # first-time fetch
    if _running_on_kaggle() and _has_kaggle_api_creds():
        _optionally_publish_as_kaggle_dataset(name, cache)  # persist across sessions
    return cache
```

- `_running_on_kaggle()` checks for `KAGGLE_KERNEL_RUN_TYPE` in the
  environment or the existence of `/kaggle/input`.
- Publishing back to Kaggle as a dataset is opt-in (behind an explicit flag
  and valid `kaggle.json` credentials) — never automatic, since it uploads
  data to your Kaggle account.
- Sources to wire up: TUS-REC2024/2025 (Zenodo, per the challenge data
  page), BUSI (or TN3K/DDTI, whichever you finalize for segmentation).
  Cite the source dataset papers in any write-up/report — see `CLAUDE.md`
  §2 for the exact repos/pages.

---

## 7. Required plots per stage (research-grade tracking)

Every stage writes figures to `outputs/figures/<stage>/`, not just numbers.
Minimum set:

| Stage | Required figures |
|---|---|
| Data / Stage 0 | grid of raw frame samples; before/after augmentation pairs |
| Stage 1 (pose) | predicted-vs-GT transform error histogram; cumulative drift vs. sequence length; 2D projection of estimated probe trajectory |
| Stage 2 (compounding) | 3D scatter of compounded point cloud, colored by source frame index |
| Stage 3 (implicit field) | training loss curve; orthogonal-slice renders of the reconstructed field at increasing training steps (shows convergence, not just a final result) |
| Stage 4b (segmentation) | 2D segmentation overlay on source frames; propagated 3D mask overlay on the Stage-3 render |

These are what go in the pitch deck *and* what make a failed run diagnosable
from the manifest + figures alone, without re-running anything.

---

## 8. GPU / data-parallelism strategy

Kept intentionally simple — no orchestration framework, just a safety-checked
wrapper, configured via CLI at launch:

```
--gpus auto|0|0,1|cpu       # which device(s) to use; "auto" = detect and decide
--parallel-strategy auto|single|ddp   # default: auto
```

`utils/device.py` logic for `auto`:
1. Enumerate visible GPUs (`torch.cuda.device_count()`).
2. For each, check free VRAM (`torch.cuda.mem_get_info()`).
3. Only use more than one GPU if **at least two** have enough free VRAM for
   the stage's estimated footprint (a rough per-stage constant, not a fold
   or a duplicate model — never launch the same model twice on one 48GB
   card if that would risk contention).
4. If only one GPU is free, or the rest are busy/low-memory, fall back to
   single-GPU execution automatically — no error, no manual intervention.
5. If nothing is free, fall back to CPU with a clear log line (this is what
   the dev/smoke-test environment always hits).

This is deliberately not more complex than that — the user launches the
script with the flags above and the script adapts; no dynamic mid-run
rebalancing, no separate scheduler process.

---

## 9. Testing

- `pytest tests/` — all tests run on synthetic data, CPU-only, fast.
- Each `test_stageN_*.py` covers: correct output shape, no NaNs, gradient
  flows to every trainable parameter (`loss.backward()` then check
  `.grad is not None`), and that the stage's manifest gets written correctly
  on both pass and induced-failure.
- `scripts/smoke_test.sh` is just `pytest tests/ -x -q` — one command,
  meant to be run before every handoff to the execution environment.

---

## 10. License / attribution

This project builds on and fine-tunes from published research code — cite
the original papers/repos listed in `CLAUDE.md` §2 in any report, poster, or
pitch material, and keep their license terms (research-use only) in mind
before any wider release.
