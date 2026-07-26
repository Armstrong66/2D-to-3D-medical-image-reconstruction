# CLAUDE.md — Trackerless Freehand Ultrasound 2D→3D Reconstruction
### ACVSS26 Hackathon · 5-day build guide

This file is the engineering source of truth for the project. It tells you (or
any Claude/Codex agent working in this repo) *what* to build, *where* to pull
each component from, and *in what order*, given a hard 5-day timeline.

---

## 1. Project statement

Reconstruct a continuous, queryable 3D volume from a **freehand, trackerless
2D ultrasound sweep** (raw acquired 2D frames — never resliced from an
existing 3D volume), with an optional pluggable segmentation overlay. No
GANs anywhere in the pipeline. Vision encoder must be swappable between ViT
and ViG (Vision GNN) behind one interface.

**Core claim to defend at pitch:** trackerless + low-cost ultrasound (already
the most deployed imaging modality in low-resource settings) turned into
continuous 3D anatomy without external tracking hardware or a CT/MRI suite.

**Explicitly out of scope for 5 days:** foundation-model-scale pretraining,
MedGemma/large-VLM backbones (domain mismatch — pretrained on X-ray/derm/
ophtho/histopathology, not ultrasound), diffusion-based augmentation, joint
end-to-end multi-task segmentation+reconstruction training, real-time/
sub-100ms inference claims.

---

## 2. Repo / resource map — where to pull from

Clone everything into `third_party/` and treat as **reference implementations
to adapt from, not black boxes to import wholesale**. Read each repo's
license before reuse (all below are research-use MICCAI/academic code).

| Component | Source | What to take from it |
|---|---|---|
| Data + official baseline | `github.com/QiLi111/TUS-REC2025-Challenge_baseline` | Data loader, EfficientNet-B1 pose-regression head, official eval metrics (global/local pixel + landmark error). **This is your Day-1 safety net — reproduce it first, verbatim, before changing anything.** |
| Data (protocol reference) | `github-pages.ucl.ac.uk/tus-rec-challenge/TUS-REC2024/data.html` | TUS-REC2024 = same forearm cohort, mostly straight-sweep protocol. Use as extra training data / easier validation subset. |
| Nonrigid pose upgrade (stretch) | `github.com/QiLi111/NR-Rec-FUS` | Co-optimized rigid transform + nonrigid deformation field + regularizer. Only attempt after the rigid baseline (above) is working end-to-end. |
| Continuous implicit volume | `github.com/pakheiyeung/ImplicitVol` + `github.com/pakheiyeung/PlaneInVol` | Coordinate-MLP reconstruction architecture + (optional) absolute frame-localization network if you want an alternative to relative pose chaining. This is your **core reconstruction backbone** — adapt directly. |
| Motion estimation alt/ensemble (stretch) | `github.com/guhong3648/MoGLo` | Global-local self-attention motion cues from speckle/echogenic regions. Swap-in alternative to the TUS-REC baseline regression head. |
| Physics-aware motion cue (stretch) | `github.com/Alphafrey946/PLPPI` | Speckle-decorrelation correlation operator — use if the plain image-regression pose head underperforms; respects ultrasound physics instead of treating frames as generic RGB. |
| 2D→3D architectural pattern only (not modality-matched) | `github.com/liukuan5625/Swin-X2S` | Reference for the "dimension-expanding module" bridging a 2D encoder to a 3D decoder — borrow the *pattern*, not the weights (X-ray, not ultrasound). |
| Coordinate-MLP-for-projection-data pattern only | `github.com/dakshshah03/neas` | Second example of a NeRF-style field fit to projection data (X-ray attenuation). Read for architecture inspiration only. |
| Segmentation fine-tune data | BUSI (Breast Ultrasound Images), or TN3K/DDTI (thyroid nodules) | Small (~780 imgs for BUSI), real lesion masks, dozens of published U-Net baselines to start from. Pick ONE, don't split effort. |

**Vision encoder (Stage 0), build yourself, don't clone:**
- ViT path: standard `timm` ViT-Small/Tiny, patch embed + MHSA blocks.
- ViG path: implement Max-Relative GraphConv blocks (see §4 math) — no
  mature off-the-shelf ultrasound-domain repo exists for this; this is your
  actual novelty surface, budget real time for it and treat as stretch.

---

## 3. Flow diagram (text/line-block — no images)

```
                         ┌───────────────────────────────┐
                         │   RAW 2D US SWEEP (frames)     │
                         │   TUS-REC2024 + 2025 corpus     │
                         └───────────────┬─────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────┐
                    │  STAGE 0 — ENCODER (swappable)        │
                    │  interface: encode(frame) -> z ∈ R^d  │
                    │  [ViT default] <---swap---> [ViG]     │
                    │  SSL-pretrained on raw frame corpus    │
                    └───────────────┬─────────────────────┘
                                         │  z_i (per-frame embedding)
                                         ▼
                    ┌─────────────────────────────────────┐
                    │  STAGE 1 — POSE / MOTION ESTIMATION   │
                    │  rigid T_i ∈ SE(3)  [TUS-REC baseline] │
                    │  + optional nonrigid d_ψ(p) [NR-Rec-FUS]│
                    │  alt/ensemble cues: MoGLo / PLPPI      │
                    │  loss: point-based distance + smooth. │
                    └───────────────┬─────────────────────┘
                                         │  {T_i, d_ψ} per frame
                                         ▼
                    ┌─────────────────────────────────────┐
                    │  STAGE 2 — COMPOUNDING                │
                    │  p_world = T_i · p_local + d_ψ(p_local)│
                    └───────────────┬─────────────────────┘
                                         │  compounded point cloud + intensities
                                         ▼
                    ┌─────────────────────────────────────┐
                    │  STAGE 3 — IMPLICIT NEURAL VOLUME      │
                    │  f_θ(p) -> intensity I(p)              │
                    │  [ImplicitVol-style coord-MLP]         │
                    │  positional encoding: Fourier/SIREN    │
                    │  --> THE "JELLY": continuous, queryable│
                    │      at arbitrary resolution, smoothly │
                    │      deformable, not a static mesh     │
                    └───────────────┬─────────────────────┘
                                         │  f_θ trained + FROZEN
                                         ▼
              ┌──────────────────────────┴───────────────────────────┐
              │                                                        │
              ▼                                                        ▼
┌───────────────────────────────┐                    ┌───────────────────────────────┐
│ STAGE 4a — RENDER (required)   │                    │ STAGE 4b — SEGMENTATION (plug) │
│ ray-march / dense sample f_θ   │                    │ 2D net fine-tuned on BUSI       │
│ output: continuous 3D render   │                    │ run on ORIGINAL 2D frames       │
└───────────────────────────────┘                    │ propagate masks via Stage-1 T_i │
                                                        │ supervise 2nd head S(p) on      │
                                                        │ SAME f_θ backbone (frozen       │
                                                        │ intensity branch)               │
                                                        └───────────────┬───────────────┘
                                                                        │
                                                                        ▼
                                                        ┌───────────────────────────────┐
                                                        │ overlay S(p) onto render (4a)  │
                                                        └───────────────────────────────┘
```

**Read this diagram as a dependency graph, not a strict pipeline** — 4b
depends on Stage 1's transforms and Stage 3's frozen backbone, but is
otherwise decoupled and removable without breaking 4a.

---

## 4. Math reference (implementation-ready)

**Stage 1 — pose loss (point-based distance, not raw parameter regression):**
```
L_pose = Σ_k ‖ T_pred·p_k − T_gt·p_k ‖²
L_reg  = ‖∇d_ψ‖²                      # nonrigid smoothness (stretch only)
```

**Stage 2 — compounding:**
```
p_world = T_i · (s·[u, v, 0]ᵀ) + d_ψ(p_local)     # s = known pixel spacing
```

**Stage 3 — implicit field, positional encoding to fight spectral bias:**
```
γ(p) = [sin(2^0πp), cos(2^0πp), ..., sin(2^{L-1}πp), cos(2^{L-1}πp)]   # Fourier
   or sinusoidal activations throughout: φ(x) = sin(ω₀(Wx + b))        # SIREN

L_recon = Σ_i Σ_(u,v) ‖ f_θ(p_world) − I_i(u,v) ‖²
```

**Stage 0 — ViG graph conv (Max-Relative), if/when you build the ViG path:**
```
x_i' = h( x_i , max_{j ∈ N(i)}(x_j − x_i) )     # N(i) = k-NN in feature space
```

**Total loss when stages are trained jointly (avoid this — see §5):**
```
L = λ_pose·L_pose + λ_reg·L_reg + λ_recon·L_recon
```
Segmentation head `S(p)` is trained **after** `f_θ` is frozen, on propagated
2D→3D labels — never folded into the joint loss above. This is the
decoupling that keeps Stage 4b genuinely pluggable/removable.

---

## 5. Staged development — 5-day milestones

Each day ends with a **working, demoable artifact**, even if it's just the
previous day's output re-run. Never end a day mid-refactor with nothing that
runs.

### Day 1 — Reproduce, don't innovate
- Clone `TUS-REC2025-Challenge_baseline`, get it training/eval'ing on a data
  subset, unmodified.
- Stand up Stage 0 encoder interface with ViT as the only implementation.
- **Checkpoint (must-hit):** TUS-REC baseline reproduced, official eval
  metrics computed on held-out scans. This is your fallback demo if nothing
  else lands.

### Day 2 — Core reconstruction backbone
- Swap in your Stage 0 embeddings as the pose-regression input (replace or
  augment the baseline's raw EfficientNet features).
- Stand up Stage 3 (ImplicitVol-style coordinate-MLP) trained on Stage 1's
  compounded output from Day 1's poses.
- **Checkpoint (must-hit):** one full scan reconstructed as a continuous
  field, renderable at at least a coarse resolution.

### Day 3 — Make it fluid + start stretch goals
- Polish Stage 4a rendering (this is where the "jelly not static mesh"
  quality actually shows up — smooth interpolated queries, not a fixed
  voxel grid).
- **Stretch, pick at most ONE:** nonrigid deformation (NR-Rec-FUS-style) OR
  ViG encoder swap OR MoGLo/PLPPI motion-cue ablation. Do not attempt more
  than one stretch item in parallel.
- **Checkpoint (must-hit):** rendering pipeline works end-to-end even if the
  stretch item is dropped.

### Day 4 — Pluggable segmentation
- Fine-tune a small 2D segmentation net on BUSI (or your chosen dataset) —
  fully independent of the reconstruction pipeline.
- Freeze `f_θ` from Day 2/3, add and train the `S(p)` head on propagated 2D
  masks via Stage 1 transforms.
- **Checkpoint (must-hit):** segmented overlay on the render for at least
  one scan; explicitly OK to ship without this if Day 3 ran long — it's
  additive, not load-bearing.

### Day 5 — Integration, honesty pass, pitch
- Consolidate the best-working configuration (drop any stretch item that
  isn't stable — don't demo something you can't explain if asked).
- Prepare the limitations slide up front: pose-drift on rotation-heavy
  scans, propagated-segmentation accuracy bounded by Stage-1 pose accuracy,
  per-scan (not amortized) implicit-field optimization cost.
- Rehearse fallback order: full pipeline → reconstruction+render only →
  Day-1 baseline reproduction. Know which one you're showing before you're
  on stage.

---

## 6. Engineering standards, repo scaffolding, and dev/execution split

Full detail lives in **`README.md`** — repo scaffolding, coding standards,
phase-gated dev workflow, data auto-download, required per-stage plots,
GPU/parallelism strategy, testing. Two things to hold in mind while working
in this repo specifically:

- **Dev environment vs. execution environment are different machines.**
  Code gets written and smoke-tested here on tiny synthetic data (CPU-only,
  no real dataset access) — real training/inference happens on Kaggle or
  the remote RTX workstation. A stage isn't "done" until it's been smoke
  tested here **and** run on a real data subset there, with the resulting
  plots/manifest reviewed before the next stage is written.
- **Every stage caches its output and writes a pass/fail manifest**
  (`outputs/<stage>/manifest.json`, traceback included on failure) — this
  is the fastest path from "it broke" to a fixable report.

## 7. Non-negotiable constraints (do not relitigate mid-build)

- No GAN components anywhere.
- No 3D-volume-reslicing shortcuts — inputs are natively-2D acquisitions.
- Encoder must stay behind a swappable interface (ViT default, ViG optional).
- Segmentation head trains only after `f_θ` is frozen — never joint end-to-end.
- If a stretch item isn't stable by its checkpoint, cut it — the fallback
  chain in §5 Day 5 is the actual deliverable, not the full wishlist.
