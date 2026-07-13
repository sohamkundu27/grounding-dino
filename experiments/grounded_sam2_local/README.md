# Grounded SAM 2 — local evaluation

Controlled, fully-local evaluation of the Grounded SAM 2 pipeline on an RTX 3080.

```
text prompt -> Grounding DINO 1.0 (local) -> box -> SAM 2.1 (local) -> mask / video track
```

**Read [`results/summary.md`](results/summary.md) for the findings and the final
classification.** The short version: PROMISING as a *category-prompted* detect-and-track
pipeline (32 FPS tracking, 2.1 GB VRAM, excellent masks); **not usable as a
referring-expression system** (22% top-1 on RefDrone — Grounding DINO 1.0 resolves the
category and ignores the referring qualifiers).

## Local only

No cloud API, no API key, no image upload, **no network at inference**.

Grounded SAM 2 is a *pipeline*, not a model — it chains two independently-trained
networks and has no joint training and no standalone paper. Upstream ships demos for
both a local path and a **DDS cloud API** path (Grounding DINO 1.5/1.6, DINO-X). Only the
local path is used here. The six cloud demos are enumerated and avoided in
[`../../configs/local_eval.yaml`](../../configs/local_eval.yaml).

Enforcement is not just convention:
- `gsam2_local.assert_no_cloud_imports()` runs after every inference call and raises if a
  `dds_cloudapi` / `dinox` module is ever in `sys.modules`.
- `gsam2_local.device()` **raises rather than run on CPU**, so no CPU timing can ever be
  reported as if it were the GPU pipeline.
- `gsam2_local` forces `HF_HUB_OFFLINE=1` at import. Without it, Grounding DINO's
  `from_pretrained("bert-base-uncased")` reaches out to `huggingface.co` on **every**
  model load to revalidate the cached weights — confirmed with
  `strace -e trace=connect`. Nothing is uploaded, but it is not air-gap clean by default.

**No upstream submodule file is modified.** Everything lives in the wrapper
[`scripts/gsam2_local.py`](scripts/gsam2_local.py), outside `third_party/`.

## Layout

| Path | |
|---|---|
| `scripts/gsam2_local.py` | shared adapter — model loading, CUDA-synced timers, cloud/offline guards |
| `scripts/check_environment.py` | records GPU, versions, CUDA-extension status, network posture |
| `scripts/run_image_demo.py` | Phase 1 — smoke test |
| `scripts/run_refdrone_sample.py` | Phase 2 — RefDrone referring expressions |
| `scripts/run_video_demo.py` | Phase 3 — one-time acquisition + SAM 2 propagation |
| `scripts/benchmark_pipeline.py` | Phase 4 — config A (Tiny) vs B (Small) |
| `scripts/summarize_results.py` | generates `results/summary.md` from the result files |
| `scripts/make_figures.py` | curates `figures/` from `outputs/` |
| `results/` | committed JSON/CSV + `summary.md` |
| `figures/` | committed, downscaled representative visualizations |
| `outputs/` | **gitignored** — full videos, every frame, every visualization |

## Run it

```bash
source .venv-grounded-sam2/bin/activate
python scripts/check_environment.py        # exits non-zero if CUDA/ext/cloud checks fail
python scripts/run_refdrone_sample.py --seed 1234
python scripts/benchmark_pipeline.py
python scripts/summarize_results.py
```

Full command list in [`results/summary.md`](results/summary.md) §12.

## Two things that will bite on a fresh machine

1. **The `MultiScaleDeformableAttention` CUDA extension must be compiled** for the target's
   compute capability. Upstream's pure-PyTorch fallback is gated on
   `torch.cuda.is_available()`, **not** on whether the extension loaded — so on a GPU box
   without it, Grounding DINO **crashes rather than degrading**. Building it needs a real
   `nvcc`; this repo does it in a user-local toolchain (no sudo, no system CUDA changes).
2. **The BERT text encoder is not in `checkpoints/`.** It lives in `~/.cache/huggingface`
   (~421 MB) and must be staged separately on an air-gapped target, or Grounding DINO will
   not load at all.
