# Grounded SAM (v1) — local evaluation

Controlled, fully-local evaluation of the **Grounded SAM (v1)** pipeline on an RTX 3080,
built as the direct sibling of [`../grounded_sam2_local`](../grounded_sam2_local) so the
two can be compared on identical inputs.

```
text prompt -> Grounding DINO 1.0 (local) -> box -> SAM v1 (local) -> mask
```

**Read [`results/summary.md`](results/summary.md) for the findings, the head-to-head
comparison, and the final classification.** The short version: **REFERENCE ONLY for this
use case.** Grounded SAM (v1) shares Grounded SAM 2's exact referring-expression weakness
(22% top-1 on RefDrone — *identical*, because the Grounding DINO detector is byte-for-byte
the same), while giving up SAM 2's one real strength: it has **no video memory/propagation
at all**, so "tracking" collapses to per-frame re-detection (~4 FPS vs 32 FPS) with no
object identity. Its masks are excellent but cost ~4× the latency and ~2.5× the VRAM of
SAM 2.1 Tiny.

## The comparison is the point

The detector is held **fixed and identical** to the SAM 2 evaluation: same
`GroundingDINO_SwinT_OGC` config, same `groundingdino_swint_ogc.pth`, same compiled
`MultiScaleDeformableAttention` CUDA extension, imported from the `grounded_sam_2` tree.
Across all 50 RefDrone pairs the predicted boxes matched with **0 mismatches**, so every
measured difference is attributable to the SAM stage alone. Only the segmentation network
changes: SAM v1 (`segment_anything`, ViT-B / ViT-H) instead of SAM 2.1.

This is what lets the summary say cleanly that the 22% referring-expression ceiling is the
**detector's**, not the segmenter's — no SAM version moves it.

## Local only

No cloud API, no API key, no image upload, **no network at inference**. Same posture and
enforcement as the SAM 2 evaluation (`HF_HUB_OFFLINE=1` forced at import; `device()` raises
rather than run on CPU; `assert_no_cloud_imports()` after every inference call). **No
upstream submodule file is modified** — everything lives in the wrapper
[`scripts/gsam_local.py`](scripts/gsam_local.py).

## The one architectural fact that drives everything

**SAM v1 is image-only.** There is no `build_sam2_video_predictor`, no `init_state`, no
`propagate_in_video`, no memory. Grounded SAM (v1) therefore cannot track: applied to
video it must run the whole detect+segment pipeline on every frame, and the "target" is
re-chosen every frame by detector confidence with nothing linking one frame to the next.
Phase 3 measures exactly this and quantifies the resulting instance-hopping.

## Layout

| Path | |
|---|---|
| `scripts/gsam_local.py` | shared adapter — GD 1.0 (fixed) + SAM v1 loading, CUDA-synced timers, cloud/offline guards |
| `scripts/check_environment.py` | records GPU, versions, CUDA-ext status, network posture, and that SAM v1 has no video predictor |
| `scripts/run_image_demo.py` | Phase 1 — smoke test |
| `scripts/run_refdrone_sample.py` | Phase 2 — RefDrone referring expressions (same 50 pairs, seed 1234) |
| `scripts/run_video_demo.py` | Phase 3 — **per-frame** detect+segment (no propagation exists) |
| `scripts/benchmark_pipeline.py` | Phase 4 — config A (ViT-B) vs B (ViT-H) |
| `scripts/summarize_results.py` | generates `results/summary.md`, including the head-to-head vs SAM 2 |
| `scripts/make_figures.py` | curates `figures/` from `outputs/` |
| `results/` | committed JSON/CSV + `summary.md` |
| `figures/` | committed, downscaled representative visualizations |
| `outputs/` | **gitignored** — full videos, every frame, every visualization |

## Run it

```bash
source .venv-grounded-sam2/bin/activate        # reuses the SAM 2 env; segment_anything is on sys.path
python scripts/check_environment.py
python scripts/run_refdrone_sample.py --seed 1234
python scripts/run_video_demo.py --video ../../third_party/grounded_sam_2/assets/tracking_car.mp4 --prompt "car"
python scripts/benchmark_pipeline.py
python scripts/make_figures.py
python scripts/summarize_results.py
```

Full command list in [`results/summary.md`](results/summary.md) §12.

## Notes for a fresh machine

Same two gotchas as the SAM 2 evaluation (the BERT text encoder is not in `checkpoints/`
— it lives in `~/.cache/huggingface`; and the `MultiScaleDeformableAttention` CUDA
extension must be compiled or Grounding DINO crashes rather than degrading), plus one of
its own: the SAM v1 weights (`checkpoints/sam/sam_vit_b_01ec64.pth`,
`sam_vit_h_4b8939.pth`) must be present. No separate virtualenv is needed — the adapter
puts `third_party/grounded_sam/segment_anything` on `sys.path`, so the existing
`.venv-grounded-sam2` runs it as-is.
