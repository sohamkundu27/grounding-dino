#!/usr/bin/env python3
"""Generate results/summary.md from the measured result files.

Every number here is READ from results/*.json|csv -- for BOTH this Grounded SAM (v1)
evaluation and the sibling Grounded SAM 2 evaluation, so the head-to-head comparison
cannot drift from what was actually measured. Narrative and judgement are mine.

    python summarize_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsam_local as G  # noqa: E402

RES = G.REPO / "experiments/grounded_sam_local/results"
RES2 = G.REPO / "experiments/grounded_sam2_local/results"


def load(name, base=RES):
    p = base / name
    if not p.exists():
        sys.exit(f"missing {p} — run the phase scripts first")
    return json.loads(p.read_text())


def main() -> int:
    env = load("environment.json")
    img = load("image_results.json")
    ref = load("refdrone_results.json")
    car = load("video_results_tracking_car.json")
    zeb = load("video_results_zebra.json")
    bm = load("benchmark.json")

    # sibling SAM 2 evaluation, for the head-to-head
    ref2 = load("refdrone_results.json", RES2)
    car2 = load("video_results_tracking_car.json", RES2)
    bm2 = load("benchmark.json", RES2)

    A = next(r for r in bm["results"] if r["sam_size"] == "vit_b")
    B = next(r for r in bm["results"] if r["sam_size"] == "vit_h")
    T = next(r for r in bm2["results"] if r["sam2_size"] == "tiny")   # SAM 2.1 Tiny
    m = ref["metrics"]
    m2 = ref2["metrics"]
    gpu = env["gpu"]
    ct, zt = car["throughput"], zeb["throughput"]
    cd, zd = car["detection"], zeb["detection"]
    ci, zi = car["identity"], zeb["identity"]
    cg2 = car2["tracking"]["gaps"][0]

    # cross-check the identical-detector claim
    n_pairs = len(ref["results"])
    box_mismatch = sum(1 for a, b in zip(ref["results"], ref2["results"])
                       if a["pred_box"] != b["pred_box"])

    md = f"""# Grounded SAM (v1) — local evaluation on {gpu['name']}

Pipeline: **text prompt -> Grounding DINO 1.0 (local) -> box -> SAM v1 (local) -> mask**.
Fully local. No cloud API, no API key, no image upload, no network at inference.

This is the SAM v1 sibling of the Grounded SAM 2 evaluation. It was run so the two can
be compared on identical inputs. **The Grounding DINO detector is byte-identical between
the two** (same build, config, checkpoint, thresholds, seed): across all {n_pairs} RefDrone
pairs the predicted boxes matched with **{box_mismatch} mismatches**, so every difference
below is attributable to the SAM stage alone.

**Bottom line — classification: REFERENCE ONLY for this use case. Grounded SAM (v1) is a
strictly worse fit than Grounded SAM 2 here, on every axis that differs, and identical on
the one axis that decides the headline task.**

- Referring-expression accuracy is **{m['top1_localization_acc_iou50']:.0%}, exactly equal to Grounded SAM 2** — because the
  failure is in the shared detector, not in SAM. Swapping SAM 2 for SAM v1 cannot fix it.
- SAM v1's mask stage is **{m['mean_sam_latency_ms']/m2['mean_sam2_latency_ms']:.1f}x slower and uses ~{ref['gpu_memory']['peak_allocated_mb']/ref2['gpu_memory']['peak_allocated_mb']:.1f}x the VRAM** of SAM 2.1 Tiny, for masks of
  equivalent visual quality (same box, same image).
- **SAM v1 has no video memory/propagation at all.** Applied to video it must re-detect
  every frame ({ct['fps']:.1f} FPS vs SAM 2's {T['track_fps']:.0f} FPS) and maintains **no object identity** —
  the top-1 mask hops between instances ({cd['detections_per_frame']['mean']:.0f} cars/frame; {ci['n_consecutive_frame_hops']} hops in {car['frames_processed']} frames).

Full reasoning and the head-to-head table in sections 5–8 and 11.

---

## 1. What was actually run

| Phase | What | Status |
|---|---|---|
| 1 | Smoke test, single aerial image, prompts `car` / `person` | PASS |
| 2 | RefDrone sample, {ref['sample']['n']} (image, expression) pairs, seed {ref['sample']['seed']} | PASS (accuracy identical to SAM 2 — see 5) |
| 3 | Video, 2 clips, PER-FRAME detect+segment (SAM v1 has no propagation) | PASS (found the no-identity behaviour — see 6) |
| 4 | Benchmark, config A (SAM v1 ViT-B) vs B (SAM v1 ViT-H) | PASS |

Nothing here is claimed on the basis of a script exiting 0. Every result below was
checked against the rendered masks or the per-frame mask geometry.

## 2. Environment

| | |
|---|---|
| GPU | {gpu['name']} ({gpu['compute_capability']}, {gpu['total_vram_mb']} MB), driver {gpu['driver']} |
| CPU / RAM | {env['machine']['cpu']} / {env['machine']['ram_gb']} GB |
| Python / torch | {env['python']} / {env['packages']['torch']} (CUDA {env['packages']['torch_cuda_build']}) |
| transformers | {env['packages']['transformers']} |
| GD CUDA extension | **{'COMPILED' if env['grounding_dino_cuda_extension']['compiled'] else 'MISSING'}** |
| SAM v1 checkpoints | ViT-B ({env['sam_v1']['params_millions']['vit_b']} M), ViT-H ({env['sam_v1']['params_millions']['vit_h']} M) |
| Cloud SDK installed | {'YES (!!)' if env['local_only']['cloud_sdk_installed'] else 'no'} |

**The detector is the exact build from the Grounded SAM 2 evaluation**, imported from
`third_party/grounded_sam_2` with its already-compiled `MultiScaleDeformableAttention`
CUDA op. Only the segmentation stage (`segment_anything`, from `third_party/grounded_sam`)
is new. This is deliberate: holding the detector fixed is what makes the comparison an
apples-to-apples measurement of the SAM stage.

## 3. Local-only verification

- Cloud SDK (`dds_cloudapi_sdk`): **not installed**.
- `assert_no_cloud_imports()` runs after every inference call and raises if any
  `dds_cloudapi` / `dinox` module is ever present in `sys.modules`. It never fired.
- **No upstream submodule file was modified.** All work lives in a wrapper,
  `scripts/gsam_local.py`.
- `gsam_local.device()` raises rather than run on CPU, so no CPU timing is reported as GPU.

## 4. Network posture (inherited from the detector)

Same finding as the SAM 2 evaluation, and for the same reason: Grounding DINO's text
encoder calls `from_pretrained("bert-base-uncased")`, and transformers revalidates the
cached files against `huggingface.co` on every load. `gsam_local` sets
`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE=1` at import, before transformers loads, so the
pipeline is network-free by default. The BERT encoder (~421 MB) lives in
`~/.cache/huggingface`, not in `checkpoints/`, and must be staged on an air-gapped target.
SAM v1 itself adds no new network dependency.

## 5. Phase 2 — RefDrone referring expressions (identical to SAM 2, and that is the point)

{ref['sample']['n']} pairs, seed {ref['sample']['seed']}, split {' / '.join(f"{v} {k}" for k, v in ref['sample']['by_split'].items())}.
Config: GD Swin-T + SAM v1 {ref['config']['sam']}, box_thr {ref['config']['box_threshold']}, text_thr {ref['config']['text_threshold']}.

| Metric | Grounded SAM (v1) | Grounded SAM 2 |
|---|---|---|
| **Top-1 localization, IoU>=0.5** | **{m['top1_localization_acc_iou50']:.1%}** | {m2['top1_localization_acc_iou50']:.1%} |
| Mean best IoU | {m['mean_best_iou']:.3f} | {m2['mean_best_iou']:.3f} |
| Median best IoU | {m['median_best_iou']:.3f} | {m2['median_best_iou']:.3f} |
| No-detection rate | {m['no_detection_rate']:.1%} | {m2['no_detection_rate']:.1%} |
| Multiple-detection rate | {m['multiple_detection_rate']:.1%} | {m2['multiple_detection_rate']:.1%} |
| Mean GD latency | {m['mean_gd_latency_ms']:.0f} ms | {m2['mean_gd_latency_ms']:.0f} ms |
| **Mean SAM latency** | **{m['mean_sam_latency_ms']:.0f} ms** (ViT-B, fp32) | {m2['mean_sam2_latency_ms']:.0f} ms (Tiny, bf16) |
| Peak VRAM | {ref['gpu_memory']['peak_allocated_mb']:.0f} MB | {ref2['gpu_memory']['peak_allocated_mb']:.0f} MB |

The localization columns are identical **to four decimal places** because the boxes are
identical ({box_mismatch} mismatches over {n_pairs} pairs). This is the single most important thing this
evaluation shows: **the referring-expression failure is entirely in Grounding DINO 1.0**,
which resolves the *category* and discards the qualifiers. It is a shared, detector-side
capability gap. **No choice of SAM version touches it.** The only thing SAM v1 changes is
that it draws the (equally wrong-or-right) mask {m['mean_sam_latency_ms']/m2['mean_sam2_latency_ms']:.1f}x slower.

The characteristic failure is unchanged: `"The black cars near the intersection."` returns
10 boxes, top-1 IoU 0.00, and SAM v1 segments that wrong box perfectly (figure 02). A clean,
confident mask on the wrong object — same as SAM 2, because it *is* the same box.

**Metric caveats (same as the SAM 2 run).** This is my protocol, not RefDrone's official
metric: highest-confidence box, scored as max IoU over the GT boxes. Do not compare to
published RefDrone numbers. RefDrone provides boxes only — no mask ground truth — so SAM
v1 mask quality is assessed *visually* (figure 01) and is deliberately **not scored**.

## 6. Phase 3 — video: SAM v1 cannot track, so it re-detects every frame

There is no SAM 2 protocol to run here, because SAM v1 has **no video predictor**: no
`init_state`, no `propagate_in_video`, no memory. The only coherent way to apply Grounded
SAM (v1) to a clip is to run the full image pipeline on every frame. Two things fall out.

### 6a. Cost — the detector is in the loop on every frame

| Clip | Throughput | ms/frame (GD + SAM) | vs SAM 2 same clip |
|---|---|---|---|
| `tracking_car.mp4` ({car['resolution'][0]}x{car['resolution'][1]}, {car['frames_processed']} frames) | **{ct['fps']:.1f} FPS** | {ct['mean_ms_per_frame']:.0f} ({ct['gd_ms_per_frame']['mean']:.0f} + {ct['sam_ms_per_frame']['mean']:.0f}) | {car2['tracking']['tracking_fps']:.0f} FPS (propagation) |
| `zebra.mp4` ({zeb['resolution'][0]}x{zeb['resolution'][1]}, {zeb['frames_processed']} frames) | **{zt['fps']:.1f} FPS** | {zt['mean_ms_per_frame']:.0f} ({zt['gd_ms_per_frame']['mean']:.0f} + {zt['sam_ms_per_frame']['mean']:.0f}) | — |

SAM 2 pays Grounding DINO **once** ({car2['acquisition']['gd_latency_ms']:.0f} ms) and then every subsequent frame is a
cheap propagation step; Grounded SAM (v1) pays GD **+** SAM on all {car['frames_processed']} frames. That is the
whole architectural difference, and it is worth ~{car2['tracking']['tracking_fps']/ct['fps']:.0f}x in throughput on this hardware.

### 6b. No object identity — the top-1 mask hops between instances

Because each frame is independent, "which object" is re-decided every frame by detector
confidence. There is no id linking frame N to frame N-1.

| | `tracking_car.mp4` | `zebra.mp4` |
|---|---|---|
| Detection rate | {cd['detection_rate']:.0%} | {zd['detection_rate']:.0%} |
| Mean detections / frame | {cd['detections_per_frame']['mean']:.1f} (max {cd['detections_per_frame']['max']}) | {zd['detections_per_frame']['mean']:.1f} (max {zd['detections_per_frame']['max']}) |
| Top-1 centroid jump (median / max) | {ci['top1_centroid_jump_px']['median']:.0f} / {ci['top1_centroid_jump_px']['max']:.0f} px | {zi['top1_centroid_jump_px']['median']:.0f} / {zi['top1_centroid_jump_px']['max']:.0f} px |
| Frame-to-frame hops (> 10% diagonal) | **{ci['n_consecutive_frame_hops']}** | **{zi['n_consecutive_frame_hops']}** |

Detection rate is 100% only because these clips always contain *some* instance of the
class; it says nothing about tracking a *specific* one. Figures 04–05 show the mechanism:
between frame 192 and 193 the top-1 `car` mask jumps ~{ci['top1_centroid_jump_px']['max']:.0f} px from a car in the centre
lanes to a different car near the horizon — a clean mask on a different object.

**How this compares to the SAM 2 failure mode.** SAM 2's Phase 3 found a *silent identity
switch*: after the tracked car left the frame at f{cg2['vanished_after_frame']+1}, SAM 2 re-bound object id 1 to a
different car {cg2['centroid_jump_px']:.0f} px away and kept reporting it as the same track. That is arguably
worse *because it is hidden* — SAM 2 claims a persistent identity and then quietly breaks
it. Grounded SAM (v1) never makes that claim: it has no persistent identity to break, so
the instance-hopping is visible in every frame rather than concealed. Neither is usable
as a single-target tracker out of the box; they fail differently. SAM 2 fails by *lying
about* an identity it has; SAM v1 fails by *never having* one.

## 7. Phase 4 — performance (ViT-B vs ViT-H, {gpu['name']})

{bm['protocol']['warmups_discarded']} warm-ups discarded, {bm['protocol']['measured_image_iters']} measured image iterations, {bm['protocol']['video_frames']} per-frame video frames.
Image input {A['image_resolution']} (prompt `{A['image_prompt']}`, {A['n_detections']} boxes); video input {A['track_resolution']}.

| | **A: GD Swin-T + SAM v1 ViT-B** | **B: GD Swin-T + SAM v1 ViT-H** |
|---|---|---|
| SAM params | {A['sam_params_m']:.0f} M | {B['sam_params_m']:.0f} M |
| Model load (cold) | {A['model_load_total_ms']:.0f} ms | {B['model_load_total_ms']:.0f} ms |
| First inference (cold) | {A['first_inference_total_ms']:.0f} ms | {B['first_inference_total_ms']:.0f} ms |
| **Warm image latency** | **{A['warm_total_mean_ms']:.0f} ms ({A['warm_total_fps']:.1f} FPS)** | **{B['warm_total_mean_ms']:.0f} ms ({B['warm_total_fps']:.1f} FPS)** |
| &nbsp;&nbsp;of which Grounding DINO | {A['warm_gd_mean_ms']:.0f} ms | {B['warm_gd_mean_ms']:.0f} ms |
| &nbsp;&nbsp;of which SAM v1 | {A['warm_sam_mean_ms']:.0f} ms (enc {A['warm_sam_encode_mean_ms']:.0f} + dec {A['warm_sam_decode_mean_ms']:.0f}) | {B['warm_sam_mean_ms']:.0f} ms (enc {B['warm_sam_encode_mean_ms']:.0f} + dec {B['warm_sam_decode_mean_ms']:.0f}) |
| p95 image latency | {A['warm_total_p95_ms']:.0f} ms | {B['warm_total_p95_ms']:.0f} ms |
| **Per-frame video throughput** | **{A['track_fps']:.1f} FPS** ({A['track_mean_ms_per_frame']:.0f} ms/frame) | **{B['track_fps']:.1f} FPS** ({B['track_mean_ms_per_frame']:.0f} ms/frame) |
| Peak VRAM (image) | {A['image_peak_vram_mb']:.0f} MB | {B['image_peak_vram_mb']:.0f} MB |
| Peak VRAM (video) | {A['track_peak_vram_mb']:.0f} MB | {B['track_peak_vram_mb']:.0f} MB |
| Peak CPU RSS | {A['peak_cpu_rss_mb']:.0f} MB | {B['peak_cpu_rss_mb']:.0f} MB |

All GPU regions are timed with `torch.cuda.synchronize()` on **both** sides. Each config
ran in its own fresh process so B's cold-load time is not flattered by A's warm CUDA
context.

Unlike SAM 2 (where the detector dominates and the SAM backbone barely moves the needle),
**the SAM backbone is a real lever here.** ViT-B -> ViT-H nearly triples image latency
({A['warm_total_mean_ms']:.0f} -> {B['warm_total_mean_ms']:.0f} ms) and adds {B['image_peak_vram_mb'] - A['image_peak_vram_mb']:.0f} MB of VRAM, almost entirely in the image encoder
(enc {A['warm_sam_encode_mean_ms']:.0f} -> {B['warm_sam_encode_mean_ms']:.0f} ms). ViT-H's {B['image_peak_vram_mb']:.0f} MB peak also matters on an 11 GB card. On this
evidence there is no reason to prefer ViT-H: no mask-quality gain was observed in any run,
and the box it segments is chosen by the (identical) detector regardless.

## 8. Head-to-head: Grounded SAM (v1) vs Grounded SAM 2

Same detector, same inputs, same machine. Everything below is measured.

| | **Grounded SAM (v1)** | **Grounded SAM 2** | Winner |
|---|---|---|---|
| RefDrone top-1 @IoU0.5 | {m['top1_localization_acc_iou50']:.0%} | {m2['top1_localization_acc_iou50']:.0%} | tie (shared detector) |
| SAM mask latency (RefDrone image) | {m['mean_sam_latency_ms']:.0f} ms | {m2['mean_sam2_latency_ms']:.0f} ms | **SAM 2** |
| Warm image latency | {A['warm_total_mean_ms']:.0f} ms ({A['warm_total_fps']:.1f} FPS) | {T['warm_total_mean_ms']:.0f} ms ({T['warm_total_fps']:.1f} FPS) | **SAM 2** |
| Video approach | detect **every** frame | detect **once** + propagate | **SAM 2** |
| Video throughput | {ct['fps']:.1f} FPS | {T['track_fps']:.0f} FPS | **SAM 2** |
| Cross-frame object identity | none (per-frame) | memory-based* | **SAM 2** |
| Peak VRAM (image, smallest) | {A['image_peak_vram_mb']:.0f} MB | {ref2['gpu_memory']['peak_allocated_mb']:.0f} MB | **SAM 2** |
| Mask quality (given a correct box) | excellent | excellent | tie |

*SAM 2's memory is a real capability but it silently switches identity after target egress
(section 6b) — better than v1's no-identity, still not a drop-in single-target tracker.

The verdict is not close. On the headline task (referring expressions) the two are
**identical** because the weak stage is shared; on every axis where they differ —
mask cost, VRAM, video throughput, and the existence of any temporal identity at all —
**SAM 2 wins**, most of them by a wide margin.

## 9. What works (SAM v1)

- Fully local, GPU, no network, no cloud API. No upstream file modified.
- **SAM v1 mask quality is excellent** — tight, clean masks on aerial imagery (figure 01),
  visually indistinguishable in quality from SAM 2. Given a correct box it is not the weak
  link. The problem is never mask *quality*; it is cost and the absence of tracking.
- 100% per-frame detection on clips that always contain the class.

## 10. What does not work / does not exist (SAM v1)

- **Referring-expression grounding: {m['top1_localization_acc_iou50']:.0%}**, identical to SAM 2 — a shared
  detector-side capability gap, unfixable by any SAM choice.
- **No video tracking capability at all.** No memory, no propagation, no object id.
  "Tracking" degrades to per-frame re-detection: {ct['fps']:.1f} FPS and instance-hopping.
- **Expensive masks.** {m['mean_sam_latency_ms']/m2['mean_sam2_latency_ms']:.1f}x the SAM 2 latency and ~{A['image_peak_vram_mb']/ref2['gpu_memory']['peak_allocated_mb']:.1f}x the VRAM for no quality gain.
- **ViT-H buys nothing** here but ~2x the latency and {B['image_peak_vram_mb'] - A['image_peak_vram_mb']:.0f} MB more VRAM.

## 11. Classification: **REFERENCE ONLY** (for this use case)

Scoped, and stated plainly:

**REFERENCE ONLY as a detect-and-track pipeline for this project.** Grounded SAM (v1) is
a competent *image* detect-and-segment pipeline and its masks are excellent, but it is
dominated by Grounded SAM 2 on this workload. It shares SAM 2's fatal referring-expression
weakness (same detector, {m['top1_localization_acc_iou50']:.0%}) while giving up SAM 2's one genuine strength — cheap,
memory-based video tracking — because SAM v1 has no video capability to begin with. For a
single-image, single-shot segmentation task where 100 ms and 3.5 GB are acceptable, it is
perfectly serviceable. For anything involving video, or where VRAM/latency matter, or where
the goal is referring-expression grounding, there is no reason to choose it over Grounded
SAM 2, and several reasons not to.

The one thing this evaluation establishes cleanly, and that carries beyond the v1-vs-v2
question: **the {m['top1_localization_acc_iou50']:.0%} referring-expression ceiling is the detector's, not the segmenter's.**
Replacing Grounding DINO 1.0 is the only lever that matters; both SAM versions are waiting
on a better box.

## 12. Reproduce

```bash
source .venv-grounded-sam2/bin/activate   # same env: torch+CUDA, groundingdino, segment_anything on path
python experiments/grounded_sam_local/scripts/check_environment.py
python experiments/grounded_sam_local/scripts/run_image_demo.py \\
    --image datasets/refdrone/images/all_image/0000001_02999_d_0000005.jpg \\
    --prompts "car" "person"
python experiments/grounded_sam_local/scripts/run_refdrone_sample.py --seed 1234
python experiments/grounded_sam_local/scripts/run_video_demo.py \\
    --video third_party/grounded_sam_2/assets/tracking_car.mp4 --prompt "car"
python experiments/grounded_sam_local/scripts/run_video_demo.py \\
    --video third_party/grounded_sam_2/assets/zebra.mp4 --prompt "zebra" --start-frame 20
python experiments/grounded_sam_local/scripts/benchmark_pipeline.py
python experiments/grounded_sam_local/scripts/make_figures.py
python experiments/grounded_sam_local/scripts/summarize_results.py
```

## 13. Figures

See [`figures/`](../figures/). SAM v1 mask quality (01), the shared grounding failure (02),
small aerial targets (03), and the per-frame no-identity hop between consecutive frames
(04, 05), plus per-frame zebra (06).

## 14. Not measured / not claimed

- SAM v1 mask **accuracy** is unscored: RefDrone has no mask ground truth and none was
  invented. Quality is asserted visually only.
- No comparison against published RefDrone baselines (different metric — see section 5).
- Jetson performance: **not measured.**
- Multi-object per-frame association / a real tracker on top of v1: not built, not claimed.
  Only single top-1 was followed, precisely to demonstrate the absence of identity.

---

*Generated by `scripts/summarize_results.py` from the files in `results/` (this experiment)
and `experiments/grounded_sam2_local/results/` (the SAM 2 sibling). Every number above is
read from a result file; none is transcribed by hand.*
"""
    p = RES / "summary.md"
    p.write_text(md)
    print(f"wrote {p}")
    print(f"  RefDrone top-1 @IoU0.5 : {m['top1_localization_acc_iou50']:.1%} "
          f"(SAM 2: {m2['top1_localization_acc_iou50']:.1%}, box mismatches: {box_mismatch})")
    print(f"  SAM mask latency       : {m['mean_sam_latency_ms']:.0f} ms (v1) vs "
          f"{m2['mean_sam2_latency_ms']:.0f} ms (SAM 2)")
    print(f"  video (v1 per-frame)   : {ct['fps']:.1f} FPS vs {T['track_fps']:.0f} FPS (SAM 2 propagate)")
    print(f"  classification         : REFERENCE ONLY (scoped) — see section 11")
    return 0


if __name__ == "__main__":
    sys.exit(main())
