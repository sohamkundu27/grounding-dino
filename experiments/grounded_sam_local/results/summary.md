# Grounded SAM (v1) — local evaluation on NVIDIA GeForce RTX 3080

Pipeline: **text prompt -> Grounding DINO 1.0 (local) -> box -> SAM v1 (local) -> mask**.
Fully local. No cloud API, no API key, no image upload, no network at inference.

This is the SAM v1 sibling of the Grounded SAM 2 evaluation. It was run so the two can
be compared on identical inputs. **The Grounding DINO detector is byte-identical between
the two** (same build, config, checkpoint, thresholds, seed): across all 50 RefDrone
pairs the predicted boxes matched with **0 mismatches**, so every difference
below is attributable to the SAM stage alone.

**Bottom line — classification: REFERENCE ONLY for this use case. Grounded SAM (v1) is a
strictly worse fit than Grounded SAM 2 here, on every axis that differs, and identical on
the one axis that decides the headline task.**

- Referring-expression accuracy is **22%, exactly equal to Grounded SAM 2** — because the
  failure is in the shared detector, not in SAM. Swapping SAM 2 for SAM v1 cannot fix it.
- SAM v1's mask stage is **4.0x slower and uses ~2.5x the VRAM** of SAM 2.1 Tiny, for masks of
  equivalent visual quality (same box, same image).
- **SAM v1 has no video memory/propagation at all.** Applied to video it must re-detect
  every frame (4.1 FPS vs SAM 2's 32 FPS) and maintains **no object identity** —
  the top-1 mask hops between instances (13 cars/frame; 27 hops in 200 frames).

Full reasoning and the head-to-head table in sections 5–8 and 11.

---

## 1. What was actually run

| Phase | What | Status |
|---|---|---|
| 1 | Smoke test, single aerial image, prompts `car` / `person` | PASS |
| 2 | RefDrone sample, 50 (image, expression) pairs, seed 1234 | PASS (accuracy identical to SAM 2 — see 5) |
| 3 | Video, 2 clips, PER-FRAME detect+segment (SAM v1 has no propagation) | PASS (found the no-identity behaviour — see 6) |
| 4 | Benchmark, config A (SAM v1 ViT-B) vs B (SAM v1 ViT-H) | PASS |

Nothing here is claimed on the basis of a script exiting 0. Every result below was
checked against the rendered masks or the per-frame mask geometry.

## 2. Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 (sm_86, 11911 MB), driver 580.95.05 |
| CPU / RAM | 12th Gen Intel(R) Core(TM) i9-12900K / 62.5 GB |
| Python / torch | 3.12.3 / 2.5.1+cu124 (CUDA 12.4) |
| transformers | 4.44.2 |
| GD CUDA extension | **COMPILED** |
| SAM v1 checkpoints | ViT-B (93.7 M), ViT-H (641.1 M) |
| Cloud SDK installed | no |

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

50 pairs, seed 1234, split 30 train / 10 val / 10 test.
Config: GD Swin-T + SAM v1 vit_b, box_thr 0.25, text_thr 0.2.

| Metric | Grounded SAM (v1) | Grounded SAM 2 |
|---|---|---|
| **Top-1 localization, IoU>=0.5** | **22.0%** | 22.0% |
| Mean best IoU | 0.212 | 0.212 |
| Median best IoU | 0.004 | 0.004 |
| No-detection rate | 4.0% | 4.0% |
| Multiple-detection rate | 84.0% | 84.0% |
| Mean GD latency | 104 ms | 104 ms |
| **Mean SAM latency** | **107 ms** (ViT-B, fp32) | 26 ms (Tiny, bf16) |
| Peak VRAM | 3518 MB | 1433 MB |

The localization columns are identical **to four decimal places** because the boxes are
identical (0 mismatches over 50 pairs). This is the single most important thing this
evaluation shows: **the referring-expression failure is entirely in Grounding DINO 1.0**,
which resolves the *category* and discards the qualifiers. It is a shared, detector-side
capability gap. **No choice of SAM version touches it.** The only thing SAM v1 changes is
that it draws the (equally wrong-or-right) mask 4.0x slower.

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
| `tracking_car.mp4` (1920x1080, 200 frames) | **4.1 FPS** | 211 (103 + 108) | 31 FPS (propagation) |
| `zebra.mp4` (1280x720, 200 frames) | **4.3 FPS** | 210 (104 + 106) | — |

SAM 2 pays Grounding DINO **once** (358 ms) and then every subsequent frame is a
cheap propagation step; Grounded SAM (v1) pays GD **+** SAM on all 200 frames. That is the
whole architectural difference, and it is worth ~8x in throughput on this hardware.

### 6b. No object identity — the top-1 mask hops between instances

Because each frame is independent, "which object" is re-decided every frame by detector
confidence. There is no id linking frame N to frame N-1.

| | `tracking_car.mp4` | `zebra.mp4` |
|---|---|---|
| Detection rate | 100% | 100% |
| Mean detections / frame | 13.0 (max 17) | 5.9 (max 7) |
| Top-1 centroid jump (median / max) | 9 / 746 px | 4 / 698 px |
| Frame-to-frame hops (> 10% diagonal) | **27** | **20** |

Detection rate is 100% only because these clips always contain *some* instance of the
class; it says nothing about tracking a *specific* one. Figures 04–05 show the mechanism:
between frame 192 and 193 the top-1 `car` mask jumps ~746 px from a car in the centre
lanes to a different car near the horizon — a clean mask on a different object.

**How this compares to the SAM 2 failure mode.** SAM 2's Phase 3 found a *silent identity
switch*: after the tracked car left the frame at f69, SAM 2 re-bound object id 1 to a
different car 953 px away and kept reporting it as the same track. That is arguably
worse *because it is hidden* — SAM 2 claims a persistent identity and then quietly breaks
it. Grounded SAM (v1) never makes that claim: it has no persistent identity to break, so
the instance-hopping is visible in every frame rather than concealed. Neither is usable
as a single-target tracker out of the box; they fail differently. SAM 2 fails by *lying
about* an identity it has; SAM v1 fails by *never having* one.

## 7. Phase 4 — performance (ViT-B vs ViT-H, NVIDIA GeForce RTX 3080)

3 warm-ups discarded, 10 measured image iterations, 120 per-frame video frames.
Image input 1920x1080 (prompt `car`, 7 boxes); video input 1280x720.

| | **A: GD Swin-T + SAM v1 ViT-B** | **B: GD Swin-T + SAM v1 ViT-H** |
|---|---|---|
| SAM params | 94 M | 641 M |
| Model load (cold) | 1941 ms | 4607 ms |
| First inference (cold) | 459 ms | 796 ms |
| **Warm image latency** | **217 ms (4.6 FPS)** | **559 ms (1.8 FPS)** |
| &nbsp;&nbsp;of which Grounding DINO | 101 ms | 102 ms |
| &nbsp;&nbsp;of which SAM v1 | 116 ms (enc 108 + dec 8) | 457 ms (enc 448 + dec 9) |
| p95 image latency | 218 ms | 560 ms |
| **Per-frame video throughput** | **4.4 FPS** (227 ms/frame) | **1.8 FPS** (570 ms/frame) |
| Peak VRAM (image) | 3518 MB | 6541 MB |
| Peak VRAM (video) | 3518 MB | 6541 MB |
| Peak CPU RSS | 1879 MB | 2018 MB |

All GPU regions are timed with `torch.cuda.synchronize()` on **both** sides. Each config
ran in its own fresh process so B's cold-load time is not flattered by A's warm CUDA
context.

Unlike SAM 2 (where the detector dominates and the SAM backbone barely moves the needle),
**the SAM backbone is a real lever here.** ViT-B -> ViT-H nearly triples image latency
(217 -> 559 ms) and adds 3023 MB of VRAM, almost entirely in the image encoder
(enc 108 -> 448 ms). ViT-H's 6541 MB peak also matters on an 11 GB card. On this
evidence there is no reason to prefer ViT-H: no mask-quality gain was observed in any run,
and the box it segments is chosen by the (identical) detector regardless.

## 8. Head-to-head: Grounded SAM (v1) vs Grounded SAM 2

Same detector, same inputs, same machine. Everything below is measured.

| | **Grounded SAM (v1)** | **Grounded SAM 2** | Winner |
|---|---|---|---|
| RefDrone top-1 @IoU0.5 | 22% | 22% | tie (shared detector) |
| SAM mask latency (RefDrone image) | 107 ms | 26 ms | **SAM 2** |
| Warm image latency | 217 ms (4.6 FPS) | 143 ms (7.0 FPS) | **SAM 2** |
| Video approach | detect **every** frame | detect **once** + propagate | **SAM 2** |
| Video throughput | 4.1 FPS | 32 FPS | **SAM 2** |
| Cross-frame object identity | none (per-frame) | memory-based* | **SAM 2** |
| Peak VRAM (image, smallest) | 3518 MB | 1433 MB | **SAM 2** |
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

- **Referring-expression grounding: 22%**, identical to SAM 2 — a shared
  detector-side capability gap, unfixable by any SAM choice.
- **No video tracking capability at all.** No memory, no propagation, no object id.
  "Tracking" degrades to per-frame re-detection: 4.1 FPS and instance-hopping.
- **Expensive masks.** 4.0x the SAM 2 latency and ~2.5x the VRAM for no quality gain.
- **ViT-H buys nothing** here but ~2x the latency and 3023 MB more VRAM.

## 11. Classification: **REFERENCE ONLY** (for this use case)

Scoped, and stated plainly:

**REFERENCE ONLY as a detect-and-track pipeline for this project.** Grounded SAM (v1) is
a competent *image* detect-and-segment pipeline and its masks are excellent, but it is
dominated by Grounded SAM 2 on this workload. It shares SAM 2's fatal referring-expression
weakness (same detector, 22%) while giving up SAM 2's one genuine strength — cheap,
memory-based video tracking — because SAM v1 has no video capability to begin with. For a
single-image, single-shot segmentation task where 100 ms and 3.5 GB are acceptable, it is
perfectly serviceable. For anything involving video, or where VRAM/latency matter, or where
the goal is referring-expression grounding, there is no reason to choose it over Grounded
SAM 2, and several reasons not to.

The one thing this evaluation establishes cleanly, and that carries beyond the v1-vs-v2
question: **the 22% referring-expression ceiling is the detector's, not the segmenter's.**
Replacing Grounding DINO 1.0 is the only lever that matters; both SAM versions are waiting
on a better box.

## 12. Reproduce

```bash
source .venv-grounded-sam2/bin/activate   # same env: torch+CUDA, groundingdino, segment_anything on path
python experiments/grounded_sam_local/scripts/check_environment.py
python experiments/grounded_sam_local/scripts/run_image_demo.py \
    --image datasets/refdrone/images/all_image/0000001_02999_d_0000005.jpg \
    --prompts "car" "person"
python experiments/grounded_sam_local/scripts/run_refdrone_sample.py --seed 1234
python experiments/grounded_sam_local/scripts/run_video_demo.py \
    --video third_party/grounded_sam_2/assets/tracking_car.mp4 --prompt "car"
python experiments/grounded_sam_local/scripts/run_video_demo.py \
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
