# Grounded SAM 2

## Overview

**A pipeline, not a new model — and the closest system-level match to what this
project is actually trying to build.** It chains Grounding DINO (open-vocabulary
target acquisition from text) into SAM 2 (segmentation *and* memory-based
tracking across video).

```
text prompt ──► Grounding DINO 1.0 ──► box ──► SAM 2 ──► masks + persistent track IDs
                (slow, run once)                (fast, every frame)
```

That split — acquire once with a heavy detector, then track cheaply — is exactly
the architecture the drone brief describes ("detection may run slowly, tracking
must hit 10+ Hz").

- **Primary paper:** SAM 2 — Ravi et al. 2024, arXiv [2408.00714](https://arxiv.org/abs/2408.00714) → [`../papers/04_grounded_sam_2/`](../papers/04_grounded_sam_2/)
- **No standalone Grounded SAM 2 paper exists** → see [`../papers/04_grounded_sam_2/NO_STANDALONE_PAPER.md`](../papers/04_grounded_sam_2/NO_STANDALONE_PAPER.md)
- **Official repo:** <https://github.com/IDEA-Research/Grounded-SAM-2>
- **Local source:** `third_party/grounded_sam_2/` @ `b7a9c29`

## Architecture

- **Detector:** Grounding DINO 1.0 (Swin-T/B + BERT) — local checkpoint
- **Image encoder (tracker):** SAM 2's **Hiera** hierarchical ViT (T / S / B+ / L)
- **Tracker:** a **memory bank** + memory-attention. SAM 2 conditions the current frame on features and mask predictions from previous frames, which is what lets a target survive occlusion and re-entry without re-detection.

The memory is the whole point, and it is also the thing that makes export hard
(see below).

## Inputs and outputs

- **In:** image or video + text prompt
- **Out:** masks, boxes, and **persistent object IDs across frames**
- This is the only system here that produces **track IDs**.

## Local availability

| | |
|---|---|
| Code | ✅ `third_party/grounded_sam_2/` |
| Grounding DINO ckpt | ✅ `checkpoints/grounding_dino/groundingdino_swint_ogc.pth` (shared) |
| SAM 2 ckpts | ✅ `checkpoints/sam2/` — `sam2.1_hiera_{tiny,small,base_plus,large}.pt` (149 MB → 857 MB) |
| License | Apache-2.0 |

## ⚠️ Local vs API — the distinction that matters

The repo mixes both, and only the local path is usable here.

**Fully local** (weights on disk, nothing leaves the machine):
`grounded_sam2_local_demo.py`, `grounded_sam2_tracking_demo.py`,
`grounded_sam2_tracking_demo_with_continuous_id.py`,
`grounded_sam2_tracking_demo_custom_video_input_gd1.0_local_model.py` ← **start here**

**API-backed — NOT local:**
The `*_gd1.5_*` and `*_dinox_*` demos import `dds_cloudapi_sdk` and require an
`API_TOKEN`. **Grounding DINO 1.5, 1.6 and DINO-X have no public local
checkpoints** — they are a hosted DeepDataSpace service, and calling them uploads
your imagery to a third party. They cannot ship on a Jetson. No token was
obtained and no API was called during this collection.

## Main strength

It is the **only system here that closes the loop**: language → target → sustained
track. `sam2.1_hiera_tiny` is 149 MB, which is genuinely plausible for real-time
tracking on Orin. SAM 2's CUDA extension is also *optional* (`USE_CUDA=0`) — a
useful escape hatch on aarch64.

## Main weakness

**The tracker is stateful, and ONNX is a stateless dataflow graph.** Exporting SAM
2 means externalising the memory bank as explicit graph I/O and driving the loop
yourself from C++ — realistically **3–4 separate engines** (image encoder, memory
attention, mask decoder, memory encoder) rather than one. The Grounding DINO half
remains as hard to export as ever.

Also: it needs **torch ≥ 2.3.1 and Python ≥ 3.10**, which conflicts with the
OpenMMLab stack. It must live in its own environment.

## Relevance to RefDrone

**Limited — and this is important.** RefDrone is **image-based**: no temporal
annotations, no track IDs, no video. It can validate the *detector* half of this
pipeline but says **nothing** about the tracker, which is the harder deployment
constraint. Evaluating detect-plus-track needs a separate video source
(VisDrone-VID/-MOT, or in-house drone footage). Do not let a good RefDrone score
create false confidence in the tracking stage.

## Relevance to Jetson Orin

**Highest.** This is the reference pipeline for the deployment target. The
detector runs once (slow is allowed); SAM 2 runs every frame (must clear 10 Hz).
Benchmark `hiera_tiny` and `hiera_small` first — they are the ones that decide
whether the frame-rate requirement is achievable at all.

## What to measure

- **Sustained tracker FPS** on real video, per Hiera size — the 10+ Hz question. Measure this *before* anything else; it gates the whole architecture.
- **How often must the detector re-fire?** The "detect once, track forever" premise assumes rarely. If the target is lost every few seconds, the "detection may be slow" allowance collapses and the plan changes.
- Track survival through **occlusion, scale change, and re-entry** — the memory bank's actual job.
- Drift over long sequences; ID switches when multiple similar targets are present ("several people, one in red").
- End-to-end latency: acquisition (detector) vs steady-state (tracker), reported separately.
