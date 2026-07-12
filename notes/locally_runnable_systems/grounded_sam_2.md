# Grounded SAM 2

**Primary paper:** none — Grounded SAM 2 is a **software integration**, not a
standalone paper. Academic foundations: Grounding DINO
(`papers/locally_runnable_systems/01_grounding_dino/2023_Liu_Grounding_DINO.pdf`)
and SAM 2 (`papers/core_foundations/2024_Ravi_SAM_2.pdf`). See
`papers/locally_runnable_systems/04_grounded_sam_2/README.md`.

## What it is
A pipeline that chains an open-set detector (Grounding DINO / Grounding DINO 1.5)
with **SAM 2** to do open-vocabulary *detect → segment → track* across images and
video. This is the architecture the drone project is reaching for.

## Type
Pipeline / software integration — **not** a new base model and **not** an
independent academic paper.

## Core architecture
Grounding DINO acquires the target from a text prompt (one slow open-vocabulary
pass) → box prompt → SAM 2 segments and then **tracks** using its streaming
memory across frames.

- **Image encoder:** Swin (Grounding DINO) + Hiera (SAM 2 image encoder).
- **Text encoder:** BERT (via Grounding DINO).

## Open-vocabulary capability
Yes — from the Grounding DINO stage.

## Local code availability
Yes — `third_party/grounded_sam_2/` (IDEA-Research/Grounded-SAM-2, pinned
`b7a9c29`). **Caveat:** several demos import `dds_cloudapi_sdk` and require a
cloud API token (they upload imagery); the local path avoids these.

## Local checkpoint availability
Yes for the local route — public Grounding DINO weights + SAM 2.1 Hiera weights
(T/S/B+/L). `sam2.1_hiera_tiny` (~149 MB) is the realistic Orin candidate.
No local weights exist for Grounding DINO 1.5/1.6 / DINO-X (API-only).

## API dependence
Partial/optional — the strongest detector variants (GD 1.5/1.6, DINO-X) are
API-only. The fully-local route uses Grounding DINO 1.0 + SAM 2.

## Segmentation capability
Yes — SAM 2 masks.

## Video tracking capability
**Yes** — SAM 2's memory-based tracker is the key addition; this is the only one
of the five systems with first-class video tracking.

## Relevance to aerial/drone detection
Highest of the five for the end goal: slow open-vocabulary acquisition once, then
fast mask-based tracking — matching the "detect slowly, track at 10+ Hz" pattern.

## Relevance to Jetson deployment
Most promising target architecture. SAM 2 tiny is plausibly real-time on Orin for
the tracking stage; the Grounding DINO acquisition stage runs infrequently. Export
of both stages to ONNX/TensorRT is the main engineering risk.

## Likely strengths
- Open-vocabulary acquisition + robust memory-based tracking in one pipeline.
- Tunable cost via SAM 2 model size (tiny → large).

## Likely weaknesses
- Best detectors are API-only; local detector is the slower GD 1.0.
- Two-stage export/quantization complexity; re-acquisition policy after track loss
  is undefined.

## Unresolved research questions
- Sustained tracking FPS of SAM 2 tiny on Jetson AGX Orin (must reach 10+ Hz).
- When/how to re-trigger the detector after occlusion or track drift.
- Aerial small-object tracking robustness of SAM 2 memory.
