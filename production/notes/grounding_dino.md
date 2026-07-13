# Grounding DINO 1.0

## Overview

The original open-vocabulary detector this whole family descends from. Given an
image and a free-form text prompt, it returns boxes with per-phrase scores — no
task-specific training, no fixed class list. It is the reference implementation
and the behavioural baseline everything else is measured against.

- **Paper:** Liu et al. 2023, arXiv [2303.05499](https://arxiv.org/abs/2303.05499) → [`../papers/01_grounding_dino/`](../papers/01_grounding_dino/)
- **Official repo:** <https://github.com/IDEA-Research/GroundingDINO>
- **Local source:** `third_party/grounding_dino/` @ `856dde2`

## Architecture

A DETR-style detector fused with a language model. The image and text branches
are cross-attended in a *feature enhancer*, then language-guided query selection
picks the queries that a cross-modality decoder turns into boxes. The key idea is
fusing vision and language **early** (neck, query init, and head) rather than
just scoring boxes against text at the end.

- **Image encoder:** Swin Transformer (Swin-T or Swin-B)
- **Text encoder:** BERT-base-uncased (via HuggingFace `transformers`)
- **Detector head:** DETR/DINO-style decoder with deformable attention

## Inputs and outputs

- **In:** an image + a text prompt (`"person wearing a red shirt ."` — phrases are period-separated)
- **Out:** boxes (cxcywh, normalised), per-box logits over the prompt's tokens, and the phrase each box grounds to

There are **no masks and no track IDs.** Boxes only.

## Local availability

| | |
|---|---|
| Code | ✅ `third_party/grounding_dino/` |
| Weights | ✅ `checkpoints/grounding_dino/` — `groundingdino_swint_ogc.pth` (662 MB), `groundingdino_swinb_cogcoor.pth` (895 MB) |
| License | Apache-2.0, code and weights |

## Main strength

Simplicity and reproducibility. It is the most-studied, most-forked model here,
so failures are diagnosable and the community has already hit most of the walls.
Swin-T is small enough to be a plausible edge detector.

## Main weakness

**Deployment.** The repo has **no official ONNX or TensorRT export** — I grepped
the tree, there is none. Its `MultiScaleDeformableAttention` CUDA op has no native
ONNX equivalent and must be rewritten (`grid_sample`, opset ≥16) or replaced with
a custom TensorRT plugin. It also compiles that op from source, so `nvcc` and
`CUDA_HOME` are hard install requirements. Accuracy has also since been beaten by
MM-Grounding-DINO.

## Relevance to RefDrone

The natural zero-shot baseline on RefDrone: run it with the referring expression
as the prompt and see what comes back. Two things to watch — RefDrone contains
**no-target** and **multi-target** expressions, and vanilla Grounding DINO has no
mechanism for "there is nothing here", so it will tend to emit a box regardless.
Expect that to hurt.

## Relevance to Jetson Orin

**Medium-high, as the detector only.** Swin-T is a plausible size for the
"detection may be slow" stage. But `TORCH_CUDA_ARCH_LIST` upstream stops at
**8.6**, and Orin is **SM 8.7** — that must be added or you get a kernel-launch
failure at runtime rather than a build error. It will not hit 10+ Hz on Orin
without serious work; that is the tracker's job.

## What to measure

- Zero-shot box AP on RefDrone val, **bucketed by object pixel area** — aggregate numbers hide small-object failure, which is the whole problem in aerial imagery.
- Behaviour on **no-target** expressions: does score thresholding ever produce a clean "nothing"?
- Attribute sensitivity: `"person"` vs `"person in a red shirt"` vs `"red shirt"` — does the attribute actually change the ranking?
- Latency at the real input resolution, Swin-T vs Swin-B.
- Prompt-format sensitivity (period-separated phrases vs a sentence).
