# MM-Grounding-DINO

## Overview

OpenMMLab's clean-room reimplementation of Grounding DINO, retrained on a much
larger and fully open data mixture. Same task, same interface — image + text
prompt → boxes — but better zero-shot numbers and, crucially, a **real
deployment story**. This is the strongest production-oriented detector candidate
in the set.

- **Paper:** Zhao et al. 2024, arXiv [2401.02361](https://arxiv.org/abs/2401.02361) → [`../papers/02_mm_grounding_dino/`](../papers/02_mm_grounding_dino/)
- **Official repo:** <https://github.com/open-mmlab/mmdetection>
- **Local source:** `third_party/mm_grounding_dino/` @ `cfd5d3a` (MMDetection v3.3.0)

> ⚠️ **There is no standalone MM-Grounding-DINO repository.** It lives *inside*
> MMDetection. The full mmdet repo is vendored so these paths resolve:
>
> | What | Path |
> |---|---|
> | Configs | `configs/mm_grounding_dino/` |
> | Model | `mmdet/models/detectors/grounding_dino.py` |
> | Usage docs | `configs/mm_grounding_dino/usage.md` |
> | Every checkpoint URL | `configs/mm_grounding_dino/metafile.yml` |

## Architecture

Architecturally the same as Grounding DINO — Swin backbone, BERT text encoder,
feature enhancer, language-guided query selection, cross-modality decoder. The
contribution is not the architecture; it is the **open, reproducible training
pipeline** and a far bigger pretraining mixture (Objects365, GoldG, GRIT-9M,
V3Det), plus a released Swin-L.

- **Image encoder:** Swin-T / Swin-B / Swin-L
- **Text encoder:** BERT-base-uncased
- **Framework:** MMDetection (mmengine + mmcv)

## Inputs and outputs

- **In:** image + text prompt
- **Out:** boxes + phrase scores. No masks, no track IDs.

## Local availability

| | |
|---|---|
| Code | ✅ `third_party/mm_grounding_dino/` (whole of mmdet) |
| Weights | ✅ `checkpoints/mm_grounding_dino/` — Swin-T ×2 (950 MB / 1.0 GB), Swin-B (1.1 GB), Swin-L (1.4 GB) |
| License | Apache-2.0, code and weights |

The Swin-T `..._obj365_goldg_v3det_...pth` is also **PET-DINO's required
initialisation** — shared, not duplicated.

## Main strength

**It is the only one of the five with a first-party deployment path.**
[MMDeploy](https://github.com/open-mmlab/mmdeploy) supports ONNX Runtime and
TensorRT backends for mmdet models and ships a **TensorRT plugin for
`MultiScaleDeformableAttention`** — precisely the op that makes exporting vanilla
Grounding DINO so painful. It also beats the original on zero-shot benchmarks and
offers a Swin-T for the size-constrained case.

MMDeploy is deliberately **not cloned** — it is not needed until the export step.

## Main weakness

**The OpenMMLab dependency stack**, which is the most brittle here.

- `mmdet/__init__.py` hard-asserts `mmcv >= 2.0.0rc4, < 2.2.0`.
- `mmcv` ships compiled CUDA ops and must match your exact torch/CUDA build.
- **There is no prebuilt `mmcv` wheel for Jetson/aarch64** — it must be compiled
  from source (~1 h, fragile). This partly offsets the MMDeploy advantage.
- The LVIS API on the eval path breaks on `numpy >= 1.24`.

Also: MMDeploy's *grounding* support is less exercised than its plain-detector
support. Expect work — just less of it than with vanilla Grounding DINO.

## Relevance to RefDrone

The best-accuracy zero-shot baseline to try on RefDrone, and the fairest
comparison against Grounding DINO since the interface is identical. Same
no-target caveat applies: no explicit mechanism for "nothing matches".

## Relevance to Jetson Orin

**Highest of the detectors** — same architecture family, better accuracy, a real
TensorRT route, and a Swin-T variant. The realistic plan is *not* to install mmdet
on the Orin, but to export to TensorRT on the workstation and run the engine from
C++, so mmcv's aarch64 problem mostly evaporates at deployment time (it still
bites during export).

## What to measure

- Zero-shot AP on RefDrone val vs Grounding DINO, **same prompts, same thresholds** — is the reported accuracy gain real on *aerial* data?
- Small-object AP bucketed by pixel area.
- Swin-T vs Swin-B vs Swin-L: accuracy-per-millisecond, to find the knee.
- **Export feasibility early:** try MMDeploy → ONNX → TensorRT on Swin-T *before* investing in accuracy tuning. If the deformable-attention plugin works, this model wins on deployment grounds almost regardless of the accuracy delta.
- Whether text embeddings can be precomputed offline (fixed prompt set) to drop BERT from the runtime graph.
