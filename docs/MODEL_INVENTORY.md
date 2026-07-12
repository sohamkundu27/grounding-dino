# Model Inventory

Five open-source systems collected for the drone open-vocabulary target-finding
project, plus one dataset-associated reference model (NGDINO).

Nothing in this repository has been run. All statements below come from reading
source, configs, and dependency files.

## Comparison

| | **Grounding DINO** | **MM-Grounding-DINO** | **Grounded-SAM** | **Grounded-SAM-2** | **PET-DINO** | *NGDINO (ref.)* |
|---|---|---|---|---|---|---|
| **Official repo** | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) | [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) | [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | [fuweifuvtoo/PET_DINO](https://github.com/fuweifuvtoo/PET_DINO) | [sunzc-sunny/refdrone](https://github.com/sunzc-sunny/refdrone) |
| **Commit** | `856dde2` | `cfd5d3a` | `126abe6` | `b7a9c29` | `7830a46` | `86314ec` |
| **Local source** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Local weights** | ✅ | ✅ | ✅ (shared) | ✅ | ✅ | ✅ |
| **Checkpoints** | `groundingdino_swint_ogc`, `groundingdino_swinb_cogcoor` | 4× `grounding_dino_swin-{t,t,b,l}_pretrain_*` | *reuses* GD + SAM | `sam2.1_hiera_{t,s,b+,l}` | `pet_dino_swin-t_8xb4_12e_obj365` | `NGDINO_T`, `NGDINO_B` |
| **Sizes** | 662 MB / 895 MB | 950 MB – 1.4 GB | — | 149 MB – 857 MB | 2.2 GB | 2.0 GB / 2.6 GB |
| **Backbone** | Swin-T / Swin-B | Swin-T / B / L | Swin-T + ViT-H | Swin-T + Hiera | Swin-T | Swin-T / Swin-B |
| **Text encoder** | BERT-base | BERT-base | BERT-base | BERT-base | BERT-base | BERT-base |
| **Detector / pipeline** | detector | detector | detector→segmenter | detector→**tracker** | detector | detector + count branch |
| **Text prompts** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Visual prompts** | ❌ | ❌ | ❌ | ⚠️ box/point → SAM only | ✅ **AFVPG** | ❌ |
| **Video tracking** | ❌ | ❌ | ❌ | ✅ **SAM 2 memory** | ❌ | ❌ |
| **Depends on an API** | ❌ | ❌ | ❌ | ⚠️ **partly — see below** | ❌ | ❌ |
| **License (code)** | Apache-2.0 | Apache-2.0 | Apache-2.0 | Apache-2.0 | Apache-2.0 | CC BY 4.0 |
| **License (weights)** | Apache-2.0 | Apache-2.0 | Apache-2.0 | Apache-2.0 | Apache-2.0 | ⚠️ VisDrone-derived → research-only |
| **Role in project** | baseline detector | **primary detector candidate** | reference pattern | **primary pipeline** | visual-prompt research | aerial-REC baseline |
| **Source status** | cloned | cloned | cloned | cloned | cloned, **verified** | cloned |
| **Checkpoint status** | downloaded | downloaded | shared, none new | downloaded | downloaded | downloaded |
| **ONNX/TRT difficulty** | HIGH | **MEDIUM** (MMDeploy) | HIGH | MIXED | HIGH | HIGH |
| **Jetson relevance** | MEDIUM-HIGH | **HIGH** | LOW | **HIGHEST** | MEDIUM | LOW (eval only) |

## Notes per system

### Grounding DINO (original)
The reference implementation everything else descends from. Swin-T and Swin-B
weights are both public GitHub-release assets. No official ONNX/TensorRT
material exists in the repo — I grepped for it. The
`MultiScaleDeformableAttention` CUDA op must be compiled and has no native ONNX
equivalent. Use as the accuracy/behaviour baseline.

### MM-Grounding-DINO
Not a separate repo — it lives **inside MMDetection** at
`configs/mm_grounding_dino/` + `mmdet/models/detectors/grounding_dino.py`. Full
mmdet v3.3.0 is vendored so those paths resolve. Beats the original Grounding
DINO on zero-shot benchmarks and is the **only one of the five with a
first-party TensorRT path** (MMDeploy). Cost: the OpenMMLab dependency stack.

### Grounded-SAM
Grounding DINO + SAM v1 = open-vocabulary *segmentation*. Bundles its own copies
of both models; **do not re-download those weights** — point it at
`checkpoints/grounding_dino/` and `checkpoints/sam/`. No video tracking, so it
cannot meet the 10+ Hz requirement. Kept for the paper and as the canonical
detect-then-segment reference.

### Grounded-SAM-2 — the one that matches the project shape
Grounding DINO 1.0 (**local**) + SAM 2 (**video tracking with a memory bank**).
This is exactly the brief's architecture: run the slow open-vocabulary detector
once to acquire the target from the prompt, then hand the box to a fast tracker.
`sam2.1_hiera_tiny` (149 MB) is the realistic 10+ Hz Orin candidate.

> ⚠️ **API-backed components — not local, not usable for deployment.**
> Six demos in this repo (`*_gd1.5_*`, `*_dinox_*`) import `dds_cloudapi_sdk`
> and require an `API_TOKEN`. **Grounding DINO 1.5, Grounding DINO 1.6 and
> DINO-X have no public local checkpoints** — they are a hosted cloud service
> (DeepDataSpace). Using them means uploading your imagery to a third party.
> No token was obtained, no API was called, and no image was sent anywhere
> during this setup. **Do not design the Jetson pipeline around them.** The
> local GD-1.0 demos are listed in `environments/grounded_sam_2/README.md`.

### PET-DINO
CVPR 2026 Highlight; an mmdet fork adding **visual prompts** alongside text —
the only system here that does both. Authorship was verified rather than assumed:
author project page (`fuweifuvtoo.github.io/pet-dino`) → GitHub repo → HF org
`fuweifu`, all consistent with corresponding author Weifu Fu (Tencent YouTu).
No unofficial reimplementation was substituted.

Only **one** checkpoint is published (Swin-T). A Swin-L config exists in-repo
with no released weight — don't plan around it. Strategically interesting for a
drone operator UI ("track *that*", pointed at rather than described), but too new
and too undeployed for the critical path.

### NGDINO (RefDrone reference model — *not* one of the five)
Grounding DINO plus an explicit object-**count** branch, built to handle
RefDrone's multi-target and no-target expressions. Stored under
`checkpoints/refdrone_ngdino/` as a dataset-associated baseline. It is trained on
VisDrone imagery and therefore inherits VisDrone's **academic/non-commercial**
restriction — see `docs/LICENSES.md`. Useful as the aerial-REC yardstick to beat;
not a deployment candidate.

## Recommended reading order

1. `environments/grounded_sam_2/README.md` — the target architecture.
2. `environments/mm_grounding_dino/README.md` — the strongest detector + the only TRT story.
3. `docs/DATASET_INVENTORY.md` — what RefDrone can and cannot tell you (it is **not** a tracking benchmark).
4. `docs/NEXT_STEPS.md`.
