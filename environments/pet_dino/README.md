# Environment — PET-DINO

**Nothing here has been installed or run.** Notes from reading the upstream
dependency files, vendored under `upstream/`.

Source: `third_party/pet_dino/` @ `7830a462f9a320f34293cce0aabdda1256d9dc15`
CVPR 2026 **Highlight**. Apache-2.0.

## What it is

A **fork of MMDetection** that extends MM-Grounding-DINO with visual prompting.
Everything in `environments/mm_grounding_dino/README.md` applies — same mmcv
bounds, same mmengine bounds, same compiled-mmcv problem on Jetson. Read that
file first.

The distinguishing feature: **PET-DINO takes text prompts *and* visual prompts**
(Alignment-Friendly Visual Prompt Generation). It is the only one of the five
systems that does.

## Upstream expectations

| | |
|---|---|
| Python | 3.10 (follows mmdet `get_started`) |
| PyTorch | 2.x |
| mmcv | `>=2.0.0rc4, <2.2.0` (identical `mminstall.txt` to mmdet) |
| mmengine | `>=0.7.1, <1.0.0` |
| Extra | `requirements/multimodal.txt`, plus `emoji`, `ddd-dataset`, `lvis-api` |

**numpy must be `==1.23`.** The README says so explicitly: the LVIS third-party
library does not support numpy ≥ 1.24. This will fight with Grounded-SAM-2,
which wants `numpy>=1.24.4`. Separate environments, non-negotiable.

## Pretrained weights it expects

PET-DINO is *initialised from* MM-Grounding-DINO, so it needs those weights too:

| Needed under `pretrained/` | Already collected? |
|---|---|
| MM-GDINO Swin-T `..._obj365_goldg_v3det_...pth` | **yes** → `checkpoints/mm_grounding_dino/` |
| MM-GDINO Swin-L `..._obj365_goldg-34dcdc53.pth` | no (Swin-L `pretrain_all` collected instead) |
| Swin-T / Swin-L ImageNet backbones | no — only needed for *training*, which is out of scope |
| `bert-base-uncased` | no — fetched by `transformers` on first run |

Symlink rather than copy when you get there.

## Released checkpoint — only one

`pet_dino_swin-t_8xb4_12e_obj365.pth` (2.3 GB) → `checkpoints/pet_dino/`
from <https://huggingface.co/fuweifu/PET-DINO> (author-owned, ungated).

The repo also ships `configs/pet_dino/pet_dino_swin-l_8xb4_12e_obj365.py`, but
**no Swin-L weight is published**. Do not plan around a Swin-L PET-DINO.

## Custom CUDA operators — YES (via mmcv), compilation required

Same as MM-Grounding-DINO. No `nvcc` compilation in *this* repo beyond mmdet's
own `setup.py`, but mmcv's compiled ops are a hard dependency.

## Inference entry points

```bash
# Text prompt
python scripts/image_demo.py images/animals.png \
    configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py \
    --weights $CKPT --texts 'zebra. giraffe. bird' -c

# Visual prompt (bounding boxes)
python scripts/image_demo.py images/animals.png \
    configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py \
    --weights $CKPT --prompt_type 'Visual' \
    --prompt_bboxes '[[1291.6, 679.9, 1536.8, 840.0]]' --prompt_bboxes_labels '[30]'

# Visual prompt (precomputed embedding) -- note --extract-visual-embedding
```

The embedding-extraction mode is interesting for the drone use case: you can
compute a visual prompt embedding **once** and reuse it, which decouples the
expensive prompt encoding from the per-frame path.

## ONNX / TensorRT difficulty — HIGH

Inherits Grounding DINO's deformable attention *plus* an extra visual-prompt
branch that upstream has never exported. MMDeploy may cover the mmdet backbone,
but the AFVPG module is novel and will not have plugin support. Assume manual
work. **Do not put this on the critical path to Jetson.**

## Jetson Orin relevance — MEDIUM (high research value, low deployment readiness)

Strategically interesting: visual prompting ("track *that* object", pointed at
rather than described) is a natural fit for a drone operator UI, and the
precomputed-embedding path is deployment-friendly. But it is a brand-new CVPR
paper with one checkpoint, no deployment tooling, and an mmdet dependency stack.
**Evaluate it on the workstation; do not build the Jetson pipeline on it yet.**
