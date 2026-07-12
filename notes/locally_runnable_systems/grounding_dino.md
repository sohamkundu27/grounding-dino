# Grounding DINO 1.0

**Primary paper:** Grounding DINO: Marrying DINO with Grounded Pre-Training for
Open-Set Object Detection (Liu et al., arXiv 2023, ECCV 2024) —
`papers/locally_runnable_systems/01_grounding_dino/2023_Liu_Grounding_DINO.pdf`

## What it is
An open-set (open-vocabulary) object **detector**: given an image and a free-form
text prompt (category names or a referring phrase), it returns boxes for the
described objects without task-specific training.

## Type
Standalone detector (single model), not a pipeline.

## Core architecture
DINO/DETR-style transformer detector fused with language at three points — a
feature enhancer (cross-modality deformable attention between image and text),
language-guided query selection, and a cross-modality decoder. Trained with
grounded pre-training that aligns regions to phrases (contrastive box↔token).

- **Image encoder:** Swin Transformer (Swin-T for the light model, Swin-B/L for larger).
- **Text encoder:** BERT (from HuggingFace) producing token embeddings for the prompt.

## Open-vocabulary capability
Yes — the defining feature. Accepts arbitrary category strings or referring
expressions at inference; not limited to a fixed label set.

## Local code availability
Yes — `third_party/grounding_dino/` (IDEA-Research/GroundingDINO, pinned `856dde2`).
Requires compiling a custom CUDA deformable-attention op.

## Local checkpoint availability
Yes — public Swin-T and Swin-B weights (tracked in the drone project's
`manifests/checkpoints.json`; not committed here).

## API dependence
None for 1.0 — fully local. (Grounding DINO **1.5/1.6** are cloud-API only; see
`direct_extensions/`.)

## Segmentation capability
No — detection (boxes) only. Segmentation comes from pairing with SAM (see
Grounded SAM).

## Video tracking capability
No native tracking; per-frame detection only.

## Relevance to aerial/drone detection
High as the "acquire the target from language" stage: it can turn *"find the
person in the red shirt"* into a box with no retraining. Small-object and
top-down aerial performance is a known open question (see weaknesses).

## Relevance to Jetson deployment
Moderate. It is the slow open-vocabulary stage — acceptable if run once to
acquire, then handed to a fast tracker. The custom CUDA op and the BERT text
branch complicate ONNX/TensorRT export; the text branch is often frozen/exported
separately.

## Likely strengths
- Strong zero-shot / open-set detection from natural language.
- Mature code, multiple backbones, widely used foundation for the whole family.

## Likely weaknesses
- Heavy for real-time on-device use (transformer + BERT).
- Export friction (custom ops, dual-modality inputs).
- Aerial/top-down and very small objects are out of its training distribution.

## Unresolved research questions
- Best way to export/quantize for Jetson without losing open-vocabulary quality.
- How much aerial-domain fine-tuning is needed for small-object drone imagery.
- Prompt strategy (category list vs referring phrase) for reliable single-target acquisition.
