# Grounded Segment Anything (Grounded SAM)

**Primary paper:** Grounded SAM: Assembling Open-World Models for Diverse Visual
Tasks (Ren et al., arXiv 2024) —
`papers/locally_runnable_systems/03_grounded_sam/2024_Ren_Grounded_SAM.pdf`

## What it is
A **pipeline** that composes an open-set detector (Grounding DINO) with a
promptable segmenter (SAM): Grounding DINO turns a text prompt into boxes, and
SAM turns each box into a mask. Enables open-vocabulary *detect → segment*.

## Type
Pipeline / assembly of two foundation models — **not** a new base detector.

## Core architecture
Grounding DINO (open-set detection) → box prompts → SAM (promptable
segmentation). Extensible with other models (e.g. captioning, inpainting) but the
detect→segment core is what matters here.

- **Image encoder:** Swin (Grounding DINO) + ViT (SAM image encoder).
- **Text encoder:** BERT (via Grounding DINO). SAM itself takes geometric prompts,
  not text.

## Open-vocabulary capability
Yes — inherited entirely from Grounding DINO; SAM adds class-agnostic masks.

## Local code availability
Yes — `third_party/grounded_sam/`
(IDEA-Research/Grounded-Segment-Anything, pinned `126abe6`).

## Local checkpoint availability
Yes — reuses Grounding DINO weights + SAM weights (ViT-H default, ViT-B for edge);
no weights of its own.

## API dependence
None for the core local pipeline (optional extensions may call other services).

## Segmentation capability
Yes — this is the point. Produces instance masks for text-described objects.

## Video tracking capability
No native video tracking (that is Grounded SAM 2's domain via SAM 2).

## Relevance to aerial/drone detection
Useful when a **mask** (not just a box) is needed, e.g. precise target extent or
auto-labeling aerial data. Inherits Grounding DINO's aerial limitations.

## Relevance to Jetson deployment
Heavier than detection alone — two large models in series. The default SAM ViT-H
is expensive; SAM ViT-B is the edge option. Better as an offline/annotation tool
than a real-time on-device stage.

## Likely strengths
- Open-vocabulary segmentation with zero task-specific training.
- Excellent for auto-annotation / dataset bootstrapping.

## Likely weaknesses
- Two heavy models in series → high latency/memory; not real-time on-device.
- Error compounding: a missed/false detection propagates to segmentation.

## Unresolved research questions
- Lightest detector+SAM combo that keeps aerial mask quality acceptable.
- Value of masks vs boxes for the downstream tracking stage.
