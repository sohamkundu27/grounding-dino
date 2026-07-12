# MM-Grounding-DINO

**Primary paper:** An Open and Comprehensive Pipeline for Unified Object
Grounding and Detection (Zhao et al., arXiv 2024) —
`papers/locally_runnable_systems/02_mm_grounding_dino/2024_Zhao_MM_Grounding_DINO.pdf`

## What it is
An open-source, from-scratch reimplementation of Grounding DINO built inside
OpenMMLab's **MMDetection**, with an expanded and fully documented training
pipeline (more grounding/detection datasets, reproducible configs).

## Type
Detector (same architecture family as Grounding DINO) delivered as part of a
framework (MMDetection). Not a segmentation/tracking pipeline.

## Core architecture
Grounding DINO architecture (feature enhancer, language-guided query selection,
cross-modality decoder) reimplemented in `mmdet`. Its contribution is the *open
pipeline* — training recipe, datasets, and configs — more than a new model design.

- **Image encoder:** Swin Transformer (Swin-T/B/L configs).
- **Text encoder:** BERT.

## Open-vocabulary capability
Yes — same open-set detection capability as Grounding DINO, with reported gains
from broader pre-training data in the paper.

## Local code availability
Yes — lives inside `third_party/mm_grounding_dino/` (open-mmlab/mmdetection,
pinned `cfd5d3a`) at `configs/mm_grounding_dino/` and
`mmdet/models/detectors/grounding_dino.py`. The whole of MMDetection is the repo.

## Local checkpoint availability
Yes — multiple public Swin-T/B/L weights published via MMDetection.

## API dependence
None — fully local.

## Segmentation capability
No (detection only). Could be composed with SAM as in Grounded SAM.

## Video tracking capability
No native tracking.

## Relevance to aerial/drone detection
High and arguably the **best first-party candidate**: MMDetection provides
mature training/eval tooling for fine-tuning on aerial datasets, and it is the
family member with the clearest first-party TensorRT/deployment path.

## Relevance to Jetson deployment
Best-supported of the family for deployment — MMDetection/MMDeploy provide a
documented export route. Still constrained by the transformer + BERT cost.

## Likely strengths
- Reproducible, well-documented training; easy fine-tuning on custom domains.
- Broad pre-training → strong zero-shot; first-party deployment tooling.

## Likely weaknesses
- Large dependency surface (all of MMDetection / OpenMMLab).
- Same fundamental detector cost as Grounding DINO for on-device real-time.

## Unresolved research questions
- Quantified aerial small-object gains from fine-tuning vs base Grounding DINO.
- End-to-end MMDeploy → TensorRT quality/latency on Jetson AGX Orin.
