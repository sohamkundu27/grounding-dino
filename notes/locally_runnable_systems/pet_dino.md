# PET-DINO

**Primary paper:** PET-DINO: Unifying Visual Cues into Grounding DINO with
Prompt-Enriched Training (Fu et al., CVPR 2026 Highlight) —
`papers/locally_runnable_systems/05_pet_dino/2026_Fu_PET_DINO.pdf`
Project page: <https://fuweifuvtoo.github.io/pet-dino/>

## What it is
An extension of Grounding DINO that unifies **textual and visual prompts**:
alongside language, it accepts visual exemplars (e.g. example crops / points) via
"prompt-enriched training", improving detection when text alone is ambiguous.

## Type
Detector (Grounding DINO derivative), not a pipeline.

## Core architecture
Grounding DINO backbone + a training scheme that injects visual cues into the
prompt/query pathway so the model can be conditioned on visual exemplars as well
as text.

- **Image encoder:** Swin (Grounding DINO backbone; the published checkpoint is Swin-T).
- **Text encoder:** BERT (Grounding DINO text branch), plus a visual-prompt path.

## Open-vocabulary capability
Yes — retains Grounding DINO's open-set text detection and adds visual prompting.

## Local code availability
Yes — `third_party/pet_dino/` (fuweifuvtoo/PET_DINO, pinned `7830a46`).

## Local checkpoint availability
Partial — one public Swin-T checkpoint at time of writing (fewer options than base
Grounding DINO).

## API dependence
None — local.

## Segmentation capability
No (detection only).

## Video tracking capability
No native tracking.

## Relevance to aerial/drone detection
Potentially high: **visual prompting** is valuable when the target is hard to name
("that specific vehicle") — the operator can point/give an exemplar instead of, or
with, a phrase. Useful for one-shot target specification in aerial scenes.

## Relevance to Jetson deployment
Similar cost profile to Grounding DINO (same backbone family) plus the visual-cue
path; same export challenges. Only a Swin-T checkpoint is published, which is the
lighter end.

## Likely strengths
- Text **and** visual prompting → more flexible target specification.
- Recent (CVPR 2026 Highlight); builds directly on a well-understood base.

## Likely weaknesses
- Fewer public checkpoints; newer/less battle-tested tooling.
- Inherits Grounding DINO's real-time and export limitations.

## Unresolved research questions
- How much visual prompting improves single-target acquisition vs text alone in aerial imagery.
- Availability of larger backbones / deployment recipes.
- Interaction of visual prompts with a downstream tracker.
