# Grounded Segment Anything

## Overview

**A pipeline, not a new model.** Grounded SAM chains Grounding DINO (boxes from
text) into SAM (masks from boxes), producing open-vocabulary *segmentation*. No
new detector is trained; the contribution is the assembly.

```
text prompt ──► Grounding DINO ──► boxes ──► SAM ──► masks
```

- **Paper:** Ren et al. 2024, arXiv [2401.14159](https://arxiv.org/abs/2401.14159) → [`../papers/03_grounded_sam/`](../papers/03_grounded_sam/)
- **Official repo:** <https://github.com/IDEA-Research/Grounded-Segment-Anything>
- **Local source:** `third_party/grounded_sam/` @ `126abe6`

## Architecture

Two published models bolted together, plus demo glue (inpainting, RAM/Tag2Text
tagging, ChatGPT-assisted labelling — none of which this project needs).

- **Image encoder (detector):** Swin-T/B, via the bundled Grounding DINO
- **Text encoder:** BERT-base-uncased
- **Segmenter:** SAM — a ViT (ViT-B/L/H) image encoder + a lightweight prompt-conditioned mask decoder

## Inputs and outputs

- **In:** image + text prompt
- **Out:** boxes **and per-object binary masks**
- **No video tracking.** Every frame would be re-detected and re-segmented from scratch.

## Local availability

| | |
|---|---|
| Code | ✅ `third_party/grounded_sam/` |
| Weights | ✅ **through its components** — no weights of its own |
| Grounding DINO ckpt | `checkpoints/grounding_dino/groundingdino_swint_ogc.pth` |
| SAM ckpt | `checkpoints/sam/sam_vit_h_4b8939.pth` (2.4 GB), `sam_vit_b_01ec64.pth` (358 MB) |
| License | Apache-2.0 (repo, and both component weight sets) |

⚠️ The repo vendors its own copies of Grounding DINO and SAM. **Do not
re-download those weights** — point it at the shared paths above.

## Main strength

The cleanest reference implementation of the detect-then-segment pattern, and the
cheapest way to answer one specific question: **do precise masks localise the
target better than boxes?** In aerial imagery a box around a person contains
mostly ground; a mask does not.

## Main weakness

**No temporal component at all.** SAM v1 has no memory and no video mode, so
maintaining a target across frames means re-running the full ViT-H encoder every
frame — hopeless against a 10+ Hz budget on Orin. It also inherits Grounding
DINO's `MultiScaleDeformableAttention` compile requirement and export difficulty.

SAM v1 *does* have an official ONNX export, but **only for the lightweight mask
decoder** — the expensive ViT image encoder is not covered by it.

For this project, Grounded SAM is **superseded by Grounded SAM 2**, which does
everything it does plus tracking.

## Relevance to RefDrone

RefDrone is a **box**-level benchmark — its annotations are boxes, not masks. So
Grounded SAM cannot be scored natively on it. Its value is diagnostic: take the
same boxes and see whether the mask reveals the detector actually found the right
object versus a plausible-looking blob.

## Relevance to Jetson Orin

**Low.** No tracking, and ViT-H is far too heavy for the fast path. Keep it as a
reference and for the segmentation-quality question; do not build the Jetson
pipeline on it.

## What to measure

- **Does the mask change the answer?** For prompts where the detector is right, does the mask tighten localisation usefully — or is a box sufficient at aerial scale?
- Mask quality on **small** aerial objects (a 20 px person) — SAM was trained on much larger objects; expect degradation and quantify it.
- Per-frame latency of SAM ViT-H vs ViT-B, to size the segmentation overhead.
- Whether segmentation adds anything beyond what Grounded SAM 2 would give you anyway. If not, **drop this system** and move on — that is a perfectly good outcome for a one-day experiment.
