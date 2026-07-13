# Production Candidates — Five Locally Runnable Systems

The five open-vocabulary systems selected for evaluation on the drone
target-finding project, in one place.

**This folder holds papers and reference documentation only.** No model code, no
checkpoints, no datasets, no environments, no outputs, no ONNX/TensorRT/Docker
assets. Those live elsewhere in the repository:

| What | Where |
|---|---|
| Upstream source (pinned submodules) | [`../third_party/`](../third_party/) |
| Model weights (~17 GB) | `../checkpoints/` |
| RefDrone + VisDrone | `../datasets/` |
| Dependency notes per system | [`../environments/`](../environments/) |
| Wider literature collection | [`../papers/`](../papers/) |

The PDFs here are **relative symlinks** to the canonical copies under
[`../papers/`](../papers/) — one payload each, no duplicates. Verify with
`sha256sum -c production/metadata/checksums.sha256`.

> Nothing has been run. No inference, no training, no benchmarks. Everything below
> comes from reading papers, source, and configs. **No benchmark numbers are
> quoted, because none have been measured.**

## Summary

| # | System | Type | Primary paper | Official GitHub | Text | Visual | Seg | Track | Code | Weights | API | Planned role |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Grounding DINO 1.0** | detector | [Liu et al. 2023](https://arxiv.org/abs/2303.05499) · [local](papers/01_grounding_dino/) | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | Baseline detector |
| 2 | **MM-Grounding-DINO** | detector | [Zhao et al. 2024](https://arxiv.org/abs/2401.02361) · [local](papers/02_mm_grounding_dino/) | [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | **Strongest production detector candidate** |
| 3 | **Grounded Segment Anything** | segmentation pipeline | [Ren et al. 2024](https://arxiv.org/abs/2401.14159) · [local](papers/03_grounded_sam/) | [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ ¹ | ❌ | Do masks improve localisation? |
| 4 | **Grounded SAM 2** | tracking pipeline | [Ravi et al. 2024 (SAM 2)](https://arxiv.org/abs/2408.00714) ² · [local](papers/04_grounded_sam_2/) | [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ ¹ | ❌ ³ | **Full acquisition + tracking baseline** |
| 5 | **PET-DINO** | detector | [Fu et al. 2026](https://arxiv.org/abs/2604.00503) · [local](papers/05_pet_dino/) | [fuweifuvtoo/PET_DINO](https://github.com/fuweifuvtoo/PET_DINO) | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Prompt-enriched detector comparison |

¹ Through its component models — these pipelines have no weights of their own.
² **No standalone Grounded SAM 2 paper exists.** SAM 2 is its primary academic
reference; see [`papers/04_grounded_sam_2/NO_STANDALONE_PAPER.md`](papers/04_grounded_sam_2/NO_STANDALONE_PAPER.md).
³ **For the local path only.** The same repo also ships API-backed Grounding DINO
1.5 / 1.6 / DINO-X demos, which have no public local weights — see
[Important distinction](#important-distinction).

Machine-readable: [`metadata/systems.json`](metadata/systems.json) ·
[`metadata/systems.csv`](metadata/systems.csv)

## The five systems

### 1. Grounding DINO 1.0 — *detector*

The original open-vocabulary detector everything else here descends from. It
fuses a Swin image backbone with a BERT text encoder early — in the neck, in
query initialisation, and in the head — so a free-form phrase can steer detection
directly, with no fixed class list and no task-specific training.

- **GitHub:** <https://github.com/IDEA-Research/GroundingDINO>
- **Paper:** [arXiv 2303.05499](https://arxiv.org/abs/2303.05499) · [`papers/01_grounding_dino/`](papers/01_grounding_dino/)
- **Kind:** standalone detector
- **Output:** boxes + phrase scores. No masks, no track IDs.
- **Supporting reading:** [DINO](https://arxiv.org/abs/2203.03605), [BERT](https://arxiv.org/abs/1810.04805), [Swin Transformer](https://arxiv.org/abs/2103.14030) — in [`../papers/core_foundations/`](../papers/core_foundations/), referenced rather than duplicated
- **Why test it:** it is the reference point. Every other system is a delta against it, so its behaviour on aerial imagery sets the baseline the others must beat.
- **Notes:** [`notes/grounding_dino.md`](notes/grounding_dino.md)

### 2. MM-Grounding-DINO — *detector*

OpenMMLab's clean-room reimplementation, retrained on a much larger open data
mixture (Objects365, GoldG, GRIT-9M, V3Det). Same interface as Grounding DINO,
better zero-shot accuracy, and — decisively — a real deployment story.

- **GitHub:** <https://github.com/open-mmlab/mmdetection> ⚠️ **not a standalone repo** — the model lives inside MMDetection at `configs/mm_grounding_dino/`
- **Paper:** [arXiv 2401.02361](https://arxiv.org/abs/2401.02361) · [`papers/02_mm_grounding_dino/`](papers/02_mm_grounding_dino/)
- **Kind:** standalone detector
- **Output:** boxes + phrase scores
- **Why test it:** it is the only one of the five with a **first-party TensorRT path** ([MMDeploy](https://github.com/open-mmlab/mmdeploy), which ships a plugin for the deformable-attention op that blocks exporting vanilla Grounding DINO). If its accuracy holds on aerial data, it is the most likely thing to actually reach the Orin.
- **Notes:** [`notes/mm_grounding_dino.md`](notes/mm_grounding_dino.md)

### 3. Grounded Segment Anything — *segmentation pipeline*

**A pipeline, not a new model.** Grounding DINO produces boxes from text; SAM
turns those boxes into masks. Nothing new is trained — the contribution is the
assembly.

- **GitHub:** <https://github.com/IDEA-Research/Grounded-Segment-Anything>
- **Paper:** [arXiv 2401.14159](https://arxiv.org/abs/2401.14159) · [`papers/03_grounded_sam/`](papers/03_grounded_sam/)
- **Foundations:** [Grounding DINO](https://arxiv.org/abs/2303.05499) + [Segment Anything](https://arxiv.org/abs/2304.02643)
- **Kind:** pipeline (detector → segmenter). **No weights of its own** — it reuses the shared Grounding DINO and SAM checkpoints.
- **Output:** boxes **and per-object masks**. No tracking.
- **Why test it:** one question only — **do precise masks localise an aerial target better than a box?** A box around a person seen from a drone is mostly ground; a mask is not. If the answer is "not usefully", drop it and move on.
- **Notes:** [`notes/grounded_sam.md`](notes/grounded_sam.md)

### 4. Grounded SAM 2 — *tracking pipeline*

**A pipeline, and the closest system-level match to what this project is actually
building.** Grounding DINO acquires the target from language; SAM 2 segments and
**tracks it across frames** using a memory bank. Acquire once with a slow
detector, then track cheaply — precisely the architecture the drone brief
describes.

- **GitHub:** <https://github.com/IDEA-Research/Grounded-SAM-2>
- **Primary paper:** [SAM 2 — arXiv 2408.00714](https://arxiv.org/abs/2408.00714) · [`papers/04_grounded_sam_2/`](papers/04_grounded_sam_2/)
- ⚠️ **No standalone Grounded SAM 2 paper exists.** It is a software integration. Rather than fabricate a citation, that absence is documented in [`papers/04_grounded_sam_2/NO_STANDALONE_PAPER.md`](papers/04_grounded_sam_2/NO_STANDALONE_PAPER.md), and the official repo README is the authoritative description of the integration itself.
- **Kind:** pipeline (detector → segmenter → tracker)
- **Output:** masks, boxes, and **persistent track IDs**. The only system here that produces track IDs.
- **Why test it:** it is the only one that closes the loop from natural language to a sustained track. `sam2.1_hiera_tiny` is 149 MB — genuinely plausible for real-time tracking on Orin.
- **Notes:** [`notes/grounded_sam_2.md`](notes/grounded_sam_2.md)

### 5. PET-DINO — *detector*

A Grounding-DINO-derived detector (an MMDetection fork) that takes **text prompts
*and* visual prompts**. The only system of the five that does both: instead of
describing the target, you can point at an example.

- **GitHub:** <https://github.com/fuweifuvtoo/PET_DINO> · **Project page:** <https://fuweifuvtoo.github.io/pet-dino/>
- **Paper:** [arXiv 2604.00503](https://arxiv.org/abs/2604.00503) — **CVPR 2026 (Highlight)** · [`papers/05_pet_dino/`](papers/05_pet_dino/)
- **Kind:** standalone detector
- **Output:** boxes + scores. No masks, no track IDs.
- **Why test it:** visual prompting is a natural fit for a drone operator UI — *"track **that**"* is often more precise than any sentence, especially for targets with no clean verbal description. Its precomputed-embedding path is also deployment-friendly.
- **Caveat:** only a **Swin-T** checkpoint is published; the in-repo Swin-L config has no released weight. It is a brand-new paper with no export tooling.
- **Notes:** [`notes/pet_dino.md`](notes/pet_dino.md)

## Important distinction

These five are **not five of the same thing.** Three are models; two are
assemblies of models. Comparing them as peers will produce nonsense.

**Detector systems — new models, trained weights of their own:**

| | |
|---|---|
| **Grounding DINO 1.0** | the original |
| **MM-Grounding-DINO** | retrained, better data, real deployment path |
| **PET-DINO** | adds visual prompts |

All three take an image + a prompt and return **boxes**. They are directly
comparable to each other: same input, same output, same metric.

**Multi-model pipelines — no new model, no weights of their own:**

| | |
|---|---|
| **Grounded SAM** | Grounding DINO → SAM. **Adds segmentation.** |
| **Grounded SAM 2** | Grounding DINO → SAM 2. **Adds segmentation *and* video tracking.** |

Both reuse a detector from the list above and bolt a segmenter onto it. Their
detection quality is *inherited*, not improved — if the detector misses the
target, no amount of segmentation recovers it.

**What this means in practice:**

- A pipeline's detection accuracy is whatever its underlying detector's is. **Do not "compare" Grounded SAM against Grounding DINO on detection** — they share the same detector; you would be measuring noise.
- The pipelines are worth testing for what they **add**: masks (Grounded SAM) and track IDs (Grounded SAM 2). Those are the only axes on which they can win.
- Choosing a detector and choosing a pipeline are **separate decisions**. The likely end state is *the best detector* feeding *the Grounded SAM 2 tracker* — which is not any one of these five off the shelf, but a recombination of them.

```
         DETECTORS                          PIPELINES
  ┌──────────────────────┐        ┌──────────────────────────────┐
  │ Grounding DINO 1.0   │        │ Grounded SAM                 │
  │ MM-Grounding-DINO    │ ──────►│   detector → SAM   (masks)   │
  │ PET-DINO             │        │                              │
  │                      │        │ Grounded SAM 2               │
  │ boxes from text      │ ──────►│   detector → SAM 2 (masks +  │
  └──────────────────────┘        │              track IDs)      │
                                  └──────────────────────────────┘
```

## Recommended testing order

1. **Grounding DINO 1.0**
2. **MM-Grounding-DINO**
3. **PET-DINO**
4. **Grounded Segment Anything**
5. **Grounded SAM 2**

The logic is cheapest-and-most-decisive first.

**Steps 1–3 — compare the box-only detectors.** Same images, same prompts, same
thresholds, one variable. This is the comparison that actually matters, because
whichever detector wins is the one both pipelines will end up wrapping. It is
also the cheapest: no segmentation, no video, no tracking state. Answer the
fundamental question here — *can any of these resolve "the person in the red
shirt" from a drone's altitude at all?* If none can, nothing downstream saves the
project, and you have found that out in days rather than months.

**Step 4 — measure the segmentation overhead.** Add SAM to the winning detector
and ask one narrow question: do masks localise an aerial target meaningfully
better than boxes, and what does that cost per frame? A perfectly good outcome is
"no, and it costs 400 ms" — that removes a whole branch from the plan.

**Step 5 — test the full detector-plus-tracker pipeline.** Only now bring in
video and state. This is the most expensive to set up and the hardest to
interpret, so it goes last — but it is also where the **10+ Hz requirement** is
either met or not. Everything before it is preparation for this measurement.

> ⚠️ **Steps 1–4 can be scored on RefDrone. Step 5 cannot.** RefDrone is
> image-based — no temporal annotations, no track IDs, no video. Evaluating
> detection-plus-tracking needs a separate video source (VisDrone-VID/-MOT, or
> in-house drone footage). Budget for that; a strong RefDrone result says nothing
> about whether the tracker holds a target.
