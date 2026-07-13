# Grounded SAM 2 — no standalone paper exists

**Grounded SAM 2 is a software integration, not an independent publication.**

There is no peer-reviewed paper and no arXiv preprint titled "Grounded SAM 2".
No citation is invented here. Instead:

- **Primary academic paper for this system:** the **SAM 2** paper
  (`2024_Ravi_SAM_2.pdf` in this folder) — that is the actual research
  contribution being used.
- **Authoritative description of the integration itself:** the official
  repository README, <https://github.com/IDEA-Research/Grounded-SAM-2>.

## What the system actually is

A chaining of two published models:

```
text prompt ──► Grounding DINO 1.0  ──► boxes ──► SAM 2 ──► masks + tracked IDs
               (local checkpoint)                (memory bank, video)
```

Grounding DINO does open-vocabulary target *acquisition* from language; SAM 2
does *segmentation and tracking* across frames. Neither is new here — the
contribution is the wiring.

## Its academic foundations

| Foundation | Paper | Where in this repo |
|---|---|---|
| **SAM 2** (primary) | Ravi et al. 2024, arXiv [2408.00714](https://arxiv.org/abs/2408.00714) | `2024_Ravi_SAM_2.pdf` (this folder) |
| Grounding DINO | Liu et al. 2023, arXiv [2303.05499](https://arxiv.org/abs/2303.05499) | [`../01_grounding_dino/`](../01_grounding_dino/) |
| Segment Anything (SAM 1) | Kirillov et al. 2023, arXiv [2304.02643](https://arxiv.org/abs/2304.02643) | `papers/core_foundations/2023_Kirillov_Segment_Anything.pdf` |

## ⚠️ Local vs API — this repo only uses the local path

The upstream repository ships demos for **both** local and cloud-hosted
detectors. Only the local ones are usable for this project.

**Local (weights on disk, nothing leaves the machine):**
Grounding DINO **1.0** + SAM 2 — e.g.
`grounded_sam2_local_demo.py`,
`grounded_sam2_tracking_demo_custom_video_input_gd1.0_local_model.py`.

**API-backed (NOT local):**
Grounding DINO **1.5**, Grounding DINO **1.6**, and **DINO-X** demos import
`dds_cloudapi_sdk` and require an `API_TOKEN`. These models run on the
DeepDataSpace **cloud** — using them uploads imagery to a third party, and
**no public local checkpoints exist for them**. They cannot be part of a Jetson
deployment. No token was obtained and no API was called during this collection.

See [`../../notes/grounded_sam_2.md`](../../notes/grounded_sam_2.md).
