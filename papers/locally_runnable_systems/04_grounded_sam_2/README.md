# 04 — Grounded SAM 2 (software integration, no standalone paper)

**Grounded SAM 2 is a software integration, not an independent academic paper.**
No standalone "Grounded SAM 2" peer-reviewed paper or arXiv preprint exists as of
this collection's date. To avoid fabricating a citation, **no PDF is placed in
this folder** — this reference documents the system and points to its academic
foundations and official repository instead.

## What it is

Grounded SAM 2 chains an open-set **detector** (Grounding DINO / Grounding DINO
1.5) with **SAM 2** to get open-vocabulary *detect → segment → track* across
images and video. It is the architecture this project is reaching for: run the
slow open-vocabulary detector once to acquire the target from a natural-language
prompt, then hand the box to SAM 2's memory-based tracker for the fast (10+ Hz)
stage.

## Academic foundations (the papers to actually read)

| Foundation | Where it lives in this repo |
|---|---|
| Grounding DINO (open-set detection) | [`papers/locally_runnable_systems/01_grounding_dino/2023_Liu_Grounding_DINO.pdf`](../01_grounding_dino/2023_Liu_Grounding_DINO.pdf) |
| SAM 2 (promptable image+video segmentation / tracking) | [`papers/core_foundations/2024_Ravi_SAM_2.pdf`](../../core_foundations/2024_Ravi_SAM_2.pdf) |
| Segment Anything (SAM 1, the predecessor) | [`papers/core_foundations/2023_Kirillov_Segment_Anything.pdf`](../../core_foundations/2023_Kirillov_Segment_Anything.pdf) |

A concrete applied example of a Grounded-SAM-2-style pipeline is catalogued under
[`papers/segmentation_integrations/2026_Korporaal_Colony_Grounded_SAM2.pdf`](../../segmentation_integrations/2026_Korporaal_Colony_Grounded_SAM2.pdf).

## Official repository documentation

- Repository: <https://github.com/IDEA-Research/Grounded-SAM-2>
- Pinned in this repo as a submodule at `third_party/grounded_sam_2/` (commit `b7a9c29`).
- Environment notes: [`environments/grounded_sam_2/README.md`](../../../environments/grounded_sam_2/README.md)

## Caveats

- Several Grounded SAM 2 demos import `dds_cloudapi_sdk` and require a **cloud API
  token** (they upload imagery to a third party). No token was obtained and no
  API was called. The **local** path uses a public Grounding DINO / SAM 2
  checkpoint instead.
- Grounding DINO 1.5 / 1.6 and DINO-X are API-only; there are no public local
  checkpoints for them.
