# Grounding DINO — Literature Collection & Open-Vocabulary Detection/Tracking Research

A curated, verified collection of the **Grounding DINO** paper family — the
papers, metadata, notes, and tooling needed to understand and compare the
ecosystem — alongside the source, dependency definitions, and inventories for
five open-vocabulary detection/tracking systems.

**Target application.** A drone-based open-vocabulary target-finding system:
given *"find the person wearing a red shirt"*, locate the target from natural
language with no task-specific training. Detection may be slow; the subsequent
**tracking must sustain 10+ Hz**. Eventual platform is an NVIDIA Jetson AGX Orin
(ONNX → TensorRT → C++, Shield AI Hivemind SDK). **Deployment is not part of this
repository.**

> **Nothing here has been executed.** No inference, no tests, no benchmarks, no
> training, no ONNX/TensorRT export, no dependency installation, no checkpoint or
> dataset download. Papers come only from official sources (arXiv / CVF / official
> publishers) and are `%PDF`-validated with SHA-256. Upstream repositories are
> unmodified.

## Repository purpose

1. **Literature collection** (`papers/`, `metadata/`, `notes/`): a curated, one-
   paper-per-file library of the Grounding DINO family, verified and organized so
   the ecosystem can be understood and compared *before* deciding what to run.
2. **Systems research collection** (`docs/`, `manifests/`, `environments/`,
   `third_party/`, `datasets/`): inventories, dependency notes and pinned upstream
   source for five locally runnable systems — for the drone application above.

## How the papers are organized

Each paper is stored in **exactly one** folder (no duplicate PDFs; relationships
are expressed in metadata). Filenames are `YYYY_FirstAuthor_Short_Title.pdf` using
the *verified* year. Selection and verification rules are in
[`docs/paper_selection_criteria.md`](docs/paper_selection_criteria.md); every move
and rename is logged in [`docs/organization_log.md`](docs/organization_log.md).

```
papers/
├── locally_runnable_systems/{01_grounding_dino … 05_pet_dino}/   the five systems' primary papers
├── core_foundations/         DETR, DINO, BERT, Swin, SAM, SAM 2
├── direct_extensions/        Grounding DINO 1.5, DINO-X
├── tracking_and_video/       VideoGrounding-DINO, Grounding DINO in Videos
├── segmentation_integrations/ annotation & Grounded-SAM(-2) pipelines
├── deployment_and_efficiency/ Dynamic-DINO
├── domain_adaptations/       ultrasound, agriculture, 3D-CT, RefDrone (aerial)
├── application_papers/       applied integrations
└── unverified_or_pending/    verified metadata, no open-access PDF
```

## The five locally runnable systems

Their **primary** papers live under
[`papers/locally_runnable_systems/`](papers/locally_runnable_systems/); shared
**foundations** live in [`papers/core_foundations/`](papers/core_foundations/) so
they are never duplicated. Per-system notes are in
[`notes/locally_runnable_systems/`](notes/locally_runnable_systems/).

| # | System | Folder | Notes |
|---|---|---|---|
| 01 | Grounding DINO 1.0 | [`01_grounding_dino/`](papers/locally_runnable_systems/01_grounding_dino/) | [notes](notes/locally_runnable_systems/grounding_dino.md) |
| 02 | MM-Grounding-DINO | [`02_mm_grounding_dino/`](papers/locally_runnable_systems/02_mm_grounding_dino/) | [notes](notes/locally_runnable_systems/mm_grounding_dino.md) |
| 03 | Grounded Segment Anything | [`03_grounded_sam/`](papers/locally_runnable_systems/03_grounded_sam/) | [notes](notes/locally_runnable_systems/grounded_sam.md) |
| 04 | Grounded SAM 2 | [`04_grounded_sam_2/`](papers/locally_runnable_systems/04_grounded_sam_2/) | [notes](notes/locally_runnable_systems/grounded_sam_2.md) |
| 05 | PET-DINO | [`05_pet_dino/`](papers/locally_runnable_systems/05_pet_dino/) | [notes](notes/locally_runnable_systems/pet_dino.md) |

### Primary papers

<!-- BEGIN:PRIMARY_TABLE -->
| System | Primary paper | Year | Venue | Source | Local PDF |
|---|---|---|---|---|---|
| 01 Grounding DINO | Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection | 2023 | ECCV 2024 (arXiv 2023) | [2303.05499](https://arxiv.org/abs/2303.05499) | `papers/locally_runnable_systems/01_grounding_dino/2023_Liu_Grounding_DINO.pdf` |
| 02 MM-Grounding-DINO | An Open and Comprehensive Pipeline for Unified Object Grounding and Detection | 2024 | arXiv (OpenMMLab / MMDetection technical report) | [2401.02361](https://arxiv.org/abs/2401.02361) | `papers/locally_runnable_systems/02_mm_grounding_dino/2024_Zhao_MM_Grounding_DINO.pdf` |
| 03 Grounded SAM | Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks | 2024 | arXiv | [2401.14159](https://arxiv.org/abs/2401.14159) | `papers/locally_runnable_systems/03_grounded_sam/2024_Ren_Grounded_SAM.pdf` |
| 04 Grounded SAM 2 | _software integration — no standalone paper_ | — | — | [repo](https://github.com/IDEA-Research/Grounded-SAM-2) | `papers/locally_runnable_systems/04_grounded_sam_2/README.md` |
| 05 PET-DINO | PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training | 2026 | CVPR 2026 (Highlight) | [2604.00503](https://arxiv.org/abs/2604.00503) | `papers/locally_runnable_systems/05_pet_dino/2026_Fu_PET_DINO.pdf` |
<!-- END:PRIMARY_TABLE -->

### Supporting foundations

<!-- BEGIN:SUPPORTING_TABLE -->
| Paper | Year | Venue | Supports | Source | Local PDF |
|---|---|---|---|---|---|
| End-to-End Object Detection with Transformers | 2020 | ECCV 2020 | grounding_dino | [2005.12872](https://arxiv.org/abs/2005.12872) | `papers/core_foundations/2020_Carion_DETR.pdf` |
| DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection | 2022 | ICLR 2023 (arXiv 2022) | grounding_dino | [2203.03605](https://arxiv.org/abs/2203.03605) | `papers/core_foundations/2022_Zhang_DINO.pdf` |
| BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | 2019 | NAACL-HLT 2019 | grounding_dino | [1810.04805](https://arxiv.org/abs/1810.04805) | `papers/core_foundations/2019_Devlin_BERT.pdf` |
| Swin Transformer: Hierarchical Vision Transformer using Shifted Windows | 2021 | ICCV 2021 | grounding_dino | [2103.14030](https://arxiv.org/abs/2103.14030) | `papers/core_foundations/2021_Liu_Swin_Transformer.pdf` |
| Segment Anything | 2023 | ICCV 2023 | grounded_sam | [2304.02643](https://arxiv.org/abs/2304.02643) | `papers/core_foundations/2023_Kirillov_Segment_Anything.pdf` |
| SAM 2: Segment Anything in Images and Videos | 2024 | ICLR 2025 (arXiv 2024) | grounded_sam_2 | [2408.00714](https://arxiv.org/abs/2408.00714) | `papers/core_foundations/2024_Ravi_SAM_2.pdf` |
<!-- END:SUPPORTING_TABLE -->

## Broader literature categories

Papers by category (generated from `metadata/papers.json`):

<!-- BEGIN:CATEGORY_COUNTS -->
| Category | Papers |
|---|---|
| locally_runnable_systems | 4 |
| core_foundations | 6 |
| direct_extensions | 2 |
| tracking_and_video | 2 |
| segmentation_integrations | 3 |
| deployment_and_efficiency | 1 |
| domain_adaptations | 4 |
| application_papers | 1 |
| unverified_or_pending | 1 |
| **total** | **24** |
<!-- END:CATEGORY_COUNTS -->

The authoritative record is [`metadata/papers.json`](metadata/papers.json)
(mirrored to [`metadata/papers.csv`](metadata/papers.csv) and
[`metadata/papers.bib`](metadata/papers.bib)).

## Recommended reading order

DETR → DINO → BERT → Swin → **Grounding DINO** → MM-Grounding-DINO → Segment
Anything → Grounded SAM → SAM 2 → **Grounded SAM 2** → PET-DINO → efficiency &
video extensions. Full annotated path with links:
[`notes/reading_order.md`](notes/reading_order.md).

## Verification status

Every committed PDF is verified: non-empty, begins with `%PDF`, not an HTML page,
and recorded with a SHA-256 in [`metadata/checksums.sha256`](metadata/checksums.sha256).
Per-paper status (present / downloaded / pending / unavailable) is in
[`metadata/download_status.md`](metadata/download_status.md). One paper (Low-Rank
Prompt Adaptation, ICCVW 2025) is **closed access** with no open-access PDF; it is
recorded as `unavailable` rather than sourced from an unofficial mirror.

## Duplicate handling

Each paper is stored once; cross-references live in metadata rather than as copied
files. [`scripts/find_duplicates.py`](scripts/find_duplicates.py) and
[`scripts/verify_papers.py`](scripts/verify_papers.py) check for duplicate
SHA-256, duplicate arXiv IDs/DOIs, similar titles/filenames, and orphan PDFs —
currently **none**.

## Two things worth knowing

**Grounded SAM and Grounded SAM 2 are pipelines, not independent base detectors.**
They reuse Grounding DINO for open-set detection and SAM / SAM 2 for masks and
(for SAM 2) memory-based tracking. **Grounded SAM 2 has no standalone paper** — it
is a software integration; see
[`papers/locally_runnable_systems/04_grounded_sam_2/README.md`](papers/locally_runnable_systems/04_grounded_sam_2/README.md).

**Some later models are API-only / have no public checkpoints.** Grounding DINO
1.5 / 1.6 and DINO-X are served as cloud APIs; no public local weights exist. Do
not design the on-device pipeline around them.

## Running the paper tooling

```bash
python scripts/download_papers.py --dry-run   # show what would be fetched (arXiv/CVF only)
python scripts/download_papers.py             # fetch any missing PDFs, %PDF-validated
python scripts/verify_papers.py               # OFFLINE integrity check (no network, no model code)
python scripts/find_duplicates.py             # report likely duplicates (never deletes)
python scripts/generate_inventory.py          # regenerate metadata/* and the tables above
```

## Systems research collection (drone application)

| Question | File |
|---|---|
| What are the five systems, and which should I use? | [`docs/MODEL_INVENTORY.md`](docs/MODEL_INVENTORY.md) |
| Where is everything? | [`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md) |
| What weights would I need? | [`docs/CHECKPOINT_INVENTORY.md`](docs/CHECKPOINT_INVENTORY.md) |
| What can I legally ship? | [`docs/LICENSES.md`](docs/LICENSES.md) |
| Why can't I use one environment? | [`docs/DEPENDENCY_NOTES.md`](docs/DEPENDENCY_NOTES.md) |
| How do the papers relate to the drone project? | [`notes/project_relevance.md`](notes/project_relevance.md) |

Upstream source is vendored as pinned Git submodules under `third_party/`
(`git submodule update --init --recursive` to hydrate); per-model dependency notes
are in `environments/`.

## Repository map

Full directory-by-directory guide: [`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md).

```
README.md      this file           metadata/     literature inventory (csv/json/bib/checksums/status)
papers/        committed PDFs      notes/        per-system notes + comparison / reading order
scripts/       collection tooling  docs/         documentation, selection criteria, org log
manifests/     drone inventories   environments/ per-model dependency notes
third_party/   upstream submodules checkpoints/  weights (IGNORED) · datasets/ (IGNORED) · outputs/ (IGNORED)
```

## What is intentionally excluded

Datasets, model checkpoints/weights, model source code, inference outputs, and any
proprietary Honeywell / Shield AI / Hivemind material are **intentionally excluded
and git-ignored** — this is a paper/literature collection, not a runtime. Only the
academic PDFs, metadata, notes, docs, scripts, and pinned upstream pointers are
tracked.

## Ground rules

- Upstream source is **never modified**; submodules stay clean at pinned commits.
- Papers come only from **official sources**; no mirrors/ResearchGate/Scribd when
  an official source exists; missing fields are left empty, never fabricated.
- Checkpoints, datasets and outputs are **never committed**; the curated paper
  PDFs **are** committed (small, redistributable, official).
- Nothing here calls `torch.load`, imports model code, or runs inference.
- **No proprietary Honeywell / Shield AI / Hivemind material in this repo.**
