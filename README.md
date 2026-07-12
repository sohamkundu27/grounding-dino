# Open-Vocabulary Detection & Tracking — Research Collection

Source, weights, papers, dependency definitions, and dataset components for five
open-vocabulary detection/tracking systems, collected and documented so they can
be compared **before** deciding what to run.

**Target application.** A drone-based open-vocabulary target-finding system:
given *"find the person wearing a red shirt"*, locate the target from natural
language with no task-specific training. Detection may be slow; the subsequent
**tracking must sustain 10+ Hz**. Eventual platform is an NVIDIA Jetson AGX Orin
(ONNX → TensorRT → C++, Shield AI Hivemind SDK). **Deployment is not part of this
repository.**

> **Nothing here has been executed.** No inference, no tests, no benchmarks, no
> training, no ONNX/TensorRT export, no dependency installation, no CUDA
> compilation. Every claim below comes from reading source, configs and
> dependency files. Upstream repositories are unmodified.

## Start here

| Question | File |
|---|---|
| What are the five systems, and which should I use? | [`docs/MODEL_INVENTORY.md`](docs/MODEL_INVENTORY.md) |
| What can RefDrone actually tell me? | [`docs/DATASET_INVENTORY.md`](docs/DATASET_INVENTORY.md) |
| Where is everything? | [`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md) |
| What weights do I have? | [`docs/CHECKPOINT_INVENTORY.md`](docs/CHECKPOINT_INVENTORY.md) |
| What can I legally ship? | [`docs/LICENSES.md`](docs/LICENSES.md) |
| What still needs downloading? | [`docs/DOWNLOAD_STATUS.md`](docs/DOWNLOAD_STATUS.md) |
| Why can't I use one environment? | [`docs/DEPENDENCY_NOTES.md`](docs/DEPENDENCY_NOTES.md) |
| What now? | [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md) |

```bash
python scripts/print_inventory.py     # status table
python scripts/verify_downloads.py    # passive integrity check
```

## The five systems

| System | Commit | Role |
|---|---|---|
| [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) | `856dde2` | baseline detector |
| [MM-Grounding-DINO](https://github.com/open-mmlab/mmdetection) (inside MMDetection) | `cfd5d3a` | **primary detector candidate** — best zero-shot, only first-party TensorRT path |
| [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) | `126abe6` | reference detect→segment pattern |
| [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | `b7a9c29` | **primary pipeline** — local detection + SAM 2 video tracking |
| [PET-DINO](https://github.com/fuweifuvtoo/PET_DINO) | `7830a46` | CVPR 2026; text **and visual** prompts |

Plus [NGDINO](https://github.com/sunzc-sunny/refdrone) (`86314ec`) as the
RefDrone-associated aerial-REC baseline — *not* one of the five.

## Three things worth knowing immediately

**1. Grounded-SAM-2 is the architecture this project is reaching for.**
Run the slow open-vocabulary detector once to acquire the target from the prompt,
then hand the box to SAM 2's memory-based tracker for the 10+ Hz stage.
`sam2.1_hiera_tiny` (149 MB) is the realistic Orin candidate.

**2. Grounding DINO 1.5, 1.6 and DINO-X are cloud APIs, not local models.**
Six demos in Grounded-SAM-2 import `dds_cloudapi_sdk` and need an API token —
they upload your imagery to a third party. **No public local checkpoints exist
for them.** No token was obtained, no API called, no image sent anywhere. Don't
design the Jetson pipeline around them.

**3. RefDrone is research-only, and it is not a tracking benchmark.**
Its CC BY 4.0 covers the *language annotations*; the VisDrone pixels underneath
are academic/non-commercial, and the NGDINO weights inherit that. Fine for
choosing a model, not shippable. And it has no temporal annotations — it cannot
validate the 10+ Hz tracking stage at all.

## Layout

```
docs/          documentation          manifests/     JSON inventories + checksums
scripts/       collection tooling     third_party/   upstream source (submodules)
checkpoints/   ~17 GB of weights *    datasets/      RefDrone OK / VisDrone missing *
papers/        7 PDFs *               environments/  per-model dependency notes
outputs/       (empty) *
```
`*` ignored by Git — reproducible via `scripts/`.

## Getting a fresh clone to this state

```bash
git clone --recurse-submodules <this repo>     # or: ./scripts/download_repositories.sh
python scripts/download_checkpoints.py         # ~17 GB, resumable, SHA-256 verified
python scripts/download_papers.py              # 7 PDFs
python scripts/verify_downloads.py             # passive check
```

**VisDrone2019-DET must be downloaded manually in a browser** — the official repo
serves it via Google Drive, which returns an HTML interstitial rather than the
ZIP. That requirement was documented, not bypassed, and no unofficial mirror was
substituted. See [`datasets/visdrone2019_det/README.md`](datasets/visdrone2019_det/README.md),
then:

```bash
python scripts/prepare_refdrone.py \
    --refdrone-annotations datasets/refdrone/annotations \
    --visdrone-root        datasets/visdrone2019_det --dry-run
```

This builds RefDrone's flat `all_image/` view as **symlinks** — the 8,536 images
are stored once, not duplicated.

## Ground rules for this repository

- Upstream source is **never modified**; submodules stay clean at pinned commits.
- Checkpoints, datasets, PDFs and outputs are **never committed** — manifests and
  checksums are, so everything is reproducible.
- Checkpoints are downloaded as **opaque bytes**. Nothing here calls `torch.load`
  or imports model code.
- **No proprietary Honeywell / Shield AI / Hivemind material in this repo.**
