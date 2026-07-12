# Repository Map

What every top-level directory holds, and what is / is not tracked in Git.

```
.
├── README.md            entry point
├── .gitignore           keeps weights, datasets and outputs out of Git (PDFs ARE tracked)
├── .gitmodules          the six upstream repos, pinned by commit
├── docs/                all documentation (tracked)
├── manifests/           drone-project machine-readable inventory + checksums (tracked)
├── metadata/            literature-collection metadata (csv/json/bib/checksums/status, tracked)
├── notes/               per-system notes + comparison / reading order / relevance (tracked)
├── scripts/             collection / verification / inventory tooling (tracked)
├── third_party/         upstream source, as Git submodules (pointers only)
├── checkpoints/         model weights (IGNORED — 17 GB on disk)
├── datasets/            RefDrone + VisDrone (IGNORED, except READMEs + small metadata JSON)
├── papers/              curated literature collection — PDFs ARE COMMITTED, by category
├── environments/        per-model dependency notes (tracked)
└── outputs/             inference results (IGNORED, empty)
```

> **Two paper records coexist by design.** `manifests/papers.json` is the
> drone-project reproducibility manifest (weights/datasets/repos live alongside
> it). `metadata/papers.json` is the expanded **literature-collection** record and
> is the source of truth for `papers/`, `metadata/*`, and the README paper tables.

## `docs/`

| File | Contents |
|---|---|
| `MODEL_INVENTORY.md` | comparison table across the five systems + NGDINO |
| `DATASET_INVENTORY.md` | RefDrone: counts, format, VisDrone requirement, limits |
| `REPOSITORY_MAP.md` | this file |
| `CHECKPOINT_INVENTORY.md` | every weight file, incl. shared/duplicate ones |
| `LICENSES.md` | code vs weights vs data licenses, tracked **separately** |
| `DOWNLOAD_STATUS.md` | what landed, what needs manual action, what failed |
| `DEPENDENCY_NOTES.md` | the five stacks compared, without installing them |
| `NEXT_STEPS.md` | the 10-step plan — none of it performed |

## `manifests/`

Machine-readable mirror of `docs/`, for scripting.

| File | Contents |
|---|---|
| `repositories.json` | name, URL, branch, **exact commit SHA**, license, notes |
| `checkpoints.json` | model, file, backbone, URL, sizes, SHA-256, license, status |
| `datasets.json` | dataset, component, split, URL, license, status, manual action |
| `papers.json` | title, authors, year, venue, arXiv id, URL, SHA-256, status |
| `checksums.sha256` | flat `sha256  path` list, `sha256sum -c`-compatible |

`checkpoints.json` also has two blocks worth knowing about:
`available_but_not_downloaded` (deliberately skipped, with reasons) and
`api_only_not_downloadable` (Grounding DINO 1.5/1.6, DINO-X — **no local weights
exist**).

## `scripts/`

None of these import a model package, load weights, or run inference.

| Script | Does |
|---|---|
| `download_repositories.sh` | (re)hydrate the six submodules at their pinned commits |
| `download_checkpoints.py` | manifest-driven streaming download; `.part` files, resume, SHA-256, size check, HTML-error-page detection. **Never deserialises a checkpoint.** |
| `download_papers.py` | **metadata-driven** literature downloader (arXiv/CVF only); streamed, `.part` temp files, `%PDF` validation, SHA-256, idempotent |
| `verify_papers.py` | **offline** literature verifier — existence, `%PDF`, zero-byte, HTML-as-PDF, checksum match, duplicate checksums, orphan PDFs |
| `find_duplicates.py` | duplicate detector — SHA-256, filename and title similarity, repeated arXiv IDs / DOIs (never deletes) |
| `generate_inventory.py` | regenerates `metadata/*` and the README paper tables from `metadata/papers.json` (preserves prose via markers) |
| `prepare_refdrone.py` | builds RefDrone's flat `all_image/` view as **symlinks** into VisDrone. Downloads nothing, modifies nothing, has `--dry-run` |
| `verify_downloads.py` | passive integrity sweep — checksums, `%PDF`, HTML-as-checkpoint, zero-byte, `.part`, duplicate payloads, broken symlinks, missing checkouts |
| `print_inventory.py` | the status table |

## `third_party/` — Git submodules

**Vendoring choice: submodules.** The parent repo was already under Git with a
clean tree, so submodules were preferred over ignored vendor clones (the brief's
stated preference). Only the commit pointer is tracked — no upstream source is
copied into this repo, and **no upstream file has been modified**.

```bash
git submodule update --init --recursive     # re-hydrate
```

| Path | Upstream | Commit |
|---|---|---|
| `grounding_dino/` | IDEA-Research/GroundingDINO | `856dde2` |
| `mm_grounding_dino/` | **open-mmlab/mmdetection** | `cfd5d3a` |
| `grounded_sam/` | IDEA-Research/Grounded-Segment-Anything | `126abe6` |
| `grounded_sam_2/` | IDEA-Research/Grounded-SAM-2 | `b7a9c29` |
| `pet_dino/` | fuweifuvtoo/PET_DINO | `7830a46` |
| `refdrone/` | sunzc-sunny/refdrone | `86314ec` |

⚠️ **`third_party/mm_grounding_dino/` is the whole of MMDetection v3.3.0.**
MM-Grounding-DINO is not a standalone repo; it lives inside mmdet at:

- `configs/mm_grounding_dino/` — configs, `usage.md`, `dataset_prepare.md`, `metafile.yml`
- `mmdet/models/detectors/grounding_dino.py` — the model

## `checkpoints/` — ignored by Git

~17 GB. Rebuild with `python scripts/download_checkpoints.py`.

```
grounding_dino/    2 files   Swin-T + Swin-B          ← shared with grounded_sam*
mm_grounding_dino/ 4 files   Swin-T ×2, Swin-B, Swin-L
sam/               2 files   ViT-H (default), ViT-B (edge)
sam2/              4 files   SAM 2.1 Hiera T/S/B+/L
pet_dino/          1 file    Swin-T (the only one published)
refdrone_ngdino/   2 files   NGDINO Swin-T / Swin-B   ← research-only license
```

Grounded-SAM and Grounded-SAM-2 have **no directory of their own** — they reuse
`grounding_dino/`, `sam/` and `sam2/`. See `docs/CHECKPOINT_INVENTORY.md`.

## `datasets/` — ignored by Git (READMEs tracked)

```
refdrone/annotations/   ✅ downloaded, 18.6 MB
refdrone/images/        ❌ empty — symlink view, needs VisDrone
refdrone/metadata/      ✅ generated (which images are required)
visdrone2019_det/       ❌ empty — MANUAL BROWSER DOWNLOAD, see its README
sample_media/           empty by design; no proprietary footage here
```

## `papers/` — curated literature collection (COMMITTED)

The Grounding DINO literature, organized one-paper-per-file (no duplicate PDFs;
relationships live in metadata). Filenames are `YYYY_FirstAuthor_Short_Title.pdf`.

```
papers/
├── locally_runnable_systems/
│   ├── 01_grounding_dino/     Grounding DINO (primary)
│   ├── 02_mm_grounding_dino/  MM-Grounding-DINO (primary)
│   ├── 03_grounded_sam/       Grounded SAM (primary)
│   ├── 04_grounded_sam_2/     README only — software integration, no paper
│   └── 05_pet_dino/           PET-DINO (primary)
├── core_foundations/          DETR, DINO, BERT, Swin, SAM, SAM 2
├── direct_extensions/         Grounding DINO 1.5, DINO-X
├── tracking_and_video/        VideoGrounding-DINO, GDINO-in-Videos
├── segmentation_integrations/ Mumuni, Pijarowski, Colony Grounded SAM2
├── deployment_and_efficiency/ Dynamic-DINO
├── domain_adaptations/        US-SAM, agriculture, 3D-CT, RefDrone
├── application_papers/         bird segmentation
└── unverified_or_pending/     README — Low-Rank Prompt Adaptation (no OA PDF)
```

All PDFs are `%PDF`-verified with SHA-256 in `metadata/checksums.sha256`.
Refetch anything missing with `python scripts/download_papers.py`, verify offline
with `python scripts/verify_papers.py`, and regenerate metadata with
`python scripts/generate_inventory.py`.

## `metadata/` — literature-collection inventory

| File | Contents |
|---|---|
| `papers.json` | source of truth — one record per paper (task schema) |
| `papers.csv` | flat table (generated) |
| `papers.bib` | one verified BibTeX entry per paper (generated; no fabricated fields) |
| `checksums.sha256` | `sha256  path`, `sha256sum -c`-compatible (generated) |
| `download_status.md` | present / downloaded / pending / unavailable (generated) |

## `notes/`

Per-system notes for the five locally runnable systems plus `comparison.md`,
`reading_order.md`, and `project_relevance.md`.

## `environments/`

One `README.md` per model: Python/PyTorch/CUDA expectations, whether custom CUDA
ops must be compiled, inference entry point, ONNX/TensorRT difficulty, Jetson
relevance. Plus `upstream/` holding the **copied** dependency files
(`requirements.txt`, `environment.yaml`, `Dockerfile`, `pyproject.toml`,
`setup.py` as `.txt`) so you can read them without leaving this repo.

**None of them have been installed or executed.** The five stacks conflict —
see `docs/DEPENDENCY_NOTES.md`.

## `outputs/`

Empty. Reserved for inference results, which are ignored by Git.
