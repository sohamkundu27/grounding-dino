# Repository Map

What every top-level directory holds, and what is / is not tracked in Git.

```
.
├── README.md            entry point
├── .gitignore           keeps weights, datasets, PDFs and outputs out of Git
├── .gitmodules          the six upstream repos, pinned by commit
├── docs/                all documentation (tracked)
├── manifests/           machine-readable inventory + checksums (tracked)
├── scripts/             collection / verification tooling (tracked)
├── third_party/         upstream source, as Git submodules (pointers only)
├── checkpoints/         model weights (IGNORED — 17 GB on disk)
├── datasets/            RefDrone + VisDrone (IGNORED, except READMEs)
├── papers/              PDFs (IGNORED — fetch with scripts/download_papers.py)
├── environments/        per-model dependency notes (tracked)
└── outputs/             inference results (IGNORED, empty)
```

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
| `download_papers.py` | arXiv/CVF only; verifies the `%PDF` magic bytes |
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

## `papers/` — ignored by Git

Seven PDFs, all `%PDF`-verified with SHA-256 in `manifests/papers.json`.
Refetch with `python scripts/download_papers.py`.

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
