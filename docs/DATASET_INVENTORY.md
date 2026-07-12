# Dataset Inventory

## RefDrone

A referring-expression-comprehension (REC) benchmark for aerial drone imagery —
the closest public benchmark to this project's actual task: *"find the person
wearing a red shirt"*, from a drone, with no task-specific training.

| | |
|---|---|
| Paper | [arXiv 2502.00392](https://arxiv.org/abs/2502.00392) → `papers/refdrone/2025_Sun_RefDrone.pdf` |
| Code | <https://github.com/sunzc-sunny/refdrone> → `third_party/refdrone/` @ `86314ec` |
| Data | <https://huggingface.co/datasets/sunzc-sunny/RefDrone> (public, ungated) |
| Weights | <https://huggingface.co/sunzc-sunny/ngdino> (public, ungated) |
| License | CC BY 4.0 — **annotations only** |

### Counts

Measured directly from the released `*_mdetr.json` files, **not** quoted from the
abstract:

| Split | Image–expression pairs | Unique images | Unique expressions | Object instances |
|---|---|---|---|---|
| train | 13,022 | 6,407 | 8,910 | 47,557 |
| val | 1,428 | 534 | 1,124 | 4,741 |
| test | 3,432 | 1,595 | 2,686 | 12,105 |
| **total** | **17,882** | **8,536** | — | **64,403** |

Splits are disjoint at the image level.

> The published abstract advertises ~31k images / ~71k expressions. The release
> does not contain that. The numbers above were counted from the actual files at
> the commit in `manifests/repositories.json`; trust them, and be careful not to
> repeat the abstract's figures in any writeup. Cause unknown (possibly a larger
> internal version, or the abstract counting something else).

### Categories

11, inherited from VisDrone: `pedestrian`, `people`, `bicycle`, `car`, `van`,
`truck`, `tricycle`, `awning-tricycle`, `bus`, `motor`, `others`.

Note these are the *object* categories. The referring expressions themselves are
free-form natural language and are not restricted to this vocabulary — which is
the whole point of the benchmark.

### Annotation format

MDETR/COCO-style JSON: `info`, `licenses`, `images`, `annotations`, `categories`.

The important subtlety: **each entry in `images` is one (image, expression)
pair**, with the expression in the `caption` field. So `len(images)` = 17,882
pairs, not 8,536 distinct images. Boxes in `annotations` join via `image_id`.

RefDrone deliberately includes **multi-target** and **no-target** expressions.
A detector that always emits at least one box will score badly. This is why
NGDINO adds an explicit count branch, and it is a genuine consideration for the
drone system: "find the person in the red shirt" must be able to answer
*"there isn't one"*.

### Image origin and required VisDrone splits

RefDrone ships **language annotations only**. The pixels are VisDrone2019-DET
frames and are not redistributed (the HF repo carries a base64 TSV of the *test*
images only).

Required splits were **derived from the annotations**, not assumed — every
referenced `file_name` was extracted and matched against the VisDrone split
structure. The mapping is now **verified**, not merely inferred: all three
archives were extracted (8,629 images indexed) and every one of the 8,536
referenced basenames resolved against them, with **0 missing**.

| RefDrone split | Images needed | VisDrone split | Archive has | Matched |
|---|---|---|---|---|
| train | 6,407 | `train` | 6,471 | ✅ 6,407 / 0 missing |
| val | 534 | `val` | 548 | ✅ 534 / 0 missing |
| test | 1,595 | `test-dev` | 1,610 | ✅ 1,595 / 0 missing |
| — | — | ~~`test-challenge`~~ | 1,580 | ⛔ **never referenced — not downloaded** |

Each RefDrone split is a strict subset of the matching VisDrone split. Because
all 8,536 images resolved from train + val + test-dev alone, **test-challenge is
confirmed unnecessary** — this is now a measured fact, not an assumption.
`verify_refdrone.py` warns if that directory is ever populated.

The `9999xxx_*` filenames (5,090 in train, 1,351 in test) are VisDrone's
static-camera images and ship inside the standard archives — nothing extra needed.

Artifact: `datasets/refdrone/metadata/required_images_by_split.json`.

### Final directory layout

```
datasets/
├── refdrone/
│   ├── annotations/       RefDrone_{train,val,test}_mdetr.json   ← 18.6 MB
│   ├── images/all_image/  8,536 SYMLINKS into visdrone2019_det/  ← COMPLETE
│   ├── metadata/          refdrone_manifest.json, stats, required_images_by_split.json
│   └── README.md
└── visdrone2019_det/
    ├── archives/          the 3 ZIPs (~1.8 GB, gitignored, deletable)
    ├── train/{images,annotations}      6,471 jpg + 6,471 txt
    ├── val/{images,annotations}          548 jpg +   548 txt
    ├── test-dev/{images,annotations}   1,610 jpg + 1,610 txt
    └── README.md
```

`images/all_image/` is built by `scripts/prepare_refdrone.py` as **symlinks**:
8,536 links, **0 regular files**. The 1.9 GB of pixels lives once in the VisDrone
tree; nothing is duplicated and the VisDrone originals are never modified.

### Download and preparation status — ✅ COMPLETE

| Component | Status |
|---|---|
| RefDrone annotations (3 JSON) | ✅ downloaded (18.6 MB) |
| RefDrone paper | ✅ downloaded |
| RefDrone repo | ✅ cloned |
| NGDINO checkpoints (T, B) | ✅ downloaded |
| VisDrone2019-DET **train** | ✅ 6,471 images — `sha256 86a77eba9313…` |
| VisDrone2019-DET **val** | ✅ 548 images — `sha256 abeea063037e…` |
| VisDrone2019-DET **test-dev** | ✅ 1,610 images — `sha256 78b0c5078a14…` |
| VisDrone2019-DET test-challenge | ⛔ **not downloaded — confirmed unnecessary** |
| `refdrone/images/all_image/` | ✅ **8,536 / 8,536 linked, 0 missing** |

`scripts/verify_refdrone.py` → **RESULT: OK** — every referenced image resolves,
0 broken symlinks, 0 links escaping the VisDrone tree, 0 duplicated payloads,
0 cross-split leakage.

The archives had to be fetched **manually in a browser**: the official VisDrone
repo distributes only via Google Drive / BaiduYun, and a plain GET against the
Drive endpoint returns Google's virus-scan interstitial HTML rather than the ZIP
— for the 0.07 GB val archive as well as the 1.44 GB train archive, so size is
not the trigger. That requirement was **documented, not bypassed** with a scraped
confirm-token, and **no unofficial mirror** (Kaggle, `Voxel51/VisDrone2019-DET`,
scraped archive) was substituted. Upstream publishes **no checksums**, so the
SHA-256 values above were computed locally and are the reproducibility anchor
from here on. → `datasets/visdrone2019_det/README.md`.

### License / usage restrictions — read before deploying

Two **different** licenses are stacked here, and this is easy to get wrong:

- **RefDrone annotations: CC BY 4.0.** Permissive. Covers the language.
- **VisDrone imagery: academic / non-commercial research.** VisDrone has no
  LICENSE file; the challenge terms restrict use. CC BY 4.0 on the annotations
  **cannot** relicense the underlying pixels.
- **NGDINO weights** are trained on VisDrone imagery and inherit that restriction.

The stated deployment target is a Jetson AGX Orin running the Shield AI Hivemind
SDK. Treat RefDrone, VisDrone, and NGDINO as **evaluation-only** unless
commercial terms are confirmed in writing with the VisDrone authors. They are
fine for deciding *which model to use*; they are not fine to ship.

### Applicability, and the limitation that matters

**Applicable:** the only public benchmark that combines aerial viewpoints,
free-form referring expressions, very small objects, and multi/no-target cases.
It directly exercises the detector's core job.

**Limitation — RefDrone is image-based and is NOT a tracking benchmark.**
It has no temporal annotations, no track IDs, and no video sequences. It can
validate the *detection* half of the system (find the target from language) and
says **nothing** about the 10+ Hz *tracking* half, which is the harder deployment
constraint. Evaluating detection-plus-tracking end to end needs a separate video
source — VisDrone-VID / VisDrone-MOT, or in-house drone footage. That gap is
tracked in `docs/NEXT_STEPS.md` (step 7).

## VisDrone2019-DET

Not downloaded. See `datasets/visdrone2019_det/README.md` for exact archive
names, official URLs, sizes, splits, terms, and manual steps. Summary:

| Archive | Size | Images | Needed? |
|---|---|---|---|
| `VisDrone2019-DET-train.zip` | 1.44 GB | 6,471 | ✅ |
| `VisDrone2019-DET-val.zip` | 0.07 GB | 548 | ✅ |
| `VisDrone2019-DET-test-dev.zip` | 0.28 GB | 1,610 | ✅ |
| `VisDrone2019-DET-test-challenge.zip` | 0.28 GB | 1,580 | ❌ |

Total required ≈ 1.79 GB. No upstream checksums are published;
`scripts/verify_downloads.py` will record SHA-256s once the files are in place.

## sample_media

Intentionally empty. Drop representative EO stills / short clips here at the
inference stage. Grounded-SAM-2 ships usable demo assets at
`third_party/grounded_sam_2/assets/` if you need something immediately.

> **Do not place proprietary Honeywell / Shield AI / Hivemind footage in this
> Git repository.**
