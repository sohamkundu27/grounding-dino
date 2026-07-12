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
structure:

| RefDrone split | Images needed | VisDrone archive | Archive size |
|---|---|---|---|
| train | 6,407 | `VisDrone2019-DET-train.zip` (6,471 imgs) | 1.44 GB |
| val | 534 | `VisDrone2019-DET-val.zip` (548 imgs) | 0.07 GB |
| test | 1,595 | `VisDrone2019-DET-test-dev.zip` (1,610 imgs) | 0.28 GB |
| — | — | ~~`test-challenge`~~ | **not needed** |

Each RefDrone split is a strict subset of the matching VisDrone split, so
**test-challenge must not be downloaded**. The `9999xxx_*` filenames (5,090 in
train, 1,351 in test) are VisDrone's static-camera images and ship inside the
standard archives — nothing extra needed.

Artifact: `datasets/refdrone/metadata/required_images_by_split.json`.

### Directory layout

```
datasets/
├── refdrone/
│   ├── annotations/       RefDrone_{train,val,test}_mdetr.json   ← DOWNLOADED (18.6 MB)
│   ├── images/all_image/  symlinks into visdrone2019_det/         ← EMPTY, needs VisDrone
│   └── metadata/          required_images_by_split.json, stats    ← GENERATED
└── visdrone2019_det/
    ├── train/  val/  test/                                        ← EMPTY, manual download
    └── README.md          exact manual download instructions
```

`images/all_image/` is built by `scripts/prepare_refdrone.py` as **symlinks**, so
the 8,536 images are stored once, in the VisDrone tree, and never duplicated.

### Download status

| Component | Status |
|---|---|
| RefDrone annotations (3 JSON) | ✅ **downloaded** |
| RefDrone paper | ✅ **downloaded** |
| RefDrone repo | ✅ **cloned** |
| NGDINO checkpoints (T, B) | ✅ **downloaded** |
| VisDrone2019-DET train/val/test-dev | ❌ **manual browser download required** |
| `refdrone/images/all_image/` | ⏳ blocked on VisDrone |

**Missing component: VisDrone imagery.** The official VisDrone repo distributes
only via Google Drive / BaiduYun, and the Drive endpoint returns an HTML
virus-scan interstitial rather than the ZIP. That is a manual-browser-access
requirement and was **not bypassed**; no unofficial mirror (Kaggle, scraped
archive) was substituted. → `datasets/visdrone2019_det/README.md`.

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
