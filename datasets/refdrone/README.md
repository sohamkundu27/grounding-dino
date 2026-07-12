# RefDrone

Referring-expression-comprehension (REC) benchmark for aerial drone imagery — the
closest public benchmark to this project's actual task: *"find the person wearing
a red shirt"*, from a drone, with no task-specific training.

- Paper: <https://arxiv.org/abs/2502.00392>
- Code: <https://github.com/sunzc-sunny/refdrone> → `third_party/refdrone/`
- Data: <https://huggingface.co/datasets/sunzc-sunny/RefDrone> (public, ungated)
- Weights: <https://huggingface.co/sunzc-sunny/ngdino> → `checkpoints/refdrone_ngdino/`
- License: CC BY 4.0 — **annotations only**. See the caveat at the bottom.

## Layout

```
datasets/refdrone/
├── annotations/                       ← downloaded (18.6 MB)
│   ├── RefDrone_train_mdetr.json
│   ├── RefDrone_val_mdetr.json
│   └── RefDrone_test_mdetr.json
├── images/
│   └── all_image/                     ← SYMLINKS into datasets/visdrone2019_det/
│       └── 0000001_02999_d_0000005.jpg -> ../../../visdrone2019_det/val/images/...
├── metadata/
│   ├── refdrone_manifest.json         ← per-split counts, generated
│   ├── refdrone_stats.json
│   ├── required_images_by_split.json  ← the 8,536 image names RefDrone needs
│   └── refdrone_image_manifest.json   ← written by prepare_refdrone.py
└── README.md
```

`images/all_image/` is the flat directory RefDrone's own code expects (see
`third_party/refdrone/README.md`). It is built as **symbolic links** into the
VisDrone tree, so the 8,536 images are stored exactly **once** — there is no
second copy of the pixels, and the VisDrone originals are never modified.

## Counts

Measured from the released `*_mdetr.json` files. **Not** quoted from the paper.

| Split | Annotation records | Unique expressions | Unique images | Object instances | From VisDrone |
|---|---|---|---|---|---|
| train | 13,022 | 8,910 | 6,407 | 47,557 | `train` |
| val | 1,428 | 1,124 | 534 | 4,741 | `val` |
| test | 3,432 | 2,686 | 1,595 | 12,105 | `test-dev` |
| **total** | **17,882** | — | **8,536** | **64,403** | |

Splits are **disjoint at the image level** (verified: zero images shared between
splits). The 9,346 "duplicate" image references are expected — a single image
carries several referring expressions.

> The paper's abstract advertises ~31k images / ~71k expressions. The release does
> not contain that. The table above is what the actual files hold. Use these
> numbers; do not repeat the abstract's figures.

Categories (11, inherited from VisDrone): `pedestrian`, `people`, `bicycle`,
`car`, `van`, `truck`, `tricycle`, `awning-tricycle`, `bus`, `motor`, `others`.
The referring expressions themselves are free-form language, not limited to these.

## Annotation format

MDETR/COCO-style JSON: `info`, `licenses`, `images`, `annotations`, `categories`.

The subtlety that trips people up: **each entry in `images` is one
(image, expression) pair**, with the expression in `caption`. So `len(images)` is
17,882 pairs, not 8,536 distinct images. Boxes in `annotations` join by `image_id`.

RefDrone deliberately includes **multi-target** and **no-target** expressions. A
detector that always emits at least one box scores badly — which is the point, and
why NGDINO adds an object-count branch. For the drone system this matters
directly: *"find the person in the red shirt"* must be able to answer
**"there isn't one."**

## Images come from VisDrone2019-DET

RefDrone ships language only; the HF repo carries a base64 TSV of the *test*
images but does **not** redistribute train/val pixels. All 8,536 images are
VisDrone2019-DET frames.

**→ `datasets/visdrone2019_det/README.md` has the download checklist.**

## Rerunning preparation

```bash
# 1. validate + extract the VisDrone archives (idempotent; reuses valid extractions)
python scripts/setup_visdrone.py --dry-run
python scripts/setup_visdrone.py

# 2. build the symlink view (dry-run first)
python scripts/prepare_refdrone.py \
    --refdrone-annotations datasets/refdrone/annotations \
    --visdrone-root        datasets/visdrone2019_det --dry-run
python scripts/prepare_refdrone.py \
    --refdrone-annotations datasets/refdrone/annotations \
    --visdrone-root        datasets/visdrone2019_det
```

`prepare_refdrone.py` never downloads, never modifies VisDrone, and defaults to
symlinks. `--link-mode hardlink` (same filesystem only) and `--link-mode copy`
(actually duplicates the bytes — avoid) exist for filesystems that need them.

Safe to re-run: existing links are left alone, broken ones are replaced.

## Validating

```bash
python scripts/verify_refdrone.py --write-manifest
```

Offline and read-only — it imports no model code, loads no checkpoint, runs no
inference, and never touches the network. It checks that the annotations parse,
that every referenced image resolves, that no symlink is broken or points outside
`datasets/visdrone2019_det/`, that no image payload was duplicated, and that no
image leaks across splits.

Exit codes: `0` OK · `2` incomplete (VisDrone not linked yet) · `1` real
integrity problem.

## Known limitations

**It is not a tracking benchmark.** RefDrone is image-based: no temporal
annotations, no track IDs, no video sequences. It can validate the *detector* —
the "find the person in the red shirt" step — and says **nothing** about the
10+ Hz tracking stage, which is the harder deployment constraint. Evaluating
detection-plus-tracking end to end needs a separate video source (VisDrone-VID /
-MOT, or in-house footage). Tracked in `docs/NEXT_STEPS.md` step 7.

**Small objects.** Aerial targets are tiny. Report mAP bucketed by object pixel
area; an aggregate number hides exactly the failure mode that matters here.

## License caveat — read before deploying

RefDrone's CC BY 4.0 covers the **language annotations**. It does not and cannot
relicense the underlying VisDrone pixels, which remain under VisDrone's
**academic / non-commercial research** terms. The NGDINO checkpoints are trained
on VisDrone imagery and inherit that restriction.

```
RefDrone annotations   CC BY 4.0            ✅ permissive
        ↓ point at
VisDrone images        non-commercial       ❌ restricted
        ↓ trained on
NGDINO weights         inherit restriction  ❌ restricted
```

Given the Jetson AGX Orin / Shield AI Hivemind deployment target, treat all of
this as **evaluation-only** unless commercial terms are confirmed with the
VisDrone authors. Fine for deciding which model to deploy; not shippable.
See `docs/LICENSES.md`.
