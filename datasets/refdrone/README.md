# RefDrone

Referring-expression-comprehension (REC) benchmark for aerial drone imagery.
This is the closest public benchmark to the project's target task — "find the
person wearing a red shirt", from a drone, with no task-specific training.

- Paper: <https://arxiv.org/abs/2502.00392> (local: `papers/refdrone/2025_Sun_RefDrone.pdf`)
- Code: <https://github.com/sunzc-sunny/refdrone> (local: `third_party/refdrone/`)
- Data: <https://huggingface.co/datasets/sunzc-sunny/RefDrone> (public, ungated)
- Weights: <https://huggingface.co/sunzc-sunny/ngdino> (public, ungated)
- License: CC BY 4.0 — **annotations only**, see the caveat at the bottom.

## What is here

| Path | Status |
|---|---|
| `annotations/RefDrone_{train,val,test}_mdetr.json` | **downloaded** (18.6 MB) |
| `metadata/required_images_by_split.json` | **generated** — the 8,536 image names RefDrone references |
| `metadata/refdrone_stats.json` | **generated** — counts below |
| `images/all_image/` | **empty** — needs VisDrone; built by `scripts/prepare_refdrone.py` |

## Counts (measured from the annotation files, not quoted from the abstract)

| Split | Image–expression pairs | Unique images | Unique expressions | Object instances |
|---|---|---|---|---|
| train | 13,022 | 6,407 | 8,910 | 47,557 |
| val | 1,428 | 534 | 1,124 | 4,741 |
| test | 3,432 | 1,595 | 2,686 | 12,105 |
| **total** | **17,882** | **8,536** | — | **64,403** |

Splits are disjoint at the image level (no leakage).

Categories (11, inherited from VisDrone): `pedestrian`, `people`, `bicycle`,
`car`, `van`, `truck`, `tricycle`, `awning-tricycle`, `bus`, `motor`, `others`.

> The published abstract advertises larger round numbers (~31k images / ~71k
> expressions). The tables above are what the released `*_mdetr.json` files
> actually contain, at the commit recorded in `manifests/datasets.json`. Trust
> these; they were counted directly. The discrepancy is unexplained — possibly a
> larger internal version, or abstract figures that count something else.

## Annotation format

MDETR/COCO-style JSON with `info`, `licenses`, `images`, `annotations`,
`categories`. Each entry in `images` is one **(image, referring-expression)
pair** — the `caption` field holds the expression — so `len(images)` is the
number of pairs (17,882), not the number of distinct images (8,536). Boxes in
`annotations` link back via `image_id`.

RefDrone deliberately includes **multi-target and no-target** expressions, which
is why NGDINO adds an explicit object-count branch. A detector that always emits
at least one box will score badly here — worth knowing before benchmarking.

## Images: not included, must come from VisDrone2019-DET

The HF dataset ships annotations plus a base64 TSV of the *test* images only.
The train/val imagery is **not redistributed**. All 8,536 images are VisDrone2019-DET
frames and must be fetched from the official VisDrone source.

**→ See `datasets/visdrone2019_det/README.md` for the manual download steps.**

Once VisDrone is in place:

```bash
python scripts/prepare_refdrone.py \
    --refdrone-annotations datasets/refdrone/annotations \
    --visdrone-root        datasets/visdrone2019_det \
    --dry-run
```

`images/all_image/` is then populated with **symlinks** into the VisDrone tree —
the 8,536 images are not duplicated, and the VisDrone originals are never touched.

## Relevance and limits for this project

**Relevant:** it is the only public benchmark combining aerial viewpoints,
natural-language referring expressions, small objects, and multi/no-target cases
— which is precisely the drone target-finding task.

**Limitation — it is image-based, not a tracking benchmark.** RefDrone provides
no temporal annotations, no track IDs, and no video sequences. It can validate
the *detector* (the "find the person in the red shirt" step) but says nothing
about the 10+ Hz *tracking* stage. Evaluating detection-plus-tracking needs a
separate video source (e.g. VisDrone-VID/-MOT, or in-house drone footage), and
that gap is called out in `docs/NEXT_STEPS.md`.

## License caveat

RefDrone's CC BY 4.0 covers the **language annotations**. It does not and cannot
relicense the underlying VisDrone pixels, which remain under VisDrone's
academic/non-commercial research terms. The NGDINO checkpoints are trained on
VisDrone imagery and inherit that restriction. Treat all of it as research-only
for a commercial/defense deployment until confirmed otherwise.
