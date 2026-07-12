# VisDrone2019-DET — MANUAL DOWNLOAD REQUIRED

RefDrone ships **language annotations only**. The underlying imagery comes from
VisDrone2019-DET and must be obtained separately, from the official source.

**Status: not downloaded. Manual browser action required.**

## Why this was not downloaded automatically

The official VisDrone repository (<https://github.com/VisDrone/VisDrone-Dataset>)
distributes the DET archives **only** via Google Drive and BaiduYun. A `HEAD`
against the Google Drive direct-download endpoint returns
`content-type: text/html` — Google's virus-scan / confirmation interstitial —
rather than the ZIP payload.

That is exactly the "requires manual browser access" case, so the download was
**not** bypassed with a scraped confirm-token, and no unofficial mirror
(Kaggle, scraped archive, third-party re-upload) was substituted. Please fetch
the archives yourself using the links below.

## Exactly which archives RefDrone needs

Derived from the RefDrone annotations themselves (not guessed) — see
`scripts/prepare_refdrone.py` and `datasets/refdrone/metadata/required_images_by_split.json`.

| RefDrone split | Unique images needed | Comes from VisDrone archive | Archive images | Size |
|---|---|---|---|---|
| train | 6,407 | `VisDrone2019-DET-train.zip` | 6,471 | 1.44 GB |
| val | 534 | `VisDrone2019-DET-val.zip` | 548 | 0.07 GB |
| test | 1,595 | `VisDrone2019-DET-test-dev.zip` | 1,610 | 0.28 GB |
| — | — | ~~`VisDrone2019-DET-test-challenge.zip`~~ | 1,580 | **NOT NEEDED** |

Total required: **~1.79 GB** of archives → 8,536 unique images.

Each RefDrone split is a strict subset of the corresponding VisDrone split, so
**test-challenge is not required** and should not be downloaded.

Note the `9999xxx_*` filenames (5,090 in train, 1,351 in test): these are
VisDrone's static-camera images. They are part of the standard train/test-dev
archives — nothing extra is needed for them.

## Official download links

From the official VisDrone repository, Task 1 (Object Detection in Images).
These Google Drive links are the **author-published** distribution channel, not
third-party mirrors.

| Archive | Google Drive | BaiduYun |
|---|---|---|
| `VisDrone2019-DET-train.zip` | <https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view> | <https://pan.baidu.com/s/1K-JtLnlHw98UuBDrYJvw3A> |
| `VisDrone2019-DET-val.zip` | <https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view> | <https://pan.baidu.com/s/1jdK_dAxRJeF2Xi50IoML1g> |
| `VisDrone2019-DET-test-dev.zip` | <https://drive.google.com/open?id=1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V> | <https://pan.baidu.com/s/1RdRfSWV-1IFK7aWljLU_LQ> |

Landing page: <https://github.com/VisDrone/VisDrone-Dataset>

No checksums are published upstream for these archives. `scripts/verify_downloads.py`
will record the SHA-256 of whatever you place here so the state is reproducible
from that point on.

## Steps

1. Download the three archives in a browser using the links above.
2. Extract them so this directory looks like:

   ```
   datasets/visdrone2019_det/
   ├── train/   ← from VisDrone2019-DET-train.zip
   │   ├── images/           (6,471 .jpg)
   │   └── annotations/      (VisDrone box labels; unused by RefDrone)
   ├── val/     ← from VisDrone2019-DET-val.zip
   │   ├── images/           (548 .jpg)
   │   └── annotations/
   └── test/    ← from VisDrone2019-DET-test-dev.zip
       ├── images/           (1,610 .jpg)
       └── annotations/
   ```

   The archives extract to `VisDrone2019-DET-train/`, `.../val/`, `.../test-dev/`
   with an inner `images/` folder. Rename the top level to `train` / `val` /
   `test` (or point the prepare script at wherever you put them — it takes explicit
   paths and will search recursively).

3. Build the flat `all_image/` view RefDrone expects — **without copying 8,536
   files**:

   ```bash
   python scripts/prepare_refdrone.py \
       --refdrone-annotations datasets/refdrone/annotations \
       --visdrone-root        datasets/visdrone2019_det \
       --dry-run                      # inspect the plan first
   python scripts/prepare_refdrone.py \
       --refdrone-annotations datasets/refdrone/annotations \
       --visdrone-root        datasets/visdrone2019_det
   ```

   This symlinks the needed images into `datasets/refdrone/images/all_image/`,
   reports anything missing or duplicated, and writes a manifest. The VisDrone
   originals are never modified.

## License / terms — read before deploying

VisDrone carries **no explicit license file** in its GitHub repository. The
dataset is released by the AISKYEYE team (Lab of Machine Learning and Data
Mining, Tianjin University) for the VisDrone challenge, and the challenge terms
restrict use to **academic / non-commercial research**.

> **This matters for this project.** The stated deployment target is a
> Jetson AGX Orin with the Shield AI Hivemind SDK. VisDrone imagery, and any
> model trained or fine-tuned on it (including the RefDrone NGDINO checkpoints,
> which are VisDrone-derived), should be treated as **research-only** unless
> commercial terms are confirmed in writing with the VisDrone authors. Use them
> for evaluation and capability assessment; do not assume they may ship.
> RefDrone's own *annotations* are CC BY 4.0, but that license covers the
> language annotations, not the underlying VisDrone pixels.
