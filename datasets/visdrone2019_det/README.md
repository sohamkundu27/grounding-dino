# VisDrone2019-DET

The image source for RefDrone. RefDrone ships **language annotations only**; all
8,536 images it references are VisDrone2019-DET frames.

**Status: 3 archives needed, manual browser download required.**

---

## ⬇️ DOWNLOAD CHECKLIST — do these three, then run one command

Download each file in a **browser** (you must be signed in to a Google account),
and save it into `datasets/visdrone2019_det/archives/` under **exactly** the
filename shown.

| # | Save as | Size | Download from |
|---|---|---|---|
| 1 | `VisDrone2019-DET-train.zip` | 1.44 GB | <https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view> |
| 2 | `VisDrone2019-DET-val.zip` | 0.07 GB | <https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view> |
| 3 | `VisDrone2019-DET-test-dev.zip` | 0.28 GB | <https://drive.google.com/open?id=1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V> |

Google Drive will show a **"Google Drive can't scan this file for viruses"**
page. Click **"Download anyway"**. That click is the whole reason this step is
manual (see below).

Destination (create it if missing — `mkdir -p datasets/visdrone2019_det/archives`):

```
datasets/visdrone2019_det/archives/
├── VisDrone2019-DET-train.zip
├── VisDrone2019-DET-val.zip
└── VisDrone2019-DET-test-dev.zip
```

⚠️ **Do NOT download `VisDrone2019-DET-test-challenge.zip`.** It is not needed —
verified against the RefDrone annotations, which reference no image from it.

### Confirm the downloads are real ZIPs, not saved web pages

The most common failure is the browser saving the *interstitial HTML page* under
a `.zip` name. Check before continuing:

```bash
file datasets/visdrone2019_det/archives/*.zip
# each must say:  Zip archive data
# if any says:    HTML document   -> re-download it
```

`scripts/setup_visdrone.py` checks this for you too (and rejects the file rather
than extracting it), so you can just run the next step and read the output.

### Then resume setup — everything after this is automatic

```bash
python scripts/setup_visdrone.py --dry-run   # validate archives, extract nothing
python scripts/setup_visdrone.py             # validate + extract + normalise

python scripts/prepare_refdrone.py \
    --refdrone-annotations datasets/refdrone/annotations \
    --visdrone-root        datasets/visdrone2019_det --dry-run
python scripts/prepare_refdrone.py \
    --refdrone-annotations datasets/refdrone/annotations \
    --visdrone-root        datasets/visdrone2019_det

python scripts/verify_refdrone.py            # must end: RESULT: OK
```

---

## Why this is not automated

The official VisDrone repository (<https://github.com/VisDrone/VisDrone-Dataset>)
distributes the DET archives **only** via Google Drive and BaiduYun. There is no
AWS, OneDrive, or direct-HTTP host. The RefDrone dataset card agrees, saying only
*"Please download the images from VisDrone."* The `aiskyeye.com` object-detection
download sub-pages return **HTTP 404**.

A plain `GET` against the Drive download endpoint returns:

```
<!DOCTYPE html><html><head><title>Google Drive - Virus scan warning</title>...
```

— for the **val** archive (0.07 GB) as well as train, so file size is not the
issue. Getting the bytes requires either clicking through that page in a browser
or scraping its confirmation token. **Scraping the token was explicitly out of
scope**, so the requirement is documented here rather than bypassed.

**No unofficial mirror was substituted.** Copies exist on Kaggle and on
third-party Hugging Face accounts (e.g. `Voxel51/VisDrone2019-DET`), but they are
not author-controlled: their contents and licensing cannot be trusted to match
the official release. They were not used and are not recommended without your
explicit approval.

## Checksums

**Upstream publishes none.** No SHA-256, MD5, or byte count is given for any
VisDrone archive, so `manifests/datasets.json` carries `null` for
`expected_size_bytes` and `sha256` rather than an invented value.

`scripts/setup_visdrone.py` computes and records the SHA-256 of whatever you
place here, so the state becomes reproducible **from that point on**. Integrity
before that point rests on the ZIP CRC check (which the script runs) and on the
extracted image count matching the expected 6,471 / 548 / 1,610.

## Layout after extraction

The archives extract to `VisDrone2019-DET-train/` etc. `setup_visdrone.py`
normalises that to the short split names, preserving the inner structure:

```
datasets/visdrone2019_det/
├── archives/                    the three ZIPs (gitignored; safe to delete after extraction)
├── train/
│   ├── images/                  6,471 .jpg
│   └── annotations/             6,471 .txt   (VisDrone boxes; unused by RefDrone)
├── val/
│   ├── images/                  548 .jpg
│   └── annotations/
├── test-dev/
│   ├── images/                  1,610 .jpg
│   └── annotations/
└── README.md
```

RefDrone's flat `all_image/` view is then built by `prepare_refdrone.py` as
**symlinks into these directories** — the 8,536 images are stored exactly once
and the VisDrone originals are never modified.

## What RefDrone actually needs

| RefDrone split | Unique images | From VisDrone split | Archive has |
|---|---|---|---|
| train | 6,407 | `train` | 6,471 |
| val | 534 | `val` | 548 |
| test | 1,595 | `test-dev` | 1,610 |
| — | — | ~~`test-challenge`~~ | **not needed** |

Each RefDrone split is a strict subset of its VisDrone split. Counts were
measured from the released annotation files (see
`datasets/refdrone/metadata/required_images_by_split.json`), not taken from the
paper.

## Disk space

| | |
|---|---|
| archives | ~1.8 GB |
| extracted images | ~1.8 GB |
| RefDrone symlink view | ~0 (symlinks, not copies) |
| **peak (archives + extracted)** | **~3.6 GB** |

Archives can be deleted after a verified extraction:
`python scripts/setup_visdrone.py --remove-archives-after`, or just
`rm datasets/visdrone2019_det/archives/*.zip`. Keep them if you have the space —
re-downloading means clicking through Google Drive again.

## License — read before deploying

VisDrone ships **no LICENSE file**. It is released by the AISKYEYE team (Lab of
Machine Learning and Data Mining, Tianjin University) for the VisDrone challenge,
and the challenge terms restrict use to **academic / non-commercial research**.

> **This matters for this project.** The deployment target is a Jetson AGX Orin
> running the Shield AI Hivemind SDK — a commercial/defense context. VisDrone
> imagery, and anything trained on it (including the RefDrone NGDINO
> checkpoints), is **research-only** unless commercial terms are confirmed in
> writing with the VisDrone authors. Use it to *decide which model to deploy*;
> do not ship it. RefDrone's own CC BY 4.0 covers the language annotations and
> cannot relicense the pixels underneath. See `docs/LICENSES.md`.
