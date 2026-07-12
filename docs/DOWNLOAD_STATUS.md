# Download Status

Generated after the collection run. Re-check at any time:

```bash
python scripts/verify_downloads.py --deep    # repos / checkpoints / papers
python scripts/verify_refdrone.py            # dataset -> RESULT: OK
python scripts/print_inventory.py
```

**Headline: nothing is outstanding.** Every repository, checkpoint, paper and
dataset component is present and verified. The VisDrone imagery — the one item
that previously required a manual browser download — has been obtained,
validated, extracted and linked.

---

## 📦 RefDrone dataset — ✅ COMPLETE

| Component | Status |
|---|---|
| RefDrone annotations (train / val / test) | ✅ downloaded, 18.6 MB, all parse |
| VisDrone2019-DET **train** | ✅ 6,471 images + 6,471 labels — `sha256 86a77eba9313…` |
| VisDrone2019-DET **val** | ✅ 548 images + 548 labels — `sha256 abeea063037e…` |
| VisDrone2019-DET **test-dev** | ✅ 1,610 images + 1,610 labels — `sha256 78b0c5078a14…` |
| VisDrone2019-DET test-challenge | ⛔ not downloaded — **confirmed unnecessary** |
| Archive extraction | ✅ all 3 — ZIP magic + CRC + zip-slip checked before extracting |
| RefDrone image links | ✅ **8,536 / 8,536** symlinks, **0 missing** |
| Missing images | ✅ **0** |
| Broken symlinks | ✅ **0** |
| Duplicated image payloads | ✅ **0** (8,536 symlinks, 0 regular files) |
| Cross-split leakage | ✅ **0** |
| Verifier | ✅ `verify_refdrone.py` → **RESULT: OK** (exit 0) |

**Split mapping — verified, not assumed.** RefDrone train → VisDrone `train`,
val → `val`, test → `test-dev`. All 8,536 referenced basenames resolved against
the 8,629 extracted images, so test-challenge is confirmed unneeded as a measured
fact.

**Archives retained** (~1.8 GB, gitignored) at
`datasets/visdrone2019_det/archives/`. Disk was never a constraint (605 GB free),
so they were kept — re-downloading means clicking through Google Drive again.
Delete with `python scripts/setup_visdrone.py --remove-archives-after` if needed.

Upstream publishes **no checksums** for the VisDrone archives, so the SHA-256
values above were computed locally on receipt and are the reproducibility anchor
from here on. They are recorded in `manifests/datasets.json` — not fabricated
beforehand.

---

## ✅ Successfully cloned

Six repositories, vendored as Git submodules at pinned commits. All checkouts
clean; **no upstream file modified**.

| Repo | Commit | Upstream |
|---|---|---|
| `third_party/grounding_dino` | `856dde20aee659246248e20734ef9ba5214f5e44` | IDEA-Research/GroundingDINO |
| `third_party/mm_grounding_dino` | `cfd5d3a985b0249de009b67d04f37263e11cdf3d` | open-mmlab/mmdetection |
| `third_party/grounded_sam` | `126abe633ffe333e16e4a0a4e946bc1003caf757` | IDEA-Research/Grounded-Segment-Anything |
| `third_party/grounded_sam_2` | `b7a9c29f196edff0eb54dbe14588d7ae5e3dde28` | IDEA-Research/Grounded-SAM-2 |
| `third_party/pet_dino` | `7830a462f9a320f34293cce0aabdda1256d9dc15` | fuweifuvtoo/PET_DINO |
| `third_party/refdrone` | `86314ec6e0db91c5a922f300c30b1a362e60bdac` | sunzc-sunny/refdrone |

## ✅ Successfully downloaded

**Checkpoints — 15 files, 16.91 GB, 0 failures.** SHA-256 for each in
`manifests/checksums.sha256`.

| Group | Files | Size |
|---|---|---|
| `grounding_dino/` | Swin-T, Swin-B | 1.5 GB |
| `mm_grounding_dino/` | Swin-T ×2, Swin-B, Swin-L | 4.4 GB |
| `sam/` | ViT-H, ViT-B | 2.7 GB |
| `sam2/` | SAM 2.1 Hiera L / B+ / S / T | 1.5 GB |
| `pet_dino/` | Swin-T | 2.2 GB |
| `refdrone_ngdino/` | NGDINO Swin-T, Swin-B | 4.6 GB |

The OpenMMLab/Meta filenames embed a hash suffix, and the computed SHA-256
prefixes match them (`e316e297`, `b448804b`, `f9818a7c`, `56d69e78`) — independent
confirmation the files are byte-identical to what upstream published.

**Papers — 7 PDFs, all `%PDF`-verified**, checksums in `manifests/papers.json`.
Grounding DINO, MM-Grounding-DINO, Grounded SAM, Segment Anything, SAM 2,
PET-DINO, RefDrone. All from arXiv.

**RefDrone annotations — 3 JSON, 18.6 MB.** From the authors' Hugging Face
dataset (public, ungated).

## ✅ Already present

Nothing was pre-existing; this was a clean collection run. The four submodules
staged by a prior session (`grounding_dino`, `mm_grounding_dino`, `grounded_sam`,
`grounded_sam_2`) were verified and committed rather than re-cloned.

## ⚠️ Manual action required

**None remaining.**

### VisDrone2019-DET imagery — resolved

This was the one gap, and it is now closed. The archives were downloaded in a
browser by the repository owner and transferred to this machine, then validated,
extracted and linked automatically.

| Archive | Bytes received | Images | SHA-256 (computed locally) |
|---|---|---|---|
| `VisDrone2019-DET-train.zip` | 1,549,875,511 | 6,471 | `86a77eba9313…` |
| `VisDrone2019-DET-val.zip` | 81,638,851 | 548 | `abeea063037e…` |
| `VisDrone2019-DET-test-dev.zip` | 311,045,829 | 1,610 | `78b0c5078a14…` |
| `VisDrone2019-DET-test-challenge.zip` | — | — | ⛔ **not downloaded, not needed** |

**Why it could not be automated:** the official VisDrone repo
(<https://github.com/VisDrone/VisDrone-Dataset>) and RefDrone's own dataset card
distribute the imagery only via Google Drive and BaiduYun. A plain `GET` against
the Drive endpoint returns Google's **"Virus scan warning"** HTML page rather than
the ZIP — for the 0.07 GB val archive as well as the 1.44 GB train archive, so
file size is not the trigger. The `aiskyeye.com` object-detection download
sub-pages return **HTTP 404**, so no alternative official host exists.

The interstitial was **not bypassed** with a scraped confirm-token, and **no
unofficial mirror** (Kaggle, `Voxel51/VisDrone2019-DET`, re-upload, scraped
archive) was substituted.

**Validation caught a real problem.** The first validation pass ran while `scp`
was still writing and correctly **rejected** the partially-transferred test-dev
archive (`file` reported "Zip archive data", but the End-of-Central-Directory
record was absent). Re-validating after the transfer settled passed cleanly. The
"reject before extracting" design did its job.

**→ Steps, for anyone rebuilding: `datasets/visdrone2019_det/README.md`**

## 🔒 Authentication required

**None.** Every source used was public and ungated:

- GitHub release assets — anonymous.
- `download.openmmlab.com`, `dl.fbaipublicfiles.com` — anonymous.
- Hugging Face (`sunzc-sunny/RefDrone`, `sunzc-sunny/ngdino`, `fuweifu/PET-DINO`)
  — checked via the HF API: `gated: False`, `private: False`. No token needed and
  none was used.

## 📜 License acceptance required

**VisDrone.** No click-through gate, but no LICENSE file either — the challenge
terms restrict use to academic / non-commercial research. Acknowledge before use;
this **conflicts with the Jetson/Hivemind commercial deployment target**. See
`docs/LICENSES.md`.

## ❌ Unavailable — API-only, no local weights exist

**Grounding DINO 1.5 Pro/Edge, Grounding DINO 1.6, DINO-X.**

Referenced by six Grounded-SAM-2 demos (`grounded_sam2_gd1.5_demo.py`,
`grounded_sam2_dinox_demo.py`, and four tracking variants) which import
`dds_cloudapi_sdk` and require an `API_TOKEN`. These models run on the
DeepDataSpace **cloud** — using them uploads imagery to a third party.

**No public checkpoints exist.** Not downloadable, at any size, by anyone. No API
credentials were obtained, no cloud API was called, and no image was sent
anywhere. They cannot be part of a Jetson deployment.

## ❓ Unverified

**None.** Every artifact traces to an official source.

The one that *needed* verifying was **PET-DINO** — the brief warned not to trust a
repo merely because its name matches. Verified chain:

> arXiv 2604.00503 → author project page `fuweifuvtoo.github.io/pet-dino`
> → GitHub `fuweifuvtoo/PET_DINO` → HF org `fuweifu`

consistent with corresponding author **Weifu Fu** (Tencent YouTu Lab), and the
paper is a **CVPR 2026 Highlight** with a CVF Open Access page. Apache-2.0.
**No unofficial reimplementation was substituted.**

## ❌ Failed

**None.** 15/15 checkpoints, 7/7 papers, 3/3 annotation files, 6/6 repositories.
No `.part` files left behind, no zero-byte files, no HTML pages saved as
checkpoints, no broken symlinks.

## ⏭️ Skipped — duplicate or unnecessary

| Skipped | Why |
|---|---|
| Grounding DINO / SAM weights for Grounded-SAM & Grounded-SAM-2 | **Shared.** Those repos want the same files already in `checkpoints/grounding_dino/` and `checkpoints/sam*/`. Nothing stored twice. |
| MM-GDINO Swin-T for PET-DINO's `pretrained/` | **Shared.** Already in `checkpoints/mm_grounding_dino/`; symlink it. |
| `refdrone_test_base64.tsv` (884 MB) | Base64 copy of the RefDrone **test** images. The same pixels arrive via VisDrone test-dev and get symlinked. |
| `VisDrone2019-DET-test-challenge.zip` | **Not referenced by RefDrone** — verified against the annotations. |
| 6 further NGDINO files (~10.8 GB) | 50-epoch variants, training inits, an auxiliary colour classifier. |
| SAM `vit_l`; SAM 2 (072824) | ViT-H/ViT-B bracket the range; SAM 2.1 supersedes SAM 2 (upstream comments it out). |
| ~40 MM-GDINO downstream fine-tunes | COCO/LVIS/ODinW/etc. URLs all in upstream `metafile.yml`. |
| **MMDeploy** | Needed only at the ONNX/TensorRT stage. Not cloned, per the brief's "don't clone unrelated giant repos". |

All recorded under `available_but_not_downloaded` in `manifests/checkpoints.json`
so any of them can be fetched later by adding an entry and re-running the script.

## Disk usage

| Directory | Size |
|---|---|
| `checkpoints/` | 17 GB |
| `third_party/` | 1.1 GB |
| `papers/` | 65 MB |
| `datasets/` | 19 MB |
| everything else | < 1 MB |
| **total** | **~18 GB** |

No cleanup or dataset deletion was necessary — see the final report.
