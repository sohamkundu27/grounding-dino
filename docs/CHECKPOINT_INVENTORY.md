# Checkpoint Inventory

Every model weight collected, including which ones are **shared** between systems.

- Machine-readable source of truth: `manifests/checkpoints.json`
- SHA-256 for every file: `manifests/checksums.sha256` (`sha256sum -c`-compatible)
- Verify at any time: `python scripts/verify_downloads.py`

Weights are **not** committed to Git (~17 GB). Rebuild with
`python scripts/download_checkpoints.py` — it skips files that already exist and
match their recorded checksum.

> Every file below was downloaded as **opaque bytes**. Nothing in this repo calls
> `torch.load`, unpickles a checkpoint, or imports model code. Sizes were checked
> against the official host before the download was accepted, and any HTML error
> page saved under a `.pth` name is rejected rather than kept.

## `checkpoints/grounding_dino/` — 1.5 GB

| File | Backbone | Size | Source | License |
|---|---|---|---|---|
| `groundingdino_swint_ogc.pth` | Swin-T | 662 MB | GitHub release `v0.1.0-alpha` | Apache-2.0 |
| `groundingdino_swinb_cogcoor.pth` | Swin-B | 895 MB | GitHub release `v0.1.0-alpha2` | Apache-2.0 |

Configs: `third_party/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py`
and `GroundingDINO_SwinB_cfg.py`.

⚠️ **Shared — do not re-download.** Grounded-SAM and Grounded-SAM-2 both need a
Grounding DINO checkpoint. Their download scripts
(`third_party/grounded_sam_2/gdino_checkpoints/download_ckpts.sh`) fetch these
exact two files from these exact two URLs. Point them at this directory instead;
that is why there is no `checkpoints/grounded_sam*/`.

## `checkpoints/mm_grounding_dino/` — 4.4 GB

| File | Backbone | Size | Training data |
|---|---|---|---|
| `grounding_dino_swin-t_pretrain_obj365_goldg_v3det_20231218_095741-e316e297.pth` | Swin-T | 950 MB | O365 + GoldG + V3Det |
| `grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth` | Swin-T | 1.0 GB | + GRIT9M — **best Swin-T** |
| `grounding_dino_swin-b_pretrain_all-f9818a7c.pth` | Swin-B | 1.1 GB | full mixture |
| `grounding_dino_swin-l_pretrain_all-56d69e78.pth` | Swin-L | 1.4 GB | full mixture — **accuracy ceiling** |

All from `download.openmmlab.com`, Apache-2.0. URLs in
`third_party/mm_grounding_dino/configs/mm_grounding_dino/metafile.yml`.

🔗 **The Swin-T `..._v3det_...` file is also PET-DINO's required initialisation**
(its README's "Pretrained Weights" table asks for exactly this file). Symlink it
into PET-DINO's `pretrained/` rather than downloading a second copy.

The full MM-GDINO zoo has ~40 more fine-tunes (COCO/LVIS/ODinW/RTTS/RUOD/…).
Deliberately skipped; every URL is in `metafile.yml`.

## `checkpoints/sam/` — 2.7 GB (SAM v1)

| File | Encoder | Size | Note |
|---|---|---|---|
| `sam_vit_h_4b8939.pth` | ViT-H | 2.4 GB | default in all Grounded-SAM demos |
| `sam_vit_b_01ec64.pth` | ViT-B | 358 MB | smallest; the only edge-plausible SAM v1 |

From `dl.fbaipublicfiles.com/segment_anything/`, Apache-2.0 (Meta).
`sam_vit_l` skipped — ViT-H and ViT-B already bracket the range.

## `checkpoints/sam2/` — 1.5 GB (SAM 2.1)

| File | Backbone | Size | Note |
|---|---|---|---|
| `sam2.1_hiera_large.pt` | Hiera-L | 857 MB | most accurate |
| `sam2.1_hiera_base_plus.pt` | Hiera-B+ | 309 MB | |
| `sam2.1_hiera_small.pt` | Hiera-S | 176 MB | |
| `sam2.1_hiera_tiny.pt` | Hiera-T | 149 MB | **primary Jetson tracking candidate** |

From `dl.fbaipublicfiles.com/segment_anything_2/092824/`, Apache-2.0 (Meta).
These are the **SAM 2.1** (Sept 2024) weights, matching
`third_party/grounded_sam_2/checkpoints/download_ckpts.sh`. The older SAM 2
(072824) weights are commented out upstream and were not fetched.

All four were taken because this is the **10+ Hz component** — the tiny/small
variants are the ones that decide whether the project's frame-rate requirement is
achievable, and they are cheap to keep.

## `checkpoints/pet_dino/` — 2.2 GB

| File | Backbone | Size | License |
|---|---|---|---|
| `pet_dino_swin-t_8xb4_12e_obj365.pth` | Swin-T | 2.2 GB | Apache-2.0 |

From <https://huggingface.co/fuweifu/PET-DINO> — author-owned, ungated, card
declares `apache-2.0`. Config: `configs/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.py`.
Supports **text and visual prompts**.

⚠️ This is the **only** checkpoint PET-DINO publishes. The repo also ships a
Swin-L config (`pet_dino_swin-l_8xb4_12e_obj365.py`) with **no released weight** —
do not plan around a Swin-L PET-DINO.

Also needs the MM-GDINO Swin-T above as its init; see that section.

## `checkpoints/refdrone_ngdino/` — 4.6 GB (reference model, **not** one of the five)

| File | Backbone | Size | Config |
|---|---|---|---|
| `NGDINO_T.pth` | Swin-T | 2.0 GB | `configs/NGDINO/ngdino_swin-t_refdrone.py` |
| `NGDINO_B.pth` | Swin-B | 2.6 GB | `configs/NGDINO/ngdino_swin-b_refdrone.py` |

From <https://huggingface.co/sunzc-sunny/ngdino> — author-owned, ungated.

🚨 **License: research-only.** RefDrone's repo says CC BY 4.0, but these weights
are trained on **VisDrone** imagery, which is academic/non-commercial. The
weights inherit that restriction regardless of what the repo's README says. Use
them as the aerial-REC yardstick; **do not ship them.** See `docs/LICENSES.md`.

Six further files on that HF repo (50-epoch variants, training inits, an
auxiliary colour classifier — ~10.8 GB) were deliberately skipped; they are
listed under `available_but_not_downloaded` in `manifests/checkpoints.json`.

## Shared / duplicate checkpoints — summary

| Checkpoint | Stored once at | Also wanted by |
|---|---|---|
| `groundingdino_swint_ogc.pth` | `checkpoints/grounding_dino/` | Grounded-SAM, **Grounded-SAM-2** |
| `groundingdino_swinb_cogcoor.pth` | `checkpoints/grounding_dino/` | Grounded-SAM |
| `sam_vit_h_4b8939.pth` | `checkpoints/sam/` | Grounded-SAM |
| `sam2.1_hiera_*.pt` | `checkpoints/sam2/` | Grounded-SAM-2 |
| MM-GDINO Swin-T `..._v3det_...pth` | `checkpoints/mm_grounding_dino/` | **PET-DINO** (as init) |

No checkpoint is stored twice. Grounded-SAM and Grounded-SAM-2 have no
checkpoint directories of their own by design — they are *pipelines* over models
that already live here. When a demo wants a weights path, give it the shared one
(symlink if the code insists on a specific layout) rather than re-downloading.

`scripts/verify_downloads.py --deep` hashes every file on disk and will report
any accidental duplicate payload.

## ❌ NOT downloadable — API-only, no local weights exist

**Grounding DINO 1.5 Pro/Edge, Grounding DINO 1.6, DINO-X.**

Grounded-SAM-2 ships six demos that use them (`grounded_sam2_gd1.5_demo.py`,
`grounded_sam2_dinox_demo.py`, and four tracking variants). Those demos import
`dds_cloudapi_sdk` and require an `API_TOKEN` — the models run on the
DeepDataSpace **cloud**, and using them uploads your imagery to a third party.

**There are no public checkpoints for these models.** No API token was obtained,
no cloud API was called, and no image was sent anywhere during this setup. They
cannot be part of a Jetson deployment. Do not let their presence in the demo list
suggest otherwise.
