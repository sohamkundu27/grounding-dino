# Licenses

**An Apache-2.0 repository does not make its checkpoints or its training data
Apache-2.0.** These three things are tracked separately below, because in this
project they genuinely differ, and the difference decides what can ship.

This is a summary for engineering triage, not legal advice. Confirm with counsel
before any commercial or defense deployment.

---

## 1. Repository / source-code licenses

| Repository | License | File |
|---|---|---|
| IDEA-Research/GroundingDINO | **Apache-2.0** | `third_party/grounding_dino/LICENSE` |
| open-mmlab/mmdetection | **Apache-2.0** | `third_party/mm_grounding_dino/LICENSE` |
| IDEA-Research/Grounded-Segment-Anything | **Apache-2.0** | `third_party/grounded_sam/LICENSE` |
| IDEA-Research/Grounded-SAM-2 | **Apache-2.0** | `third_party/grounded_sam_2/LICENSE` (+ `LICENSE_sam2`, `LICENSE_groundingdino`, `LICENSE_cctorch` for the bundled components) |
| fuweifuvtoo/PET_DINO | **Apache-2.0** (OpenMMLab-derived) | `third_party/pet_dino/LICENSE` |
| sunzc-sunny/refdrone | **CC BY 4.0** | stated in README |

All permissive. Attribution required. No copyleft in the set. Verified by reading
the actual `LICENSE` files, not inferred from README badges.

> Grounded-SAM-2 correctly ships **separate** license files for its bundled
> components (`LICENSE_sam2`, `LICENSE_groundingdino`, `LICENSE_cctorch`). Keep
> all of them if you redistribute — this is the one repo here that already models
> the code-vs-component distinction properly.

> RefDrone has **no LICENSE file at all**; CC BY 4.0 is only a README statement.
> Applying CC BY 4.0 to source code is also unusual — CC licenses are not designed
> for software. It is an mmdet fork, so the Apache-2.0 obligations from the
> upstream code it contains persist regardless. Attribute both.

---

## 2. Model-weight licenses

| Weights | License | Confidence |
|---|---|---|
| `groundingdino_swint_ogc`, `groundingdino_swinb_cogcoor` | Apache-2.0 | released with the Apache-2.0 repo as GitHub release assets; no separate weight license published |
| MM-GDINO `grounding_dino_swin-{t,b,l}_pretrain_*` | Apache-2.0 | OpenMMLab model zoo |
| SAM v1 `sam_vit_{h,b}` | **Apache-2.0** | explicit — Meta licenses SAM weights Apache-2.0 |
| SAM 2.1 `sam2.1_hiera_*` | **Apache-2.0** | explicit |
| PET-DINO `pet_dino_swin-t_*` | Apache-2.0 | HF model card `license: apache-2.0` |
| **NGDINO `NGDINO_T`, `NGDINO_B`** | ⚠️ **encumbered — see below** | RefDrone says CC BY 4.0, but the weights are VisDrone-trained |

### ⚠️ The NGDINO problem

RefDrone's repo declares CC BY 4.0. But NGDINO is **trained on VisDrone
imagery**, which is licensed for academic / non-commercial research only. A
project cannot grant a permissive license to a model derived from data it does
not own that permissively.

**Treat the NGDINO checkpoints as research-only.** They are a benchmark yardstick,
not a deployment candidate. Nothing about the project's plan depends on shipping
them, so this is a documentation issue, not a blocker.

### Training-data provenance caveat (applies to the permissive weights too)

Grounding DINO and MM-Grounding-DINO were pretrained on mixtures including
**Objects365, GoldG, Cap4M, GRIT, V3Det**. Some of those corpora are themselves
research-licensed or have web-scraped provenance. The Apache-2.0 grant on the
weights is what the authors offer; it does not launder the upstream data terms,
and no upstream author indemnifies you. This is normal for the field and is
usually accepted, but for a **defense/commercial** deployment it deserves an
explicit decision rather than a shrug.

---

## 3. Dataset licenses

| Dataset | License | Commercial use? |
|---|---|---|
| **RefDrone annotations** (the 3 JSON files) | **CC BY 4.0** | ✅ yes, with attribution |
| **VisDrone2019-DET images** (the actual pixels) | **academic / non-commercial research** | ❌ **no** |

VisDrone ships **no LICENSE file**. It is released by the AISKYEYE team (Lab of
Machine Learning and Data Mining, Tianjin University) for the VisDrone challenge,
and the challenge terms restrict use to academic / non-commercial research.

### The stacking trap

RefDrone's CC BY 4.0 covers **the language annotations only**. It cannot and does
not relicense the VisDrone pixels those annotations point at. So:

```
RefDrone annotations   CC BY 4.0            ✅ permissive
        ↓ point at
VisDrone images        non-commercial       ❌ restricted
        ↓ trained on
NGDINO weights         inherit restriction  ❌ restricted
```

Anyone reading only RefDrone's README would conclude the whole benchmark is
CC BY 4.0. It isn't.

---

## What this means for the project

The deployment target is a Jetson AGX Orin running the Shield AI Hivemind SDK —
a commercial/defense context.

✅ **Clear to build on:** Grounding DINO, MM-Grounding-DINO, Grounded-SAM,
Grounded-SAM-2, SAM, SAM 2, PET-DINO. All Apache-2.0, code and weights. Keep the
NOTICE/attribution files.

⚠️ **Evaluation only:** RefDrone, VisDrone, NGDINO. Use them to *decide which
model to deploy* — that is a research activity and is fine. Do not ship the data,
do not ship NGDINO, and do not fine-tune a production model on VisDrone imagery
without first obtaining commercial terms from the VisDrone authors.

📋 **Action if you later want to fine-tune on aerial data for production:** either
license VisDrone commercially, or collect/label in-house drone imagery. Worth
knowing now, before anyone builds a plan around fine-tuning on RefDrone.

---

## Attribution

If anything derived from these is published or shipped, cite: Grounding DINO
(Liu et al. 2023), MM-Grounding-DINO (Zhao et al. 2024), Grounded SAM (Ren et al.
2024), Segment Anything (Kirillov et al. 2023), SAM 2 (Ravi et al. 2024),
PET-DINO (Fu et al. 2026), RefDrone (Sun et al. 2025), VisDrone (Zhu et al.).
Full records in `manifests/papers.json`.
