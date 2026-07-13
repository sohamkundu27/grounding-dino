# PET-DINO

## Overview

A Grounding-DINO-derived detector that accepts **text prompts *and* visual
prompts**. It is the only system of the five that does both. Instead of only
describing the target in words, you can *point at* an example — a box, or a
precomputed visual embedding — and ask for more like it.

- **Paper:** Fu et al. 2026, arXiv [2604.00503](https://arxiv.org/abs/2604.00503) — **CVPR 2026 (Highlight)** → [`../papers/05_pet_dino/`](../papers/05_pet_dino/)
- **Official repo:** <https://github.com/fuweifuvtoo/PET_DINO>
- **Project page:** <https://fuweifuvtoo.github.io/pet-dino/>
- **Local source:** `third_party/pet_dino/` @ `7830a46`

**Authorship was verified, not assumed** (the name alone proves nothing):
arXiv → author project page → GitHub `fuweifuvtoo/PET_DINO` → HF org `fuweifu`,
all consistent with corresponding author **Weifu Fu** (Tencent YouTu Lab). No
unofficial reimplementation was substituted.

## Architecture

A **fork of MMDetection** extending MM-Grounding-DINO. Everything in
[`mm_grounding_dino.md`](mm_grounding_dino.md) about the mmcv/mmengine stack
applies here too — read that first.

The additions:

- **AFVPG** (Alignment-Friendly Visual Prompt Generation) — turns input coordinates into visual prompts that live in the same space as the text embeddings, so both prompt types can drive the same query-selection module.
- **IBP** (Intra-Batch Parallel Prompting) and **DMD** (Dynamic Memory-Driven Prompting) — training strategies that expose the model to multiple prompt routes at once, so the text pathway is not degraded by adding the visual one.

Both prompts act as **location priors for the 900 decoder queries**.

- **Image encoder:** Swin-T (Swin-L config exists but has **no released weight**)
- **Text encoder:** BERT-base-uncased
- **Framework:** MMDetection fork

## Inputs and outputs

- **In:** image + a text prompt **or** a visual prompt (bounding boxes, or a precomputed embedding)
- **Out:** boxes + scores. **No masks, no track IDs.**
- Two visual modes: **Visual-I** (in-image exemplar) and **Visual-G** (generalised, from an extracted embedding)

## Local availability

| | |
|---|---|
| Code | ✅ `third_party/pet_dino/` |
| Weights | ✅ `checkpoints/pet_dino/pet_dino_swin-t_8xb4_12e_obj365.pth` (2.2 GB) — the **only** one published |
| Init weights | needs MM-GDINO Swin-T, already at `checkpoints/mm_grounding_dino/` (shared) |
| License | Apache-2.0 (repo and HF model card) |

⚠️ Verified current as of this collection: repo live, HF repo public and ungated.
The **Swin-L config has no released checkpoint** — do not plan around it.

## Main strength

**Visual prompting is a genuinely good fit for a drone operator UI.** "Track
*that*" — pointed at on a screen — is often more natural and more precise than
describing a target in words, especially when the target has no clean verbal
description ("that specific vehicle", not "a white van").

The **precomputed-embedding path is deployment-friendly**: extract the visual
prompt embedding once, offline, then reuse it. That decouples expensive prompt
encoding from the per-frame path — the same trick that makes text-embedding
precomputation attractive for ONNX export.

## Main weakness

**Deployment immaturity.** It inherits Grounding DINO's deformable attention
*plus* adds a novel AFVPG branch that upstream has never exported. MMDeploy may
cover the mmdet backbone, but it will have no plugin for AFVPG. One checkpoint,
one backbone, a brand-new paper, and the full OpenMMLab dependency stack
(including `numpy==1.23`, which conflicts with Grounded-SAM-2's `numpy>=1.24.4`).

## Relevance to RefDrone

Evaluate its **text** pathway on RefDrone for a like-for-like comparison against
Grounding DINO and MM-Grounding-DINO — the paper's claim is that adding visual
prompts *preserves* text performance, and RefDrone is a fair place to check that.

Its **visual** pathway cannot be scored on RefDrone (RefDrone is a
referring-*expression* benchmark; there is no visual-exemplar protocol). Assess
that qualitatively instead.

## Relevance to Jetson Orin

**Medium — high research value, low deployment readiness.** Strategically the
most interesting of the five for the operator-facing product, but it is the
*least* ready to ship. Evaluate it on the workstation; **do not put it on the
critical path to Jetson.**

## What to measure

- **Text pathway** on RefDrone vs Grounding DINO and MM-GDINO, same prompts — is the text capability really preserved?
- **Visual pathway, qualitatively:** given one box around a target in frame 1, how well does it find that target elsewhere? This is the drone-relevant question.
- Visual-I vs Visual-G — does the precomputed-embedding route lose accuracy versus the in-image exemplar?
- Cost of embedding extraction, and whether it can be done once and cached.
- Whether visual prompting beats text prompting for targets that are **hard to describe** — the case where it would actually earn its place.
