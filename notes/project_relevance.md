# Project relevance

How this literature maps onto a drone-based, natural-language target-finding
system. This is general architecture reasoning only — **no proprietary project
details are included.**

## Natural-language target acquisition
The core requirement — *"find the person wearing a red shirt"* → a box — is exactly
what the Grounding DINO family provides. Grounding DINO / MM-Grounding-DINO turn a
free-form phrase into detections with no task-specific training. PET-DINO adds
**visual** prompting for targets that are easier to show than to name.

## Zero-shot detection
Open-vocabulary/open-set detection removes the need to pre-train on a fixed target
list. New target descriptions work at inference time — essential when targets are
not known in advance.

## Aerial imagery
The general-purpose pre-training of these models is not aerial/top-down. RefDrone
(`papers/domain_adaptations/2025_Sun_RefDrone.pdf`) is the closest catalogued
benchmark for referring-expression comprehension in drone scenes; its NGDINO
baseline is a Grounding DINO derivative. Expect a domain gap; some aerial
fine-tuning is likely required.

## Small-object detection
Aerial targets are often tiny. This is a known weak spot for transformer detectors
at fixed input resolution — a primary risk to validate, and a reason to look at
efficiency/backbone choices rather than assuming out-of-the-box performance.

## Detector + tracker architecture
The recurring pattern: run the **slow** open-vocabulary detector once to *acquire*
the target, then hand the box to a **fast** tracker. Grounded SAM 2 embodies this
directly (Grounding DINO acquire → SAM 2 memory tracker). The detection stage may
be slow; the **tracking stage must sustain 10+ Hz.**

## EO vs IR limitations
All catalogued models are trained on RGB/EO imagery. Infrared (IR/thermal) is out
of distribution — appearance cues and text-image alignment learned on EO may not
transfer. IR support would need domain adaptation or a dedicated pipeline; nothing
here is validated on thermal.

## Jetson AGX Orin deployment
The eventual platform is embedded (Orin). Implications: prefer the lighter
backbones (Swin-T) and small SAM 2 (Hiera tiny, ~149 MB) for the real-time stage;
treat the detector as an infrequent acquire step. MM-Grounding-DINO has the
clearest first-party deployment tooling; Grounded SAM 2 is the most promising
end-to-end target.

## ONNX and TensorRT relevance
On-device speed will come from ONNX → TensorRT. The main engineering risks are the
custom CUDA deformable-attention op and the dual-modality (image + BERT text)
inputs; the text branch is often exported/frozen separately. This is deployment
work and is **out of scope for this repository** (papers only).

## Hivemind integration context
The downstream autonomy/integration layer is external and proprietary; it is
deliberately **not** part of this collection. This repo stops at the open
literature and organization needed to *choose* an approach — no integration,
inference, training, or deployment is performed here.
