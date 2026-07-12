# Next Steps

**None of this has been performed.** The collection task deliberately stopped at
organize-document-inspect. This is the plan for what comes after, in order.

Prerequisite for everything below: **download VisDrone2019-DET** (manual browser
step — `datasets/visdrone2019_det/README.md`), then run
`scripts/prepare_refdrone.py`. Nothing dataset-related can proceed without it.

---

### 1. Create isolated environments

Four of them — they conflict and cannot be merged. See `docs/DEPENDENCY_NOTES.md`.

| Env | Covers |
|---|---|
| `ovd-gsam2` | Grounded-SAM-2 + SAM 2 ← **start here** |
| `ovd-gdino` | Grounding DINO, Grounded-SAM |
| `ovd-mmdet` | MM-Grounding-DINO, NGDINO (`mmcv==2.1.0`, **not** the 2.2.0 RefDrone's README tells you to use) |
| `ovd-petdino` | PET-DINO (`numpy==1.23`) |

Needs a full CUDA toolkit (`nvcc` + `CUDA_HOME`) — every model compiles
`MultiScaleDeformableAttention`. Budget real time for `mmcv`.

### 2. Select sample EO images

Assemble a small, honest evaluation set in `datasets/sample_media/`:
- RefDrone test images spanning easy → hard (small targets, occlusion, crowds)
- **no-target** and **multi-target** cases — the ones that break naive detectors
- if available, in-house EO stills at the real operating altitude and sensor

Keep it small (~50 images). The goal is a fast qualitative loop, not a benchmark.
**No proprietary footage in this Git repo.**

### 3. Run workstation inference

First real run. Suggested order:
1. Grounded-SAM-2 local demo (`grounded_sam2_local_demo.py`) — sanity check.
2. Grounding DINO Swin-T, then MM-GDINO Swin-T, on the same images.
3. PET-DINO, including its **visual-prompt** path.

Use the actual target prompt — *"the person wearing a red shirt"* — not COCO
nouns. Attribute-conditioned referring is a different task from category
detection, and this is where the models will separate.

### 4. Standardize the output schema

Before comparing anything, agree one JSON schema across all models: image id,
prompt, boxes (xyxy, absolute px), scores, labels, latency, model+commit+checkpoint
hash. Write per-model adapters into it. Without this, step 5–7 become an
unfalsifiable mess of incompatible formats. Land the results in `outputs/`.

### 5. Benchmark prompt robustness

The system's whole premise is natural language, so test the language:
- paraphrase (*"red shirt person"* / *"the man in red"* / *"person, red top"*)
- attributes (colour, clothing, action, relative position)
- negation and absence — **does it correctly return nothing?**
- multi-target (*"all the vans"*)
- distractors (several people, one in red)

RefDrone's no-target/multi-target design makes it the right vehicle. Expect this
to be the most informative experiment in the list, and the one most likely to
disqualify a model.

### 6. Benchmark aerial small-object performance

Aerial targets are tiny. Measure mAP **bucketed by object pixel area**, not just
overall — an aggregate number will hide exactly the failure that matters.
Evaluate on RefDrone val/test. Also test sliced/tiled inference (SAHI-style),
since these detectors are trained on ~800px inputs and drone frames are much larger.

### 7. Evaluate detection **plus** tracking

⚠️ **RefDrone cannot do this.** It is image-based: no temporal annotations, no
track IDs, no video. You need a separate video source — VisDrone-VID/-MOT, or
in-house drone footage.

The thing to actually measure: detect once with the slow open-vocabulary
detector, hand the box to SAM 2, and track. Then ask:
- does SAM 2 hold the target through occlusion / scale change / re-entry?
- what is the sustained tracker FPS (the **10+ Hz** requirement)?
- how often must the detector re-fire, and what does that cost?

This is the experiment that decides whether the architecture works.

### 8. Inspect ONNX export

Inspect, then attempt. In increasing order of pain:
- **SAM 2 image encoder** (Hiera) — plain transformer, should export.
- **SAM 2 memory attention / mask decoder** — the tracker is *stateful*; the
  memory bank must become explicit graph I/O. Expect 3–4 separate engines
  orchestrated from C++, not one.
- **MM-Grounding-DINO via MMDeploy** — the only first-party path, and it has a
  TensorRT plugin for `MultiScaleDeformableAttention`. Clone MMDeploy **at this
  step**, not before.
- **Grounding DINO** — no official exporter exists. Deformable attention has no
  native ONNX op; needs a `grid_sample` rewrite (opset ≥16) or a custom plugin.

Key trick: **precompute BERT text embeddings offline.** If the prompt set is
known ahead of time, the text encoder becomes a lookup table and only the vision
branch needs exporting. This removes most of the difficulty.

### 9. Inspect TensorRT compatibility

Per candidate engine: supported ops, plugins required, dynamic shapes, FP16 vs
INT8 accuracy loss, memory footprint. `MultiScaleDeformableAttention` is the
single op that decides this — settle it first, because if it cannot be made to
work the model selection changes.

### 10. Profile on Jetson Orin

Only now touch the hardware. Measure end to end, on the real sensor feed:
detector latency (allowed to be slow), **tracker FPS (must clear 10 Hz)**, power,
thermals, memory. Remember **SM 8.7** — the upstream `TORCH_CUDA_ARCH_LIST`
values stop at 8.6 and will fail at runtime, not build time.

Target: the Orin runs **TensorRT engines from C++ under Hivemind**, with no
Python ML stack on the device.

---

## Open questions worth settling early

- **Does any of these actually resolve "red shirt" from a drone?** Attribute-level
  referring at aerial scale, where a person is 20 px tall, is a real open question.
  Step 5 answers it, and it is cheap. Do it before investing in export work.
- **Is one detection enough?** The architecture assumes detect-once-then-track.
  If the target must be re-acquired often, the "detection may be slow" premise
  weakens and the whole plan shifts.
- **Licensing.** RefDrone/VisDrone/NGDINO are **research-only**
  (`docs/LICENSES.md`). Fine for choosing a model; not shippable. If the plan
  ever grows a fine-tuning step on aerial data, that needs resolving first — it is
  much cheaper to know now than after training.
