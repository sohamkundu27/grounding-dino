# Environment — MM-Grounding-DINO (OpenMMLab / MMDetection)

**Nothing here has been installed or run.** Notes from reading the upstream
dependency files, vendored under `upstream/`.

Source: `third_party/mm_grounding_dino/` @ `cfd5d3a985b0249de009b67d04f37263e11cdf3d`
(this is the **full MMDetection repo**, v3.3.0 — MM-GDINO lives inside it)

## Where MM-Grounding-DINO actually is

| What | Path inside `third_party/mm_grounding_dino/` |
|---|---|
| Model code | `mmdet/models/detectors/grounding_dino.py` |
| Configs | `configs/mm_grounding_dino/` |
| Usage docs | `configs/mm_grounding_dino/usage.md` |
| Dataset prep | `configs/mm_grounding_dino/dataset_prepare.md` |
| Model index | `configs/mm_grounding_dino/metafile.yml` ← every checkpoint URL |

## Upstream expectations

| | |
|---|---|
| Python | 3.7–3.11 |
| PyTorch | ≥ 1.8 (practically 2.x; RefDrone/PET-DINO forks assume 2.1) |
| CUDA | 11.8 / 12.1 (whatever the mmcv wheel was built for) |
| mmengine | `>=0.7.1, <1.0.0` |
| mmcv | **`>=2.0.0rc4, <2.2.0`** |

Extra for MM-GDINO (`requirements/multimodal.txt`): `transformers`, `nltk`,
`fairscale`, `jsonlines`, `pycocoevalcap`.

## The version trap — read this before installing

`mmdet/__init__.py` contains a **hard assertion**:

```python
mmcv_minimum_version = '2.0.0rc4'
mmcv_maximum_version = '2.2.0'
assert mmcv_version >= ... and mmcv_version < digit_version(mmcv_maximum_version)
```

`mmcv` **2.2.0 is excluded**. Note that `third_party/refdrone/README.md` tells
you to run `mim install "mmcv==2.2.0"` — **that instruction contradicts the
assertion in the very code it ships** and will abort on import. Use
`mmcv==2.1.0`.

Beyond that: mmcv must be installed via `mim` against your exact torch/CUDA
build (it also has compiled ops), and the LVIS API used by the eval path
**breaks on numpy ≥ 1.24** — pin `numpy==1.23`.

## Custom CUDA operators — YES (inside mmcv, not this repo)

MMDetection itself is mostly pure Python, but it depends on **mmcv**, which
ships compiled CUDA ops (including MultiScaleDeformableAttention). You get these
from a prebuilt mmcv wheel on x86; **on Jetson there is no prebuilt wheel and
mmcv must be compiled from source**, which is slow (~1 h) and fragile.

Text encoder: BERT-base-uncased via `transformers` (see
`configs/mm_grounding_dino/usage.md` → "Download BERT weight").

## Inference entry point

```
demo/image_demo.py  --texts 'person wearing a red shirt.'
```

## ONNX / TensorRT difficulty — MEDIUM (best of the five)

This is the **only** system of the five with a first-party deployment path:
**MMDeploy** (<https://github.com/open-mmlab/mmdeploy>) supports ONNX Runtime and
TensorRT backends for MMDetection models and provides a TRT plugin for
MultiScaleDeformableAttention — which is exactly the op that makes the original
Grounding DINO so painful.

**MMDeploy is NOT cloned here** — it is not needed for source inspection, and the
brief said not to clone unrelated giant repos. Clone it only when you reach the
ONNX-export step (`docs/NEXT_STEPS.md` step 8). Caveat: MM-GDINO is a
text-conditioned model, and MMDeploy's grounding support is less exercised than
its plain-detector support — expect work, just less of it than with vanilla
Grounding DINO.

## Jetson Orin relevance — HIGH

The strongest detector candidate for this project: same architecture family as
Grounding DINO, better zero-shot numbers, an existing TensorRT story via
MMDeploy, and a Swin-T variant. The cost is the OpenMMLab dependency stack,
which is the most brittle of the five (mmcv-from-source on Jetson).
