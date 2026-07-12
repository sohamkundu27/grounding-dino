# Environment — Grounding DINO (original, IDEA-Research)

**Nothing here has been installed or run.** These are notes from reading the
upstream dependency files, vendored under `upstream/`.

Source: `third_party/grounding_dino/` @ `856dde20aee659246248e20734ef9ba5214f5e44`

## Upstream expectations

| | |
|---|---|
| Python | 3.9–3.10 (`environment.yaml` pins a 3.9 conda lock) |
| PyTorch | unpinned in `requirements.txt`; the official Dockerfile uses **torch 2.1.2** |
| CUDA | **12.1** per `FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime` |
| Arch list | `TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"` |

Runtime deps (`requirements.txt`): `torch`, `torchvision`, `transformers`,
`addict`, `yapf`, `timm`, `numpy`, `opencv-python`, `supervision>=0.22.0`,
`pycocotools`.

## Custom CUDA operators — YES, compilation required

`setup.py` builds a `CUDAExtension` (falls back to `CppExtension` with no GPU):

```
groundingdino/models/GroundingDINO/csrc/MsDeformAttn/*   ->  groundingdino._C
```

This is **MultiScaleDeformableAttention**, the deformable-attention op from
Deformable DETR. Consequences:

- `pip install -e .` requires a **full CUDA toolkit** (`nvcc`), not just a
  runtime, with `CUDA_HOME` set. A torch wheel alone is not enough.
- The op is compiled against a specific torch ABI. Changing torch version means
  rebuilding.
- On Jetson, this must be recompiled for **SM 8.7** (Orin). The upstream arch
  list above stops at 8.6 and therefore **does not cover Orin** — you must add
  `8.7` to `TORCH_CUDA_ARCH_LIST`.

The text encoder is **BERT-base-uncased** via HuggingFace `transformers`, so
first run wants network access (or a pre-staged HF cache).

## Inference entry point

```
third_party/grounding_dino/demo/inference_on_a_image.py
groundingdino/util/inference.py  ->  load_model(), predict(), annotate()
```
Config: `groundingdino/config/GroundingDINO_SwinT_OGC.py` (Swin-T) or
`GroundingDINO_SwinB_cfg.py` (Swin-B). Weights: `checkpoints/grounding_dino/`.

## ONNX / TensorRT difficulty — HIGH

**There is no official ONNX or TensorRT export material in this repository.**
I grepped the whole tree; the only hit is an incidental mention in
`groundingdino/util/misc.py`. Any exporter you find is community-maintained.

Why it is hard:
1. **MultiScaleDeformableAttention** has no native ONNX op. It must be replaced
   with a decomposed `grid_sample` implementation or a custom TensorRT plugin.
   `grid_sample` needs opset ≥ 16 and is slow in TRT.
2. **Text-conditioned graph.** The BERT branch and the text–image cross-attention
   mean the graph is not a fixed-shape image-in/boxes-out CNN. Either export
   with dynamic axes for the token dimension, or **precompute text embeddings
   offline** and export only the vision+fusion path.
3. Post-processing (token-to-phrase mapping) is Python and must be reimplemented
   in C++ for the Hivemind pipeline.

**Practical route for Jetson:** freeze the prompt set, precompute BERT
embeddings on the host, export only the image branch with the deformable
attention rewritten, and treat the text encoder as a lookup. This is a real
engineering project, not an export flag.

## Jetson Orin relevance — MEDIUM-HIGH (as the detector, not the tracker)

Swin-T at ~700 MB is plausible on Orin's 64 GB for the *slow* detection stage,
which the project explicitly permits. It will **not** hit 10+ Hz on Orin without
significant work — that is what the SAM 2 tracker is for. Recompile the CUDA op
for SM 8.7.
