# Environment — Grounded-Segment-Anything (Grounding DINO + SAM v1)

**Nothing here has been installed or run.** Notes from reading the upstream
dependency files, vendored under `upstream/`.

Source: `third_party/grounded_sam/` @ `126abe633ffe333e16e4a0a4e946bc1003caf757`

## Upstream expectations

| | |
|---|---|
| Python | 3.8–3.10 |
| PyTorch | unpinned; the Dockerfile targets CUDA 11.x-era torch |
| CUDA | 11.3+ (`Dockerfile` sets `TORCH_CUDA_ARCH_LIST`, `BUILD_WITH_CUDA`) |
| Also ships | `cog.yaml` (Replicate) |

Deps (`requirements.txt`) include `torch`, `torchvision`, `transformers`,
`diffusers`, `gradio`, `onnxruntime`, `supervision`, `timm`, `litellm`,
`fairscale`, `nltk`, `pycocotools`.

Note `diffusers`, `gradio` and `litellm` are pulled in by the *extra* demos
(inpainting, ChatGPT-assisted labelling, RAM/Tag2Text). The core
detect-then-segment path does not need them. This is a demo-heavy repo — it is a
**pipeline assembly**, not a new model.

## Custom CUDA operators — YES (inherited)

This repo **vendors its own copy** of Grounding DINO under `GroundingDINO/`, so
it carries the same `MultiScaleDeformableAttention` CUDA extension and the same
`nvcc` + `CUDA_HOME` build requirement. See
`environments/grounding_dino/README.md` — everything there applies.

SAM v1 itself (`segment_anything/`) is **pure PyTorch with no custom ops**, which
is why SAM exports to ONNX comparatively easily.

## Shared checkpoints — do not re-download

The demos want a Grounding DINO checkpoint *and* a SAM checkpoint. Both are
already here; point at them rather than duplicating:

| Demo expects | Use |
|---|---|
| `groundingdino_swint_ogc.pth` | `checkpoints/grounding_dino/groundingdino_swint_ogc.pth` |
| `sam_vit_h_4b8939.pth` | `checkpoints/sam/sam_vit_h_4b8939.pth` |

## Inference entry point

```
grounded_sam_demo.py         --input_image ... --text_prompt "person wearing a red shirt"
grounded_sam_simple_demo.py  (shortest path)
```

## ONNX / TensorRT difficulty — HIGH for the pipeline, LOW for SAM alone

SAM v1 has an **official ONNX export** for its lightweight mask decoder
(`scripts/export_onnx_model.py` upstream) — but only the decoder. The ViT-H
*image encoder* is the expensive part (~2.4 GB, ~0.5 s/frame on good hardware)
and is not covered by that export. The Grounding DINO half remains as hard as
ever.

## Jetson Orin relevance — LOW

Superseded by Grounded-SAM-2 for this project. SAM v1 has **no video tracking**
and no memory mechanism, so hitting 10+ Hz would mean re-segmenting every frame
with a ViT-H encoder — not viable on Orin. Keep it as a reference implementation
of the detect-then-segment pattern and for the original SAM paper; build on
Grounded-SAM-2 instead.
