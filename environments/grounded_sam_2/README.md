# Environment — Grounded-SAM-2 (Grounding DINO 1.0 + SAM 2 video tracking)

**Nothing here has been installed or run.** Notes from reading the upstream
dependency files, vendored under `upstream/`.

Source: `third_party/grounded_sam_2/` @ `b7a9c29f196edff0eb54dbe14588d7ae5e3dde28`

> **This is the most directly relevant repository in the project.** It is the
> only one that pairs local open-vocabulary detection with a real video tracker.

## Upstream expectations

| | |
|---|---|
| Python | **≥ 3.10** (`setup.py`) |
| PyTorch | **≥ 2.3.1** (`torch>=2.3.1`, `torchvision>=0.18.1`) |
| CUDA | **12.1** (`FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel`) |
| numpy | ≥ 1.24.4 |
| Also | `hydra-core>=1.3.2` (SAM 2 configs), `iopath`, `pillow`, `tqdm` |
| Video extras | `eva-decord`, `av>=13.0.0`, `opencv-python` |

Note the torch floor (**≥2.3.1**) is *higher* than Grounding DINO's Docker pin
(2.1.2) and clashes with the OpenMMLab stack's mmcv wheels. **Do not try to
share one environment with MM-GDINO / PET-DINO / RefDrone.**

## Custom CUDA operators — YES for Grounding DINO, OPTIONAL for SAM 2

- The bundled `grounding_dino/` copy needs the same `MultiScaleDeformableAttention`
  extension (`nvcc`, `CUDA_HOME`). Same story as everywhere else.
- SAM 2 builds an **optional** CUDA extension for connected components
  (`BUILD_WITH_CUDA` / `USE_CUDA=0` in the Dockerfile). SAM 2 runs without it;
  you lose only some mask post-processing. Useful escape hatch on Jetson.

## Local vs cloud-API — important

The repo mixes both. **Only the local demos are usable for this project.**

**Fully local** (weights on disk, nothing leaves the machine):

| Script | What |
|---|---|
| `grounded_sam2_local_demo.py` | GD 1.0 (local ckpt) + SAM 2, image |
| `grounded_sam2_hf_model_demo.py` | GD 1.0 via HF hub + SAM 2, image |
| `grounded_sam2_tracking_demo.py` | **video tracking** |
| `grounded_sam2_tracking_demo_with_continuous_id.py` | video tracking, persistent IDs |
| `grounded_sam2_tracking_demo_custom_video_input_gd1.0_local_model.py` | **your own video, fully local** ← start here |
| `grounded_sam2_tracking_camera_with_continuous_id.py` | live camera |

**Cloud-API-backed — NOT local, requires a DDS token, sends your images to a
remote server:**

`grounded_sam2_gd1.5_demo.py`, `grounded_sam2_dinox_demo.py`,
`grounded_sam2_tracking_demo_with_gd1.5.py`,
`grounded_sam2_tracking_demo_with_continuous_id_gd1.5.py`,
`grounded_sam2_tracking_demo_custom_video_input_gd1.5.py`,
`grounded_sam2_tracking_demo_custom_video_input_dinox.py`

These import `dds_cloudapi_sdk` and need `API_TOKEN`. **Grounding DINO 1.5, 1.6
and DINO-X have no public local checkpoints** — they are a hosted service. No
token was obtained, no API was called, no image was sent anywhere. Do not plan
the Jetson pipeline around them.

## Inference entry point

```
grounded_sam2_tracking_demo_custom_video_input_gd1.0_local_model.py
```
Weights: `checkpoints/grounding_dino/groundingdino_swint_ogc.pth` +
`checkpoints/sam2/sam2.1_hiera_*.pt`. SAM 2 configs are Hydra YAMLs under
`sam2/configs/`.

## ONNX / TensorRT difficulty — MIXED

- **SAM 2 image encoder (Hiera):** plain transformer, exports reasonably.
- **SAM 2 memory-attention / memory-bank:** this is the hard part. The tracker is
  *stateful* — the graph carries a memory bank across frames. ONNX is a
  stateless dataflow graph, so you must externalise the memory as explicit
  I/O tensors and drive the loop from C++. Expect to export **three or four
  separate engines** (image encoder, memory attention, mask decoder, memory
  encoder) and orchestrate them yourself.
- **Grounding DINO half:** unchanged, still hard (see the grounding_dino notes).

## Jetson Orin relevance — HIGHEST

This is the architecture the project is actually reaching for: run the heavy
open-vocabulary detector **once** (slow is fine, per the brief) to acquire the
target from the prompt, then hand the box to **SAM 2 tracking**, which is the
10+ Hz component. `sam2.1_hiera_tiny` (156 MB) and `sam2.1_hiera_small` (184 MB)
are the realistic Orin candidates; benchmark those first.
