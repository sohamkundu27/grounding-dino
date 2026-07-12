# Dependency Notes

The five stacks compared **without installing any of them**. Everything below
comes from reading `requirements.txt`, `environment.yaml`, `pyproject.toml`,
`setup.py` and `Dockerfile` in each upstream repo (copies under
`environments/<model>/upstream/`).

Per-model detail lives in `environments/<model>/README.md`. This file is the
cross-cutting view: **what conflicts, and why you cannot have one environment.**

## The five stacks

| | Grounding DINO | MM-GDINO | Grounded-SAM | **Grounded-SAM-2** | PET-DINO |
|---|---|---|---|---|---|
| Python | 3.9–3.10 | 3.7–3.11 | 3.8–3.10 | **≥ 3.10** | 3.10 |
| PyTorch | 2.1.2 (Docker) | ≥1.8, practically 2.x | unpinned | **≥ 2.3.1** | 2.x |
| CUDA | 12.1 | 11.8 / 12.1 | 11.3+ | **12.1** | 11.8 / 12.1 |
| numpy | unpinned | unpinned | unpinned | **≥ 1.24.4** | **== 1.23** |
| Framework | standalone | **OpenMMLab** | standalone | standalone | **OpenMMLab** |
| Custom CUDA op | ✅ compile | ✅ (via mmcv) | ✅ compile | ✅ GD; optional SAM 2 | ✅ (via mmcv) |
| Text encoder | BERT (`transformers`) | BERT | BERT | BERT | BERT |

They split cleanly into two families:

- **Standalone / IDEA-Research**: Grounding DINO, Grounded-SAM, Grounded-SAM-2.
  Ordinary pip stacks. The pain is compiling `MultiScaleDeformableAttention`.
- **OpenMMLab**: MM-Grounding-DINO, PET-DINO (and RefDrone/NGDINO). Add
  `mmengine` + `mmcv`, which are strictly version-bounded and ship their own
  compiled CUDA ops. The pain is `mmcv`.

## Hard conflicts — these are not negotiable

### 1. numpy: PET-DINO `==1.23` vs Grounded-SAM-2 `>=1.24.4`

Directly contradictory. PET-DINO's README says so explicitly (the LVIS API used
on its eval path breaks on numpy ≥ 1.24); Grounded-SAM-2's `setup.py` floors at
1.24.4. **One environment cannot satisfy both.**

### 2. mmcv `<2.2.0` — and RefDrone's README is wrong

`mmdet/__init__.py` (v3.3.0, shared by MM-GDINO, PET-DINO and RefDrone):

```python
mmcv_minimum_version = '2.0.0rc4'
mmcv_maximum_version = '2.2.0'
assert mmcv_version >= ... and mmcv_version < digit_version(mmcv_maximum_version)
```

`third_party/refdrone/README.md` line 32 says:

```bash
mim install "mmcv==2.2.0"      # ← excluded by the assertion above
```

**Following RefDrone's own install instructions produces a stack that aborts on
`import mmdet`.** Use `mmcv==2.1.0`. This is a genuine upstream bug, not a
misreading — the bound is exclusive.

### 3. PyTorch: 2.1.x (Grounding DINO / mmcv wheels) vs ≥2.3.1 (Grounded-SAM-2)

Not merely a preference. `mmcv` wheels are built against a **specific torch
version**, and the deformable-attention extension is compiled against a specific
torch ABI. Mixing means recompiling, and mmcv-from-source is the slowest,
most failure-prone build in the set.

## Recommended layout — four environments

Do **not** try to merge these.

| Env | Covers | Key pins |
|---|---|---|
| `ovd-gsam2` | **Grounded-SAM-2** (+ SAM 2) | py3.10, torch 2.3.1+cu121, numpy≥1.24 |
| `ovd-gdino` | Grounding DINO, Grounded-SAM | py3.10, torch 2.1.2+cu121 |
| `ovd-mmdet` | MM-Grounding-DINO, RefDrone/NGDINO | py3.10, torch 2.1.0+cu121, **mmcv==2.1.0**, mmengine<1.0 |
| `ovd-petdino` | PET-DINO | as `ovd-mmdet` but **numpy==1.23** |

`ovd-mmdet` and `ovd-petdino` differ only by numpy, but that one pin is load-bearing —
keep them apart rather than fighting it.

Start with **`ovd-gsam2`**. It is the target architecture and the least entangled.

## Cross-cutting requirements

**A full CUDA toolkit, not just a runtime.** Every one of the five needs `nvcc`
and `CUDA_HOME` at install time, because they all compile
`MultiScaleDeformableAttention` (directly, or inside mmcv). A `pytorch-cuda` conda
package alone is not enough — this is the single most common install failure in
this family of repos.

**BERT-base-uncased is downloaded at first run** by `transformers`. Pre-stage the
HF cache for an offline or air-gapped machine.

**`TORCH_CUDA_ARCH_LIST` does not cover Orin.** Upstream defaults stop at `8.6`
(Grounding DINO: `"6.0 6.1 7.0 7.5 8.0 8.6+PTX"`; Grounded-SAM-2: `"7.0;7.5;8.0;8.6"`).
**Jetson AGX Orin is SM 8.7.** You must add `8.7` before compiling, or you will
get a kernel-launch failure at runtime rather than a build error — which is a
miserable way to find out.

## Jetson-specific warnings

1. **No prebuilt `mmcv` wheel exists for Jetson/aarch64.** It must be compiled
   from source (~1 hour, fragile). This is the biggest practical argument against
   putting the OpenMMLab stack on the Orin, and it partly offsets MMDeploy's
   TensorRT advantage.
2. **Use NVIDIA's L4T PyTorch containers**, not pip wheels — `pip install torch`
   fetches an x86 build or a CPU-only aarch64 build.
3. **SAM 2's CUDA extension is optional** (`USE_CUDA=0`). It only affects mask
   post-processing. Genuinely useful escape hatch on Orin.
4. The realistic Jetson path is **not** "install these five repos on the Orin".
   It is: pick a model on the workstation, export ONNX → TensorRT, and run the
   engine from C++ with no Python ML stack on the device at all. The dependency
   analysis above matters for the *workstation* phase; the Orin should ideally
   see only engines.

## Bottom line

The dependency situation is unremarkable for this field — two families, one
compiled op that everything depends on, and one genuinely wrong install
instruction in RefDrone's README. Budget setup time for `mmcv` and for the
CUDA-toolkit requirement, keep four environments, and expect the Orin to run
TensorRT engines rather than any of these stacks.
