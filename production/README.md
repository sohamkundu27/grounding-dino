# Production Candidates — Five Locally Runnable Systems

The five open-vocabulary systems selected for evaluation on the drone
target-finding project, in one place.

**This folder holds papers and reference documentation only.** No model code, no
checkpoints, no datasets, no environments, no outputs, no ONNX/TensorRT/Docker
assets. Those live elsewhere in the repository:

| What | Where |
|---|---|
| Upstream source (pinned submodules) | [`../third_party/`](../third_party/) |
| Model weights (~17 GB) | `../checkpoints/` |
| RefDrone + VisDrone | `../datasets/` |
| Dependency notes per system | [`../environments/`](../environments/) |
| Wider literature collection | [`../papers/`](../papers/) |

The PDFs here are **relative symlinks** to the canonical copies under
[`../papers/`](../papers/) — one payload each, no duplicates. Verify with
`sha256sum -c production/metadata/checksums.sha256`.

> Nothing has been run. No inference, no training, no benchmarks. Everything below
> comes from reading papers, source, and configs. **No benchmark numbers are
> quoted, because none have been measured.**

## Summary

| # | System | Type | Primary paper | Official GitHub | Text | Visual | Seg | Track | Code | Weights | API | Planned role |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Grounding DINO 1.0** | detector | [Liu et al. 2023](https://arxiv.org/abs/2303.05499) · [local](papers/01_grounding_dino/) | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | Baseline detector |
| 2 | **MM-Grounding-DINO** | detector | [Zhao et al. 2024](https://arxiv.org/abs/2401.02361) · [local](papers/02_mm_grounding_dino/) | [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | **Strongest production detector candidate** |
| 3 | **Grounded Segment Anything** | segmentation pipeline | [Ren et al. 2024](https://arxiv.org/abs/2401.14159) · [local](papers/03_grounded_sam/) | [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ ¹ | ❌ | Do masks improve localisation? |
| 4 | **Grounded SAM 2** | tracking pipeline | [Ravi et al. 2024 (SAM 2)](https://arxiv.org/abs/2408.00714) ² · [local](papers/04_grounded_sam_2/) | [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ ¹ | ❌ ³ | **Full acquisition + tracking baseline** |
| 5 | **PET-DINO** | detector | [Fu et al. 2026](https://arxiv.org/abs/2604.00503) · [local](papers/05_pet_dino/) | [fuweifuvtoo/PET_DINO](https://github.com/fuweifuvtoo/PET_DINO) | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Prompt-enriched detector comparison |

¹ Through its component models — these pipelines have no weights of their own.
² **No standalone Grounded SAM 2 paper exists.** SAM 2 is its primary academic
reference; see [`papers/04_grounded_sam_2/NO_STANDALONE_PAPER.md`](papers/04_grounded_sam_2/NO_STANDALONE_PAPER.md).
³ **For the local path only.** The same repo also ships API-backed Grounding DINO
1.5 / 1.6 / DINO-X demos, which have no public local weights — see
[Important distinction](#important-distinction).

Machine-readable: [`metadata/systems.json`](metadata/systems.json) ·
[`metadata/systems.csv`](metadata/systems.csv)
