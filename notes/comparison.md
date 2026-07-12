# Comparison of the five locally runnable systems

All rows are qualitative and come from reading the papers, code and configs — **no
metrics were run or invented.** "Local weights" means public checkpoints usable
without a cloud API.

| System | Primary paper | Detector or pipeline | Text prompting | Visual prompting | Segmentation | Video tracking | Local code | Local weights | API dependence | Likely embedded suitability | Main research value |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Grounding DINO 1.0** | Liu et al. 2023 | Detector | Yes | No | No | No | Yes | Yes (Swin-T/B) | None | Medium (slow acquire stage) | The open-set detector the whole family is built on |
| **MM-Grounding-DINO** | Zhao et al. 2024 | Detector (in MMDetection) | Yes | No | No | No | Yes | Yes (Swin-T/B/L) | None | Medium–High (best deploy tooling) | Reproducible training + first-party TensorRT path |
| **Grounded SAM** | Ren et al. 2024 | Pipeline (GDINO+SAM) | Yes | Via boxes | Yes | No | Yes | Reuses GDINO+SAM | None (core) | Low (two heavy models) | Open-vocab detect→segment; auto-annotation |
| **Grounded SAM 2** | *no paper* (integration) | Pipeline (GDINO+SAM 2) | Yes | Via boxes | Yes | **Yes** | Yes | Reuses GDINO+SAM 2 | Partial (best detectors API-only) | **Highest** (SAM 2 tiny tracker) | Detect→segment→**track**; the target architecture |
| **PET-DINO** | Fu et al. 2026 | Detector | Yes | **Yes** | No | No | Yes | Partial (Swin-T) | None | Medium | Unified text+visual prompting |

## One-line takeaways
- **Best "acquire from language" model with deploy tooling:** MM-Grounding-DINO.
- **Best end-to-end target architecture (acquire once, track fast):** Grounded SAM 2.
- **Best when the target is easier to *show* than to *name*:** PET-DINO.
- **Grounded SAM / Grounded SAM 2 are pipelines**, not independent base detectors —
  they reuse Grounding DINO for detection and SAM/SAM 2 for masks/tracking.
- **API-only, no local weights:** Grounding DINO 1.5/1.6 and DINO-X (do not design
  the on-device pipeline around them).
