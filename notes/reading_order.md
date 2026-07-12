# Recommended reading order

A path from the transformer-detection and language foundations up to the five
locally runnable systems and their extensions. Paths in brackets are where each
item lives in this repo.

1. **DETR** — End-to-End Object Detection with Transformers (Carion et al., 2020)
   `papers/core_foundations/2020_Carion_DETR.pdf`
   *Why:* the transformer set-prediction detector everything below descends from.

2. **DINO** — DETR with Improved DeNoising Anchor Boxes (Zhang et al., 2022)
   `papers/core_foundations/2022_Zhang_DINO.pdf`
   *Why:* the exact detector Grounding DINO builds on (denoising, query selection).

3. **BERT** — Pre-training of Deep Bidirectional Transformers (Devlin et al., 2019)
   `papers/core_foundations/2019_Devlin_BERT.pdf`
   *Why:* the text encoder that embeds the language prompt.

4. **Swin Transformer** — Hierarchical Vision Transformer (Liu et al., 2021)
   `papers/core_foundations/2021_Liu_Swin_Transformer.pdf`
   *Why:* the image backbone used by Grounding DINO / MM-Grounding-DINO.

5. **Grounding DINO** (Liu et al., 2023)
   `papers/locally_runnable_systems/01_grounding_dino/2023_Liu_Grounding_DINO.pdf`
   *Why:* system #1 — marries DINO + grounded language pre-training.

6. **MM-Grounding-DINO** (Zhao et al., 2024)
   `papers/locally_runnable_systems/02_mm_grounding_dino/2024_Zhao_MM_Grounding_DINO.pdf`
   *Why:* system #2 — open, reproducible reimplementation in MMDetection.

7. **Segment Anything** (Kirillov et al., 2023)
   `papers/core_foundations/2023_Kirillov_Segment_Anything.pdf`
   *Why:* the promptable segmenter used by Grounded SAM.

8. **Grounded SAM** (Ren et al., 2024)
   `papers/locally_runnable_systems/03_grounded_sam/2024_Ren_Grounded_SAM.pdf`
   *Why:* system #3 — detect→segment pipeline (GDINO + SAM).

9. **SAM 2** (Ravi et al., 2024)
   `papers/core_foundations/2024_Ravi_SAM_2.pdf`
   *Why:* adds memory-based video segmentation/tracking.

10. **Grounded SAM 2 documentation** (software integration, no paper)
    `papers/locally_runnable_systems/04_grounded_sam_2/README.md`
    *Why:* system #4 — detect→segment→**track**; the target architecture.

11. **PET-DINO** (Fu et al., 2026)
    `papers/locally_runnable_systems/05_pet_dino/2026_Fu_PET_DINO.pdf`
    *Why:* system #5 — unifies text + visual prompts.

12. **Relevant efficiency and video extensions**
    - Dynamic-DINO (real-time MoE) — `papers/deployment_and_efficiency/2025_Lu_Dynamic_DINO.pdf`
    - VideoGrounding-DINO — `papers/tracking_and_video/2024_Wasim_VideoGrounding_DINO.pdf`
    - Grounding DINO in Videos (parameter-efficient) — `papers/tracking_and_video/2026_Wang_Grounding_DINO_in_Videos.pdf`
    - Grounding DINO 1.5 / DINO-X (API-only successors) — `papers/direct_extensions/`
    - Domain adaptations (aerial/medical/agri) — `papers/domain_adaptations/`
