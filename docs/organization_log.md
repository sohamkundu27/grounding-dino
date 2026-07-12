# Organization log

Record of how the curated literature collection was built: file moves/renames,
downloads, and the one policy change. Commit hashes are short SHAs on `main`.

## Policy change (documented deviation)
The repository originally **git-ignored all PDFs** (`papers/**`), treating papers
as "redistributable by link, reproduce with a script." For the curated literature
collection this was reversed: the paper PDFs under `papers/` are now **committed**
(they are small, redistributable academic PDFs from official sources). Model
weights and datasets remain untracked. See commit `3e52a29`
(`chore: stop git-ignoring paper PDFs...`). The drone-project reproducibility
manifest (`manifests/papers.json`) and its tooling were preserved; its paper
paths were re-pointed at the new taxonomy.

## Migrated from the pre-existing local collection (reused, verified, renamed, moved)
These PDFs were already downloaded locally (byte-identical to
`manifests/papers.json` checksums where overlapping) and were reused rather than
re-downloaded; each was renamed to `YYYY_FirstAuthor_Short_Title.pdf` and moved
into the taxonomy.

| Original filename | New path | Action | Commit |
|---|---|---|---|
| `2023_Liu_Grounding_DINO.pdf` | `papers/locally_runnable_systems/01_grounding_dino/2023_Liu_Grounding_DINO.pdf` | Moved | `7ee44b9` |
| `2024_Zhao_Open_Comprehensive_Pipeline_Unified_Object.pdf` | `papers/locally_runnable_systems/02_mm_grounding_dino/2024_Zhao_MM_Grounding_DINO.pdf` | Renamed + moved | `c123b56` |
| `2024_Ren_Grounded_SAM.pdf` | `papers/locally_runnable_systems/03_grounded_sam/2024_Ren_Grounded_SAM.pdf` | Moved | `ef602d3` |
| `2026_Fu_PET_DINO_Unifying_Visual_Cues.pdf` | `papers/locally_runnable_systems/05_pet_dino/2026_Fu_PET_DINO.pdf` | Renamed + moved | `7da646f` |
| `2022_Zhang_DINO_DETR_Improved_DeNoising_Anchor.pdf` | `papers/core_foundations/2022_Zhang_DINO.pdf` | Renamed + moved | `90aed86` |
| `2024_Ren_Grounding_DINO_1_5.pdf` | `papers/direct_extensions/2024_Ren_Grounding_DINO_1_5.pdf` | Moved | `76ab574` |
| `2024_Ren_DINO_X_Unified_Vision_Model.pdf` | `papers/direct_extensions/2024_Ren_DINO_X.pdf` | Renamed + moved | `1f170b9` |
| `2024_Wasim_VideoGrounding_DINO_Open_Vocabulary_Spatio.pdf` | `papers/tracking_and_video/2024_Wasim_VideoGrounding_DINO.pdf` | Renamed + moved | `1f00afa` |
| `2026_Wang_Unlocking_Potential_Grounding_DINO_Videos.pdf` | `papers/tracking_and_video/2026_Wang_Grounding_DINO_in_Videos.pdf` | Renamed + moved | `1b44a24` |
| `2024_Mumuni_Segment_Anything_Model_automated_image.pdf` | `papers/segmentation_integrations/2024_Mumuni_SAM_Annotation_Grounding_DINO.pdf` | Renamed + moved | `aabcad2` |
| `2024_Pijarowski_Utilizing_Grounded_SAM_self.pdf` | `papers/segmentation_integrations/2024_Pijarowski_Utilizing_Grounded_SAM.pdf` | Renamed + moved | `12a1ea7` |
| `2026_Korporaal_Colony_Grounded_SAM2.pdf` | `papers/segmentation_integrations/2026_Korporaal_Colony_Grounded_SAM2.pdf` | Moved | `abdfbf9` |
| `2025_Lu_Dynamic_DINO_Fine_Grained_Mixture.pdf` | `papers/deployment_and_efficiency/2025_Lu_Dynamic_DINO.pdf` | Renamed + moved | `659a6a1` |
| `2025_Rasaee_Grounding_DINO.pdf` | `papers/domain_adaptations/2025_Rasaee_Grounding_DINO_US_SAM.pdf` | Renamed + moved | `10330ea` |
| `2025_Singh_Few_Shot_Adaptation_Grounding_DINO.pdf` | `papers/domain_adaptations/2025_Singh_Few_Shot_Grounding_DINO_Agriculture.pdf` | Renamed + moved | `6e1911b` |
| `2026_Chen_Pseudo_Text_Conditioned_3D_Grounding.pdf` | `papers/domain_adaptations/2026_Chen_Pseudo_Text_3D_Grounding_DINO_CT.pdf` | Renamed + moved | `4527c03` |
| `2026_Munagala_Zero_Shot_Supervised_Bird_Image.pdf` | `papers/application_papers/2026_Munagala_Bird_Segmentation.pdf` | Renamed + moved | `afa00e8` |

## Downloaded (missing papers fetched from arXiv, %PDF-validated)

| Paper | New path | Action | Commit |
|---|---|---|---|
| DETR (2005.12872) | `papers/core_foundations/2020_Carion_DETR.pdf` | Downloaded | `4832dc0` |
| BERT (1810.04805) | `papers/core_foundations/2019_Devlin_BERT.pdf` | Downloaded | `63658e7` |
| Swin Transformer (2103.14030) | `papers/core_foundations/2021_Liu_Swin_Transformer.pdf` | Downloaded | `8bb5995` |
| Segment Anything (2304.02643) | `papers/core_foundations/2023_Kirillov_Segment_Anything.pdf` | Downloaded | `b8ce9ac` |
| SAM 2 (2408.00714) | `papers/core_foundations/2024_Ravi_SAM_2.pdf` | Downloaded | `e3bf2a0` |
| RefDrone (2502.00392) | `papers/domain_adaptations/2025_Sun_RefDrone.pdf` | Downloaded | `4103396` |

## No standalone paper
- **Grounded SAM 2** — software integration, no paper. Recorded as a Markdown
  reference at `papers/locally_runnable_systems/04_grounded_sam_2/README.md`; its
  academic foundations are Grounding DINO + SAM 2.

## Unavailable
- **Low-Rank Prompt Adaptation for Open-Vocabulary Object Detection** (ICCVW 2025,
  IEEE, DOI `10.1109/ICCVW69036.2025.00443`) — closed access, no open-access PDF.
  Recorded in `papers/unverified_or_pending/README.md` and `metadata/papers.json`
  (`download_status: unavailable`). No source fabricated.

## Duplicates
None. `scripts/find_duplicates.py` and `scripts/verify_papers.py` report no
duplicate SHA-256, no duplicate arXiv IDs/DOIs, and no orphan PDFs. No duplicate
PDFs were removed because none were created (each paper is stored once).

## Merge / divergence events
The remote `main` advanced during this work (a concurrent drone-project session
pushed RefDrone/VisDrone metadata and scripts). Resolved by `git rebase
origin/main` (no content conflicts — disjoint files; notes/papers vs
datasets/scripts) and normal push. **No force-push, no history rewrite of remote.**
