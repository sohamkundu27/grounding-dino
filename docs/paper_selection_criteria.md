# Paper selection criteria

What qualifies a paper for the curated Grounding DINO literature collection under
`papers/`, and how it is placed.

## Inclusion — a paper is included if it does at least one of:
- introduces Grounding DINO;
- directly modifies Grounding DINO;
- extends Grounding DINO;
- reproduces or reimplements Grounding DINO;
- adapts Grounding DINO in a technically meaningful way;
- benchmarks Grounding DINO as a central subject;
- integrates Grounding DINO into a substantial detection, segmentation, or
  tracking system;
- proposes an efficient or deployable variant;
- is a **foundational** paper required to understand the five locally runnable
  systems (DETR, DINO, BERT, Swin, SAM, SAM 2).

## Exclusion
- Papers that merely use Grounding DINO as a minor annotation utility, unless they
  were already present and are genuinely useful.
- Anything requiring a fabricated citation, DOI, venue, or PDF source.
- Random mirrors / Scribd / ResearchGate / unverified uploads when an official
  source exists.
- This is a curated collection, **not** a dump of every paper that mentions
  Grounding DINO once.

## Verification (every paper)
1. Prefer official sources, in order: arXiv → CVF Open Access → official
   proceedings → official publisher → official author/university page.
2. Verify: exact title, authors, year, venue, arXiv ID, DOI, official page, PDF URL.
3. Use the newest official version.
4. Never fabricate a missing field (leave it empty).
5. PDF validity: non-empty, begins with `%PDF`, not an HTML error page, SHA-256 recorded.

## Placement rules
- Each paper is stored in **exactly one** folder (no duplicate PDFs across
  folders); relationships are expressed via metadata, not copies.
- The five locally runnable systems' **primary** papers live under
  `papers/locally_runnable_systems/NN_*/`.
- Shared **foundations** (DETR, DINO, BERT, Swin, SAM, SAM 2) live in
  `papers/core_foundations/` even though several systems depend on them.
- Everything else is filed by category: `direct_extensions`, `tracking_and_video`,
  `segmentation_integrations`, `deployment_and_efficiency`, `domain_adaptations`,
  `application_papers`, `unverified_or_pending`.

## Filenames
`YYYY_FirstAuthor_Short_Title.pdf`, using the **verified** year (arXiv submission
year and conference year are not assumed identical).

## Categories in this collection
See `metadata/papers.csv` / `metadata/papers.json` for the authoritative mapping,
and `metadata/download_status.md` for per-paper status.
