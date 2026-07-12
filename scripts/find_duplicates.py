#!/usr/bin/env python3
"""Report likely-duplicate papers. Read-only: never deletes anything.

Detects potential duplicates three ways:
  1. Identical SHA-256 (same bytes) among PDFs under papers/.
  2. Near-identical normalized filenames.
  3. Near-identical normalized titles in metadata/papers.json
     (also flags repeated arXiv IDs and DOIs).

Deletion is intentionally left to a human; this only surfaces candidates.

Usage:
    python scripts/find_duplicates.py
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
METADATA = REPO / "metadata" / "papers.json"
PAPERS_DIR = REPO / "papers"

_NON = re.compile(r"[^a-z0-9]+")


def norm(s: str) -> str:
    return _NON.sub(" ", s.lower()).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    found = False

    print("== identical file content (SHA-256) ==")
    hashes: dict[str, list[str]] = defaultdict(list)
    if PAPERS_DIR.is_dir():
        for pdf in sorted(PAPERS_DIR.rglob("*.pdf")):
            hashes[sha256_file(pdf)].append(str(pdf.relative_to(REPO)))
    dupes = {h: ps for h, ps in hashes.items() if len(ps) > 1}
    if dupes:
        found = True
        for h, ps in dupes.items():
            print(f"  DUP sha256 {h[:12]}...")
            for p in ps:
                print(f"      {p}")
    else:
        print("  none")

    print("\n== similar filenames ==")
    names = [str(p.relative_to(REPO)) for p in PAPERS_DIR.rglob("*.pdf")] if PAPERS_DIR.is_dir() else []
    flagged = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = Path(names[i]).stem, Path(names[j]).stem
            if SequenceMatcher(None, norm(a), norm(b)).ratio() >= 0.9:
                print(f"  similar: {names[i]}  <->  {names[j]}")
                flagged = found = True
    if not flagged:
        print("  none")

    print("\n== metadata: repeated arXiv IDs / DOIs / similar titles ==")
    data = json.loads(METADATA.read_text())
    papers = data["papers"]
    by_arxiv: dict[str, list[str]] = defaultdict(list)
    by_doi: dict[str, list[str]] = defaultdict(list)
    for p in papers:
        if p.get("arxiv_id"):
            by_arxiv[p["arxiv_id"]].append(p["title"])
        if p.get("doi"):
            by_doi[p["doi"].lower()].append(p["title"])
    meta_flagged = False
    for k, v in by_arxiv.items():
        if len(v) > 1:
            print(f"  repeated arXiv {k}: {v}")
            meta_flagged = found = True
    for k, v in by_doi.items():
        if len(v) > 1:
            print(f"  repeated DOI {k}: {v}")
            meta_flagged = found = True
    titles = [p["title"] for p in papers]
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            if SequenceMatcher(None, norm(titles[i]), norm(titles[j])).ratio() >= 0.9:
                print(f"  similar titles:\n      {titles[i]}\n      {titles[j]}")
                meta_flagged = found = True
    if not meta_flagged:
        print("  none")

    print(f"\n{'='*60}")
    print("duplicates found" if found else "no duplicates detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
