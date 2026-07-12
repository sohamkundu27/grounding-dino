#!/usr/bin/env python3
"""Offline, read-only integrity check for the literature-collection PDFs.

Strictly offline: it never contacts the network and never imports a model
package, deserialises a checkpoint, or runs inference. It only reads the PDFs
listed in metadata/papers.json and the files on disk under papers/.

Checks performed:
  * every metadata record with a local_path points to a file that exists
  * every PDF begins with the %PDF magic bytes
  * no zero-byte PDFs
  * no HTML pages saved with a .pdf extension
  * recorded SHA-256 matches the file on disk
  * duplicate checksums (same bytes under two paths)
  * every PDF on disk has a metadata record (no orphans)
  * broken/missing local paths for records marked present/downloaded

Exit code is non-zero if any hard problem is found.

Usage:
    python scripts/verify_papers.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
METADATA = REPO / "metadata" / "papers.json"
PAPERS_DIR = REPO / "papers"

problems: list[str] = []
notes: list[str] = []
ok_count = 0


def ok(msg: str) -> None:
    global ok_count
    ok_count += 1
    print(f"  ok    {msg}")


def bad(msg: str) -> None:
    problems.append(msg)
    print(f"  FAIL  {msg}")


def note(msg: str) -> None:
    notes.append(msg)
    print(f"  note  {msg}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def head(path: Path, n: int = 512) -> bytes:
    with path.open("rb") as fh:
        return fh.read(n)


def is_html(path: Path) -> bool:
    return head(path).lstrip().lower().startswith((b"<!doctype html", b"<html", b"<?xml"))


def main() -> int:
    data = json.loads(METADATA.read_text())
    papers = data["papers"]

    print("== metadata records ==")
    recorded_paths: dict[str, str] = {}   # abs relpath -> title
    hashes: dict[str, list[str]] = defaultdict(list)

    for p in papers:
        title = p.get("title", "<untitled>")
        local = p.get("local_path")
        status = p.get("download_status", "")
        if not local:
            if status in ("unavailable", "pending") or p.get("verification_status", "").startswith("verified_metadata"):
                note(f"{title[:60]}: no local PDF (status={status})")
            else:
                bad(f"{title[:60]}: record has no local_path but status={status}")
            continue

        recorded_paths[local] = title
        path = REPO / local
        if not path.exists():
            (bad if status in ("present", "downloaded") else note)(
                f"{Path(local).name}: absent (status={status})")
            continue
        if path.stat().st_size == 0:
            bad(f"{Path(local).name}: zero bytes")
            continue
        if head(path, 5).lstrip()[:4] != b"%PDF":
            bad(f"{Path(local).name}: does not begin with %PDF")
            continue
        if is_html(path):
            bad(f"{Path(local).name}: looks like HTML saved as PDF")
            continue
        digest = sha256_file(path)
        hashes[digest].append(local)
        if p.get("sha256") and p["sha256"] != digest:
            bad(f"{Path(local).name}: SHA-256 mismatch (metadata != disk)")
            continue
        if not p.get("sha256"):
            note(f"{Path(local).name}: present but no sha256 recorded (run generate_inventory.py)")
        ok(f"{Path(local).name} ({path.stat().st_size/1024:.0f} KB)")

    print("\n== duplicate checksums ==")
    dupes = {h: ps for h, ps in hashes.items() if len(ps) > 1}
    if dupes:
        for h, ps in dupes.items():
            bad(f"duplicate payload sha256 {h[:12]}...: {', '.join(ps)}")
    else:
        ok("no duplicate checksums")

    print("\n== orphan PDFs on disk (no metadata record) ==")
    orphans = 0
    if PAPERS_DIR.is_dir():
        for pdf in sorted(PAPERS_DIR.rglob("*.pdf")):
            rel = str(pdf.relative_to(REPO))
            if rel not in recorded_paths:
                bad(f"orphan (no metadata record): {rel}")
                orphans += 1
    if not orphans:
        ok("every PDF on disk has a metadata record")

    print("\n== stray temp files ==")
    strays = [str(p.relative_to(REPO)) for p in PAPERS_DIR.rglob("*.part")] if PAPERS_DIR.is_dir() else []
    if strays:
        for s in strays:
            bad(f"incomplete download present: {s}")
    else:
        ok("no .part files")

    print(f"\n{'='*60}")
    print(f"passed: {ok_count}   problems: {len(problems)}   notes: {len(notes)}")
    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print(f"  - {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
