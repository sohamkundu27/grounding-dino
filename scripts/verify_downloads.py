#!/usr/bin/env python3
"""Passive integrity check over everything collected in this repository.

Strictly read-only. It never imports a model package, never deserialises a
checkpoint, never runs inference, and never deletes anything. Safe to run at
any time.

Checks:
  * manifest-listed files exist
  * SHA-256 matches the recorded checksum
  * PDFs begin with %PDF
  * checkpoints/archives are not HTML error pages saved under a binary name
  * no zero-byte files
  * no leftover .part files
  * duplicate payloads (same SHA-256 under different paths)
  * broken symlinks
  * expected third_party/ repositories are present and are real checkouts

Usage:
    python scripts/verify_downloads.py            # manifest files only (fast)
    python scripts/verify_downloads.py --deep     # + hash every file on disk
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFESTS = REPO / "manifests"
SCAN_DIRS = ("checkpoints", "datasets", "papers", "outputs")
EXPECTED_REPOS = (
    "grounding_dino", "mm_grounding_dino", "grounded_sam",
    "grounded_sam_2", "pet_dino", "refdrone",
)

ok_count = 0
problems: list[str] = []
notes: list[str] = []


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


def load(name: str) -> dict:
    p = MANIFESTS / name
    return json.loads(p.read_text()) if p.exists() else {}


def check_repositories() -> None:
    print("\n== third_party repositories ==")
    for name in EXPECTED_REPOS:
        path = REPO / "third_party" / name
        if not path.is_dir():
            bad(f"third_party/{name} is missing")
        elif not (path / ".git").exists():
            bad(f"third_party/{name} exists but is not a git checkout")
        elif not any(p for p in path.iterdir() if p.name != ".git"):
            bad(f"third_party/{name} is an empty checkout (submodule not initialised?)")
        else:
            ok(f"third_party/{name}")


def check_checkpoints() -> None:
    print("\n== checkpoints ==")
    data = load("checkpoints.json")
    for e in data.get("checkpoints", []):
        path = REPO / e["local_path"]
        label = f"{e['model']}/{e['checkpoint']}"
        if not path.exists():
            (bad if e.get("download_status") in ("downloaded", "present")
             else note)(f"{label}: absent (status={e.get('download_status')})")
            continue
        size = path.stat().st_size
        if size == 0:
            bad(f"{label}: zero bytes")
            continue
        if is_html(path):
            bad(f"{label}: HTML page saved as a checkpoint")
            continue
        if e.get("expected_size_bytes") and size != e["expected_size_bytes"]:
            bad(f"{label}: size {size} != expected {e['expected_size_bytes']}")
            continue
        if e.get("sha256"):
            if sha256_file(path) != e["sha256"]:
                bad(f"{label}: SHA-256 mismatch")
                continue
            ok(f"{label} ({size/1024**2:.0f} MB, sha256 verified)")
        else:
            note(f"{label}: present but no checksum on record")


def check_papers() -> None:
    print("\n== papers ==")
    for p in load("papers.json").get("papers", []):
        path = REPO / p["local_path"]
        name = Path(p["local_path"]).name
        if not path.exists():
            (bad if p.get("status") in ("downloaded", "present") else note)(f"{name}: absent")
            continue
        if head(path, 5) != b"%PDF-":
            bad(f"{name}: does not begin with %PDF")
            continue
        if p.get("sha256") and sha256_file(path) != p["sha256"]:
            bad(f"{name}: SHA-256 mismatch")
            continue
        ok(f"{name} ({path.stat().st_size/1024:.0f} KB)")


def check_datasets() -> None:
    print("\n== datasets ==")
    for d in load("datasets.json").get("datasets", []):
        if not d.get("local_path"):
            continue
        path = REPO / d["local_path"]
        label = f"{d['dataset']}/{d['component']}"
        if d["download_status"] == "downloaded":
            if not path.exists():
                bad(f"{label}: marked downloaded but absent")
            elif path.stat().st_size == 0:
                bad(f"{label}: zero bytes")
            else:
                ok(f"{label}")
        elif d["download_status"] in ("manual_action_required", "blocked_on_visdrone"):
            note(f"{label}: {d['download_status']} -- {(d.get('manual_action') or '')[:70]}")


def scan_tree(deep: bool) -> None:
    print("\n== filesystem scan ==")
    hashes: dict[str, list[Path]] = defaultdict(list)
    parts = zeros = broken = 0

    for root in SCAN_DIRS:
        base = REPO / root
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_symlink() and not path.exists():
                bad(f"broken symlink: {path.relative_to(REPO)}")
                broken += 1
                continue
            if not path.is_file() or path.is_symlink():
                continue
            if path.suffix == ".part":
                bad(f"incomplete download: {path.relative_to(REPO)}")
                parts += 1
                continue
            if path.stat().st_size == 0 and path.name != ".gitkeep":
                bad(f"zero-byte file: {path.relative_to(REPO)}")
                zeros += 1
                continue
            if deep and path.stat().st_size > 1024:
                hashes[sha256_file(path)].append(path)

    if not (parts or zeros or broken):
        ok("no .part files, no zero-byte files, no broken symlinks")

    if deep:
        dupes = {h: ps for h, ps in hashes.items() if len(ps) > 1}
        if dupes:
            print()
            for h, ps in dupes.items():
                note(f"duplicate payload (sha256 {h[:12]}...):")
                for p in ps:
                    print(f"          {p.relative_to(REPO)}")
        else:
            ok("no duplicate payloads")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deep", action="store_true",
                    help="hash every file on disk to find duplicates (slow)")
    args = ap.parse_args()

    print(f"verifying {REPO}")
    check_repositories()
    check_checkpoints()
    check_papers()
    check_datasets()
    scan_tree(args.deep)

    print(f"\n{'='*60}")
    print(f"passed: {ok_count}   problems: {len(problems)}   notes: {len(notes)}")
    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print(f"  - {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
