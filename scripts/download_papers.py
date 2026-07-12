#!/usr/bin/env python3
"""Download the papers listed in manifests/papers.json.

Sources are restricted to arXiv / CVF Open Access / official project pages.
Every download is checked for the %PDF magic bytes, so an HTML interstitial or
a rate-limit page can never masquerade as a paper.

Usage:
    python scripts/download_papers.py --dry-run
    python scripts/download_papers.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("This script needs 'requests' (pip install requests).")

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "manifests" / "papers.json"

# arXiv asks automated clients to identify themselves and go easy on the rate.
HEADERS = {"User-Agent": "grounding-dino-research-repo/1.0 (paper archival; contact repo owner)"}
DELAY = 3.0
ALLOWED_HOSTS = ("arxiv.org", "openaccess.thecvf.com", "proceedings.neurips.cc", "openreview.net")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_pdf(path: Path) -> bool:
    with path.open("rb") as fh:
        return fh.read(5) == b"%PDF-"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = json.loads(MANIFEST.read_text())
    changed = False

    for p in data["papers"]:
        target = REPO / p["local_path"]
        print(f"{p['local_path']}")

        host = p["pdf_url"].split("/")[2]
        if not any(host.endswith(h) for h in ALLOWED_HOSTS):
            print(f"      REFUSED: {host} is not an approved paper host\n")
            p["status"] = "refused_unapproved_host"
            changed = True
            continue

        if target.exists() and is_pdf(target):
            p.update(sha256=sha256_file(target), status="present")
            changed = True
            print(f"      present\n")
            continue

        if args.dry_run:
            print(f"      DRY RUN -> would GET {p['pdf_url']}\n")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(".pdf.part")
        try:
            r = requests.get(p["pdf_url"], headers=HEADERS, timeout=(30, 120), allow_redirects=True)
            r.raise_for_status()
            part.write_bytes(r.content)

            if not is_pdf(part):
                head = part.read_bytes()[:80]
                print(f"      !! not a PDF (starts with {head[:40]!r}); discarding")
                part.unlink()
                p["status"] = "failed_not_a_pdf"
            else:
                part.rename(target)
                p.update(sha256=sha256_file(target), status="downloaded")
                print(f"      ok  {target.stat().st_size/1024:.0f} KB\n")
        except requests.RequestException as exc:
            print(f"      FAILED: {exc}\n")
            p["status"] = f"failed: {exc}"
            if part.exists():
                part.unlink()
        changed = True
        time.sleep(DELAY)

    if changed and not args.dry_run:
        MANIFEST.write_text(json.dumps(data, indent=2) + "\n")
        print(f"updated {MANIFEST.relative_to(REPO)}")

    return 1 if any(str(p["status"]).startswith(("failed", "refused")) for p in data["papers"]) else 0


if __name__ == "__main__":
    sys.exit(main())
