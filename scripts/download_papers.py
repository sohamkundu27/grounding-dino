#!/usr/bin/env python3
"""Download the curated literature-collection PDFs listed in metadata/papers.json.

Only official hosts are allowed (arXiv / CVF Open Access / official proceedings).
Every download is streamed to a ``.part`` file, checked for the ``%PDF`` magic
bytes (so an HTML interstitial or rate-limit page can never masquerade as a
paper), hashed with SHA-256, and only then moved into place. Valid existing
files are skipped, so the script is idempotent and safe to re-run.

This script NEVER executes, imports, or deserialises anything it downloads.

Usage:
    python scripts/download_papers.py --dry-run
    python scripts/download_papers.py
    python scripts/download_papers.py --only 2005.12872 1810.04805
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
except ImportError:  # pragma: no cover
    sys.exit("This script needs 'requests' (pip install requests).")

REPO = Path(__file__).resolve().parent.parent
METADATA = REPO / "metadata" / "papers.json"

HEADERS = {
    "User-Agent": (
        "grounding-dino-literature-collection/1.0 "
        "(academic paper archival; +https://github.com/sohamkundu27/grounding-dino)"
    )
}
ALLOWED_HOSTS = (
    "arxiv.org",
    "openaccess.thecvf.com",
    "proceedings.neurips.cc",
    "openreview.net",
    "proceedings.mlr.press",
)
TIMEOUT = (30, 180)          # (connect, read)
MAX_RETRIES = 4
RETRY_BACKOFF = 3.0
DELAY = 3.0                  # polite pause between downloads
CHUNK = 1 << 16
MIN_PDF_BYTES = 5000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_pdf(path: Path) -> bool:
    with path.open("rb") as fh:
        return fh.read(5).lstrip()[:4] == b"%PDF"


def host_of(url: str) -> str:
    return url.split("/")[2] if "//" in url else ""


def approved(url: str) -> bool:
    host = host_of(url)
    return any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS)


def fetch(url: str) -> bytes | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, headers=HEADERS, timeout=TIMEOUT,
                              stream=True, allow_redirects=True) as r:
                if r.status_code == 200:
                    return b"".join(r.iter_content(CHUNK))
                if r.status_code in (429, 500, 502, 503, 504):
                    wait = RETRY_BACKOFF * attempt
                    print(f"      HTTP {r.status_code}; retry {attempt}/{MAX_RETRIES} in {wait:.0f}s")
                    time.sleep(wait)
                    continue
                print(f"      HTTP {r.status_code}; giving up")
                return None
        except requests.RequestException as exc:
            wait = RETRY_BACKOFF * attempt
            print(f"      error: {exc}; retry {attempt}/{MAX_RETRIES} in {wait:.0f}s")
            time.sleep(wait)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="show actions, download nothing")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these arXiv IDs (or path substrings)")
    args = ap.parse_args()

    data = json.loads(METADATA.read_text())
    papers = data["papers"]
    changed = False
    failures = 0

    for p in papers:
        local = p.get("local_path")
        if not local:
            continue  # e.g. unverified_or_pending records with no PDF
        if args.only and not any(
            tok in (p.get("arxiv_id", "") or "") or tok in local for tok in args.only
        ):
            continue

        target = REPO / local
        print(local)

        if target.exists() and is_pdf(target):
            digest = sha256_file(target)
            if p.get("sha256") and p["sha256"] != digest:
                print(f"      WARNING: on-disk sha256 differs from metadata record")
            p["sha256"], p["download_status"] = digest, "present"
            changed = True
            print("      present (valid PDF, skipped)")
            continue

        pdf_url = p.get("pdf_url", "")
        if not pdf_url:
            print("      no pdf_url on record; skipping (see notes)")
            continue
        if not approved(pdf_url):
            print(f"      REFUSED: {host_of(pdf_url)} is not an approved host")
            p["download_status"] = "refused_unapproved_host"
            changed = True
            failures += 1
            continue

        if args.dry_run:
            print(f"      DRY RUN -> would GET {pdf_url}")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        body = fetch(pdf_url)
        if body is None:
            p["download_status"] = "failed_download"
            changed = True
            failures += 1
            continue

        part = target.with_suffix(target.suffix + ".part")
        part.write_bytes(body)
        if part.read_bytes()[:5].lstrip()[:4] != b"%PDF" or part.stat().st_size < MIN_PDF_BYTES:
            head = part.read_bytes()[:60]
            print(f"      !! not a valid PDF (head={head[:40]!r}); discarding")
            part.unlink(missing_ok=True)
            p["download_status"] = "failed_not_a_pdf"
            changed = True
            failures += 1
            continue

        part.replace(target)
        p["sha256"], p["download_status"] = sha256_file(target), "downloaded"
        changed = True
        print(f"      ok  {target.stat().st_size/1024:.0f} KB  sha256={p['sha256'][:12]}...")
        time.sleep(DELAY)

    if changed and not args.dry_run:
        METADATA.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        print(f"updated {METADATA.relative_to(REPO)}")

    print(f"\ndone; failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
