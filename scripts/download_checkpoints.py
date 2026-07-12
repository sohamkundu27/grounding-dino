#!/usr/bin/env python3
"""Download model checkpoints listed in manifests/checkpoints.json.

Safety contract (deliberate, do not "optimise" away):
  * Files are streamed to disk and NEVER deserialised. No torch.load, no pickle,
    no importing of any model package. A .pth file is treated as opaque bytes.
  * Downloads land in <target>.part and are only renamed into place once the
    byte count and (when known) the expected size agree.
  * A file that already exists and matches its recorded SHA-256 is skipped.
  * Broken partial files are REPORTED before they are removed.

Usage:
    python scripts/download_checkpoints.py --dry-run
    python scripts/download_checkpoints.py                 # everything pending
    python scripts/download_checkpoints.py --model sam2    # one model group
    python scripts/download_checkpoints.py --recheck       # re-hash what exists
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("This script needs 'requests' (pip install requests).")

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "manifests" / "checkpoints.json"
CHECKSUMS = REPO / "manifests" / "checksums.sha256"

CHUNK = 1 << 20  # 1 MiB
RETRIES = 5
TIMEOUT = (30, 120)  # (connect, read)


def human(n: int | None) -> str:
    if n is None:
        return "unknown"
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f} {unit}" if unit != "B" else f"{int(f)} B"
        f /= 1024
    return f"{f:.1f} TB"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def looks_like_html(path: Path) -> bool:
    """Catch login walls / error pages saved under a .pth name."""
    with path.open("rb") as fh:
        head = fh.read(512).lstrip().lower()
    return head.startswith((b"<!doctype html", b"<html", b"<?xml"))


def free_bytes() -> int:
    st = os.statvfs(REPO)
    return st.f_bavail * st.f_frsize


def download(url: str, target: Path, expected: int | None) -> tuple[bool, str]:
    """Stream url -> target. Returns (ok, message). Resumes a .part if allowed."""
    part = target.with_suffix(target.suffix + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {}
        mode = "wb"
        if have:
            # Only resume if the server honours ranges; verified below via 206.
            headers["Range"] = f"bytes={have}-"

        try:
            with requests.get(
                url, stream=True, timeout=TIMEOUT, headers=headers, allow_redirects=True
            ) as r:
                if have and r.status_code == 206:
                    mode = "ab"
                elif have and r.status_code == 200:
                    # Server ignored Range: restart cleanly rather than corrupt.
                    print(f"      server ignored resume; restarting from 0")
                    have, mode = 0, "wb"
                r.raise_for_status()

                total = None
                if "content-length" in r.headers:
                    total = int(r.headers["content-length"]) + have

                done = have
                last = time.time()
                with part.open(mode) as fh:
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 5:
                            pct = f"{100 * done / total:5.1f}%" if total else "  ?  "
                            print(f"      {pct}  {human(done)}", flush=True)
                            last = time.time()

            size = part.stat().st_size
            if expected is not None and size != expected:
                return False, f"size mismatch: got {size}, expected {expected}"
            if size == 0:
                return False, "zero-byte download"
            if looks_like_html(part):
                # Report, then remove -- an HTML error page is never a checkpoint.
                print(f"      !! HTML page received instead of a binary; discarding")
                part.unlink()
                return False, "server returned an HTML page (auth wall or dead URL?)"

            part.rename(target)
            return True, "ok"

        except (requests.RequestException, OSError) as exc:
            wait = min(2**attempt, 30)
            print(f"      attempt {attempt}/{RETRIES} failed: {exc}; retry in {wait}s")
            if attempt == RETRIES:
                return False, f"failed after {RETRIES} attempts: {exc}"
            time.sleep(wait)

    return False, "unreachable"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="only this model group (e.g. sam2)")
    ap.add_argument("--dry-run", action="store_true", help="plan only, download nothing")
    ap.add_argument("--recheck", action="store_true", help="re-hash existing files")
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=20.0,
        help="refuse to start a download that would drop free space below this",
    )
    args = ap.parse_args()

    data = json.loads(MANIFEST.read_text())
    entries = data["checkpoints"]
    if args.model:
        entries = [e for e in entries if e["model"] == args.model]
        if not entries:
            return print(f"no checkpoints for model '{args.model}'") or 1

    print(f"free space: {human(free_bytes())}\n")
    changed = False

    for e in entries:
        target = REPO / e["local_path"]
        expected = e.get("expected_size_bytes")
        print(f"[{e['model']}] {e['checkpoint']}  ({human(expected)})")

        if target.exists():
            size = target.stat().st_size
            if e.get("sha256") and not args.recheck:
                print(f"      present, checksum on record -> skip\n")
                continue
            digest = sha256_file(target)
            if e.get("sha256") and digest != e["sha256"]:
                print(f"      !! CHECKSUM MISMATCH on disk\n")
                e["download_status"] = "corrupt"
                changed = True
                continue
            e.update(actual_size_bytes=size, sha256=digest, download_status="present")
            changed = True
            print(f"      present, sha256={digest[:16]}...\n")
            continue

        if args.dry_run:
            print(f"      DRY RUN -> would GET {e['official_url']}\n")
            continue

        need = (expected or 0) + int(args.min_free_gb * 1024**3)
        if free_bytes() < need:
            print(f"      SKIPPED: would breach the {args.min_free_gb:.0f} GB buffer\n")
            e["download_status"] = "skipped_insufficient_space"
            changed = True
            continue

        ok, msg = download(e["official_url"], target, expected)
        if ok:
            size = target.stat().st_size
            digest = sha256_file(target)
            e.update(
                actual_size_bytes=size, sha256=digest, download_status="downloaded"
            )
            print(f"      done {human(size)}  sha256={digest[:16]}...\n")
        else:
            e["download_status"] = f"failed: {msg}"
            print(f"      FAILED: {msg}\n")
        changed = True

    if changed and not args.dry_run:
        MANIFEST.write_text(json.dumps(data, indent=2) + "\n")
        lines = [
            f"{e['sha256']}  {e['local_path']}"
            for e in data["checkpoints"]
            if e.get("sha256")
        ]
        CHECKSUMS.write_text("\n".join(sorted(lines)) + "\n" if lines else "")
        print(f"updated {MANIFEST.relative_to(REPO)} and {CHECKSUMS.relative_to(REPO)}")

    bad = [e for e in entries if str(e.get("download_status", "")).startswith("failed")]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
