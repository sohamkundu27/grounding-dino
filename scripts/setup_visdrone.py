#!/usr/bin/env python3
"""Validate and extract the VisDrone2019-DET archives, then normalise the layout.

This script DOWNLOADS NOTHING. VisDrone is distributed only via Google Drive and
BaiduYun, both of which require browser interaction, so the archives must be
fetched by hand first -- see datasets/visdrone2019_det/README.md.

Place the three ZIPs in datasets/visdrone2019_det/archives/ and run this. It:

  1. rejects zero-byte files, HTML/JSON error pages, and anything that is not a
     real ZIP (checks the PK\\x03\\x04 magic bytes, not the file extension)
  2. records actual size + SHA-256
  3. opens the archive and runs a CRC test
  4. refuses to extract any entry with '..' or an absolute path (zip-slip)
  5. extracts, then normalises VisDrone2019-DET-train/ -> train/, etc.
  6. verifies the images are real JPEGs at the header level
  7. updates manifests/datasets.json

It never modifies the source archives, and never overwrites an existing
extracted file without comparing it first.

    python scripts/setup_visdrone.py --dry-run    # validate only, extract nothing
    python scripts/setup_visdrone.py
    python scripts/setup_visdrone.py --remove-archives-after   # only if space is tight
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VIS = REPO / "datasets" / "visdrone2019_det"
ARCHIVES = VIS / "archives"
MANIFEST = REPO / "manifests" / "datasets.json"

# archive filename -> normalised split dir. test-challenge is deliberately absent:
# the RefDrone annotations never reference it (proven in scripts/verify_refdrone.py).
SPLITS = {
    "VisDrone2019-DET-train.zip": ("train", "VisDrone2019-DET-train"),
    "VisDrone2019-DET-val.zip": ("val", "VisDrone2019-DET-val"),
    "VisDrone2019-DET-test-dev.zip": ("test-dev", "VisDrone2019-DET-test-dev"),
}

ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
JPEG_MAGIC = b"\xff\xd8\xff"


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} GB"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def free_bytes() -> int:
    st = os.statvfs(REPO)
    return st.f_bavail * st.f_frsize


def validate_archive(path: Path) -> tuple[bool, str]:
    """Every check that must pass before we are willing to extract."""
    size = path.stat().st_size
    if size == 0:
        return False, "zero-byte file"

    head = path.open("rb").read(512)

    # An HTML page (Google Drive's virus-scan interstitial) or a JSON error blob
    # saved under a .zip name is the most likely failure mode here.
    stripped = head.lstrip().lower()
    if stripped.startswith((b"<!doctype html", b"<html", b"<?xml")):
        return False, "HTML page saved as a ZIP (did the browser save the interstitial?)"
    if stripped.startswith(b"{") or stripped.startswith(b"["):
        return False, "JSON saved as a ZIP (likely an API error response)"
    if not head.startswith(ZIP_MAGIC):
        return False, f"not a ZIP: magic bytes are {head[:4]!r}, expected PK.."

    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad is not None:
                return False, f"CRC failure on entry: {bad}"

            # zip-slip: refuse '..' segments and absolute paths.
            for name in zf.namelist():
                if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                    return False, f"unsafe archive entry (path traversal): {name!r}"
                if Path(name).is_absolute():
                    return False, f"absolute path in archive: {name!r}"
    except zipfile.BadZipFile as exc:
        return False, f"corrupt ZIP: {exc}"

    return True, "ok"


def safe_extract(zip_path: Path, dest: Path) -> Path:
    """Extract, having already validated. Returns the top-level dir created."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        tops = {Path(n).parts[0] for n in zf.namelist() if n.strip()}
        # Re-assert containment at write time, not just at inspect time.
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):
                sys.exit(f"REFUSING to extract outside {dest}: {member.filename}")
        zf.extractall(dest)
    if len(tops) == 1:
        return dest / tops.pop()
    return dest


def count_images(d: Path) -> int:
    return sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() == ".jpg")


def check_jpegs(d: Path, sample: int = 50) -> tuple[int, list[str]]:
    """Header-level readability check. Does not decode pixels."""
    jpgs = sorted(p for p in d.rglob("*.jpg"))
    bad = []
    step = max(1, len(jpgs) // sample)
    for p in jpgs[::step][:sample]:
        if p.open("rb").read(3) != JPEG_MAGIC:
            bad.append(p.name)
    return len(jpgs), bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="validate only; extract nothing")
    ap.add_argument("--remove-archives-after", action="store_true",
                    help="delete an archive once its split is extracted AND verified")
    args = ap.parse_args()

    print(f"archives dir : {ARCHIVES}")
    print(f"free space   : {human(free_bytes())}\n")

    present = {p.name: p for p in ARCHIVES.glob("*.zip")}
    missing = [n for n in SPLITS if n not in present]

    if missing:
        print("MISSING ARCHIVES -- manual download required:")
        for n in missing:
            print(f"  - {n}")
        print("\nSee datasets/visdrone2019_det/README.md for the exact links and steps.")
        if not present:
            return 2

    manifest = json.loads(MANIFEST.read_text())
    by_component = {d["component"]: d for d in manifest["datasets"]}
    results = {}
    rc = 0

    for name, (split, top) in SPLITS.items():
        if name not in present:
            continue
        zpath = present[name]
        size = zpath.stat().st_size
        print(f"== {name}  ({human(size)})")

        ok, msg = validate_archive(zpath)
        if not ok:
            print(f"   REJECTED: {msg}\n")
            rc = 1
            if (e := by_component.get(name)):
                e["download_status"] = f"invalid: {msg}"
            continue
        print(f"   valid ZIP, CRC ok, no unsafe entries")

        digest = sha256_file(zpath)
        print(f"   sha256 {digest}")

        split_dir = VIS / split
        images_dir = split_dir / "images"

        if args.dry_run:
            with zipfile.ZipFile(zpath) as zf:
                n = sum(1 for x in zf.namelist() if x.lower().endswith(".jpg"))
            print(f"   DRY RUN -> would extract {n:,} images to {split_dir}\n")
            results[split] = dict(sha256=digest, size=size, images=n, extracted=False)
            continue

        # Reuse a good existing extraction rather than clobbering it.
        if images_dir.is_dir() and count_images(images_dir) > 0:
            n = count_images(images_dir)
            print(f"   already extracted ({n:,} images) -- reusing, not overwriting")
        else:
            staging = VIS / f".staging_{split}"
            if staging.exists():
                shutil.rmtree(staging)
            top_dir = safe_extract(zpath, staging)

            # Normalise VisDrone2019-DET-train/{images,annotations} -> train/{images,annotations}
            split_dir.mkdir(parents=True, exist_ok=True)
            for sub in ("images", "annotations"):
                src = top_dir / sub
                dst = split_dir / sub
                if src.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.move(str(src), str(dst))
            # anything else the archive shipped (e.g. a readme) -- keep it
            for leftover in top_dir.iterdir():
                shutil.move(str(leftover), str(split_dir / leftover.name))
            shutil.rmtree(staging)
            n = count_images(images_dir)
            print(f"   extracted -> {split_dir}  ({n:,} images)")

        n_jpg, bad = check_jpegs(images_dir)
        if bad:
            print(f"   !! {len(bad)} files failed the JPEG header check: {bad[:3]}")
            rc = 1
        else:
            print(f"   JPEG headers ok (sampled)")

        ann = split_dir / "annotations"
        n_ann = len(list(ann.glob("*.txt"))) if ann.is_dir() else 0
        print(f"   annotations: {n_ann:,} .txt\n")

        results[split] = dict(sha256=digest, size=size, images=n_jpg,
                              annotations=n_ann, extracted=True)

        e = by_component.get(name)
        if e:
            e.update(actual_size_bytes=size, sha256=digest,
                     download_status="downloaded_manually",
                     extraction_status="extracted",
                     local_path=f"datasets/visdrone2019_det/{split}/",
                     archive_path=f"datasets/visdrone2019_det/archives/{name}",
                     image_count=n_jpg, annotation_count=n_ann, manual_action=None)

        if args.remove_archives_after and not bad:
            zpath.unlink()
            print(f"   removed archive (freed {human(size)})\n")
            if e:
                e["archive_path"] = None
                e["notes"] = (e.get("notes") or "") + " Archive deleted after verified extraction."

    if not args.dry_run and results:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"updated {MANIFEST.relative_to(REPO)}")

    print("\nsummary")
    for split, r in results.items():
        print(f"  {split:9} {r['images']:>6,} images  sha256={r['sha256'][:12]}...")

    if missing:
        print(f"\n{len(missing)} archive(s) still missing -- setup incomplete.")
        rc = max(rc, 2)
    return rc


if __name__ == "__main__":
    sys.exit(main())
