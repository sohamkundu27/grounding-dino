#!/usr/bin/env python3
"""Build the flat `all_image/` view RefDrone expects, without duplicating images.

RefDrone ships language annotations only; the pixels live in VisDrone2019-DET.
Upstream's layout wants every referenced image in one flat directory. Copying
8,536 JPEGs to satisfy that is wasteful, so by default this script creates
symlinks into the VisDrone tree. The VisDrone originals are never modified.

This script DOWNLOADS NOTHING. Fetch VisDrone yourself first --
see datasets/visdrone2019_det/README.md.

Typical use:

    # Inspect the plan; touches nothing.
    python scripts/prepare_refdrone.py \
        --refdrone-annotations datasets/refdrone/annotations \
        --visdrone-root        datasets/visdrone2019_det \
        --dry-run

    # Build the symlink view.
    python scripts/prepare_refdrone.py \
        --refdrone-annotations datasets/refdrone/annotations \
        --visdrone-root        datasets/visdrone2019_det

    # Filesystem without symlinks (e.g. some network/Windows mounts):
    ...  --link-mode hardlink      # same filesystem only
    ...  --link-mode copy          # last resort; actually duplicates the bytes
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_required(ann_dir: Path) -> dict[str, set[str]]:
    """file_names referenced by each RefDrone split."""
    required: dict[str, set[str]] = {}
    for split in SPLITS:
        path = ann_dir / f"RefDrone_{split}_mdetr.json"
        if not path.exists():
            sys.exit(f"missing annotation file: {path}\n(see datasets/refdrone/README.md)")
        data = json.loads(path.read_text())
        # Each 'images' entry is one (image, expression) pair, so names repeat.
        required[split] = {img["file_name"] for img in data["images"]}
    return required


def index_visdrone(root: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """Map basename -> path for every image under root. Also flag repeated basenames.

    Searched recursively, so it does not matter whether the archives were
    extracted as train/images/ or VisDrone2019-DET-train/images/.
    """
    index: dict[str, Path] = {}
    seen: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            seen[path.name].append(path)
    for name, paths in seen.items():
        index[name] = sorted(paths)[0]  # deterministic pick
    duplicates = {n: p for n, p in seen.items() if len(p) > 1}
    return index, duplicates


def link_one(src: Path, dst: Path, mode: str) -> str:
    """Materialise dst from src. Returns the action taken."""
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and not dst.exists():
            dst.unlink()  # broken symlink from an earlier run
        else:
            return "exists"

    if mode == "symlink":
        # Relative target keeps the tree portable if the repo moves.
        dst.symlink_to(os.path.relpath(src, dst.parent))
        return "symlink"
    if mode == "hardlink":
        os.link(src, dst)
        return "hardlink"
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    raise ValueError(mode)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--refdrone-annotations", type=Path, required=True,
                    help="dir holding RefDrone_{train,val,test}_mdetr.json")
    ap.add_argument("--visdrone-root", type=Path, required=True,
                    help="dir holding the extracted VisDrone2019-DET images (searched recursively)")
    ap.add_argument("--out", type=Path, default=REPO / "datasets/refdrone/images/all_image",
                    help="flat view to build (default: datasets/refdrone/images/all_image)")
    ap.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink",
                    help="symlink (default) | hardlink (same fs) | copy (duplicates bytes)")
    ap.add_argument("--manifest", type=Path,
                    default=REPO / "datasets/refdrone/metadata/refdrone_image_manifest.json")
    ap.add_argument("--dry-run", action="store_true", help="report only; create nothing")
    args = ap.parse_args()

    ann_dir: Path = args.refdrone_annotations
    vis_root: Path = args.visdrone_root

    if not vis_root.is_dir():
        sys.exit(f"VisDrone root not found: {vis_root}\nSee datasets/visdrone2019_det/README.md")

    required = load_required(ann_dir)
    all_required: set[str] = set().union(*required.values())
    print(f"RefDrone references {len(all_required):,} unique images "
          f"(train {len(required['train']):,} / val {len(required['val']):,} / test {len(required['test']):,})")

    index, duplicates = index_visdrone(vis_root)
    print(f"VisDrone index: {len(index):,} images under {vis_root}")

    if not index:
        sys.exit(f"\nNo images found under {vis_root}. VisDrone is not downloaded yet.\n"
                 f"See datasets/visdrone2019_det/README.md for the manual steps.")

    if duplicates:
        print(f"\n!! {len(duplicates)} basenames appear more than once in the VisDrone tree.")
        print("   The first path (sorted) is used for each. Sample:")
        for name, paths in list(duplicates.items())[:5]:
            print(f"     {name}")
            for p in paths[:3]:
                print(f"        {p.relative_to(vis_root)}")

    present = {n for n in all_required if n in index}
    missing = sorted(all_required - present)

    print(f"\nresolved {len(present):,}/{len(all_required):,} required images")
    if missing:
        by_split = {s: sum(1 for n in missing if n in required[s]) for s in SPLITS}
        print(f"!! MISSING {len(missing):,} images "
              f"(train {by_split['train']} / val {by_split['val']} / test {by_split['test']})")
        print("   Likely an un-extracted or wrong VisDrone split. First few:")
        for n in missing[:8]:
            print(f"     {n}")
        if len(missing) > 8:
            print(f"     ... and {len(missing)-8:,} more")

    if args.dry_run:
        print(f"\nDRY RUN: would {args.link_mode} {len(present):,} images into {args.out}")
        print("Nothing was created or modified.")
        return 1 if missing else 0

    args.out.mkdir(parents=True, exist_ok=True)
    actions: dict[str, int] = defaultdict(int)
    for name in sorted(present):
        try:
            actions[link_one(index[name], args.out / name, args.link_mode)] += 1
        except OSError as exc:
            actions["failed"] += 1
            if actions["failed"] <= 3:
                print(f"   !! {name}: {exc}")
                if args.link_mode == "hardlink":
                    print("      (hardlinks require src and dst on the same filesystem)")

    print(f"\n{args.out}:")
    for action, count in sorted(actions.items()):
        print(f"   {action:9} {count:,}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "link_mode": args.link_mode,
        "visdrone_root": str(vis_root),
        "all_image_dir": str(args.out),
        "required_unique_images": len(all_required),
        "resolved": len(present),
        "missing": len(missing),
        "missing_files": missing,
        "duplicate_basenames_in_visdrone": {n: [str(p) for p in ps] for n, ps in duplicates.items()},
        "counts_by_split": {s: len(required[s]) for s in SPLITS},
        "actions": dict(actions),
    }, indent=2) + "\n")
    print(f"\nmanifest -> {args.manifest}")
    print("VisDrone originals were not modified.")

    return 1 if missing or actions["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
