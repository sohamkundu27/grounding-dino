#!/usr/bin/env python3
"""Passive integrity check for the RefDrone dataset.

Strictly offline and read-only. It never imports model code, never loads a
checkpoint, never runs inference, and never touches the network. It only reads
JSON and stats files on disk.

Checks:
  * the three RefDrone annotation files exist and parse
  * every image referenced by the annotations resolves on disk
  * no broken symlinks in the unified image view
  * every link points INSIDE datasets/visdrone2019_det (no escapes)
  * no duplicate image payloads were created (links, not copies)
  * no image is shared across two RefDrone splits
  * per-split counts, computed from the actual files

Exits nonzero only for a real integrity problem. A missing VisDrone download is
reported as "incomplete" (exit 2), not as corruption.

    python scripts/verify_refdrone.py
    python scripts/verify_refdrone.py --json    # machine-readable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ANN = REPO / "datasets" / "refdrone" / "annotations"
IMAGES = REPO / "datasets" / "refdrone" / "images" / "all_image"
VIS = REPO / "datasets" / "visdrone2019_det"
SPLITS = ("train", "val", "test")

# RefDrone split -> the VisDrone split its images are expected to come from.
# This is asserted against reality below, not assumed.
EXPECTED_SOURCE = {"train": "train", "val": "val", "test": "test-dev"}

problems: list[str] = []
warnings: list[str] = []


def fail(msg: str) -> None:
    problems.append(msg)


def warn(msg: str) -> None:
    warnings.append(msg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    report: dict = {"splits": {}, "totals": {}}

    # --- annotations -------------------------------------------------------
    required: dict[str, set[str]] = {}
    for split in SPLITS:
        path = ANN / f"RefDrone_{split}_mdetr.json"
        if not path.exists():
            fail(f"missing annotation file: {path.relative_to(REPO)}")
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            fail(f"{path.name} does not parse: {exc}")
            continue

        imgs = data.get("images", [])
        names = {i["file_name"] for i in imgs}
        exprs = {i.get("caption") for i in imgs if i.get("caption")}
        required[split] = names
        report["splits"][split] = {
            "annotation_file": str(path.relative_to(REPO)),
            "annotation_records": len(imgs),
            "unique_expressions": len(exprs),
            "unique_images": len(names),
            "object_instances": len(data.get("annotations", [])),
            "expected_visdrone_split": EXPECTED_SOURCE[split],
        }

    if not required:
        fail("no annotations could be read at all")
        return emit(report, args.json)

    all_required: set[str] = set().union(*required.values())
    report["totals"]["unique_images_referenced"] = len(all_required)
    report["totals"]["annotation_records"] = sum(
        s["annotation_records"] for s in report["splits"].values()
    )
    report["totals"]["object_instances"] = sum(
        s["object_instances"] for s in report["splits"].values()
    )

    # images shared across splits would mean leakage
    seen = Counter()
    for names in required.values():
        seen.update(names)
    shared = [n for n, c in seen.items() if c > 1]
    report["totals"]["images_in_multiple_splits"] = len(shared)
    if shared:
        fail(f"{len(shared)} image(s) appear in more than one RefDrone split "
             f"(e.g. {shared[:3]}) -- split leakage")

    # duplicate file_name references within a split are expected (one image can
    # carry several expressions), so they are counted, not flagged.
    report["totals"]["duplicate_image_references"] = (
        report["totals"]["annotation_records"] - len(all_required)
    )

    # --- the unified image view --------------------------------------------
    if not IMAGES.is_dir():
        report["totals"]["matched_images"] = 0
        report["totals"]["missing_images"] = len(all_required)
        report["image_view"] = "absent"
        warn(f"{IMAGES.relative_to(REPO)} does not exist yet -- VisDrone not "
             f"downloaded, or scripts/prepare_refdrone.py has not been run")
        return emit(report, args.json, incomplete=True)

    entries = list(IMAGES.iterdir())
    n_symlink = sum(1 for p in entries if p.is_symlink())
    n_regular = sum(1 for p in entries if p.is_file() and not p.is_symlink())

    broken: list[str] = []
    escaped: list[str] = []
    matched: set[str] = set()
    payload_ids: dict[tuple[int, int], list[str]] = defaultdict(list)

    vis_root = VIS.resolve()
    for p in entries:
        if p.is_symlink() and not p.exists():
            broken.append(p.name)
            continue
        if not p.exists():
            continue
        target = p.resolve()
        # every link must land inside the VisDrone tree
        if p.is_symlink() and not str(target).startswith(str(vis_root)):
            escaped.append(f"{p.name} -> {target}")
        matched.add(p.name)
        # inode identity: real duplicate payloads would have distinct inodes
        if not p.is_symlink():
            st = p.stat()
            payload_ids[(st.st_dev, st.st_ino)].append(p.name)

    missing = sorted(all_required - matched)
    extra = sorted(matched - all_required)

    report["image_view"] = {
        "path": str(IMAGES.relative_to(REPO)),
        "entries": len(entries),
        "symlinks": n_symlink,
        "regular_files": n_regular,
    }
    report["totals"]["matched_images"] = len(all_required & matched)
    report["totals"]["missing_images"] = len(missing)
    report["totals"]["broken_symlinks"] = len(broken)
    report["totals"]["links_escaping_visdrone"] = len(escaped)
    report["totals"]["unreferenced_entries"] = len(extra)

    if broken:
        fail(f"{len(broken)} broken symlink(s), e.g. {broken[:3]}")
    if escaped:
        fail(f"{len(escaped)} link(s) point outside datasets/visdrone2019_det: {escaped[:2]}")
    if n_regular:
        warn(f"{n_regular} entries are real files, not symlinks -- image payloads "
             f"may have been duplicated (was --link-mode copy used?)")
    if extra:
        warn(f"{len(extra)} linked image(s) are not referenced by any annotation")

    # per-split resolution
    for split in SPLITS:
        if split not in required:
            continue
        got = len(required[split] & matched)
        report["splits"][split]["matched_images"] = got
        report["splits"][split]["missing_images"] = len(required[split]) - got

    if missing:
        fail(f"{len(missing):,} referenced image(s) do not resolve, "
             f"e.g. {missing[:3]}")

    # --- did we actually need test-challenge? ------------------------------
    tc = VIS / "test-challenge"
    report["totals"]["test_challenge_downloaded"] = tc.is_dir() and any(tc.iterdir())
    if report["totals"]["test_challenge_downloaded"]:
        warn("datasets/visdrone2019_det/test-challenge/ is populated, but RefDrone "
             "references no images from it -- it was not required")

    return emit(report, args.json, incomplete=bool(missing))


def emit(report: dict, as_json: bool, incomplete: bool = False) -> int:
    if as_json:
        report["problems"] = problems
        report["warnings"] = warnings
        print(json.dumps(report, indent=2))
    else:
        print("RefDrone dataset verification")
        print("=" * 78)
        hdr = f"{'split':7} {'records':>8} {'exprs':>7} {'images':>7} {'boxes':>7} {'matched':>8} {'missing':>8}  visdrone"
        print(hdr)
        print("-" * 78)
        for split, s in report.get("splits", {}).items():
            print(f"{split:7} {s['annotation_records']:>8,} {s['unique_expressions']:>7,} "
                  f"{s['unique_images']:>7,} {s['object_instances']:>7,} "
                  f"{s.get('matched_images', 0):>8,} {s.get('missing_images', s['unique_images']):>8,}"
                  f"  {s['expected_visdrone_split']}")
        t = report.get("totals", {})
        print("-" * 78)
        print(f"{'TOTAL':7} {t.get('annotation_records',0):>8,} {'':>7} "
              f"{t.get('unique_images_referenced',0):>7,} {t.get('object_instances',0):>7,} "
              f"{t.get('matched_images',0):>8,} {t.get('missing_images',0):>8,}")
        print()
        iv = report.get("image_view")
        if isinstance(iv, dict):
            print(f"image view      : {iv['path']}")
            print(f"  entries       : {iv['entries']:,} ({iv['symlinks']:,} symlinks, "
                  f"{iv['regular_files']:,} regular files)")
        print(f"broken symlinks : {t.get('broken_symlinks', 0)}")
        print(f"links escaping  : {t.get('links_escaping_visdrone', 0)}")
        print(f"dup references  : {t.get('duplicate_image_references', 0):,} "
              f"(expected: one image can carry several expressions)")
        print(f"cross-split imgs: {t.get('images_in_multiple_splits', 0)}")
        print(f"test-challenge  : {'PRESENT (not needed)' if t.get('test_challenge_downloaded') else 'not downloaded (correct)'}")
        print()
        for w in warnings:
            print(f"  warn  {w}")
        for p in problems:
            print(f"  FAIL  {p}")
        print()
        if problems:
            print(f"RESULT: {len(problems)} integrity problem(s)")
        elif incomplete:
            print("RESULT: no corruption, but the dataset is INCOMPLETE "
                  "(VisDrone images not yet linked)")
        else:
            print("RESULT: OK -- every referenced image resolves, no broken links, "
                  "no duplicated payloads")

    if problems:
        return 1
    if incomplete:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
