#!/usr/bin/env python3
"""Print a concise status table for everything in this repository.

Reads only the manifests and file sizes. Imports no model package, loads no
weights, runs nothing.

Usage:
    python scripts/print_inventory.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFESTS = REPO / "manifests"


def human(n: float | None) -> str:
    if not n:
        return "-"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return "-"


def load(name: str) -> dict:
    p = MANIFESTS / name
    return json.loads(p.read_text()) if p.exists() else {}


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "-" * 100)


def dir_size(path: Path) -> int:
    """Bytes on disk, not following symlinks (so the RefDrone view counts ~0)."""
    return sum(p.stat().st_size for p in path.rglob("*")
               if p.is_file() and not p.is_symlink())


def main() -> None:
    rule("REPOSITORIES")
    print(f"{'name':20} {'commit':10} {'status':10} {'license':14} upstream")
    for r in load("repositories.json").get("repositories", []):
        print(f"{r['name']:20} {(r['commit_sha'] or '')[:9]:10} {r['clone_status']:10} "
              f"{r['license'][:13]:14} {r['official_url']}")

    rule("CHECKPOINTS")
    print(f"{'model':18} {'checkpoint':52} {'size':>9}  status")
    total = 0
    for c in load("checkpoints.json").get("checkpoints", []):
        size = c.get("actual_size_bytes") or 0
        total += size
        name = c["checkpoint"]
        if len(name) > 51:
            name = name[:48] + "..."
        print(f"{c['model']:18} {name:52} {human(size):>9}  {c['download_status']}")
    print(f"{'':18} {'TOTAL':52} {human(total):>9}")

    api = load("checkpoints.json").get("api_only_not_downloadable", [])
    for a in api:
        print(f"\n  ! API-ONLY, no local weights exist: {a['model']}")

    rule("PAPERS")
    print(f"{'paper':46} {'venue':22} status")
    for p in load("papers.json").get("papers", []):
        title = p["title"][:44] + ".." if len(p["title"]) > 45 else p["title"]
        print(f"{title:46} {p['venue'][:21]:22} {p['status']}")

    rule("DATASETS")
    print(f"{'dataset':18} {'component':40} status")
    for d in load("datasets.json").get("datasets", []):
        comp = d["component"][:38]
        print(f"{d['dataset']:18} {comp:40} {d['download_status']}")

    rule("DISK USAGE")
    grand = 0
    for name in ("third_party", "checkpoints", "datasets", "papers", "manifests",
                 "scripts", "docs", "environments", "outputs"):
        path = REPO / name
        if path.is_dir():
            size = dir_size(path)
            grand += size
            print(f"  {name:16} {human(size):>10}")
    print(f"  {'TOTAL':16} {human(grand):>10}")

    try:
        free = subprocess.run(["df", "-h", str(REPO)], capture_output=True, text=True,
                              check=True).stdout.splitlines()[1].split()
        print(f"\n  filesystem {free[0]}: {free[3]} free of {free[1]} ({free[4]} used)")
    except (subprocess.SubprocessError, IndexError):
        pass
    print()


if __name__ == "__main__":
    main()
