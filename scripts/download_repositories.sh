#!/usr/bin/env bash
#
# Hydrate the six upstream repositories at their pinned commits.
#
# They are vendored as GIT SUBMODULES, so this is mostly a wrapper around
# `git submodule update`. It exists so a fresh clone has one obvious command,
# and so the pinned SHAs are printed for confirmation.
#
# This script clones only. It does not build, install, patch, or run anything,
# and it never modifies upstream source.
#
#   ./scripts/download_repositories.sh            # hydrate + report
#   ./scripts/download_repositories.sh --verify   # report only, change nothing
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VERIFY_ONLY=0
[[ "${1:-}" == "--verify" ]] && VERIFY_ONLY=1

# name -> upstream URL. Pinned commits live in the git index, not here, so this
# file can never drift from what is actually checked out.
REPOS=(
  "grounding_dino|https://github.com/IDEA-Research/GroundingDINO.git"
  "mm_grounding_dino|https://github.com/open-mmlab/mmdetection.git"
  "grounded_sam|https://github.com/IDEA-Research/Grounded-Segment-Anything.git"
  "grounded_sam_2|https://github.com/IDEA-Research/Grounded-SAM-2.git"
  "pet_dino|https://github.com/fuweifuvtoo/PET_DINO.git"
  "refdrone|https://github.com/sunzc-sunny/refdrone.git"
)

if [[ $VERIFY_ONLY -eq 0 ]]; then
  echo "==> hydrating submodules at their pinned commits"
  git submodule update --init --recursive
  echo
fi

printf '%-20s %-42s %s\n' "REPO" "COMMIT" "STATUS"
printf '%-20s %-42s %s\n' "----" "------" "------"

status=0
for entry in "${REPOS[@]}"; do
  name="${entry%%|*}"
  url="${entry#*|}"
  path="third_party/${name}"

  # In a submodule, .git is a FILE (a gitlink), not a directory -- test -e.
  if [[ ! -e "$path/.git" ]]; then
    printf '%-20s %-42s %s\n' "$name" "-" "MISSING"
    status=1
    continue
  fi

  sha="$(git -C "$path" rev-parse HEAD)"
  # Clean means: no local edits to upstream source. We never patch upstream.
  if [[ -n "$(git -C "$path" status --porcelain)" ]]; then
    printf '%-20s %-42s %s\n' "$name" "$sha" "MODIFIED (upstream should be clean!)"
    status=1
  else
    printf '%-20s %-42s %s\n' "$name" "$sha" "clean"
  fi
done

echo
echo "Upstream URLs:"
for entry in "${REPOS[@]}"; do
  printf '  %-20s %s\n' "${entry%%|*}" "${entry#*|}"
done

echo
echo "Note: third_party/mm_grounding_dino is the FULL MMDetection repo."
echo "      MM-Grounding-DINO lives inside it at:"
echo "        configs/mm_grounding_dino/            (configs, usage.md, metafile.yml)"
echo "        mmdet/models/detectors/grounding_dino.py"

exit $status
