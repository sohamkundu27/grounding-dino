#!/usr/bin/env python3
"""Curate the representative figures cited in results/summary.md.

outputs/ is gitignored (it holds full videos and every intermediate frame).
This copies a small, downscaled, reviewable subset into figures/, which IS
committed, so the findings in the summary can be checked without re-running
anything.

    python make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsam2_local as G  # noqa: E402

OUT = G.REPO / "experiments/grounded_sam2_local/outputs/images"
FIG = G.REPO / "experiments/grounded_sam2_local/figures"

MAX_W = 1280   # downscale cap
JPEG_Q = 85    # PNG kept these at ~0.5-1.2 MB each; JPEG q85 is ~10x smaller
               # and lossless-enough for reviewing box/mask placement

# (source, destination, what it shows)
FIGURES = [
    ("refdrone_hit_000_9999999_00564_d_0000254.png",
     "01_detection_success.jpg",
     "SUCCESS. 'The blue vehicles in the image.' -> 1 box, IoU 0.84 vs GT. "
     "SAM 2 mask is tight on the vehicle."),

    ("refdrone_miss_002_0000309_04401_d_0000355.png",
     "02_detection_failure_qualifier_ignored.jpg",
     "FAILURE, and the characteristic one. 'The black cars near the intersection.' "
     "-> 10 boxes, top-1 IoU 0.00. GD resolved the CATEGORY (cars) and ignored the "
     "referring qualifiers ('black', 'near the intersection'). SAM 2 then segmented "
     "the wrong box perfectly -- a confident, clean mask on the wrong object."),

    ("refdrone_miss_001_9999962_00000_d_0000022.png",
     "03_prompt_ambiguity_same_expression.jpg",
     "AMBIGUOUS PROMPT. The SAME expression as figure 01 -- 'The blue vehicles in "
     "the image.' -- on a different frame scores IoU 0.00. Identical prompt, "
     "0.84 vs 0.00. The expression is not the controlling variable; the scene is."),

    ("smoke_0000001_02999_d_0000005_person.png",
     "04_small_aerial_targets.jpg",
     "SMALL AERIAL TARGETS. 'person' on a 1920x1080 VisDrone aerial frame -> 13 "
     "detections, top score 0.48 (vs 0.67 for 'car' on the same frame). Confidence "
     "degrades sharply as target size shrinks."),

    ("refdrone_miss_007_9999991_00000_d_0000015.png",
     "05_multiple_similar_targets.jpg",
     "MULTIPLE SIMILAR TARGETS. 'The blue cars park along the street and curbside.' "
     "-> 8 boxes for 8 GT objects, yet top-1 IoU 0.01. The detector finds the right "
     "NUMBER of the right class and still cannot pick the referent. Taking the "
     "highest-confidence box is close to arbitrary here."),

    ("track_zebra_tiny_frame00199.jpg",
     "06_tracking_success.jpg",
     "TRACKING SUCCESS. zebra.mp4, frame 199 of 200. One-time GD acquisition on "
     "frame 0, then SAM 2.1 Tiny propagation with NO re-detection. 200/200 frames "
     "masked, zero empty, coherent single-animal mask, survived a mutual-occlusion "
     "event at f28-30."),

    ("track_tracking_car_tiny_frame00176.jpg",
     "07_tracking_failure_identity_switch.jpg",
     "TRACKING FAILURE -- the important one. tracking_car.mp4 frame 176. The car GD "
     "acquired on frame 0 drove out of the BOTTOM of the frame at f69 and is gone "
     "for good. At f176 SAM 2 silently re-bound object id 1 to a DIFFERENT car 953 px "
     "away, near the horizon, and reports it with full confidence. SAM 2 has no "
     "terminal 'object is gone' state and no identity check, and the detector never "
     "re-runs, so nothing corrects it. A naive empty-mask metric scores this false "
     "positive as a successful RECOVERY."),
]


def main() -> int:
    import cv2

    FIG.mkdir(parents=True, exist_ok=True)
    lines = ["# Figures", "",
             "Curated from `outputs/` (which is gitignored). Regenerate with",
             "`python scripts/make_figures.py`.", ""]
    missing = []

    for src, dst, caption in FIGURES:
        s = OUT / src
        if not s.exists():
            missing.append(src)
            print(f"  !! MISSING {src}")
            continue
        img = cv2.imread(str(s))
        if img is None:
            missing.append(src)
            continue
        h, w = img.shape[:2]
        if w > MAX_W:
            img = cv2.resize(img, (MAX_W, int(h * MAX_W / w)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(FIG / dst), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        kb = (FIG / dst).stat().st_size / 1024
        print(f"  {dst}  ({kb:.0f} KB)")
        lines += [f"### {dst}", "", f"![{dst}]({dst})", "", caption, ""]

    (FIG / "README.md").write_text("\n".join(lines))
    if missing:
        print(f"\n!! {len(missing)} source figure(s) missing — re-run the phase scripts")
        return 1
    print(f"\nwrote {len(FIGURES)} figures + README to {FIG.relative_to(G.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
