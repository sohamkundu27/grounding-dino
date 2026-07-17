#!/usr/bin/env python3
"""Dense class-agnostic video annotation with a local SAM 2 checkpoint.

This is the prompt-free, *no-download* companion to panoptic_annotate_video.py. It uses
SAM 2's automatic mask generator to carve each frame into many regions from a grid of
point prompts, then colors them. It needs no text prompt and no new weights: it runs on
the sam2.1_hiera_*.pt checkpoints already in checkpoints/sam2/.

IMPORTANT — what this is NOT
---------------------------
SAM 2 is a *segmentation* model, not a *panoptic* one. It knows where object boundaries
are, but it has no class vocabulary, so it cannot tell you that a region is "road" or
"car". Regions are therefore labeled "region #N", never "road" or "car #3", and they are
not split into thing/stuff. If you want real semantic labels, use
panoptic_annotate_video.py (Mask2Former) instead — this script exists for the case where
downloading panoptic weights is not an option.

Coverage also differs: SAM 2 tends to cover salient objects and leaves some low-texture
background (open sky, uniform road) unclaimed. --fill-background paints the leftovers as a
single neutral region so the frame still reads as fully covered.

Usage
-----
    python sam2_dense_annotate_video.py \
        --input artifacts/demo_input.mp4 \
        --output artifacts/sam2_dense_demo_annotated.mp4 \
        --config configs/sam2.1/sam2.1_hiera_l.yaml \
        --checkpoint checkpoints/sam2/sam2.1_hiera_large.pt \
        --device cuda
"""

from __future__ import annotations

import argparse
import colorsys
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"  # hydra name, see resolve_sam2_config
DEFAULT_SAM2_CHECKPOINT = REPO_ROOT / "checkpoints/sam2/sam2.1_hiera_large.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense class-agnostic video annotation using a local SAM 2 checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Input video file.")
    parser.add_argument("--output", type=Path, required=True, help="Output annotated video file.")
    parser.add_argument(
        "--config", default=DEFAULT_SAM2_CONFIG,
        help="Local SAM 2 config: a hydra name relative to the sam2 package "
        "(configs/sam2.1/sam2.1_hiera_l.yaml) or a path to a .yaml file.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT,
        help="Local SAM 2 .pt checkpoint (must match --config).",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Torch device: 'auto' (cuda if available else cpu), or e.g. cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.8,
        help="SAM 2 pred_iou_thresh: minimum predicted mask quality to keep a region.",
    )
    parser.add_argument(
        "--stability-score-threshold", type=float, default=0.9,
        help="SAM 2 stability_score_thresh: minimum mask stability to keep a region.",
    )
    parser.add_argument(
        "--points-per-side", type=int, default=24,
        help="Grid density of point prompts. Higher = more regions, slower.",
    )
    parser.add_argument(
        "--min-area", type=int, default=1500,
        help="Regions smaller than this (in pixels) are still colored but get no text label.",
    )
    parser.add_argument(
        "--min-region-area", type=int, default=200,
        help="Regions smaller than this are dropped entirely (SAM 2 min_mask_region_area).",
    )
    parser.add_argument(
        "--opacity", type=float, default=0.55,
        help="Mask blend strength: output = original*(1-opacity) + color*opacity.",
    )
    parser.add_argument(
        "--iou-match-threshold", type=float, default=0.3,
        help="Mask IoU above which a region is considered the same one as in the previous "
        "frame, and reuses its stable color id.",
    )
    parser.add_argument(
        "--fill-background", action="store_true", default=True,
        help="Paint pixels no region claimed as a single neutral 'background' region.",
    )
    parser.add_argument(
        "--no-fill-background", dest="fill_background", action="store_false",
        help="Leave unclaimed pixels untouched.",
    )
    parser.add_argument("--codec", default="mp4v", help="FourCC code for the VideoWriter.")
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Stop after N frames (0 = whole video). For quick smoke tests.",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA autocast (AMP).")
    parser.add_argument("--no-labels", action="store_true", help="Do not draw text labels.")
    parser.add_argument(
        "--no-boundaries", action="store_true", help="Do not draw region boundary outlines.",
    )
    parser.add_argument(
        "--progress-every", type=int, default=10, help="Print progress every N frames.",
    )
    args = parser.parse_args(argv)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not 0.0 <= args.opacity <= 1.0:
        parser.error("--opacity must be in [0, 1].")
    if not 0.0 <= args.iou_match_threshold <= 1.0:
        parser.error("--iou-match-threshold must be in [0, 1].")
    return args


def resolve_sam2_config(config_arg: str) -> tuple[str, Path | None]:
    """Map --config onto a hydra config name.

    SAM 2 composes configs through hydra against the `sam2` package, so a filesystem path
    is not directly usable. Returns (hydra_name, extra_search_dir); the search dir is
    non-None only for a config living outside the sam2 package.
    """
    import sam2

    pkg_dir = Path(sam2.__file__).resolve().parent
    candidate = Path(config_arg)

    if candidate.is_file():
        resolved = candidate.resolve()
        try:
            return resolved.relative_to(pkg_dir).as_posix(), None
        except ValueError:
            return resolved.name, resolved.parent
    if (pkg_dir / config_arg).is_file():
        return config_arg, None

    available = sorted(
        p.relative_to(pkg_dir).as_posix() for p in (pkg_dir / "configs").rglob("*.yaml")
    )
    raise FileNotFoundError(
        f"SAM 2 config {config_arg!r} is neither an existing file nor a config name "
        f"inside the sam2 package.\nAvailable:\n  " + "\n  ".join(available)
    )


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise FileNotFoundError(f"Input video not found: {args.input}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"SAM 2 checkpoint not found: {args.checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA but torch.cuda.is_available() is False.")
    args.output.parent.mkdir(parents=True, exist_ok=True)


def build_generator(args: argparse.Namespace):
    from hydra import initialize_config_dir
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    hydra_name, search_dir = resolve_sam2_config(args.config)
    if search_dir is not None:
        # A config outside the sam2 package needs its directory on hydra's search path.
        with initialize_config_dir(config_dir=str(search_dir), version_base="1.2"):
            model = build_sam2(hydra_name, str(args.checkpoint), device=args.device)
    else:
        model = build_sam2(hydra_name, str(args.checkpoint), device=args.device)

    generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.score_threshold,
        stability_score_thresh=args.stability_score_threshold,
        min_mask_region_area=args.min_region_area,
        output_mode="binary_mask",
    )
    print(f"[load] SAM 2  config={hydra_name}  ckpt={args.checkpoint.name}  device={args.device}")
    return generator


def deterministic_color(key: int) -> tuple[int, int, int]:
    """Stable, well-spread BGR color for an integer key (golden-ratio hue stepping)."""
    hue = (key * 0.618033988749895) % 1.0
    sat = 0.65 + ((key * 7) % 3) * 0.10
    val = 0.75 + ((key * 5) % 3) * 0.08
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(b * 255), int(g * 255), int(r * 255))


class RegionTracker:
    """Keeps region colors stable across adjacent frames via mask IoU.

    SAM 2's automatic generator re-derives regions independently per frame, so without
    this every region would flicker to a new color each frame. Regions are class-agnostic,
    so unlike the panoptic tracker there is no same-class constraint on matching.
    """

    def __init__(self, iou_threshold: float) -> None:
        self.iou_threshold = iou_threshold
        self._prev: list[tuple[np.ndarray, int]] = []  # (mask, track_id)
        self._next_track_id = 1

    def assign(self, masks: list[np.ndarray]) -> list[int]:
        track_ids: list[int] = []
        used_prev: set[int] = set()
        for mask in masks:
            best_iou, best_idx = 0.0, -1
            for idx, (prev_mask, _) in enumerate(self._prev):
                if idx in used_prev:
                    continue
                inter = np.count_nonzero(mask & prev_mask)
                if inter == 0:
                    continue
                union = np.count_nonzero(mask | prev_mask)
                iou = inter / union if union else 0.0
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_idx >= 0 and best_iou >= self.iou_threshold:
                used_prev.add(best_idx)
                track_ids.append(self._prev[best_idx][1])
            else:
                track_ids.append(self._next_track_id)
                self._next_track_id += 1
        self._prev = list(zip(masks, track_ids))
        return track_ids


def label_anchor(mask: np.ndarray) -> tuple[int, int]:
    """Point furthest from any edge: always interior, even for concave regions."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = mask[y0:y1, x0:x1].astype(np.uint8)
    if sub.size > 4_000_000:
        return (int(xs.mean()), int(ys.mean()))
    padded = cv2.copyMakeBorder(sub, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)
    _, _, _, max_loc = cv2.minMaxLoc(dist)
    return (int(x0 + max_loc[0] - 1), int(y0 + max_loc[1] - 1))


def draw_label(frame: np.ndarray, text: str, anchor: tuple[int, int]) -> None:
    """White text on a dark filled rectangle, clamped to stay inside the frame."""
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    h, w = frame.shape[:2]
    pad = 3
    x = int(np.clip(anchor[0] - tw // 2, pad, max(pad, w - tw - pad)))
    y = int(np.clip(anchor[1] + th // 2, th + pad, max(th + pad, h - baseline - pad)))
    cv2.rectangle(
        frame, (x - pad, y - th - pad), (x + tw + pad, y + baseline), (0, 0, 0), cv2.FILLED,
    )
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def render_frame(frame: np.ndarray, regions: list[dict], args: argparse.Namespace) -> np.ndarray:
    h, w = frame.shape[:2]
    color_layer = np.zeros((h, w, 3), dtype=np.uint8)
    covered = np.zeros((h, w), dtype=bool)

    for reg in regions:  # already sorted largest-first, so small regions land on top
        mask = reg["mask"]
        color_layer[mask] = reg["color"]
        covered |= mask

    if args.fill_background and not covered.all():
        color_layer[~covered] = (128, 128, 128)
        covered[:] = True

    out = frame.copy()
    blended = cv2.addWeighted(frame, 1.0 - args.opacity, color_layer, args.opacity, 0.0)
    out[covered] = blended[covered]

    if not args.no_boundaries:
        for reg in regions:
            contours, _ = cv2.findContours(
                reg["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(out, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

    if not args.no_labels:
        for reg in regions:
            if reg["area"] < args.min_area:
                continue
            draw_label(out, reg["text"], label_anchor(reg["mask"]))
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    generator = build_generator(args)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {args.input}  {width}x{height}  {fps:.2f} fps  ~{total} frames")

    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*args.codec), fps, (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for: {args.output} (codec {args.codec})")

    tracker = RegionTracker(args.iou_match_threshold)
    use_amp = args.device.startswith("cuda") and not args.no_amp
    frames = 0
    coverage_sum = 0.0
    region_sum = 0
    start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and frames >= args.max_frames:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with torch.inference_mode():
                if use_amp:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        raw = generator.generate(rgb)
                else:
                    raw = generator.generate(rgb)

            # Largest first so small regions draw on top of the big ones they sit inside.
            raw.sort(key=lambda m: m["area"], reverse=True)
            masks = [m["segmentation"].astype(bool) for m in raw]
            track_ids = tracker.assign(masks)

            regions = [
                {
                    "mask": mask,
                    "area": int(np.count_nonzero(mask)),
                    "color": deterministic_color(track_id),
                    "text": f"region #{track_id}",
                }
                for mask, track_id in zip(masks, track_ids)
            ]

            writer.write(render_frame(frame, regions, args))

            union = np.zeros((height, width), dtype=bool)
            for m in masks:
                union |= m
            coverage_sum += np.count_nonzero(union) / (height * width)
            region_sum += len(regions)
            frames += 1
            if args.progress_every and frames % args.progress_every == 0:
                print(f"[run] frame {frames}  regions={len(regions)}")
    finally:
        cap.release()
        writer.release()

    runtime = time.perf_counter() - start
    if frames == 0:
        raise RuntimeError("No frames were read from the input video.")

    check = cv2.VideoCapture(str(args.output))
    if not check.isOpened():
        raise RuntimeError(f"Output written but could not be reopened: {args.output}")
    out_w = int(check.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(check.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = check.get(cv2.CAP_PROP_FPS)
    out_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    check.release()

    print("\n=== sam2_dense_annotate_video summary ===")
    print(f"model         : SAM 2 automatic mask generator (class-agnostic, no semantics)")
    print(f"checkpoint    : {args.checkpoint}")
    print(f"device        : {args.device}  (autocast={'on' if use_amp else 'off'})")
    print(f"frames        : {frames} in, {out_frames} out")
    print(f"resolution    : {out_w}x{out_h} (input {width}x{height})")
    print(f"fps           : {out_fps:.2f} (input {fps:.2f})")
    print(f"runtime       : {runtime:.1f} s")
    print(f"avg fps       : {frames / runtime:.2f}")
    print(f"mean regions  : {region_sum / frames:.1f} per frame")
    print(f"mean coverage : {100.0 * coverage_sum / frames:.1f}% of pixels claimed by SAM 2"
          f"{' (rest filled as background)' if args.fill_background else ''}")
    print(f"output        : {args.output.resolve()}")

    if (out_w, out_h) != (width, height):
        raise RuntimeError(f"Output resolution {out_w}x{out_h} != input {width}x{height}")
    if out_frames != frames:
        raise RuntimeError(f"Output frame count {out_frames} != {frames} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
