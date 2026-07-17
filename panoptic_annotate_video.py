#!/usr/bin/env python3
"""Dense panoptic video annotation with a local Mask2Former checkpoint.

This is a *prompt-free* companion to annotate_video.py. Where that script takes a text
prompt and tracks only the objects it names (Grounding DINO + SAM 2), this one runs a true
panoptic segmentation model over every pixel of every frame: things (car, person, bus,
truck, ...) are segmented per instance, and stuff (road, sidewalk, building, vegetation,
sky, pole, traffic sign, ...) is segmented as regions. There is no text prompt at all.

Model
-----
Mask2Former (transformers' Mask2FormerForUniversalSegmentation), which was the first
choice in the requested priority order. The default checkpoint is the Cityscapes-panoptic
variant, whose 19 classes line up almost exactly with street scenes (KITTI): road,
sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain,
sky, person, rider, car, truck, bus, train, motorcycle, bicycle.

Everything is read from local disk. Nothing is fetched at run time.

Usage
-----
    python panoptic_annotate_video.py \
        --input artifacts/demo_input.mp4 \
        --output artifacts/panoptic_demo_annotated.mp4 \
        --config checkpoints/mask2former/mask2former-swin-large-cityscapes-panoptic \
        --checkpoint checkpoints/mask2former/mask2former-swin-large-cityscapes-panoptic \
        --device cuda

Notes
-----
  * --config and --checkpoint are separate flags per the CLI contract, but a HuggingFace
    Mask2Former keeps config.json, preprocessor_config.json and the weights in one
    directory, so both normally point at the same place. A file path (config.json /
    model.safetensors) is accepted too and resolved to its parent directory.
  * Panoptic models number their segments independently per frame, so segment id 3 in
    frame N has nothing to do with segment id 3 in frame N+1. Thing instances are matched
    frame-to-frame by mask IoU (--iou-match-threshold) to keep instance colors stable.
    Stuff classes never need matching: their color is a pure function of the class.
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

DEFAULT_MODEL_DIR = REPO_ROOT / "checkpoints/mask2former/mask2former-swin-large-cityscapes-panoptic"

# Cityscapes "thing" classes: countable objects that get per-instance ids and IoU tracking.
# Every other class is "stuff" (road, sky, building, ...) and is colored purely by class.
CITYSCAPES_THING_NAMES = frozenset(
    {"person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"}
)

# COCO-panoptic ids 0..79 are things, 80..132 are stuff. Used only if the checkpoint is a
# COCO one rather than the Cityscapes default, so the script degrades sensibly.
COCO_PANOPTIC_THING_ID_MAX = 79


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense panoptic video annotation using a local Mask2Former checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Input video file.")
    parser.add_argument("--output", type=Path, required=True, help="Output annotated video file.")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_MODEL_DIR,
        help="Local model config: a directory holding config.json, or the config.json itself.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_MODEL_DIR,
        help="Local weights: a directory holding model.safetensors/pytorch_model.bin, "
        "or the weight file itself. Usually the same directory as --config.",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Torch device: 'auto' (cuda if available else cpu), or e.g. cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.5,
        help="Minimum per-segment probability to keep a predicted mask.",
    )
    parser.add_argument(
        "--min-area", type=int, default=1500,
        help="Segments smaller than this (in pixels) are still colored but get no text label, "
        "so tiny specks do not clutter the frame.",
    )
    parser.add_argument(
        "--opacity", type=float, default=0.55,
        help="Mask blend strength: output = original*(1-opacity) + color*opacity.",
    )
    parser.add_argument(
        "--iou-match-threshold", type=float, default=0.3,
        help="Mask IoU above which a thing instance is considered the same object as one in "
        "the previous frame, and reuses its stable color id.",
    )
    parser.add_argument(
        "--overlap-area-threshold", type=float, default=0.8,
        help="Mask2Former's overlap_mask_area_threshold: merges/discards small disconnected "
        "parts within a predicted segment.",
    )
    parser.add_argument(
        "--codec", default="mp4v", help="FourCC code for the VideoWriter.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Stop after N frames (0 = whole video). For quick smoke tests.",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA autocast (AMP).")
    parser.add_argument("--no-labels", action="store_true", help="Do not draw text labels.")
    parser.add_argument(
        "--no-boundaries", action="store_true", help="Do not draw segment boundary outlines.",
    )
    parser.add_argument(
        "--progress-every", type=int, default=10, help="Print progress every N frames.",
    )
    args = parser.parse_args(argv)

    if args.device == "auto":  # resolve now so everything downstream sees a real device
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not 0.0 <= args.opacity <= 1.0:
        parser.error("--opacity must be in [0, 1].")
    if not 0.0 <= args.iou_match_threshold <= 1.0:
        parser.error("--iou-match-threshold must be in [0, 1].")
    if args.min_area < 0:
        parser.error("--min-area must be >= 0.")
    return args


def resolve_model_dir(path: Path, what: str) -> Path:
    """Accept either a model directory or a file inside it; return the directory."""
    path = path.expanduser()
    if path.is_dir():
        return path
    if path.is_file():
        return path.parent
    raise FileNotFoundError(
        f"{what} not found: {path}\n"
        f"Expected a local Mask2Former directory (config.json + weights), e.g.\n"
        f"  {DEFAULT_MODEL_DIR}"
    )


def validate_args(args: argparse.Namespace) -> None:
    """Fail early, with actionable messages, on any bad path or device."""
    if not args.input.is_file():
        raise FileNotFoundError(f"Input video not found: {args.input}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA but torch.cuda.is_available() is False.")
    args.output.parent.mkdir(parents=True, exist_ok=True)


def load_model(config_dir: Path, checkpoint_dir: Path, device: str):
    """Load Mask2Former + its image processor from local directories only."""
    from transformers import (
        Mask2FormerConfig,
        Mask2FormerForUniversalSegmentation,
        Mask2FormerImageProcessor,
    )

    config = Mask2FormerConfig.from_pretrained(str(config_dir), local_files_only=True)
    processor = Mask2FormerImageProcessor.from_pretrained(str(config_dir), local_files_only=True)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        str(checkpoint_dir), config=config, local_files_only=True,
    )
    model.to(device)
    model.eval()
    print(f"[load] Mask2Former  config={config_dir.name}  weights={checkpoint_dir.name}  device={device}")
    return model, processor, config


def build_label_tables(config) -> tuple[dict[int, str], set[int]]:
    """Return id->label name, and the set of label ids that are 'thing' (instance) classes."""
    id2label = {int(k): str(v) for k, v in config.id2label.items()}
    names = {n.lower() for n in id2label.values()}
    # Cityscapes checkpoints are identified by their class vocabulary, not by filename.
    if {"road", "sidewalk", "vegetation"} <= names:
        thing_ids = {i for i, n in id2label.items() if n.lower() in CITYSCAPES_THING_NAMES}
        flavor = "cityscapes"
    else:
        thing_ids = {i for i in id2label if i <= COCO_PANOPTIC_THING_ID_MAX}
        flavor = "coco-panoptic"
    print(f"[load] label space: {len(id2label)} classes ({flavor}), "
          f"{len(thing_ids)} thing / {len(id2label) - len(thing_ids)} stuff")
    return id2label, thing_ids


def deterministic_color(key: int) -> tuple[int, int, int]:
    """Stable, well-spread BGR color for an integer key.

    The golden-ratio hue step keeps neighbouring keys visually far apart, and the value is
    a pure function of the key, so a class (or a stable track id) keeps its color for the
    whole video and across runs.
    """
    hue = (key * 0.618033988749895) % 1.0
    sat = 0.65 + ((key * 7) % 3) * 0.10   # 0.65 / 0.75 / 0.85
    val = 0.75 + ((key * 5) % 3) * 0.08   # 0.75 / 0.83 / 0.91
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(b * 255), int(g * 255), int(r * 255))  # OpenCV is BGR


class InstanceTracker:
    """Keeps thing-instance colors stable across adjacent frames via mask IoU.

    Panoptic models re-number segments every frame, so without this a car would flicker
    through a new color on each frame. Stuff classes bypass the tracker entirely.
    """

    def __init__(self, iou_threshold: float) -> None:
        self.iou_threshold = iou_threshold
        self._prev: list[tuple[int, np.ndarray, int]] = []  # (label_id, mask, track_id)
        self._next_track_id = 0
        self._per_class_slot: dict[int, int] = {}  # label_id -> next "#N" shown to the user
        self._display_num: dict[int, int] = {}     # track_id -> "#N"

    def assign(self, segments: list[tuple[int, np.ndarray]]) -> list[int]:
        """Match this frame's thing masks to the previous frame's; return one track id each."""
        track_ids: list[int] = []
        used_prev: set[int] = set()
        for label_id, mask in segments:
            best_iou, best_idx = 0.0, -1
            for idx, (prev_label, prev_mask, _) in enumerate(self._prev):
                if idx in used_prev or prev_label != label_id:
                    continue  # only ever match within the same semantic class
                inter = np.count_nonzero(mask & prev_mask)
                if inter == 0:
                    continue
                union = np.count_nonzero(mask | prev_mask)
                iou = inter / union if union else 0.0
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_idx >= 0 and best_iou >= self.iou_threshold:
                used_prev.add(best_idx)
                track_ids.append(self._prev[best_idx][2])
            else:
                track_ids.append(self._new_track(label_id))
        self._prev = [(lbl, m, tid) for (lbl, m), tid in zip(segments, track_ids)]
        return track_ids

    def _new_track(self, label_id: int) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        slot = self._per_class_slot.get(label_id, 0) + 1
        self._per_class_slot[label_id] = slot
        self._display_num[track_id] = slot
        return track_id

    def display_number(self, track_id: int) -> int:
        return self._display_num.get(track_id, 0)


def label_anchor(mask: np.ndarray) -> tuple[int, int]:
    """Pick a point well inside the mask to anchor its text.

    The centroid can fall outside a concave region (a road wrapping around a car), so use
    the point furthest from any edge; that is always interior and visually centered.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = mask[y0:y1, x0:x1].astype(np.uint8)
    if sub.size > 4_000_000:  # keep the transform cheap on very large stuff regions
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


def render_frame(
    frame: np.ndarray,
    segmentation: np.ndarray,
    segments: list[dict],
    args: argparse.Namespace,
) -> np.ndarray:
    """Blend every predicted segment onto the frame, then outline and label it."""
    h, w = frame.shape[:2]
    color_layer = np.zeros((h, w, 3), dtype=np.uint8)
    covered = np.zeros((h, w), dtype=bool)

    for seg in segments:
        mask = segmentation == seg["id"]
        if not mask.any():
            continue
        color_layer[mask] = seg["color"]
        covered |= mask

    # output = original*(1 - opacity) + color*opacity, applied only where a segment exists,
    # so unlabeled pixels (if any) stay untouched rather than fading to black.
    out = frame.copy()
    blended = cv2.addWeighted(frame, 1.0 - args.opacity, color_layer, args.opacity, 0.0)
    out[covered] = blended[covered]

    if not args.no_boundaries:
        for seg in segments:
            mask = (segmentation == seg["id"]).astype(np.uint8)
            if not mask.any():
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

    if not args.no_labels:
        for seg in segments:
            if seg["area"] < args.min_area:
                continue  # tiny regions would just clutter the frame
            draw_label(out, seg["text"], label_anchor(segmentation == seg["id"]))
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    config_dir = resolve_model_dir(args.config, "--config")
    checkpoint_dir = resolve_model_dir(args.checkpoint, "--checkpoint")

    model, processor, config = load_model(config_dir, checkpoint_dir, args.device)
    id2label, thing_ids = build_label_tables(config)

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

    tracker = InstanceTracker(args.iou_match_threshold)
    use_amp = args.device.startswith("cuda") and not args.no_amp
    frames = 0
    coverage_sum = 0.0
    start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and frames >= args.max_frames:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=rgb, return_tensors="pt").to(args.device)
            with torch.inference_mode():
                if use_amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)

            result = processor.post_process_panoptic_segmentation(
                outputs,
                threshold=args.score_threshold,
                overlap_mask_area_threshold=args.overlap_area_threshold,
                # Fuse stuff: one "road"/"sky" region per frame instead of fragments.
                label_ids_to_fuse=set(id2label) - thing_ids,
                target_sizes=[(height, width)],
            )[0]

            segmentation = result["segmentation"].cpu().numpy()
            info = result["segments_info"]

            # Stable colors: things go through IoU matching, stuff is colored by class.
            thing_entries = [s for s in info if s["label_id"] in thing_ids]
            thing_masks = [(s["label_id"], segmentation == s["id"]) for s in thing_entries]
            track_ids = tracker.assign(thing_masks)
            track_of_segment = {s["id"]: t for s, t in zip(thing_entries, track_ids)}

            segments = []
            for s in info:
                seg_id, label_id = s["id"], s["label_id"]
                name = id2label.get(label_id, f"class_{label_id}")
                area = int(np.count_nonzero(segmentation == seg_id))
                if area == 0:
                    continue
                if label_id in thing_ids:
                    track_id = track_of_segment[seg_id]
                    # Offset keeps thing colors from colliding with stuff class colors.
                    color = deterministic_color(label_id * 1000 + 97 * track_id + 1)
                    text = f"{name} #{tracker.display_number(track_id)}"
                else:
                    color = deterministic_color(label_id)
                    text = name
                segments.append(
                    {"id": seg_id, "color": color, "text": text, "area": area}
                )

            # Largest first, so a small object's label is not overdrawn by the road behind it.
            segments.sort(key=lambda s: s["area"], reverse=True)
            writer.write(render_frame(frame, segmentation, segments, args))

            # post_process_panoptic_segmentation leaves unassigned pixels at -1.
            coverage_sum += np.count_nonzero(segmentation != -1) / (height * width)
            frames += 1
            if args.progress_every and frames % args.progress_every == 0:
                print(f"[run] frame {frames}  segments={len(segments)}")
    finally:
        cap.release()
        writer.release()

    runtime = time.perf_counter() - start
    if frames == 0:
        raise RuntimeError("No frames were read from the input video.")

    # Validate by reopening the file we just wrote.
    check = cv2.VideoCapture(str(args.output))
    if not check.isOpened():
        raise RuntimeError(f"Output written but could not be reopened: {args.output}")
    out_w = int(check.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(check.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = check.get(cv2.CAP_PROP_FPS)
    out_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    check.release()

    print("\n=== panoptic_annotate_video summary ===")
    print(f"model         : Mask2Former ({config_dir.name})")
    print(f"checkpoint    : {checkpoint_dir}")
    print(f"device        : {args.device}  (autocast={'on' if use_amp else 'off'})")
    print(f"frames        : {frames} in, {out_frames} out")
    print(f"resolution    : {out_w}x{out_h} (input {width}x{height})")
    print(f"fps           : {out_fps:.2f} (input {fps:.2f})")
    print(f"runtime       : {runtime:.1f} s")
    print(f"avg fps       : {frames / runtime:.2f}")
    print(f"mean coverage : {100.0 * coverage_sum / frames:.1f}% of pixels segmented")
    print(f"output        : {args.output.resolve()}")

    if (out_w, out_h) != (width, height):
        raise RuntimeError(f"Output resolution {out_w}x{out_h} != input {width}x{height}")
    if out_frames != frames:
        raise RuntimeError(f"Output frame count {out_frames} != {frames} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
