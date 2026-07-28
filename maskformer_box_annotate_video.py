#!/usr/bin/env python3
"""Bounding-box video annotation driven by a local MaskFormer checkpoint.

This is the *box* companion to panoptic_annotate_video.py. That script paints every pixel
of every frame (dense masks, one colour per segment). This one runs the same family of
mask-classification models but throws the pixels away: each predicted instance mask is
reduced to its tight axis-aligned bounding box, and the frame is annotated with rectangles
plus `class #track score` labels. Nothing is blended over the image, so the underlying
video stays fully visible.

There is no text prompt anywhere in this pipeline — the model detects everything in its
label space (80 COCO thing classes for the default checkpoint).

Model
-----
MaskFormer (transformers' MaskFormerForInstanceSegmentation) via AutoModelForUniversal-
Segmentation, so a Mask2Former directory works in exactly the same way and needs no flag.
The default checkpoint is facebook/maskformer-swin-large-coco, whose 133-class COCO
panoptic label space contains the 80 countable "thing" classes that boxes make sense for
(person, car, bus, truck, traffic light, bicycle, dog, ...). Stuff classes (road, sky,
wall, ...) are region-shaped and get no box.

Everything is read from local disk. Nothing is fetched at run time.

Usage
-----
    python maskformer_box_annotate_video.py \
        --config checkpoints/maskformer/maskformer-swin-large-coco \
        --checkpoint checkpoints/maskformer/maskformer-swin-large-coco \
        --device cuda

Notes
-----
  * The input and output videos are not flags: they are the INPUT_VIDEO / OUTPUT_VIDEO
    globals near the top of this file. Edit them to run on a different clip.
  * --config and --checkpoint are separate flags per the CLI contract, but a HuggingFace
    MaskFormer keeps config.json, preprocessor_config.json and the weights in one
    directory, so both normally point at the same place. A file path (config.json /
    model.safetensors) is accepted too and resolved to its parent directory.
  * Boxes come from the masks, not from a box head: MaskFormer has no box regression, so
    each kept instance mask is converted to the tight rectangle enclosing it. This is why
    a box is always exactly as tight as the segmentation underneath it.
  * Instances are re-numbered from scratch by the model on every frame, so boxes are
    matched frame-to-frame by box IoU (--iou-match-threshold) to keep each object's colour
    and #number stable while it is visible.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL_DIR = REPO_ROOT / "checkpoints/maskformer/maskformer-swin-large-coco"

# Input/output video paths are fixed here rather than passed on the command line.
# Edit these two to point the run at a different clip.
INPUT_VIDEO = REPO_ROOT / "outputs/maskformer_boxes/street_traffic_input.webm"
OUTPUT_VIDEO = REPO_ROOT / "outputs/maskformer_boxes/street_traffic_boxes.mp4"

# Cityscapes "thing" classes, so a Cityscapes checkpoint (e.g. the one used by
# panoptic_annotate_video.py) also yields sensible boxes instead of boxing the sky.
CITYSCAPES_THING_NAMES = frozenset(
    {"person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"}
)

# COCO-panoptic ids 0..79 are things, 80..132 are stuff. Boxes are only drawn for things:
# a rectangle around "sky" or "road" carries no information.
COCO_PANOPTIC_THING_ID_MAX = 79


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bounding-box video annotation using a local MaskFormer checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
        "--score-threshold", type=float, default=0.7,
        help="Minimum per-instance probability to keep a detection.",
    )
    parser.add_argument(
        "--mask-threshold", type=float, default=0.5,
        help="Probability above which a pixel belongs to an instance mask, before the mask "
        "is reduced to its bounding box.",
    )
    parser.add_argument(
        "--overlap-area-threshold", type=float, default=0.8,
        help="MaskFormer's overlap_mask_area_threshold: discards instances whose mask is "
        "mostly claimed by another, higher-scoring instance.",
    )
    parser.add_argument(
        "--min-box-area", type=int, default=400,
        help="Boxes smaller than this (in pixels) are dropped, so specks in the distance do "
        "not fill the frame with rectangles.",
    )
    parser.add_argument(
        "--min-fill", type=float, default=0.10,
        help="Drop a detection whose mask covers less than this fraction of its own box. "
        "Such masks are usually two unrelated fragments sharing one instance id, whose "
        "enclosing rectangle spans mostly empty image.",
    )
    parser.add_argument(
        "--nms-iou", type=float, default=0.8,
        help="Within a class, drop the lower-scoring of two boxes overlapping above this "
        "IoU. 1.0 disables suppression.",
    )
    parser.add_argument(
        "--iou-match-threshold", type=float, default=0.3,
        help="Box IoU above which a detection is considered the same object as one in the "
        "previous frame, and reuses its stable colour and #number.",
    )
    parser.add_argument(
        "--box-thickness", type=int, default=2, help="Rectangle line thickness in pixels.",
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
    parser.add_argument("--no-scores", action="store_true", help="Omit scores from labels.")
    parser.add_argument(
        "--no-json", action="store_true",
        help="Do not write the per-frame box coordinates alongside the output video. They "
        "are written by default because pixels-only boxes cannot be read back out.",
    )
    parser.add_argument(
        "--progress-every", type=int, default=25, help="Print progress every N frames.",
    )
    args = parser.parse_args(argv)

    if args.device == "auto":  # resolve now so everything downstream sees a real device
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in ("score_threshold", "mask_threshold", "min_fill", "nms_iou",
                 "iou_match_threshold"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1].")
    if args.min_box_area < 0:
        parser.error("--min-box-area must be >= 0.")
    if args.box_thickness < 1:
        parser.error("--box-thickness must be >= 1.")
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
        f"Expected a local MaskFormer directory (config.json + weights), e.g.\n"
        f"  {DEFAULT_MODEL_DIR}"
    )


def validate_args(args: argparse.Namespace) -> None:
    """Fail early, with actionable messages, on any bad path or device."""
    if not INPUT_VIDEO.is_file():
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA but torch.cuda.is_available() is False.")
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)


def load_model(config_dir: Path, checkpoint_dir: Path, device: str):
    """Load a MaskFormer-family model + its image processor from local directories only."""
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForUniversalSegmentation

    config = AutoConfig.from_pretrained(str(config_dir), local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(str(config_dir), local_files_only=True)
    model = AutoModelForUniversalSegmentation.from_pretrained(
        str(checkpoint_dir), config=config, local_files_only=True,
    )
    model.to(device)
    model.eval()
    print(f"[load] {config.model_type}  config={config_dir.name}  "
          f"weights={checkpoint_dir.name}  device={device}")
    return model, processor, config


def build_label_tables(config) -> tuple[dict[int, str], set[int]]:
    """Return id->label name, and the set of label ids that get boxes ('thing' classes)."""
    id2label = {int(k): str(v) for k, v in config.id2label.items()}
    names = {n.lower() for n in id2label.values()}
    # Checkpoints are identified by their class vocabulary, not by filename.
    if {"road", "sidewalk", "vegetation"} <= names:
        thing_ids = {i for i, n in id2label.items() if n.lower() in CITYSCAPES_THING_NAMES}
        flavor = "cityscapes"
    else:
        thing_ids = {i for i in id2label if i <= COCO_PANOPTIC_THING_ID_MAX}
        flavor = "coco-panoptic"
    print(f"[load] label space: {len(id2label)} classes ({flavor}), "
          f"{len(thing_ids)} boxable thing classes")
    return id2label, thing_ids


def deterministic_color(key: int) -> tuple[int, int, int]:
    """Stable, well-spread BGR color for an integer key.

    The golden-ratio hue step keeps neighbouring keys visually far apart, and the value is
    a pure function of the key, so a track keeps its color for as long as it lives and
    across runs.
    """
    hue = (key * 0.618033988749895) % 1.0
    sat = 0.65 + ((key * 7) % 3) * 0.10   # 0.65 / 0.75 / 0.85
    val = 0.75 + ((key * 5) % 3) * 0.08   # 0.75 / 0.83 / 0.91
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(b * 255), int(g * 255), int(r * 255))  # OpenCV is BGR


def mask_to_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight (x0, y0, x1, y1) around a binary mask, exclusive on x1/y1. None if empty."""
    cols = np.flatnonzero(mask.any(axis=0))
    if cols.size == 0:
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two (x0, y0, x1, y1) boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def suppress_overlaps(detections: list[dict], nms_iou: float) -> list[dict]:
    """Class-wise NMS over mask-derived boxes.

    The model already assigns each pixel to one instance, so exact duplicates are rare, but
    a single object split across two queries can still yield two near-identical boxes.
    Detections are assumed sorted by descending score.
    """
    if nms_iou >= 1.0:
        return detections
    kept: list[dict] = []
    for det in detections:
        if any(
            k["label_id"] == det["label_id"] and box_iou(k["box"], det["box"]) > nms_iou
            for k in kept
        ):
            continue
        kept.append(det)
    return kept


class BoxTracker:
    """Keeps colors and #numbers stable across adjacent frames via box IoU.

    The model re-numbers its instances every frame, so without this a car would flicker
    through a new color on each frame. Matching is greedy, highest-scoring detection first,
    and only ever within the same semantic class.
    """

    def __init__(self, iou_threshold: float) -> None:
        self.iou_threshold = iou_threshold
        self._prev: list[tuple[int, tuple[int, int, int, int], int]] = []  # label, box, track
        self._next_track_id = 0
        self._per_class_slot: dict[int, int] = {}  # label_id -> next "#N" shown to the user
        self._display_num: dict[int, int] = {}     # track_id -> "#N"

    def assign(self, detections: list[dict]) -> list[int]:
        """Match this frame's boxes to the previous frame's; return one track id each."""
        track_ids: list[int] = []
        used_prev: set[int] = set()
        for det in detections:
            best_iou, best_idx = 0.0, -1
            for idx, (prev_label, prev_box, _) in enumerate(self._prev):
                if idx in used_prev or prev_label != det["label_id"]:
                    continue  # only ever match within the same semantic class
                iou = box_iou(det["box"], prev_box)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_idx >= 0 and best_iou >= self.iou_threshold:
                used_prev.add(best_idx)
                track_ids.append(self._prev[best_idx][2])
            else:
                track_ids.append(self._new_track(det["label_id"]))
        self._prev = [(d["label_id"], d["box"], t) for d, t in zip(detections, track_ids)]
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

    @property
    def total_tracks(self) -> int:
        return self._next_track_id


def draw_detection(
    frame: np.ndarray, det: dict, thickness: int, draw_label: bool,
) -> None:
    """Draw one rectangle, plus its label on a filled tab in the same color."""
    x0, y0, x1, y1 = det["box"]
    color = det["color"]
    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), color, thickness, cv2.LINE_AA)
    if not draw_label:
        return

    font, scale, text_thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), baseline = cv2.getTextSize(det["text"], font, scale, text_thickness)
    h, w = frame.shape[:2]
    pad = 3
    tab_h = th + baseline + 2 * pad
    # Prefer a tab sitting just above the box; flip inside when the box touches the top.
    tab_x = int(np.clip(x0, 0, max(0, w - tw - 2 * pad)))
    tab_y = y0 - tab_h if y0 - tab_h >= 0 else min(y0, max(0, h - tab_h))
    cv2.rectangle(
        frame, (tab_x, tab_y), (tab_x + tw + 2 * pad, tab_y + tab_h), color, cv2.FILLED,
    )
    # Dark text on bright fills, white on dark ones, so labels stay legible on every hue.
    b, g, r = color
    luma = 0.114 * b + 0.587 * g + 0.299 * r
    text_color = (0, 0, 0) if luma > 140 else (255, 255, 255)
    cv2.putText(
        frame, det["text"], (tab_x + pad, tab_y + pad + th), font, scale, text_color,
        text_thickness, cv2.LINE_AA,
    )


def detect_frame(
    frame: np.ndarray,
    model,
    processor,
    id2label: dict[int, str],
    thing_ids: set[int],
    args: argparse.Namespace,
    use_amp: bool,
) -> list[dict]:
    """Run the model on one BGR frame and return its kept detections, best score first."""
    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        if use_amp:
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        overlap_mask_area_threshold=args.overlap_area_threshold,
        target_sizes=[(height, width)],
        # One binary mask per instance, rather than one id-map: each mask becomes one box.
        return_binary_maps=True,
    )[0]

    info = result["segments_info"]
    if not info:
        return []
    masks = result["segmentation"].cpu().numpy().astype(bool)
    if masks.ndim == 2:  # a single instance is returned unstacked
        masks = masks[None]

    detections: list[dict] = []
    for seg, mask in zip(info, masks):
        label_id = int(seg["label_id"])
        if label_id not in thing_ids:
            continue  # stuff classes are region-shaped; a box around them says nothing
        box = mask_to_box(mask)
        if box is None:
            continue
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        if box_area < args.min_box_area:
            continue
        mask_area = int(np.count_nonzero(mask))
        if mask_area / box_area < args.min_fill:
            continue  # fragmented mask whose enclosing rectangle is mostly empty image
        detections.append(
            {
                "label_id": label_id,
                "name": id2label.get(label_id, f"class_{label_id}"),
                "score": float(seg["score"]),
                "box": box,
                "mask_area": mask_area,
            }
        )

    detections.sort(key=lambda d: d["score"], reverse=True)
    return suppress_overlaps(detections, args.nms_iou)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    config_dir = resolve_model_dir(args.config, "--config")
    checkpoint_dir = resolve_model_dir(args.checkpoint, "--checkpoint")

    model, processor, config = load_model(config_dir, checkpoint_dir, args.device)
    id2label, thing_ids = build_label_tables(config)

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {INPUT_VIDEO}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {INPUT_VIDEO}  {width}x{height}  {fps:.2f} fps  ~{total} frames")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*args.codec), fps, (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for: {OUTPUT_VIDEO} (codec {args.codec})")

    json_path = OUTPUT_VIDEO.with_suffix(".boxes.jsonl")
    json_file = None if args.no_json else json_path.open("w", encoding="utf-8")

    tracker = BoxTracker(args.iou_match_threshold)
    use_amp = args.device.startswith("cuda") and not args.no_amp
    frames = 0
    detection_count = 0
    per_class_frames: dict[str, int] = {}   # class -> boxes drawn over the whole clip
    per_class_tracks: dict[str, set[int]] = {}
    start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and frames >= args.max_frames:
                break

            detections = detect_frame(
                frame, model, processor, id2label, thing_ids, args, use_amp
            )
            track_ids = tracker.assign(detections)

            for det, track_id in zip(detections, track_ids):
                number = tracker.display_number(track_id)
                det["track_id"] = track_id
                det["color"] = deterministic_color(det["label_id"] * 1000 + 97 * track_id + 1)
                det["text"] = f"{det['name']} #{number}"
                if not args.no_scores:
                    det["text"] += f" {det['score']:.2f}"
                per_class_frames[det["name"]] = per_class_frames.get(det["name"], 0) + 1
                per_class_tracks.setdefault(det["name"], set()).add(track_id)

            # Largest first, so a small box in front of a big one stays on top and readable.
            annotated = frame.copy()
            for det in sorted(
                detections,
                key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]),
                reverse=True,
            ):
                draw_detection(annotated, det, args.box_thickness, not args.no_labels)
            writer.write(annotated)

            if json_file is not None:
                json_file.write(json.dumps({
                    "frame": frames,
                    "detections": [
                        {
                            "class": d["name"],
                            "label_id": d["label_id"],
                            "track_id": d["track_id"],
                            "score": round(d["score"], 4),
                            "box_xyxy": list(d["box"]),
                        }
                        for d in detections
                    ],
                }) + "\n")

            detection_count += len(detections)
            frames += 1
            if args.progress_every and frames % args.progress_every == 0:
                print(f"[run] frame {frames}  boxes={len(detections)}")
    finally:
        cap.release()
        writer.release()
        if json_file is not None:
            json_file.close()

    runtime = time.perf_counter() - start
    if frames == 0:
        raise RuntimeError("No frames were read from the input video.")

    # Validate by reopening the file we just wrote.
    check = cv2.VideoCapture(str(OUTPUT_VIDEO))
    if not check.isOpened():
        raise RuntimeError(f"Output written but could not be reopened: {OUTPUT_VIDEO}")
    out_w = int(check.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(check.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = check.get(cv2.CAP_PROP_FPS)
    out_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    check.release()

    print("\n=== maskformer_box_annotate_video summary ===")
    print(f"model         : {config.model_type} ({config_dir.name})")
    print(f"checkpoint    : {checkpoint_dir}")
    print(f"device        : {args.device}  (autocast={'on' if use_amp else 'off'})")
    print(f"frames        : {frames} in, {out_frames} out")
    print(f"resolution    : {out_w}x{out_h} (input {width}x{height})")
    print(f"fps           : {out_fps:.2f} (input {fps:.2f})")
    print(f"runtime       : {runtime:.1f} s")
    print(f"avg fps       : {frames / runtime:.2f}")
    print(f"boxes         : {detection_count} total, {detection_count / frames:.1f} per frame")
    print(f"tracks        : {tracker.total_tracks} distinct objects")
    for name in sorted(per_class_frames, key=lambda n: per_class_frames[n], reverse=True):
        print(f"  {name:<16} {per_class_frames[name]:>6} boxes  "
              f"{len(per_class_tracks[name]):>4} tracks")
    print(f"output        : {OUTPUT_VIDEO.resolve()}")
    if not args.no_json:
        print(f"boxes jsonl   : {json_path.resolve()}")

    if (out_w, out_h) != (width, height):
        raise RuntimeError(f"Output resolution {out_w}x{out_h} != input {width}x{height}")
    if out_frames != frames:
        raise RuntimeError(f"Output frame count {out_frames} != {frames} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
