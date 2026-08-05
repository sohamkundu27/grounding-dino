#!/usr/bin/env python3
"""Export MaskFormer box detections from a video as one RefDrone/MDETR-COCO JSON file.

This is the *dataset export* companion to maskformer_box_annotate_video.py. That script
burns rectangles into an output video; this one throws the pixels away entirely and writes
only coordinates, in the same JSON schema as
datasets/refdrone/annotations/RefDrone_*_mdetr.json.

The detection pipeline is not reimplemented here. Frames are handed to
maskformer_box_annotate_video.detect_frame(), so the model loading, the "thing"-class
filter, the mask -> tight-box reduction (mask_to_box), the min-area / min-fill rejects and
the class-wise NMS are byte-for-byte the ones that produce the annotated videos. Only the
serialisation is new.

There is no text prompt anywhere in this pipeline. MaskFormer detects everything in its
label space, so every emitted image gets an empty caption ("") and no tokens_positive
spans. The output is therefore RefDrone-*shaped* but not a referring-expression dataset:
it is class-agnostic box supervision in a loader-compatible container.

Config
------
Everything is a module-level global below -- there is no argparse and no CLI. Edit the
globals and re-run.

Usage
-----
    .venv-grounded-sam2/bin/python maskformer_to_refdrone.py

Notes
-----
  * Offline enforcement is belt-and-braces, matching annotate_video.py and
    experiments/*/scripts/gsam*_local.py rather than the thinner local_files_only=True
    that maskformer_box_annotate_video.py relies on: HF_HUB_OFFLINE /
    TRANSFORMERS_OFFLINE / HF_DATASETS_OFFLINE are set at import, before transformers is
    ever imported.
    Measured, so the reasoning is not folklore: loading THIS checkpoint with
    local_files_only=True and no env vars attempts no connections -- a from_pretrained
    pointed at a local directory never reaches the hub. The leak the repo actually hit
    was a load by *repo id* (Grounding DINO's from_pretrained("bert-base-uncased")),
    which revalidates against huggingface.co on every call even when fully cached. The
    env vars cost nothing and are the only thing that would stop that class of call if a
    repo-id load is ever introduced downstream of this script; local_files_only keeps the
    failure a clean exception rather than a silent download. Verified end to end by
    running this file with socket.connect hard-blocked: it completes.
  * This writes ONE file: the JSON. It does not extract the sampled frames to disk, so the
    file_name fields point at images that do not exist yet. Decode them from VIDEO_PATH
    with the same FRAME_STRIDE if a consumer needs the pixels.
  * Inference only. Nothing here trains, fine-tunes or updates a checkpoint.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------------------
# Offline enforcement -- MUST precede any transformers import (which happens lazily inside
# maskformer_box_annotate_video.load_model, i.e. after this block runs).
# --------------------------------------------------------------------------------------

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import cv2  # noqa: E402  (import order is deliberate: env vars first)

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The detection half of this script. Imported, never copied -- see module docstring.
import maskformer_box_annotate_video as mbav  # noqa: E402

# --------------------------------------------------------------------------------------
# Config -- module-level globals, no CLI.
# --------------------------------------------------------------------------------------

VIDEO_PATH = REPO_ROOT / "outputs/maskformer_boxes/street_traffic_input.webm"
MASKFORMER_MODEL_PATH = REPO_ROOT / "checkpoints/maskformer/maskformer-swin-large-coco"
OUTPUT_JSON_PATH = REPO_ROOT / "street_traffic_refdrone.json"

# Sample every Nth frame. 1 = every frame.
FRAME_STRIDE = 1

# Torch device: "auto" (cuda if available else cpu), or e.g. "cuda", "cuda:0", "cpu".
DEVICE = "auto"

# Stop after this many *sampled* frames. 0 = the whole video.
MAX_FRAMES = 0

# Detection thresholds. None means "inherit maskformer_box_annotate_video's default", so
# the exported boxes match what that script would have drawn. Set a number to override.
SCORE_THRESHOLD = None        # min per-instance probability to keep a detection
MASK_THRESHOLD = None         # pixel probability cutoff, before the mask becomes a box
OVERLAP_AREA_THRESHOLD = None # discard instances mostly claimed by a higher-scoring one
MIN_BOX_AREA = None           # drop boxes smaller than this many pixels
MIN_FILL = None               # drop masks covering less than this fraction of their box
NMS_IOU = None                # within-class IoU above which the lower score is dropped

# Disable CUDA autocast (AMP). Ignored on CPU.
NO_AMP = False

# file_name pattern for the sampled frames. Zero-padded so lexical order == frame order.
FRAME_NAME_TEMPLATE = "frame_{index:06d}.jpg"

# Stamped into "info"/"licenses" and each image's "dataset_name". Kept at "RefDrone"
# because MDETR-family loaders branch on dataset_name to pick their collate path; change
# it only if the consumer keys off a different name.
DATASET_NAME = "RefDrone"

# category_id used by the placeholder annotation on a frame with zero detections. -1 is
# what the real RefDrone files use for their `empty` rows.
EMPTY_CATEGORY_ID = -1

PROGRESS_EVERY = 25


# --------------------------------------------------------------------------------------
# Detection config bridge
# --------------------------------------------------------------------------------------

def build_detect_args():
    """Assemble the Namespace that maskformer_box_annotate_video.detect_frame() expects.

    Its defaults are read straight out of that module's parser, so the two scripts cannot
    drift apart: tuning a threshold there changes the export here unless a global above
    explicitly overrides it.
    """
    args = mbav.parse_args([])  # defaults only; never touches sys.argv

    args.device = "cuda" if DEVICE == "auto" and _cuda_available() else (
        "cpu" if DEVICE == "auto" else DEVICE
    )
    overrides = {
        "score_threshold": SCORE_THRESHOLD,
        "mask_threshold": MASK_THRESHOLD,
        "overlap_area_threshold": OVERLAP_AREA_THRESHOLD,
        "min_box_area": MIN_BOX_AREA,
        "min_fill": MIN_FILL,
        "nms_iou": NMS_IOU,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(args, name, value)
    return args


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def validate_config() -> None:
    """Fail early, with actionable messages, before a model is loaded."""
    if not Path(VIDEO_PATH).is_file():
        raise FileNotFoundError(f"VIDEO_PATH not found: {VIDEO_PATH}")
    if not Path(MASKFORMER_MODEL_PATH).exists():
        raise FileNotFoundError(
            f"MASKFORMER_MODEL_PATH not found: {MASKFORMER_MODEL_PATH}\n"
            "Expected a local MaskFormer directory (config.json + preprocessor_config.json "
            "+ weights). Nothing is downloaded: this script runs with HF_HUB_OFFLINE=1."
        )
    if FRAME_STRIDE < 1:
        raise ValueError(f"FRAME_STRIDE must be >= 1, got {FRAME_STRIDE}")
    if MAX_FRAMES < 0:
        raise ValueError(f"MAX_FRAMES must be >= 0, got {MAX_FRAMES}")
    Path(OUTPUT_JSON_PATH).parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# RefDrone/MDETR-COCO record builders
# --------------------------------------------------------------------------------------

def make_image_record(image_id: int, file_name: str, width: int, height: int) -> dict:
    """One "images" entry.

    Field order and the original_id/tokens_positive/dataset_name extras mirror the real
    RefDrone_*_mdetr.json rows so a loader written against those files sees no missing key.
    The caption is empty because MaskFormer is prompt-free -- there is no referring
    expression to attach, and an empty caption means an empty tokens_positive too.
    """
    return {
        "file_name": file_name,
        "height": int(height),
        "width": int(width),
        "id": int(image_id),
        "original_id": -1,
        "caption": "",
        "dataset_name": DATASET_NAME,
        "tokens_positive": [],
    }


def make_annotation_record(ann_id: int, image_id: int, det: dict) -> dict:
    """One "annotations" entry for a real detection.

    det["box"] is (x0, y0, x1, y1) exclusive on x1/y1, straight out of mbav.mask_to_box;
    COCO wants [x, y, w, h]. RefDrone stores these as floats and sets `area` to the box
    area (not the mask area) -- verified against the shipped val split -- so both are
    matched here.
    """
    x0, y0, x1, y1 = det["box"]
    w, h = x1 - x0, y1 - y0
    return {
        "area": float(w * h),
        "iscrowd": 0,
        "image_id": int(image_id),
        "category_id": int(det["label_id"]),
        "id": int(ann_id),
        "empty": False,
        "bbox": [float(x0), float(y0), float(w), float(h)],
        "original_id": -1,
        "tokens_positive": [],
    }


def make_empty_annotation_record(ann_id: int, image_id: int) -> dict:
    """Placeholder "annotations" entry for a frame where nothing was detected.

    RefDrone keeps a row for every image, so a zero-detection frame is a negative example
    rather than an absent one. Mirrors the sentinel values (area/iscrowd/category_id = -1)
    used by the `empty: true` rows in the shipped splits.
    """
    return {
        "area": -1,
        "iscrowd": -1,
        "image_id": int(image_id),
        "category_id": int(EMPTY_CATEGORY_ID),
        "id": int(ann_id),
        "empty": True,
        "bbox": [0, 0, 0, 0],
        "original_id": -1,
        "tokens_positive": [],
    }


def make_categories(used_label_ids: set[int], id2label: dict[int, str]) -> list[dict]:
    """The MaskFormer COCO-panoptic id/name pairs that actually appear in annotations.

    The RefDrone loader ignores this list, but it is what makes the file self-describing:
    without it a category_id is an opaque integer in someone else's label space. The empty
    sentinel is deliberately not listed -- it labels nothing.
    """
    return [
        {"id": int(i), "name": id2label.get(int(i), f"class_{i}")}
        for i in sorted(used_label_ids)
    ]


# --------------------------------------------------------------------------------------

def main() -> int:
    validate_config()
    args = build_detect_args()

    model_dir = mbav.resolve_model_dir(Path(MASKFORMER_MODEL_PATH), "MASKFORMER_MODEL_PATH")
    # Config and weights live in the same directory for a HuggingFace MaskFormer, which is
    # why the reference script's two flags normally point at one path.
    model, processor, config = mbav.load_model(model_dir, model_dir, args.device)
    id2label, thing_ids = mbav.build_label_tables(config)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {VIDEO_PATH}  {width}x{height}  {fps:.2f} fps  ~{total} frames  "
          f"stride={FRAME_STRIDE}")

    use_amp = args.device.startswith("cuda") and not NO_AMP

    images: list[dict] = []
    annotations: list[dict] = []
    used_label_ids: set[int] = set()
    per_class_counts: dict[str, int] = {}

    next_ann_id = 0        # increments globally across the file, not per image
    source_index = 0       # frame position in the video
    sampled = 0            # frames actually run through the model
    empty_frames = 0
    start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if source_index % FRAME_STRIDE != 0:
                source_index += 1
                continue
            if MAX_FRAMES and sampled >= MAX_FRAMES:
                break

            image_id = sampled
            # Named by source frame number, so a name maps back to a seek position in the
            # video; with FRAME_STRIDE=1 this equals image_id.
            file_name = FRAME_NAME_TEMPLATE.format(index=source_index)
            images.append(make_image_record(image_id, file_name, width, height))

            detections = mbav.detect_frame(
                frame, model, processor, id2label, thing_ids, args, use_amp
            )

            if detections:
                for det in detections:
                    annotations.append(make_annotation_record(next_ann_id, image_id, det))
                    next_ann_id += 1
                    used_label_ids.add(int(det["label_id"]))
                    per_class_counts[det["name"]] = per_class_counts.get(det["name"], 0) + 1
            else:
                annotations.append(make_empty_annotation_record(next_ann_id, image_id))
                next_ann_id += 1
                empty_frames += 1

            sampled += 1
            source_index += 1
            if PROGRESS_EVERY and sampled % PROGRESS_EVERY == 0:
                print(f"[run] sampled {sampled}  frame {source_index - 1}  "
                      f"boxes={len(detections)}")
    finally:
        cap.release()

    runtime = time.perf_counter() - start
    if sampled == 0:
        raise RuntimeError("No frames were read from the input video.")

    dataset = {
        "info": DATASET_NAME,
        "licenses": DATASET_NAME,
        "images": images,
        "annotations": annotations,
        "categories": make_categories(used_label_ids, id2label),
    }
    with Path(OUTPUT_JSON_PATH).open("w", encoding="utf-8") as fh:
        json.dump(dataset, fh)

    # Validate by reading back the file we just wrote, not the in-memory dict.
    with Path(OUTPUT_JSON_PATH).open(encoding="utf-8") as fh:
        check = json.load(fh)
    ann_ids = [a["id"] for a in check["annotations"]]
    image_ids = [i["id"] for i in check["images"]]
    if ann_ids != list(range(len(ann_ids))):
        raise RuntimeError("Annotation ids are not globally sequential from 0.")
    if image_ids != list(range(len(image_ids))):
        raise RuntimeError("Image ids are not sequential from 0.")
    if len(set(a["image_id"] for a in check["annotations"])) != len(image_ids):
        raise RuntimeError("Some image has no annotation row (empty rows are mandatory).")

    box_count = len(annotations) - empty_frames
    print("\n=== maskformer_to_refdrone summary ===")
    print(f"model         : {config.model_type} ({model_dir.name})")
    print(f"checkpoint    : {model_dir}")
    print(f"device        : {args.device}  (autocast={'on' if use_amp else 'off'})")
    print(f"offline       : HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')} "
          f"TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')}")
    print(f"video         : {VIDEO_PATH}")
    print(f"frames        : {sampled} sampled (stride {FRAME_STRIDE}, {source_index} read)")
    print(f"runtime       : {runtime:.1f} s")
    print(f"avg fps       : {sampled / runtime:.2f}")
    print(f"images        : {len(images)}")
    print(f"annotations   : {len(annotations)} ({box_count} boxes, "
          f"{empty_frames} empty placeholders)")
    print(f"categories    : {len(check['categories'])} used")
    for name in sorted(per_class_counts, key=lambda n: per_class_counts[n], reverse=True):
        print(f"  {name:<16} {per_class_counts[name]:>6} boxes")
    print(f"output        : {Path(OUTPUT_JSON_PATH).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
