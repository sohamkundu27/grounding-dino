#!/usr/bin/env python3
"""Fully-offline open-vocabulary video annotation: Grounding DINO + SAM 2.

Grounding DINO turns a natural-language prompt into boxes on the first frame; those boxes
seed the SAM 2 video predictor, which propagates masks across every remaining frame. The
output MP4 preserves the source width, height, FPS and frame order exactly.

Every weight is read from local disk: the Grounding DINO checkpoint/config, the SAM 2
checkpoint/config, and the bert-base-uncased directory used by Grounding DINO's text
encoder. No Hugging Face Hub, torch.hub, REST or hosted-inference call is made at any
point -- HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / HF_DATASETS_OFFLINE are set before
transformers is imported, and the BERT directory is validated up front.

USAGE
-----
1. Minimal command (all model paths default to this repository's local assets):

     python annotate_video.py \
         --input artifacts/demo_input.mp4 \
         --output artifacts/demo_annotated.mp4 \
         --text-prompt "car"

2. Complete command (every path explicit, as used for the recorded demo):

     python annotate_video.py \
         --input artifacts/demo_input.mp4 \
         --output artifacts/demo_annotated.mp4 \
         --text-prompt "car" \
         --grounding-dino-config third_party/grounded_sam_2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py \
         --grounding-dino-checkpoint checkpoints/grounding_dino/groundingdino_swint_ogc.pth \
         --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
         --sam2-checkpoint checkpoints/sam2/sam2.1_hiera_large.pt \
         --bert-model-path checkpoints/bert-base-uncased \
         --box-threshold 0.35 \
         --text-threshold 0.25 \
         --device cuda \
         --codec mp4v \
         --redetect-interval 0

3. Expected output behaviour:
     - Prints resolved local model paths, video metadata (WxH, FPS, frame count), the
       detections found on frame 0, then per-frame propagation progress.
     - Writes an MP4 to --output (parent directories created automatically) in which
       every tracked object carries a translucent mask, a mask-derived box, a label with
       the detected phrase + stable track ID + detection confidence, and a colour that is
       constant across frames for a given track ID.
     - Frame count, order and resolution of the output match the input exactly.
     - If the prompt matches nothing, a valid MP4 of the UNMODIFIED frames is still
       written and a warning is printed (exit code stays 0).

NOTES ON THIS CHECKOUT (verified against the code, not assumed)
---------------------------------------------------------------
  * Local BERT is the repository's own supported path: `util/get_tokenlizer.py` accepts a
    DIRECTORY for `text_encoder_type` (`os.path.isdir(...)`) in both `get_tokenlizer` and
    `get_pretrained_language_model`. `load_model` hard-codes no BERT id -- it reads
    `text_encoder_type` from the SLConfig -- so `load_grounding_dino` below rebuilds
    `load_model`'s five lines with that field overridden to --bert-model-path. No BERT
    integration is invented and no upstream file is modified.
  * `get_tokenlizer` does not forward **kwargs, so `local_files_only=True` cannot be
    passed through it; a local directory + the offline env vars are what make the load
    network-free. `validate_bert_dir` does use `local_files_only=True` explicitly.
  * SAM 2 configs are resolved by HYDRA, not by the filesystem: `sam2/__init__.py` calls
    `initialize_config_module("sam2")`, so a config is named relative to the `sam2`
    package (e.g. "configs/sam2.1/sam2.1_hiera_l.yaml"). A filesystem path is accepted
    here too and translated (see `resolve_sam2_config`).
  * Local checkpoints are SAM 2.1 (sam2.1_hiera_large.pt), pairing with configs/sam2.1/*
    -- not the older sam2_hiera_* names.
  * Grounded-SAM-2 vendors Grounding DINO under `grounding_dino.groundingdino.*`, whose
    modules use absolute imports, so the repo root must be on sys.path.

TRACKING LIMITATIONS (see --redetect-interval)
----------------------------------------------
  * Default (--redetect-interval 0): Grounding DINO runs ONCE on frame 0. Objects that
    enter the scene later are never picked up, and a track lost to full occlusion is not
    recovered. This is fine for short clips where the targets are present from the start.
  * Objects that leave the frame simply stop being drawn (their mask goes empty); their ID
    is not reused, so a re-entering object keeps its identity only if SAM 2's memory still
    holds it.
  * With --redetect-interval N > 0, this SAM 2 build cannot add objects mid-propagation
    ("Cannot add new object id ... after tracking starts"), so the upstream
    `grounded_sam2_tracking_demo_with_continuous_id.py` pattern is used: every N frames the
    state is reset, Grounding DINO re-runs, and all objects are re-registered, with IDs
    carried across chunks by box IoU. Consequences: identity can be reassigned when objects
    overlap or IoU matching fails; SAM 2's memory is cleared at each chunk boundary, so
    long-range identity is weaker than a single pass; and each chunk re-seeds from a box
    prompt, which can drift from the mask. Re-detection is OFF by default.
"""

from __future__ import annotations

import argparse
import colorsys
import os
import sys
import tempfile
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

# --------------------------------------------------------------------------------------
# Offline enforcement -- MUST precede any transformers import (done inside the loaders).
# --------------------------------------------------------------------------------------

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# --------------------------------------------------------------------------------------
# Repository layout / import bootstrap (must happen before grounding_dino & sam2 imports)
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
GSAM2_ROOT = Path(os.environ.get("GROUNDED_SAM2_ROOT", REPO_ROOT / "third_party" / "grounded_sam_2"))

DEFAULT_GD_CONFIG = GSAM2_ROOT / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DEFAULT_GD_CHECKPOINT = REPO_ROOT / "checkpoints/grounding_dino/groundingdino_swint_ogc.pth"
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"  # hydra name, relative to sam2 pkg
DEFAULT_SAM2_CHECKPOINT = REPO_ROOT / "checkpoints/sam2/sam2.1_hiera_large.pt"
DEFAULT_BERT_DIR = REPO_ROOT / "checkpoints/bert-base-uncased"

# Grounded-SAM-2's own modules import as `grounding_dino.*` / `sam2.*` and rely on the
# repo root being importable (its demos are run from that directory).
if str(GSAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(GSAM2_ROOT))

import torch  # noqa: E402  (imported after sys.path bootstrap, before repo modules)

MASK_ALPHA = 0.45  # opacity of the segmentation overlay
FONT = cv2.FONT_HERSHEY_SIMPLEX
REDETECT_IOU_MATCH = 0.5  # IoU above which a re-detection keeps an existing track ID

# BERT assets. Names vary across transformers/Grounding DINO versions, so each entry is a
# set of acceptable alternatives; at least one member of each must exist.
BERT_REQUIRED_ASSETS: tuple[tuple[str, ...], ...] = (
    ("config.json",),
    ("pytorch_model.bin", "model.safetensors"),
    ("vocab.txt", "tokenizer.json"),
)
BERT_OPTIONAL_ASSETS = ("tokenizer_config.json", "special_tokens_map.json", "tokenizer.json")


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoMeta:
    """Metadata of the source video. `frame_count` is None when unreported."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int | None

    def __str__(self) -> str:
        n = self.frame_count if self.frame_count is not None else "unknown"
        return f"{self.width}x{self.height} @ {self.fps:.3f} FPS, {n} frames"


@dataclass(frozen=True)
class Detection:
    """One Grounding DINO hit, in absolute XYXY pixels."""

    box_xyxy: np.ndarray  # shape (4,), float32
    confidence: float
    label: str


@dataclass(frozen=True)
class Track:
    """One object's state on a single frame."""

    track_id: int
    mask: np.ndarray  # bool, (H, W)
    box_xyxy: tuple[int, int, int, int]
    label: str
    confidence: float | None


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and lightly normalise command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="annotate_video.py",
        description=(
            "Annotate a video with open-vocabulary detections, fully offline: Grounding "
            "DINO finds objects matching a text prompt on the first frame, SAM 2 segments "
            "and tracks them through the rest of the video."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "example:\n"
            '  python annotate_video.py --input in.mp4 --output out.mp4 --text-prompt "car"'
        ),
    )
    parser.add_argument("--input", type=Path, required=True, help="Input video file.")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path.")
    parser.add_argument(
        "--text-prompt",
        required=True,
        help='Natural-language prompt, e.g. "person in a red shirt". Grounding DINO '
        "lowercases it and appends '.' if missing; use '.' to separate several classes.",
    )
    parser.add_argument(
        "--grounding-dino-config", type=Path, default=DEFAULT_GD_CONFIG,
        help="Local Grounding DINO model config .py.",
    )
    parser.add_argument(
        "--grounding-dino-checkpoint", type=Path, default=DEFAULT_GD_CHECKPOINT,
        help="Local Grounding DINO .pth checkpoint.",
    )
    parser.add_argument(
        "--sam2-config", default=DEFAULT_SAM2_CONFIG,
        help="Local SAM 2 config: a hydra name relative to the sam2 package "
        "(configs/sam2.1/sam2.1_hiera_l.yaml) or a path to a .yaml file.",
    )
    parser.add_argument(
        "--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT,
        help="Local SAM 2 .pt checkpoint (must match --sam2-config).",
    )
    parser.add_argument(
        "--bert-model-path", type=Path, default=DEFAULT_BERT_DIR,
        help="Local bert-base-uncased directory for Grounding DINO's text encoder. "
        "Used verbatim as the config's text_encoder_type, so nothing is fetched remotely.",
    )
    parser.add_argument("--box-threshold", type=float, default=0.35, help="Grounding DINO box threshold.")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="Grounding DINO text threshold.")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda, cuda:0 or cpu.",
    )
    parser.add_argument("--codec", default="mp4v", help="FourCC code for the VideoWriter.")
    parser.add_argument(
        "--redetect-interval", type=int, default=0, metavar="N",
        help="Re-run Grounding DINO every N frames to catch objects that appear later "
        "(0 = detect once on the first frame). See TRACKING LIMITATIONS in the module "
        "docstring: N>0 resets SAM 2 state every N frames and rematches IDs by box IoU.",
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=95,
        help="Quality of the temporary JPEGs handed to SAM 2 (does not affect output pixels).",
    )
    parser.add_argument(
        "--max-objects", type=int, default=None,
        help="Keep only the N highest-confidence detections.",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA autocast (AMP).")

    args = parser.parse_args(argv)
    if len(args.codec) != 4:
        parser.error(f"--codec must be a 4-character FourCC code, got {args.codec!r}")
    if args.max_objects is not None and args.max_objects < 1:
        parser.error("--max-objects must be >= 1")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    if args.redetect_interval < 0:
        parser.error("--redetect-interval must be >= 0")
    return args


def validate_bert_dir(bert_dir: Path) -> None:
    """Fail before any inference if the local BERT directory is unusable.

    Never falls back to a download: a missing asset is an error, with the one-time
    command needed to materialise the directory.
    """
    populate_hint = (
        "  Populate it ONCE on a networked machine, then re-run offline:\n"
        '    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "from transformers import '
        "BertModel, BertTokenizer; "
        f"BertModel.from_pretrained('bert-base-uncased').save_pretrained('{bert_dir}'); "
        f"BertTokenizer.from_pretrained('bert-base-uncased').save_pretrained('{bert_dir}')\""
    )
    if not bert_dir.is_dir():
        raise FileNotFoundError(
            f"--bert-model-path is not a directory: {bert_dir}\n"
            "  Grounding DINO's text encoder needs a LOCAL bert-base-uncased directory; "
            "this script never downloads one.\n" + populate_hint
        )

    missing = [
        " or ".join(group) for group in BERT_REQUIRED_ASSETS
        if not any((bert_dir / name).is_file() for name in group)
    ]
    if missing:
        present = sorted(p.name for p in bert_dir.iterdir() if p.is_file())
        raise FileNotFoundError(
            f"Local BERT directory {bert_dir} is missing required asset(s): {', '.join(missing)}.\n"
            f"  Present: {', '.join(present) or '(empty)'}\n" + populate_hint
        )

    # A real local load (local_files_only=True) is the only proof the assets are coherent.
    from transformers import AutoTokenizer

    try:
        AutoTokenizer.from_pretrained(str(bert_dir), local_files_only=True)
    except Exception as exc:  # noqa: BLE001 -- surfaced verbatim, never retried online
        raise RuntimeError(
            f"Local BERT tokenizer at {bert_dir} could not be loaded offline: {exc}\n" + populate_hint
        ) from exc

    optional_missing = [n for n in BERT_OPTIONAL_ASSETS if not (bert_dir / n).is_file()]
    detail = f" (optional files absent: {', '.join(optional_missing)})" if optional_missing else ""
    print(f"[offline] Local BERT validated: {bert_dir}{detail}")


def validate_paths(args: argparse.Namespace) -> None:
    """Fail early, with actionable messages, on any bad path or device."""
    if not args.input.is_file():
        raise FileNotFoundError(f"Input video not found: {args.input}")
    for label, path in (
        ("Grounding DINO config", args.grounding_dino_config),
        ("Grounding DINO checkpoint", args.grounding_dino_checkpoint),
        ("SAM 2 checkpoint", args.sam2_checkpoint),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{label} not found: {path}\n"
                f"  Expected a file. Repository defaults live under {REPO_ROOT / 'checkpoints'}."
            )
    if not GSAM2_ROOT.is_dir():
        raise FileNotFoundError(
            f"Grounded-SAM-2 checkout not found at {GSAM2_ROOT}.\n"
            "  Run `git submodule update --init third_party/grounded_sam_2` or set "
            "GROUNDED_SAM2_ROOT."
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA but torch.cuda.is_available() is False.")
    if args.output.suffix.lower() != ".mp4":
        print(f"[warn] --output {args.output.name} does not end in .mp4; writing MP4 data anyway.")
    validate_bert_dir(args.bert_model_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Model loading (each model is built exactly once per run)
# --------------------------------------------------------------------------------------


def load_grounding_dino(config_path: Path, checkpoint_path: Path, device: str, bert_dir: Path):
    """Build Grounding DINO with its text encoder pinned to a LOCAL BERT directory.

    This mirrors the repo's own `util.inference.load_model` (SLConfig -> build_model ->
    load_state_dict) with one change: `text_encoder_type` is set to the local directory
    before `build_model`. That field is what `get_tokenlizer`/`get_pretrained_language_model`
    consume, and both already accept a directory, so this is the repository's supported
    local path rather than a bespoke BERT integration.
    """
    from grounding_dino.groundingdino.models import build_model
    from grounding_dino.groundingdino.util.misc import clean_state_dict
    from grounding_dino.groundingdino.util.slconfig import SLConfig

    args = SLConfig.fromfile(str(config_path))
    args.device = device
    args.text_encoder_type = str(bert_dir)  # was "bert-base-uncased" (a remote HF id)
    model = build_model(args)

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    model.to(device)
    print(f"[load] Grounding DINO  cfg={config_path.name}  ckpt={checkpoint_path.name}  bert={bert_dir.name}")
    return model


def resolve_sam2_config(config_arg: str) -> tuple[str, Path | None]:
    """Map --sam2-config onto a hydra config name.

    SAM 2 composes configs through hydra against the `sam2` package, so a filesystem
    path is not directly usable. Returns (hydra_name, extra_search_dir); the search dir
    is non-None only for a config living outside the sam2 package.
    """
    import sam2

    pkg_dir = Path(sam2.__file__).resolve().parent
    candidate = Path(config_arg)

    if candidate.is_file():
        resolved = candidate.resolve()
        try:  # inside the package -> plain relative hydra name
            return resolved.relative_to(pkg_dir).as_posix(), None
        except ValueError:  # outside -> hydra must be pointed at its directory
            return resolved.name, resolved.parent

    if (pkg_dir / config_arg).is_file():  # already a hydra name
        return config_arg, None

    available = sorted(p.relative_to(pkg_dir).as_posix() for p in (pkg_dir / "configs").rglob("*.yaml"))
    raise FileNotFoundError(
        f"SAM 2 config {config_arg!r} is neither an existing file nor a config name "
        f"known to the sam2 package at {pkg_dir}.\n  Available names:\n    "
        + "\n    ".join(available)
    )


@contextmanager
def _hydra_search_dir(extra_dir: Path | None) -> Iterator[None]:
    """Temporarily point hydra at `extra_dir`, restoring the sam2 package search path."""
    if extra_dir is None:
        yield
        return

    from hydra import initialize_config_dir, initialize_config_module
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(extra_dir), version_base="1.2"):
            yield
    finally:
        GlobalHydra.instance().clear()
        initialize_config_module("sam2", version_base="1.2")  # restore sam2's default


def load_sam2_video_predictor(config_arg: str, checkpoint_path: Path, device: str):
    """Build the SAM 2 video predictor via the repo's `build_sam2_video_predictor`."""
    from sam2.build_sam import build_sam2_video_predictor

    hydra_name, extra_dir = resolve_sam2_config(config_arg)
    print(f"[load] SAM 2 video     cfg={hydra_name}  ckpt={checkpoint_path.name}")
    with _hydra_search_dir(extra_dir):
        return build_sam2_video_predictor(hydra_name, str(checkpoint_path), device=device)


def amp_context(device: str, enabled: bool):
    """CUDA autocast in bfloat16 (fp16 on pre-Ampere); a no-op elsewhere."""
    if not enabled or not device.startswith("cuda"):
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def enable_tf32_if_ampere(device: str) -> None:
    """TF32 matmuls are a free speedup on Ampere+ (as the upstream demos do)."""
    if device.startswith("cuda") and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# --------------------------------------------------------------------------------------
# Video I/O
# --------------------------------------------------------------------------------------


def read_video_metadata(video_path: Path) -> VideoMeta:
    """Read width/height/FPS/frame-count, tolerating missing or bogus metadata."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {video_path}\n"
                "  The file may be corrupt or its codec unsupported by this OpenCV build."
            )
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        raw_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:  # some containers only reveal size on first decode
            ok, probe = cap.read()
            if not ok or probe is None:
                raise RuntimeError(f"Could not decode any frame from {video_path}")
            height, width = probe.shape[:2]
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
            print("[warn] FPS metadata missing/invalid; defaulting output to 30.0 FPS.")
    finally:
        cap.release()

    return VideoMeta(video_path, width, height, fps, raw_count if raw_count > 0 else None)


def _fourcc(codec: str) -> int:
    """FourCC across OpenCV 4 (VideoWriter_fourcc) and 5 (VideoWriter.fourcc)."""
    fn = getattr(cv2.VideoWriter, "fourcc", None) or cv2.VideoWriter_fourcc
    return int(fn(*codec))


def open_video_writer(output_path: Path, meta: VideoMeta, codec: str) -> cv2.VideoWriter:
    """Open a VideoWriter at the source's exact resolution and FPS, or fail loudly."""
    writer = cv2.VideoWriter(str(output_path), _fourcc(codec), meta.fps, (meta.width, meta.height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            f"Could not initialise the video writer for {output_path} with codec {codec!r}.\n"
            "  The codec may be unavailable in this OpenCV build; try --codec avc1 or "
            "--codec XVID (with a .avi output)."
        )
    return writer


def extract_frames_to_dir(video_path: Path, frames_dir: Path, jpeg_quality: int) -> int:
    """Decode the video into zero-padded JPEGs, which is what SAM 2 ingests.

    `load_video_frames` in this checkout also accepts an .mp4 path, but that uses SAM 2's
    own decoder: any disagreement with OpenCV's frame count would silently misalign masks
    from the frames we render onto. Extracting the frames WE decoded keeps SAM 2's frame
    indices identical to ours. Names must be "<index>.jpg" -- SAM 2 sorts them with
    int(os.path.splitext(p)[0]).
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if not cv2.imwrite(str(frames_dir / f"{count:05d}.jpg"), frame, params):
                raise RuntimeError(f"Failed writing temporary frame {count} to {frames_dir}")
            count += 1
    finally:
        cap.release()

    if count == 0:
        raise RuntimeError(f"No frames could be decoded from {video_path}")
    return count


class SequentialFrameReader:
    """Pull original (non-recompressed) frames by index, seeking only when necessary."""

    def __init__(self, video_path: Path) -> None:
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not re-open video for rendering: {video_path}")
        self._next_idx = 0

    def read(self, index: int) -> np.ndarray | None:
        """Return frame `index` in BGR, or None past the end."""
        if index < self._next_idx:  # rewind (propagation normally never goes backwards)
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._next_idx = index
        while self._next_idx < index:  # skip forward
            if not self._cap.grab():
                return None
            self._next_idx += 1
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        self._next_idx += 1
        return frame

    def release(self) -> None:
        self._cap.release()


# --------------------------------------------------------------------------------------
# Detection (Grounding DINO)
# --------------------------------------------------------------------------------------


def detect_objects(
    model,
    frame_path: Path,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
    max_objects: int | None = None,
) -> list[Detection]:
    """Run Grounding DINO on ONE frame and return absolute-XYXY detections.

    Grounding DINO emits normalised cxcywh; SAM 2 wants absolute xyxy pixels, so we
    scale by (w, h, w, h) and convert -- the conversion the upstream demos perform.
    """
    from torchvision.ops import box_convert

    from grounding_dino.groundingdino.util.inference import load_image, predict

    image_source, image_tensor = load_image(str(frame_path))  # (H,W,3) RGB, transformed tensor
    height, width = image_source.shape[:2]

    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )
    if boxes.numel() == 0:
        return []

    boxes_abs = boxes.float() * torch.tensor([width, height, width, height], dtype=torch.float32)
    xyxy = box_convert(boxes_abs, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xyxy[:, 0::2] = xyxy[:, 0::2].clip(0, width - 1)  # clamp to the frame
    xyxy[:, 1::2] = xyxy[:, 1::2].clip(0, height - 1)
    scores = logits.float().numpy().tolist()

    detections = [
        Detection(box.astype(np.float32), float(score), (phrase or text_prompt).strip())
        for box, score, phrase in zip(xyxy, scores, phrases)
        if box[2] > box[0] and box[3] > box[1]  # drop degenerate boxes
    ]
    detections.sort(key=lambda d: d.confidence, reverse=True)
    if max_objects is not None:
        detections = detections[:max_objects]
    return detections


# --------------------------------------------------------------------------------------
# Tracking (SAM 2 video predictor)
# --------------------------------------------------------------------------------------


def initialize_video_tracking(predictor, frames_dir: Path, detections: Sequence[Detection], seed_frame_idx: int = 0):
    """Seed SAM 2 with one box per detection; returns (state, id->label, id->confidence).

    Track IDs are 1-based and assigned in detection order, matching the upstream demos.
    """
    inference_state = predictor.init_state(video_path=str(frames_dir))
    id_to_label: dict[int, str] = {}
    id_to_conf: dict[int, float] = {}

    for track_id, det in enumerate(detections, start=1):
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=seed_frame_idx,
            obj_id=track_id,
            box=det.box_xyxy,
        )
        id_to_label[track_id] = det.label
        id_to_conf[track_id] = det.confidence
    return inference_state, id_to_label, id_to_conf


def mask_to_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight XYXY box around a boolean mask, or None when the mask is empty."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return int(x0), int(y0), int(x1), int(y1)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two XYXY boxes; 0.0 when they do not overlap."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def get_color_for_id(track_id: int) -> tuple[int, int, int]:
    """Deterministic, well-separated BGR colour for a track ID (golden-ratio hue)."""
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_annotations(frame: np.ndarray, tracks: Sequence[Track]) -> np.ndarray:
    """Draw masks, boxes and labels onto a copy of `frame` (uint8 BGR in, uint8 BGR out)."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    font_scale = max(0.4, min(0.85, height / 1000.0))
    thickness = max(1, round(height / 400))

    # Masks first, so boxes and text stay crisp on top of the overlay.
    for track in tracks:
        color = np.array(get_color_for_id(track.track_id), dtype=np.float32)
        region = annotated[track.mask].astype(np.float32)
        annotated[track.mask] = (region * (1.0 - MASK_ALPHA) + color * MASK_ALPHA).astype(np.uint8)

    for track in tracks:
        color = get_color_for_id(track.track_id)
        x0, y0, x1, y1 = track.box_xyxy
        cv2.rectangle(annotated, (x0, y0), (x1, y1), color, thickness)

        label = f"#{track.track_id} {track.label}"
        if track.confidence is not None:
            label += f" {track.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, FONT, font_scale, thickness)
        pad = max(2, thickness + 1)

        # Prefer a label box above the detection; drop it inside when there is no room.
        top = y0 - text_h - baseline - 2 * pad
        top = top if top >= 0 else min(y0, height - text_h - baseline - 2 * pad - 1)
        top = max(0, top)
        left = min(max(0, x0), max(0, width - text_w - 2 * pad))

        # Dark filled plate keeps text legible over any background, then a colour rule
        # tying the label to its mask.
        cv2.rectangle(
            annotated,
            (left, top),
            (min(left + text_w + 2 * pad, width - 1), min(top + text_h + baseline + 2 * pad, height - 1)),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.rectangle(annotated, (left, top), (min(left + text_w + 2 * pad, width - 1), min(top + 2, height - 1)), color, cv2.FILLED)
        cv2.putText(
            annotated,
            label,
            (left + pad, top + text_h + pad + 1),
            FONT,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return annotated


def _tracks_from_logits(obj_ids: Sequence[int], mask_logits, id_to_label, id_to_conf, shape: tuple[int, int]) -> list[Track]:
    """Threshold SAM 2 logits at > 0 and build per-object Track records."""
    height, width = shape
    tracks: list[Track] = []
    for i, obj_id in enumerate(obj_ids):
        mask = (mask_logits[i] > 0.0).cpu().numpy()
        mask = np.squeeze(mask)  # (1, H, W) -> (H, W)
        if mask.ndim != 2 or mask.shape != (height, width) or not mask.any():
            continue  # object absent/occluded on this frame
        box = mask_to_box(mask)
        if box is None:
            continue
        tracks.append(
            Track(int(obj_id), mask.astype(bool), box, id_to_label.get(int(obj_id), "object"), id_to_conf.get(int(obj_id)))
        )
    return tracks


class _FrameEmitter:
    """Renders and writes frames in order, ignoring indices already written."""

    def __init__(self, reader: SequentialFrameReader, writer: cv2.VideoWriter, total_frames: int) -> None:
        self._reader = reader
        self._writer = writer
        self._total = total_frames
        self.written = 0
        self.frames_with_objects = 0
        self._started = time.perf_counter()

    def emit(self, frame_idx: int, tracks: Sequence[Track]) -> bool:
        """Write frame `frame_idx`; returns False if the source frame is unreadable."""
        if frame_idx < self.written:
            return True  # chunk overlap: already written
        frame = self._reader.read(frame_idx)
        if frame is None:
            print(f"[warn] Source frame {frame_idx} could not be decoded; stopping early.")
            return False
        self._writer.write(np.ascontiguousarray(draw_annotations(frame, tracks), dtype=np.uint8))
        self.written += 1
        if tracks:
            self.frames_with_objects += 1

        if self.written == 1 or self.written % 25 == 0 or self.written == self._total:
            elapsed = time.perf_counter() - self._started
            rate = self.written / elapsed if elapsed > 0 else 0.0
            print(f"  frame {self.written}/{self._total}  objects={len(tracks)}  {rate:.1f} FPS", flush=True)
        return True

    def drain(self) -> None:
        """Copy any remaining source frames through unmodified (keeps frame count exact)."""
        while self.written < self._total:
            frame = self._reader.read(self.written)
            if frame is None:
                break
            self._writer.write(np.ascontiguousarray(frame, dtype=np.uint8))
            self.written += 1


def _propagate_single_pass(predictor, state, emitter: _FrameEmitter, meta: VideoMeta, id_to_label, id_to_conf) -> None:
    """Detect-once mode: one propagation over the whole video, rendered as it streams."""
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        tracks = _tracks_from_logits(obj_ids, mask_logits, id_to_label, id_to_conf, (meta.height, meta.width))
        if not emitter.emit(frame_idx, tracks):
            break


def _propagate_with_redetection(
    predictor, state, emitter: _FrameEmitter, meta: VideoMeta, frames_dir: Path,
    detections: Sequence[Detection], grounding_model, args: argparse.Namespace, total_frames: int,
) -> tuple[dict[int, str], dict[int, float]]:
    """Re-detect every N frames, following the upstream continuous-id demo's pattern.

    This SAM 2 build refuses new object ids once tracking has started, so each chunk does
    reset_state -> re-register every live object -> propagate N frames. Existing IDs are
    preserved by matching fresh detections to each track's last known box with IoU.
    """
    interval = args.redetect_interval
    id_to_label: dict[int, str] = {}
    id_to_conf: dict[int, float] = {}
    live: dict[int, dict] = {}  # track_id -> {"box", "label", "conf"}
    next_id = 1

    for det in detections:  # seed from the frame-0 detections
        live[next_id] = {"box": np.asarray(det.box_xyxy, dtype=np.float32), "label": det.label, "conf": det.confidence}
        next_id += 1

    chunk_start = 0
    while chunk_start < total_frames:
        if chunk_start > 0:  # re-detect and reconcile identities
            fresh = detect_objects(
                grounding_model, frames_dir / f"{chunk_start:05d}.jpg", args.text_prompt,
                args.box_threshold, args.text_threshold, args.device, args.max_objects,
            )
            claimed: set[int] = set()
            for det in fresh:
                best_id, best_iou = None, REDETECT_IOU_MATCH
                for tid, st in live.items():
                    if tid in claimed:
                        continue
                    iou = box_iou(det.box_xyxy, st["box"])
                    if iou >= best_iou:
                        best_id, best_iou = tid, iou
                if best_id is None:  # genuinely new object
                    best_id = next_id
                    next_id += 1
                    live[best_id] = {"label": det.label, "conf": det.confidence, "box": None}
                    print(f"  [redetect@{chunk_start}] new object #{best_id} {det.label!r} conf={det.confidence:.2f}")
                live[best_id].update(box=np.asarray(det.box_xyxy, dtype=np.float32), conf=det.confidence)
                claimed.add(best_id)

        live = {tid: st for tid, st in live.items() if st["box"] is not None}
        if not live:  # nothing to track in this chunk: pass frames through untouched
            for idx in range(chunk_start, min(chunk_start + interval, total_frames)):
                if not emitter.emit(idx, []):
                    return id_to_label, id_to_conf
            chunk_start += interval
            continue

        predictor.reset_state(state)  # required: ids cannot be added after tracking starts
        for tid, st in live.items():
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=chunk_start, obj_id=tid, box=st["box"],
            )
            id_to_label[tid] = st["label"]
            id_to_conf[tid] = st["conf"]

        # max_frame_num_to_track is inclusive of the start frame, so N-1 yields N frames.
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
            state, start_frame_idx=chunk_start, max_frame_num_to_track=max(interval - 1, 0)
        ):
            tracks = _tracks_from_logits(obj_ids, mask_logits, id_to_label, id_to_conf, (meta.height, meta.width))
            for track in tracks:  # keep last known boxes fresh for the next IoU match
                live[track.track_id]["box"] = np.asarray(track.box_xyxy, dtype=np.float32)
            if not emitter.emit(frame_idx, tracks):
                return id_to_label, id_to_conf
        chunk_start += interval

    return id_to_label, id_to_conf


def write_annotated_video(
    predictor, state, reader: SequentialFrameReader, writer: cv2.VideoWriter, meta: VideoMeta,
    total_frames: int, id_to_label: dict[int, str], id_to_conf: dict[int, float],
    args: argparse.Namespace, frames_dir: Path, detections: Sequence[Detection], grounding_model,
) -> tuple[int, int]:
    """Propagate masks and stream annotated frames to disk.

    Rendering inside the propagation loop keeps memory flat (only the current frame's
    masks are held) and preserves frame order, which SAM 2 yields ascending from the
    seeded frame. Returns (frames_written, frames_containing_at_least_one_object).
    """
    emitter = _FrameEmitter(reader, writer, total_frames)
    amp_enabled = not args.no_amp and args.device.startswith("cuda")

    with amp_context(args.device, amp_enabled):
        if args.redetect_interval > 0:
            _propagate_with_redetection(
                predictor, state, emitter, meta, frames_dir, detections, grounding_model, args, total_frames
            )
        else:
            _propagate_single_pass(predictor, state, emitter, meta, id_to_label, id_to_conf)

    emitter.drain()  # keep the output frame count identical to the input
    return emitter.written, emitter.frames_with_objects


def write_passthrough_video(reader: SequentialFrameReader, writer: cv2.VideoWriter, total_frames: int) -> int:
    """Copy every source frame through unmodified (used when nothing is detected)."""
    emitter = _FrameEmitter(reader, writer, total_frames)
    emitter.drain()
    return emitter.written


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def _run(args: argparse.Namespace) -> int:
    validate_paths(args)

    meta = read_video_metadata(args.input)
    print(f"[video] {meta.path.name}: {meta}")
    enable_tf32_if_ampere(args.device)
    amp_enabled = not args.no_amp and args.device.startswith("cuda")
    print(f"[setup] device={args.device}  amp={'on' if amp_enabled else 'off'}  codec={args.codec}  "
          f"redetect_interval={args.redetect_interval}")

    reader: SequentialFrameReader | None = None
    writer: cv2.VideoWriter | None = None
    started = time.perf_counter()
    objects_tracked = 0
    frames_with_objects = 0

    # The temp dir, capture and writer are all released in the finally blocks below,
    # including when inference raises.
    with tempfile.TemporaryDirectory(prefix="annotate_video_frames_") as tmp:
        frames_dir = Path(tmp)
        total_frames = extract_frames_to_dir(args.input, frames_dir, args.jpeg_quality)
        if meta.frame_count is None:
            print(f"[video] Frame count metadata was missing; decoded {total_frames} frames.")
        elif meta.frame_count != total_frames:
            print(f"[warn] Metadata claimed {meta.frame_count} frames; decoded {total_frames}. Using the decoded count.")

        try:
            reader = SequentialFrameReader(args.input)
            writer = open_video_writer(args.output, meta, args.codec)

            with torch.inference_mode():
                grounding_model = load_grounding_dino(
                    args.grounding_dino_config, args.grounding_dino_checkpoint, args.device, args.bert_model_path
                )
                print(f"[detect] Grounding DINO on frame 0 | prompt={args.text_prompt!r} "
                      f"box_thr={args.box_threshold} text_thr={args.text_threshold}")
                detections = detect_objects(
                    grounding_model, frames_dir / "00000.jpg", args.text_prompt,
                    args.box_threshold, args.text_threshold, args.device, args.max_objects,
                )

                if not detections:
                    print(f"[warn] No objects matched {args.text_prompt!r} on the first frame. "
                          "Writing the original frames unmodified; try lowering --box-threshold.")
                    written = write_passthrough_video(reader, writer, total_frames)
                else:
                    print(f"[detect] {len(detections)} object(s):")
                    for track_id, det in enumerate(detections, start=1):
                        x0, y0, x1, y1 = det.box_xyxy.tolist()
                        print(f"    #{track_id} {det.label!r} conf={det.confidence:.3f} "
                              f"box=({x0:.0f}, {y0:.0f}, {x1:.0f}, {y1:.0f})")
                    objects_tracked = len(detections)

                    predictor = load_sam2_video_predictor(args.sam2_config, args.sam2_checkpoint, args.device)
                    with amp_context(args.device, amp_enabled):
                        state, id_to_label, id_to_conf = initialize_video_tracking(predictor, frames_dir, detections)
                    print(f"[track] Propagating {len(detections)} object(s) across {total_frames} frames...")
                    written, frames_with_objects = write_annotated_video(
                        predictor, state, reader, writer, meta, total_frames,
                        id_to_label, id_to_conf, args, frames_dir, detections, grounding_model,
                    )
        finally:
            if writer is not None:
                writer.release()
            if reader is not None:
                reader.release()

    elapsed = time.perf_counter() - started
    if written != total_frames:
        print(f"[warn] Wrote {written} of {total_frames} frames.")
    size_mb = args.output.stat().st_size / 1024**2 if args.output.exists() else 0.0
    print(f"[done] {written} frames @ {meta.fps:.3f} FPS ({meta.width}x{meta.height}, {size_mb:.1f} MB)")
    print(f"[done] objects_initialized={objects_tracked}  frames_with_objects={frames_with_objects}")
    print(f"[done] runtime={elapsed:.1f}s  avg_processing_fps={written / elapsed if elapsed else 0:.2f}")
    print(f"[done] Output written to: {args.output.resolve()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Expected failures print one clear line instead of a traceback."""
    args = parse_args(argv)
    try:
        return _run(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[error] Interrupted; the output file may be incomplete.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
