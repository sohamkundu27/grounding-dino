"""Shared local adapter for Grounded SAM 2 (Grounding DINO 1.0 + SAM 2.1).

This lives OUTSIDE the submodule on purpose: no upstream file is modified.
It wires up the same fully-local code path used by
`grounded_sam2_local_demo.py`, and nothing else.

HARD RULE — LOCAL ONLY:
  * no dds_cloudapi_sdk, no DINO-X, no Grounding DINO 1.5/1.6
  * no API keys, no cloud inference, no image upload
  * every weight is read from checkpoints/ on this disk
`assert_no_cloud_imports()` enforces this at runtime.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GSAM2 = REPO / "third_party" / "grounded_sam_2"

# Grounded-SAM-2's own modules import as `grounding_dino.*` and `sam2.*`,
# which requires its root on sys.path (its demos rely on being run from there).
if str(GSAM2) not in sys.path:
    sys.path.insert(0, str(GSAM2))

GD_CONFIG = GSAM2 / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_CKPT = REPO / "checkpoints/grounding_dino/groundingdino_swint_ogc.pth"
SAM2_CKPTS = {
    "tiny": REPO / "checkpoints/sam2/sam2.1_hiera_tiny.pt",
    "small": REPO / "checkpoints/sam2/sam2.1_hiera_small.pt",
    "base_plus": REPO / "checkpoints/sam2/sam2.1_hiera_base_plus.pt",
    "large": REPO / "checkpoints/sam2/sam2.1_hiera_large.pt",
}
SAM2_CFGS = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}

FORBIDDEN = ("dds_cloudapi_sdk", "dinox", "dds_cloudapi")

# Grounding DINO's text encoder calls transformers' from_pretrained("bert-base-uncased").
# Even with the weights already in ~/.cache/huggingface, that REVALIDATES them against
# huggingface.co on every load -- strace shows real outbound :443 connections to the HF
# CDN. Nothing is uploaded, but the pipeline is not network-free by default, which is
# fatal for an air-gapped target. Force offline before transformers is ever imported.
# Escape hatch for first-time setup on a machine with no HF cache: GSAM2_ALLOW_NETWORK=1.
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
BERT_CACHED = (HF_CACHE / "hub/models--bert-base-uncased").exists() or \
              (HF_CACHE / "models--bert-base-uncased").exists()

if os.environ.get("GSAM2_ALLOW_NETWORK") != "1":
    for _v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.setdefault(_v, "1")
    if not BERT_CACHED:
        raise RuntimeError(
            "bert-base-uncased is not in the local HuggingFace cache, and this adapter "
            "runs offline by default, so Grounding DINO's text encoder cannot load.\n"
            "Populate the cache ONCE on a networked machine with:\n"
            "  GSAM2_ALLOW_NETWORK=1 python -c \"from transformers import BertModel, "
            "BertTokenizer; BertModel.from_pretrained('bert-base-uncased'); "
            "BertTokenizer.from_pretrained('bert-base-uncased')\"\n"
            "then re-run offline. (Do NOT set GSAM2_ALLOW_NETWORK on an air-gapped target.)"
        )

OFFLINE_ENFORCED = os.environ.get("HF_HUB_OFFLINE") == "1"


def assert_no_cloud_imports() -> None:
    """Fail loudly if any cloud-API module ever gets imported."""
    bad = [m for m in sys.modules if any(f in m.lower() for f in FORBIDDEN)]
    if bad:
        raise RuntimeError(f"CLOUD API MODULE LOADED — aborting: {bad}")


def device() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Refusing to run on CPU: CPU timings would "
            "misrepresent the GPU pipeline."
        )
    return "cuda"


@contextmanager
def cuda_timer(store: list):
    """Wall time around a GPU op, with CUDA synchronised on BOTH sides.

    Without the syncs the kernel is still queued when the timer stops and the
    number is fiction.
    """
    import torch

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize()
    store.append((time.perf_counter() - t0) * 1000.0)  # ms


def load_grounding_dino():
    from grounding_dino.groundingdino.util.inference import load_model

    assert GD_CKPT.exists(), f"missing {GD_CKPT}"
    model = load_model(str(GD_CONFIG), str(GD_CKPT), device=device())
    assert_no_cloud_imports()
    return model


def load_sam2_image(size: str = "tiny"):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    ckpt = SAM2_CKPTS[size]
    assert ckpt.exists(), f"missing {ckpt}"
    # SAM 2 resolves its config through Hydra relative to the sam2 package.
    model = build_sam2(SAM2_CFGS[size], str(ckpt), device=device())
    assert_no_cloud_imports()
    return SAM2ImagePredictor(model)


def load_sam2_video(size: str = "tiny"):
    from sam2.build_sam import build_sam2_video_predictor

    ckpt = SAM2_CKPTS[size]
    assert ckpt.exists(), f"missing {ckpt}"
    p = build_sam2_video_predictor(SAM2_CFGS[size], str(ckpt), device=device())
    assert_no_cloud_imports()
    return p


def gd_predict(model, image_path: Path, prompt: str, box_thr=0.35, text_thr=0.25):
    """Run Grounding DINO. Returns (boxes_xyxy_abs, scores, phrases, ms, wh)."""
    import torch
    from torchvision.ops import box_convert
    from grounding_dino.groundingdino.util.inference import load_image, predict

    image_source, image = load_image(str(image_path))
    h, w = image_source.shape[:2]

    times: list[float] = []
    with cuda_timer(times):
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=box_thr,
            text_threshold=text_thr,
        )

    if boxes.numel() == 0:
        return None, [], [], times[0], (w, h)

    # GD returns cxcywh normalised -> xyxy absolute
    boxes_abs = boxes * torch.tensor([w, h, w, h])
    xyxy = box_convert(boxes_abs, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    return xyxy, logits.numpy().tolist(), phrases, times[0], (w, h)


def sam2_masks(predictor, image_rgb, boxes_xyxy, split: bool = False):
    """SAM 2 image segmentation from boxes. Returns (masks, ms[, breakdown]).

    `ms` is the END-TO-END SAM 2 cost: image encode + mask decode.

    set_image() is what runs the Hiera image backbone, and it is the dominant,
    backbone-dependent stage. Timing only predict() measures the mask decoder,
    which is nearly identical across Tiny/Small/Large and makes a bigger
    checkpoint look as fast as a smaller one. Both stages are timed here; pass
    split=True to get them separately.
    """
    import torch

    enc: list[float] = []
    dec: list[float] = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with cuda_timer(enc):
            predictor.set_image(image_rgb)          # Hiera image encoder
        with cuda_timer(dec):
            masks, scores, _ = predictor.predict(   # prompt encoder + mask decoder
                point_coords=None, point_labels=None,
                box=boxes_xyxy, multimask_output=False,
            )
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    total = enc[0] + dec[0]
    if split:
        return masks.astype(bool), total, {"encode_ms": enc[0], "decode_ms": dec[0]}
    return masks.astype(bool), total


def gpu_mem_mb() -> dict:
    import torch

    return {
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
    }


def reset_gpu_mem():
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    print(f"wrote {path}")
