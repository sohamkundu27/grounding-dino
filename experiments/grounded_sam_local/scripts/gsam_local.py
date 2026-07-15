"""Shared local adapter for Grounded SAM (Grounding DINO 1.0 + SAM v1).

This is the SAM v1 sibling of `grounded_sam2_local/scripts/gsam2_local.py`.
It lives OUTSIDE the submodules on purpose: no upstream file is modified.

DETECTOR IS HELD FIXED. The Grounding DINO stage is the *identical* build used in
the SAM 2 evaluation -- same config, same `groundingdino_swint_ogc.pth`, same
compiled `MultiScaleDeformableAttention` CUDA extension, loaded from the
`third_party/grounded_sam_2` tree. Only the segmentation stage changes (SAM v1
ViT-B / ViT-H via `third_party/grounded_sam/segment_anything`). Any difference
between this evaluation and the SAM 2 one is therefore attributable to the SAM
stage alone, not to a different detector.

HARD RULE -- LOCAL ONLY:
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
GSAM = REPO / "third_party" / "grounded_sam"          # SAM v1 (segment_anything)
GSAM2 = REPO / "third_party" / "grounded_sam_2"       # reused for the IDENTICAL detector

# Grounding DINO imports as `grounding_dino.*` from the grounded_sam_2 tree (this is
# the exact build the SAM 2 eval used, with the compiled _C extension). SAM v1 imports
# as `segment_anything` from the grounded_sam tree. Both roots go on sys.path.
for _p in (str(GSAM2), str(GSAM / "segment_anything")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GD_CONFIG = GSAM2 / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GD_CKPT = REPO / "checkpoints/grounding_dino/groundingdino_swint_ogc.pth"

# SAM v1 checkpoints present on disk. ViT-B (91 M) and ViT-H (636 M); no ViT-L here.
SAM_CKPTS = {
    "vit_b": REPO / "checkpoints/sam/sam_vit_b_01ec64.pth",
    "vit_h": REPO / "checkpoints/sam/sam_vit_h_4b8939.pth",
}
SAM_PARAMS_M = {"vit_b": 93.7, "vit_h": 641.1}  # measured at load, recorded for the table

FORBIDDEN = ("dds_cloudapi_sdk", "dinox", "dds_cloudapi")

# Same network posture as the SAM 2 adapter: Grounding DINO's text encoder calls
# transformers' from_pretrained("bert-base-uncased"), which REVALIDATES the cached
# files against huggingface.co on every load (real outbound :443 -- confirmed by strace
# in the SAM 2 run). Nothing is uploaded, but it is not network-free by default. Force
# offline before transformers is ever imported. Escape hatch: GSAM_ALLOW_NETWORK=1.
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
BERT_CACHED = (HF_CACHE / "hub/models--bert-base-uncased").exists() or \
              (HF_CACHE / "models--bert-base-uncased").exists()

if os.environ.get("GSAM_ALLOW_NETWORK") != "1":
    for _v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.setdefault(_v, "1")
    if not BERT_CACHED:
        raise RuntimeError(
            "bert-base-uncased is not in the local HuggingFace cache, and this adapter "
            "runs offline by default, so Grounding DINO's text encoder cannot load.\n"
            "Populate the cache ONCE on a networked machine with:\n"
            "  GSAM_ALLOW_NETWORK=1 python -c \"from transformers import BertModel, "
            "BertTokenizer; BertModel.from_pretrained('bert-base-uncased'); "
            "BertTokenizer.from_pretrained('bert-base-uncased')\"\n"
            "then re-run offline. (Do NOT set GSAM_ALLOW_NETWORK on an air-gapped target.)"
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


def load_sam_image(size: str = "vit_b"):
    """Build a SAM v1 model + SamPredictor. size in {vit_b, vit_h}."""
    from segment_anything import SamPredictor, sam_model_registry

    ckpt = SAM_CKPTS[size]
    assert ckpt.exists(), f"missing {ckpt}"
    sam = sam_model_registry[size](checkpoint=str(ckpt)).to(device())
    sam.eval()
    assert_no_cloud_imports()
    return SamPredictor(sam)


def gd_predict(model, image_path: Path, prompt: str, box_thr=0.35, text_thr=0.25):
    """Run Grounding DINO. Returns (boxes_xyxy_abs, scores, phrases, ms, wh).

    Byte-for-byte the same call the SAM 2 adapter makes -- same model, same
    thresholds, same pre-processing -- so the boxes handed to SAM v1 here are the
    same boxes SAM 2.1 received there.
    """
    import torch
    from grounding_dino.groundingdino.util.inference import load_image, predict
    from torchvision.ops import box_convert

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

    boxes_abs = boxes * torch.tensor([w, h, w, h])
    xyxy = box_convert(boxes_abs, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    return xyxy, logits.numpy().tolist(), phrases, times[0], (w, h)


def sam_masks(predictor, image_rgb, boxes_xyxy, split: bool = False):
    """SAM v1 image segmentation from boxes. Returns (masks_bool, ms[, breakdown]).

    `ms` is the END-TO-END SAM v1 cost: ViT image encode + mask decode.

    set_image() runs the ViT-B/ViT-H image backbone -- the dominant, backbone-
    dependent stage, exactly analogous to SAM 2's set_image()/Hiera encoder. Timing
    only predict_torch() would measure the mask decoder alone (near-identical across
    backbones) and make ViT-H look as cheap as ViT-B. Both stages are timed here.

    SAM v1 is run at its standard fp32 precision (no autocast). This is intentional:
    it is how SAM v1 is normally deployed, and the precision difference vs SAM 2's
    bf16 path is part of the honest cost comparison, not a bug to be papered over.
    """
    import numpy as np
    import torch

    enc: list[float] = []
    dec: list[float] = []
    with cuda_timer(enc):
        predictor.set_image(image_rgb)              # ViT image encoder
    boxes_t = torch.as_tensor(np.asarray(boxes_xyxy), dtype=torch.float, device=device())
    tboxes = predictor.transform.apply_boxes_torch(boxes_t, image_rgb.shape[:2])
    with cuda_timer(dec):
        masks, scores, _ = predictor.predict_torch(  # prompt encoder + mask decoder
            point_coords=None, point_labels=None,
            boxes=tboxes, multimask_output=False,
        )
    masks = masks.squeeze(1).detach().cpu().numpy().astype(bool)  # (N, H, W)
    total = enc[0] + dec[0]
    if split:
        return masks, total, {"encode_ms": enc[0], "decode_ms": dec[0]}
    return masks, total


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
