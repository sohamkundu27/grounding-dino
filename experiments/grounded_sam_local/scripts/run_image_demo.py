#!/usr/bin/env python3
"""PHASE 1 — smoke test: one image through the fully-local Grounded SAM (v1) pipeline.

  text prompt -> Grounding DINO 1.0 (local) -> box -> SAM v1 (local) -> mask

Writes a side-by-side visualisation, the raw boxes/scores/masks, and timings.

    python run_image_demo.py --image X.jpg --prompts "person" "car"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsam_local as G  # noqa: E402

OUT = G.REPO / "experiments/grounded_sam_local/outputs"
RES = G.REPO / "experiments/grounded_sam_local/results"


def visualise(img_rgb, boxes, scores, phrases, masks, prompt, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax in axes:
        ax.imshow(img_rgb)
        ax.axis("off")
    axes[0].set_title("input", fontsize=13)

    axes[1].set_title(f"Grounding DINO boxes — '{prompt}'", fontsize=13)
    if boxes is not None:
        for b, s, ph in zip(boxes, scores, phrases):
            x0, y0, x1, y1 = b
            axes[1].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                                fill=False, color="lime", lw=2))
            axes[1].text(x0, max(y0 - 5, 8), f"{ph} {s:.2f}", color="black", fontsize=9,
                         bbox=dict(facecolor="lime", alpha=0.85, pad=1, edgecolor="none"))

    axes[2].set_title("SAM v1 masks", fontsize=13)
    if masks is not None:
        overlay = np.zeros((*img_rgb.shape[:2], 4))
        rng = np.random.default_rng(0)
        for m in masks:
            c = np.concatenate([rng.random(3), [0.55]])
            overlay[m] = c
        axes[2].imshow(overlay)
        for b in (boxes if boxes is not None else []):
            x0, y0, x1, y1 = b
            axes[2].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                                fill=False, color="white", lw=1.2, ls="--"))

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=95, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--sam", default="vit_b", choices=list(G.SAM_CKPTS))
    ap.add_argument("--box-thr", type=float, default=0.35)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--tag", default="smoke")
    args = ap.parse_args()

    import cv2
    import torch

    dev = G.device()
    print(f"device: {dev}  ({torch.cuda.get_device_name(0)})")
    G.reset_gpu_mem()

    t = []
    with G.cuda_timer(t):
        gd = G.load_grounding_dino()
        sam = G.load_sam_image(args.sam)
    load_ms = t[0]
    print(f"model load: {load_ms:.0f} ms  (GD Swin-T + SAM v1 {args.sam})")

    img_bgr = cv2.imread(str(args.image))
    if img_bgr is None:
        sys.exit(f"cannot read image: {args.image}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    records = []
    for prompt in args.prompts:
        boxes, scores, phrases, gd_ms, (w, h) = G.gd_predict(
            gd, args.image, prompt, args.box_thr, args.text_thr
        )
        n = 0 if boxes is None else len(boxes)
        sam_ms, masks = None, None
        if n:
            masks, sam_ms = G.sam_masks(sam, img_rgb, boxes)

        G.assert_no_cloud_imports()

        rec = {
            "image": str(args.image.relative_to(G.REPO)) if args.image.is_absolute()
                     else str(args.image),
            "resolution": [w, h],
            "prompt": prompt,
            "sam_size": args.sam,
            "n_detections": n,
            "scores": [round(float(s), 4) for s in scores],
            "phrases": phrases,
            "boxes_xyxy": [[round(float(v), 1) for v in b] for b in (boxes if n else [])],
            "gd_latency_ms": round(gd_ms, 1),
            "sam_latency_ms": round(sam_ms, 1) if sam_ms else None,
            "total_latency_ms": round(gd_ms + (sam_ms or 0), 1),
            "mask_pixels": [int(m.sum()) for m in masks] if masks is not None else [],
        }
        records.append(rec)
        print(f"  '{prompt}': {n} det  gd={gd_ms:.0f}ms"
              + (f"  sam={sam_ms:.0f}ms" if sam_ms else "  (no box -> SAM skipped)")
              + (f"  top={max(scores):.2f}" if n else ""))

        slug = "".join(c if c.isalnum() else "_" for c in prompt)[:32]
        vis = OUT / "images" / f"{args.tag}_{args.image.stem}_{slug}.png"
        visualise(img_rgb, boxes, scores, phrases, masks, prompt, vis)
        rec["visualisation"] = str(vis.relative_to(G.REPO))

    mem = G.gpu_mem_mb()
    out = {
        "config": {"gd": "Swin-T", "sam": args.sam,
                   "box_threshold": args.box_thr, "text_threshold": args.text_thr},
        "model_load_ms": round(load_ms, 1),
        "gpu_memory": mem,
        "device": torch.cuda.get_device_name(0),
        "results": records,
    }
    G.save_json(out, RES / "image_results.json")
    print(f"peak VRAM: {mem['peak_allocated_mb']} MB allocated / "
          f"{mem['peak_reserved_mb']} MB reserved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
