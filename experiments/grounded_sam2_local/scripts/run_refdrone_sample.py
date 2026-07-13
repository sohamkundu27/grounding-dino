#!/usr/bin/env python3
"""PHASE 2 — deterministic RefDrone sample through the local pipeline.

For each sampled (image, referring-expression) pair:
  expression -> Grounding DINO -> highest-confidence box -> SAM 2 -> mask

Metrics computed against RefDrone's GROUND-TRUTH BOXES only. RefDrone provides
no mask ground truth, so SAM 2 mask quality is NOT scored here -- it is assessed
visually. No segmentation ground truth is invented.

    python run_refdrone_sample.py --n-train 30 --n-val 10 --n-test 10 --seed 1234
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsam2_local as G  # noqa: E402

ANN = G.REPO / "datasets/refdrone/annotations"
IMGS = G.REPO / "datasets/refdrone/images/all_image"
OUT = G.REPO / "experiments/grounded_sam2_local/outputs"
RES = G.REPO / "experiments/grounded_sam2_local/results"


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def load_split(split: str):
    """Return list of records: (file_name, caption, [gt boxes xyxy])."""
    d = json.loads((ANN / f"RefDrone_{split}_mdetr.json").read_text())
    by_img = {i["id"]: i for i in d["images"]}
    boxes = defaultdict(list)
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]              # COCO xywh
        boxes[a["image_id"]].append([x, y, x + w, y + h])
    out = []
    for iid, im in by_img.items():
        cap = (im.get("caption") or "").strip()
        if not cap:
            continue
        out.append({"file_name": im["file_name"], "caption": cap,
                    "gt_boxes": boxes.get(iid, []), "image_id": iid})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--n-val", type=int, default=10)
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sam2", default="tiny", choices=list(G.SAM2_CKPTS))
    ap.add_argument("--box-thr", type=float, default=0.25)
    ap.add_argument("--text-thr", type=float, default=0.20)
    ap.add_argument("--save-vis", type=int, default=12)
    args = ap.parse_args()

    import cv2
    import torch

    G.device()
    G.reset_gpu_mem()
    rng = random.Random(args.seed)

    sample = []
    for split, n in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test)):
        recs = [r for r in load_split(split) if r["gt_boxes"]]  # need GT to score
        recs.sort(key=lambda r: (r["file_name"], r["caption"]))  # determinism
        for r in rng.sample(recs, min(n, len(recs))):
            r["split"] = split
            sample.append(r)
    print(f"sampled {len(sample)} (image, expression) pairs  seed={args.seed}")

    gd = G.load_grounding_dino()
    sam = G.load_sam2_image(args.sam2)

    rows, n_vis = [], 0
    for i, r in enumerate(sample):
        img_path = IMGS / r["file_name"]
        if not img_path.exists():
            print(f"  !! missing image {r['file_name']}")
            continue

        boxes, scores, phrases, gd_ms, (w, h) = G.gd_predict(
            gd, img_path, r["caption"], args.box_thr, args.text_thr
        )
        n_det = 0 if boxes is None else len(boxes)

        best_iou, best_box, sam_ms, mask_px = 0.0, None, None, None
        if n_det:
            top = int(np.argmax(scores))          # highest-confidence box
            best_box = [float(v) for v in boxes[top]]
            best_iou = max((iou(best_box, g) for g in r["gt_boxes"]), default=0.0)
            img_rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            masks, sam_ms = G.sam2_masks(sam, img_rgb, np.array([best_box]))
            mask_px = int(masks[0].sum())

            if n_vis < args.save_vis:
                from run_image_demo import visualise
                tag = "hit" if best_iou >= 0.5 else "miss"
                vis = OUT / "images" / f"refdrone_{tag}_{i:03d}_{img_path.stem}.png"
                visualise(img_rgb, np.array([best_box]), [scores[top]],
                          [r["caption"][:40]], masks, r["caption"][:60], vis)
                n_vis += 1

        G.assert_no_cloud_imports()
        rows.append({
            "split": r["split"], "file_name": r["file_name"],
            "expression": r["caption"], "resolution": [w, h],
            "n_gt_boxes": len(r["gt_boxes"]), "n_detections": n_det,
            "top_score": round(float(max(scores)), 4) if n_det else None,
            "pred_box": [round(v, 1) for v in best_box] if best_box else None,
            "best_iou": round(best_iou, 4),
            "hit_at_50": bool(best_iou >= 0.5),
            "gd_latency_ms": round(gd_ms, 1),
            "sam2_latency_ms": round(sam_ms, 1) if sam_ms else None,
            "mask_pixels": mask_px,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sample)}")

    n = len(rows)
    det = [r for r in rows if r["n_detections"] > 0]
    ious = [r["best_iou"] for r in rows]
    summary = {
        "sample": {"n": n, "seed": args.seed,
                   "by_split": {s: sum(1 for r in rows if r["split"] == s)
                                for s in ("train", "val", "test")}},
        "config": {"gd": "Swin-T", "sam2": args.sam2,
                   "box_threshold": args.box_thr, "text_threshold": args.text_thr},
        "metrics": {
            "top1_localization_acc_iou50": round(sum(r["hit_at_50"] for r in rows) / n, 4),
            "mean_best_iou": round(float(np.mean(ious)), 4),
            "median_best_iou": round(float(np.median(ious)), 4),
            "no_detection_rate": round(sum(1 for r in rows if r["n_detections"] == 0) / n, 4),
            "multiple_detection_rate": round(sum(1 for r in rows if r["n_detections"] > 1) / n, 4),
            "mean_detections_when_found": round(
                float(np.mean([r["n_detections"] for r in det])), 2) if det else None,
            "mean_gd_latency_ms": round(float(np.mean([r["gd_latency_ms"] for r in rows])), 1),
            "mean_sam2_latency_ms": round(
                float(np.mean([r["sam2_latency_ms"] for r in det])), 1) if det else None,
            "mean_total_latency_ms": round(float(np.mean(
                [r["gd_latency_ms"] + (r["sam2_latency_ms"] or 0) for r in rows])), 1),
        },
        "gpu_memory": G.gpu_mem_mb(),
        "mask_ground_truth": "NONE — RefDrone provides boxes only. SAM 2 mask quality "
                             "is assessed visually and is NOT scored here.",
        "results": rows,
    }
    G.save_json(summary, RES / "refdrone_results.json")

    m = summary["metrics"]
    print(f"\n  top-1 localization (IoU>=0.5): {m['top1_localization_acc_iou50']:.1%}")
    print(f"  mean best IoU                : {m['mean_best_iou']:.3f}")
    print(f"  no-detection rate            : {m['no_detection_rate']:.1%}")
    print(f"  multiple-detection rate      : {m['multiple_detection_rate']:.1%}")
    print(f"  mean GD / SAM2 latency       : {m['mean_gd_latency_ms']:.0f} / "
          f"{m['mean_sam2_latency_ms']:.0f} ms")
    print(f"  peak VRAM                    : {summary['gpu_memory']['peak_allocated_mb']} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
