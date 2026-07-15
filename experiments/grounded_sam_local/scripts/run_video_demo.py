#!/usr/bin/env python3
"""PHASE 3 — video with Grounded SAM (v1): PER-FRAME detect-and-segment.

  every frame:  text prompt -> Grounding DINO -> highest-confidence box -> SAM v1 -> mask

This is NOT the SAM 2 protocol, and it cannot be. SAM v1 has no memory / no video
propagation: there is no `build_sam2_video_predictor`, no `init_state`, no
`propagate_in_video`. The ONLY way to apply Grounded SAM (v1) to a clip is to re-run
the whole image pipeline on every frame. So this script measures exactly that, and it
records the two things that fall out of it:

  1. COST: the detector is in the loop on every frame (vs SAM 2's detect-once), so the
     per-frame cost is GD + SAM every time -- no cheap steady state.
  2. NO IDENTITY: each frame is segmented independently. There is no object id carried
     across frames; "which car" is re-decided every frame by detector confidence. We
     quantify this as the frame-to-frame jump of the top-1 box centroid. A large jump
     with the target still clearly present is the top-1 hopping between instances --
     the failure mode that a memory-based tracker (SAM 2) is built to avoid, and that
     SAM v1 structurally cannot avoid because it has no memory to begin with.

    python run_video_demo.py --video X.mp4 --prompt "car" --sam vit_b
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsam_local as G  # noqa: E402

OUT = G.REPO / "experiments/grounded_sam_local/outputs"
RES = G.REPO / "experiments/grounded_sam_local/results"


def extract_frames(video: Path, dst: Path, max_frames: int,
                   start: int = 0) -> tuple[int, int, int, float]:
    import cv2

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for _ in range(start):          # skip leading frames (e.g. a fade-in)
        if not cap.read()[0]:
            break
    i = 0
    while i < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        cv2.imwrite(str(dst / f"{i:05d}.jpg"), fr)
        i += 1
    cap.release()
    return i, w, h, fps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=lambda x: Path(x).resolve())
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--sam", default="vit_b", choices=list(G.SAM_CKPTS))
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--start-frame", type=int, default=0,
                    help="skip N leading frames (e.g. a fade-in) before frame 0")
    ap.add_argument("--box-thr", type=float, default=0.35)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    import cv2
    import torch

    G.device()
    G.reset_gpu_mem()

    frames_dir = OUT / "videos" / f"frames_{args.video.stem}"
    n_frames, W, H, fps = extract_frames(args.video, frames_dir, args.max_frames,
                                         args.start_frame)
    print(f"{args.video.name}: {W}x{H}, {n_frames} frames @ {fps:.0f} fps")

    gd = G.load_grounding_dino()
    sam = G.load_sam_image(args.sam)

    # ---- per-frame detect + segment -----------------------------------------
    per_frame = []          # dict per frame
    masks_by_frame = {}
    gd_ms_all, sam_ms_all = [], []
    edge = 2

    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for fidx in range(n_frames):
        fpath = frames_dir / f"{fidx:05d}.jpg"
        boxes, scores, phrases, gd_ms, _ = G.gd_predict(gd, fpath, args.prompt,
                                                        args.box_thr, args.text_thr)
        gd_ms_all.append(gd_ms)
        n_det = 0 if boxes is None else len(boxes)

        rec = {"frame": fidx, "n_detections": n_det, "gd_ms": round(gd_ms, 1)}
        if n_det:
            top = int(np.argmax(scores))
            box = [float(v) for v in boxes[top]]
            img_rgb = cv2.cvtColor(cv2.imread(str(fpath)), cv2.COLOR_BGR2RGB)
            masks, sam_ms = G.sam_masks(sam, img_rgb, np.array([box]))
            sam_ms_all.append(sam_ms)
            m = masks[0]
            masks_by_frame[fidx] = m
            ys, xs = np.where(m)
            area = int(m.sum())
            if len(xs):
                x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
                cent = (float(xs.mean()), float(ys.mean()))
                touched = {"left": x0 <= edge, "top": y0 <= edge,
                           "right": x1 >= W - 1 - edge, "bottom": y1 >= H - 1 - edge}
            else:
                x0 = y0 = x1 = y1 = 0
                cent, touched = None, None
            rec.update({
                "top_score": round(float(scores[top]), 4),
                "top_box_xyxy": [round(v, 1) for v in box],
                "sam_ms": round(sam_ms, 1),
                "mask_area_px": area,
                "mask_bbox_xyxy": [x0, y0, x1, y1],
                "centroid": [round(v, 1) for v in cent] if cent else None,
                "touching_border": [k for k, on in (touched or {}).items() if on] or None,
            })
        else:
            rec.update({"top_score": None, "top_box_xyxy": None, "sam_ms": None,
                        "mask_area_px": 0, "mask_bbox_xyxy": None,
                        "centroid": None, "touching_border": None})
        per_frame.append(rec)
        if (fidx + 1) % 50 == 0:
            print(f"  {fidx+1}/{n_frames}  (det_rate so far "
                  f"{sum(1 for r in per_frame if r['n_detections']) / len(per_frame):.0%})")
    torch.cuda.synchronize()
    wall = time.perf_counter() - t_start

    G.assert_no_cloud_imports()

    # ---- derived metrics -----------------------------------------------------
    detected = [r for r in per_frame if r["n_detections"] > 0]
    det_rate = len(detected) / n_frames if n_frames else 0.0
    areas = np.array([r["mask_area_px"] for r in per_frame])
    n_counts = [r["n_detections"] for r in per_frame]

    # on/off flicker: transitions between detected and not-detected
    on = [r["n_detections"] > 0 for r in per_frame]
    flicker = sum(1 for a, b in zip(on, on[1:]) if a != b)

    # contiguous detected runs (parallel to SAM 2's "visible runs")
    runs, cur = [], None
    for i, o in enumerate(on):
        if o and cur is None:
            cur = i
        elif not o and cur is not None:
            runs.append([cur, i - 1]); cur = None
    if cur is not None:
        runs.append([cur, n_frames - 1])

    # NO-IDENTITY evidence: frame-to-frame centroid jump of the top-1 box, over
    # CONSECUTIVE frames that both had a detection. A memory tracker keeps this small
    # while the target is present; per-frame top-1 does not have to.
    jumps = []
    for a, b in zip(per_frame, per_frame[1:]):
        if a["centroid"] and b["centroid"]:
            jumps.append(float(np.hypot(b["centroid"][0] - a["centroid"][0],
                                        b["centroid"][1] - a["centroid"][1])))
    jumps = np.array(jumps) if jumps else np.array([0.0])
    # a "hop" = consecutive-frame top-1 centroid jump larger than 10% of the diagonal
    diag = float(np.hypot(W, H))
    hop_thr = 0.10 * diag
    n_hops = int((jumps > hop_thr).sum())

    def stat(xs):
        xs = np.asarray(xs, dtype=float)
        if not len(xs):
            return None
        return {"mean": round(float(xs.mean()), 1), "median": round(float(np.median(xs)), 1),
                "p95": round(float(np.percentile(xs, 95)), 1),
                "max": round(float(xs.max()), 1)}

    total_ms = [r["gd_ms"] + (r["sam_ms"] or 0) for r in per_frame]

    # ---- annotated output video ---------------------------------------------
    name = args.out_name or f"perframe_{args.video.stem}_{args.sam}"
    outv = OUT / "videos" / f"{name}.mp4"
    vw = cv2.VideoWriter(str(outv), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    colour = np.array([30, 220, 60], dtype=np.uint8)
    for fidx in range(n_frames):
        fr = cv2.imread(str(frames_dir / f"{fidx:05d}.jpg"))
        m = masks_by_frame.get(fidx)
        state = "no detection"
        if m is not None and m.any():
            fr[m] = (0.55 * fr[m] + 0.45 * colour).astype(np.uint8)
            ys, xs = np.where(m)
            cv2.rectangle(fr, (xs.min(), ys.min()), (xs.max(), ys.max()), (30, 220, 60), 2)
            state = f"{per_frame[fidx]['n_detections']} det, top1 shown"
        cv2.putText(fr, f"{args.prompt} | f{fidx} | {state}",
                    (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        vw.write(fr)
    vw.release()

    # sample frames: first, mid, last, plus the largest-hop frame pair
    keys = {0, n_frames // 2, n_frames - 1}
    if len(jumps) and jumps.max() > hop_thr:
        hi = int(np.argmax(jumps))
        keys |= {hi, hi + 1}
    for k in sorted(keys):
        if k in masks_by_frame:
            fr = cv2.imread(str(frames_dir / f"{k:05d}.jpg"))
            m = masks_by_frame[k]
            fr[m] = (0.55 * fr[m] + 0.45 * colour).astype(np.uint8)
            cv2.imwrite(str(OUT / "images" / f"{name}_frame{k:05d}.jpg"), fr)

    mem = G.gpu_mem_mb()
    res = {
        "video": str(args.video.relative_to(G.REPO)),
        "start_frame": args.start_frame,
        "resolution": [W, H], "source_fps": round(fps, 1),
        "frames_processed": n_frames,
        "prompt": args.prompt,
        "config": {"gd": "Swin-T", "sam": args.sam},
        "architecture": {
            "sam_v1_has_video_propagation": False,
            "detector_run_once": False,
            "detector_run_every_frame": True,
            "object_identity_across_frames": False,
            "note": ("SAM v1 has no memory/propagation. Every frame is an independent "
                     "detect+segment; there is no object id and no temporal consistency. "
                     "Contrast with the SAM 2 pipeline, which detects once and propagates."),
        },
        "throughput": {
            "wall_seconds": round(wall, 3),
            "fps": round(n_frames / wall, 2) if wall else 0.0,
            "mean_ms_per_frame": round(float(np.mean(total_ms)), 1),
            "median_ms_per_frame": round(float(np.median(total_ms)), 1),
            "p95_ms_per_frame": round(float(np.percentile(total_ms, 95)), 1),
            "gd_ms_per_frame": stat(gd_ms_all),
            "sam_ms_per_frame": stat(sam_ms_all),
        },
        "detection": {
            "detection_rate": round(det_rate, 4),
            "frames_with_no_detection": int(sum(1 for o in on if not o)),
            "on_off_flicker_transitions": flicker,
            "detected_runs": runs,
            "detections_per_frame": {
                "mean": round(float(np.mean(n_counts)), 2),
                "median": int(np.median(n_counts)),
                "max": int(max(n_counts)),
            },
        },
        "identity": {
            "note": ("top-1 box is re-chosen every frame by detector confidence; no memory "
                     "links it to the previous frame. Large consecutive-frame centroid jumps "
                     "with the target still present = the top-1 hopping between instances."),
            "top1_centroid_jump_px": stat(jumps),
            "hop_threshold_px": round(hop_thr, 1),
            "n_consecutive_frame_hops": n_hops,
            "hop_definition": "consecutive-frame top-1 centroid jump > 10% of image diagonal",
        },
        "per_frame_mask_area_px": [int(a) for a in areas],
        "per_frame": per_frame,
        "gpu_memory": mem,
        "outputs": {"video": str(outv.relative_to(G.REPO)),
                    "note": "video is gitignored (large); sample frames in outputs/images/"},
    }
    G.save_json(res, RES / f"video_results_{args.video.stem}.json")

    t = res["throughput"]; d = res["detection"]; idn = res["identity"]
    print(f"\n  per-frame detect+segment (SAM v1 has NO propagation)")
    print(f"  throughput        : {t['fps']:.1f} FPS   ({t['mean_ms_per_frame']:.0f} ms/frame"
          f" = GD {t['gd_ms_per_frame']['mean']:.0f} + SAM {t['sam_ms_per_frame']['mean']:.0f})")
    print(f"  detection rate    : {d['detection_rate']:.1%}  "
          f"({d['frames_with_no_detection']} frames with no box)")
    print(f"  on/off flicker    : {d['on_off_flicker_transitions']} transitions")
    print(f"  detections/frame  : mean {d['detections_per_frame']['mean']}, "
          f"max {d['detections_per_frame']['max']}")
    print(f"  top-1 centroid jump: median {idn['top1_centroid_jump_px']['median']:.0f}px, "
          f"max {idn['top1_centroid_jump_px']['max']:.0f}px  "
          f"({idn['n_consecutive_frame_hops']} hops > {idn['hop_threshold_px']:.0f}px)")
    print(f"  peak VRAM         : {mem['peak_allocated_mb']} MB")
    print(f"  video             : {outv.relative_to(G.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
