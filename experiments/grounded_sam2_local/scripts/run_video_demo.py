#!/usr/bin/env python3
"""PHASE 3 — video tracking: one-time semantic acquisition, then SAM 2 propagation.

  frame 0: text prompt -> Grounding DINO (ONCE) -> box
  frames 1..N:          -> SAM 2.1 memory-based propagation (NO re-detection)

This deliberately isolates acquisition cost from steady-state tracking cost.
Grounding DINO is NOT re-run per frame -- that is the whole point of the
architecture and of this measurement.

    python run_video_demo.py --video X.mp4 --prompt "car" --sam2 tiny
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsam2_local as G  # noqa: E402

OUT = G.REPO / "experiments/grounded_sam2_local/outputs"
RES = G.REPO / "experiments/grounded_sam2_local/results"


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
        # SAM 2's video predictor expects zero-padded JPEG frames in a directory
        cv2.imwrite(str(dst / f"{i:05d}.jpg"), fr)
        i += 1
    cap.release()
    return i, w, h, fps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=lambda x: Path(x).resolve())
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--sam2", default="tiny", choices=list(G.SAM2_CKPTS))
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--start-frame", type=int, default=0,
                    help="skip N leading frames (e.g. a fade-in) before acquisition")
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

    # ---- 1. one-time semantic acquisition on frame 0 -------------------------
    gd = G.load_grounding_dino()
    f0 = frames_dir / "00000.jpg"
    boxes, scores, phrases, gd_ms, _ = G.gd_predict(gd, f0, args.prompt,
                                                    args.box_thr, args.text_thr)
    if boxes is None or not len(boxes):
        sys.exit(f"Grounding DINO found nothing for '{args.prompt}' on frame 0 — "
                 f"cannot initialise tracking.")
    top = int(np.argmax(scores))
    init_box = boxes[top]
    print(f"acquisition: '{args.prompt}' -> {len(boxes)} box(es), "
          f"top score {scores[top]:.2f}, GD {gd_ms:.0f} ms")
    del gd
    torch.cuda.empty_cache()

    # ---- 2. SAM 2 init -------------------------------------------------------
    predictor = G.load_sam2_video(args.sam2)
    t = []
    with G.cuda_timer(t):
        state = predictor.init_state(video_path=str(frames_dir))
        _, _, _ = predictor.add_new_points_or_box(
            inference_state=state, frame_idx=0, obj_id=1, box=init_box
        )
    init_ms = t[0]
    print(f"SAM 2 init: {init_ms:.0f} ms")

    # ---- 3. propagate (no re-detection) -------------------------------------
    per_frame_ms, masks_by_frame = [], {}
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    prev = t_start
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for fidx, obj_ids, logits in predictor.propagate_in_video(state):
            torch.cuda.synchronize()
            now = time.perf_counter()
            per_frame_ms.append((now - prev) * 1000.0)
            prev = now
            masks_by_frame[fidx] = (logits[0] > 0.0).cpu().numpy()[0]
    torch.cuda.synchronize()
    wall = time.perf_counter() - t_start

    G.assert_no_cloud_imports()

    tracked = len(masks_by_frame)
    order = sorted(masks_by_frame)
    areas = np.array([int(masks_by_frame[k].sum()) for k in order])
    fps_track = tracked / wall if wall else 0.0

    # Per-frame mask geometry. Area alone cannot tell these three apart:
    #   (a) the tracker flickered off and back onto the SAME object,
    #   (b) the object left the frame and never returned,
    #   (c) the object left the frame and SAM 2 later re-bound the object id to a
    #       DIFFERENT object (silent identity switch).
    # Centroid + which image border the mask was touching when it vanished do.
    cents, bboxes, touched = [], [], []
    for k in order:
        m = masks_by_frame[k]
        ys, xs = np.where(m)
        if not len(xs):
            cents.append(None); bboxes.append(None); touched.append(None); continue
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        mh, mw = m.shape[:2]
        edge = 2  # px tolerance
        cents.append((float(xs.mean()), float(ys.mean())))
        bboxes.append([x0, y0, x1, y1])
        touched.append({"left": x0 <= edge, "top": y0 <= edge,
                        "right": x1 >= mw - 1 - edge, "bottom": y1 >= mh - 1 - edge})

    # contiguous runs of visibility
    runs, cur = [], None
    for i, a in enumerate(areas):
        if a > 0 and cur is None:
            cur = i
        elif a == 0 and cur is not None:
            runs.append([cur, i - 1]); cur = None
    if cur is not None:
        runs.append([cur, tracked - 1])

    lost = int((areas == 0).sum())
    nz = np.nonzero(areas)[0]
    last_visible = int(nz[-1]) if len(nz) else -1

    # Analyse each gap between visible runs.
    gaps = []
    for a, b in zip(runs, runs[1:]):
        vanish_at, reappear_at = a[1], b[0]
        left_via = [s for s, on in (touched[vanish_at] or {}).items() if on]
        ca, cb = cents[vanish_at], cents[reappear_at]
        dist = float(np.hypot(cb[0] - ca[0], cb[1] - ca[1]))
        shrinking = bool(areas[vanish_at] < areas[max(vanish_at - 3, 0)])
        # Egress signature: last mask before the gap was shrinking AND clipped by a
        # border => the object walked/drove out of frame. It cannot legitimately
        # "come back" as the same instance if it exited and the camera is static.
        egress = bool(left_via) and shrinking
        gaps.append({
            "vanished_after_frame": vanish_at,
            "reappeared_at_frame": reappear_at,
            "gap_frames": reappear_at - vanish_at - 1,
            "mask_touching_border_when_vanished": left_via or None,
            "mask_area_before_vanishing": [int(x) for x in areas[max(vanish_at - 3, 0):vanish_at + 1]],
            "area_was_shrinking": shrinking,
            "classified_as_target_egress": egress,
            "centroid_at_vanish": [round(v, 1) for v in ca],
            "centroid_at_reappear": [round(v, 1) for v in cb],
            "centroid_jump_px": round(dist, 1),
            "identity_switch_suspected": bool(egress),
            "why": ("target exited the frame (mask shrinking and clipped by border), so the "
                    "re-appearing mask CANNOT be the same instance -- SAM 2 silently re-bound "
                    "the object id to a different object" if egress else
                    "mask vanished mid-frame (not clipped/shrinking at a border) and returned: "
                    "genuine momentary dropout on the same target"),
        })

    egress_gaps = [g for g in gaps if g["classified_as_target_egress"]]
    true_dropouts = sum(g["gap_frames"] for g in gaps if not g["classified_as_target_egress"])
    id_switches = len(egress_gaps)

    # ---- 4. annotated output video ------------------------------------------
    name = args.out_name or f"track_{args.video.stem}_{args.sam2}"
    outv = OUT / "videos" / f"{name}.mp4"
    vw = cv2.VideoWriter(str(outv), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    colour = np.array([30, 220, 60], dtype=np.uint8)
    for fidx in sorted(masks_by_frame):
        fr = cv2.imread(str(frames_dir / f"{fidx:05d}.jpg"))
        m = masks_by_frame[fidx]
        if m.shape[:2] != fr.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        fr[m] = (0.55 * fr[m] + 0.45 * colour).astype(np.uint8)
        ys, xs = np.where(m)
        if len(xs):
            cv2.rectangle(fr, (xs.min(), ys.min()), (xs.max(), ys.max()), (30, 220, 60), 2)
        cv2.putText(fr, f"{args.prompt} | f{fidx} | {'LOST' if not m.any() else 'tracking'}",
                    (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        vw.write(fr)
    vw.release()

    # sample frames as PNG: frame 0, midpoint, last, plus both sides of every gap
    keys = {0, tracked // 2, tracked - 1}
    for g in gaps:
        keys |= {g["vanished_after_frame"], g["vanished_after_frame"] + 1,
                 g["reappeared_at_frame"]}
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
        "frames_tracked": tracked,
        "prompt": args.prompt,
        "config": {"gd": "Swin-T", "sam2": args.sam2},
        "acquisition": {
            "gd_latency_ms": round(gd_ms, 1),
            "n_boxes": len(boxes),
            "top_score": round(float(scores[top]), 4),
            "init_box_xyxy": [round(float(v), 1) for v in init_box],
            "detector_run_once": True,
            "redetection_used": False,
        },
        "sam2_init_ms": round(init_ms, 1),
        "tracking": {
            "wall_seconds": round(wall, 3),
            "tracking_fps": round(fps_track, 2),
            "mean_per_frame_ms": round(float(np.mean(per_frame_ms)), 1),
            "median_per_frame_ms": round(float(np.median(per_frame_ms)), 1),
            "p95_per_frame_ms": round(float(np.percentile(per_frame_ms, 95)), 1),
            "empty_mask_frames_total": lost,
            "last_visible_frame": last_visible,
            "visible_runs": runs,
            "gaps": gaps,
            "target_egress_events": len(egress_gaps),
            "identity_switches_suspected": id_switches,
            "dropout_frames_on_same_target": true_dropouts,
            "metric_note": (
                "An empty mask is not automatically a tracking failure and a re-appearing "
                "mask is not automatically a recovery. Each gap is classified by whether the "
                "mask was shrinking and clipped by an image border when it vanished (= the "
                "target left the frame). If it was, a later mask under the same object id is "
                "a DIFFERENT object: SAM 2 has no terminal 'object is gone' state and no "
                "identity check, and this pipeline never re-runs the detector to correct it."
            ),
            "per_frame_mask_area_px": [int(a) for a in areas],
            "per_frame_mask_bbox_xyxy": bboxes,
            "mask_area_first": int(areas[0]) if len(areas) else 0,
            "mask_area_last": int(areas[-1]) if len(areas) else 0,
            "mask_area_min": int(areas.min()) if len(areas) else 0,
            "mask_area_cv": round(float(areas.std() / areas.mean()), 3) if areas.mean() else None,
        },
        "gpu_memory": mem,
        "outputs": {"video": str(outv.relative_to(G.REPO)),
                    "note": "video is gitignored (large); sample frames in outputs/images/"},
    }
    G.save_json(res, RES / f"video_results_{args.video.stem}.json")

    print(f"\n  tracking FPS      : {fps_track:.1f}   ({np.mean(per_frame_ms):.1f} ms/frame)")
    print(f"  frames tracked    : {tracked}")
    print(f"  visible runs      : {runs}")
    print(f"  empty-mask frames : {lost} ({lost/tracked:.1%})")
    print(f"  dropouts on SAME target : {true_dropouts} frames")
    print(f"  target-egress events    : {len(egress_gaps)}")
    print(f"  IDENTITY SWITCHES (susp): {id_switches}")
    for g in gaps:
        print(f"    gap f{g['vanished_after_frame']}->f{g['reappeared_at_frame']} "
              f"({g['gap_frames']} empty)  border={g['mask_touching_border_when_vanished']} "
              f"shrinking={g['area_was_shrinking']}  centroid jump {g['centroid_jump_px']:.0f}px "
              f"=> {'IDENTITY SWITCH' if g['identity_switch_suspected'] else 'same-target dropout'}")
    print(f"  peak VRAM         : {mem['peak_allocated_mb']} MB")
    print(f"  video             : {outv.relative_to(G.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
