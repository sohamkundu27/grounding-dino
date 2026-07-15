#!/usr/bin/env python3
"""PHASE 4 — performance benchmark of the fully-local Grounded SAM (v1) pipeline.

  config A: Grounding DINO Swin-T + SAM v1 ViT-B
  config B: Grounding DINO Swin-T + SAM v1 ViT-H

Protocol (per config):
  * model load timed from cold
  * FIRST inference timed separately -- it includes lazy CUDA kernel/cuDNN
    autotuning and is not representative of steady state
  * 3 warm-up iterations, discarded
  * >=10 measured image iterations -> mean/median/p95 per stage
  * >=100 frames of PER-FRAME detect+segment for the "video" number. SAM v1 has no
    propagation, so the video cost is just the image pipeline run on every frame --
    there is no cheap steady state to measure, and that is the whole point.

Every GPU region is wrapped in torch.cuda.synchronize() on both sides (G.cuda_timer).
Each config runs in its OWN process so config B does not inherit config A's warm CUDA
context / page cache, which would understate its cold model-load time.

    python benchmark_pipeline.py
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsam_local as G  # noqa: E402

OUT = G.REPO / "experiments/grounded_sam_local/outputs"
RES = G.REPO / "experiments/grounded_sam_local/results"

WARMUPS = 3
MEASURED = 10
TRACK_FRAMES = 120  # >= 100 required


def cpu_rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return round(int(line.split()[1]) / 1024, 1)
    return -1.0


def pct(xs, p):
    return round(float(np.percentile(xs, p)), 1)


def stats(xs) -> dict:
    return {"mean": round(float(np.mean(xs)), 1), "median": round(float(np.median(xs)), 1),
            "p95": pct(xs, 95), "min": round(float(min(xs)), 1), "max": round(float(max(xs)), 1)}


def bench_config(sam_size: str, image: Path, prompt: str, frames_dir: Path,
                 n_frames: int, track_prompt: str) -> dict:
    """One full config. Returns a record; never raises -- failures are recorded."""
    import cv2
    import torch

    rec: dict = {"config": f"GD-SwinT + SAM-v1-{sam_size}", "sam_size": sam_size,
                 "sam_params_m": G.SAM_PARAMS_M.get(sam_size), "success": False, "error": None}

    try:
        G.reset_gpu_mem()
        gc.collect()
        torch.cuda.empty_cache()
        rss0 = cpu_rss_mb()

        # ---- model load (cold) ----------------------------------------------
        t = []
        with G.cuda_timer(t):
            gd = G.load_grounding_dino()
        rec["gd_load_ms"] = round(t[0], 1)
        t = []
        with G.cuda_timer(t):
            sam = G.load_sam_image(sam_size)
        rec["sam_load_ms"] = round(t[0], 1)
        rec["model_load_total_ms"] = round(rec["gd_load_ms"] + rec["sam_load_ms"], 1)

        img_bgr = cv2.imread(str(image))
        if img_bgr is None:
            raise RuntimeError(f"cannot read {image}")
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rec["image_resolution"] = f"{w}x{h}"
        rec["image_prompt"] = prompt

        # ---- FIRST inference (cold) -- reported separately -------------------
        boxes, scores, _, gd_ms, _ = G.gd_predict(gd, image, prompt)
        if boxes is None or not len(boxes):
            raise RuntimeError(f"no detection for '{prompt}' — cannot benchmark SAM stage")
        _, sam_ms, _ = G.sam_masks(sam, img_rgb, boxes, split=True)
        rec["first_inference_gd_ms"] = round(gd_ms, 1)
        rec["first_inference_sam_ms"] = round(sam_ms, 1)
        rec["first_inference_total_ms"] = round(gd_ms + sam_ms, 1)

        # ---- warm-ups (discarded) -------------------------------------------
        for _ in range(WARMUPS):
            b, _, _, _, _ = G.gd_predict(gd, image, prompt)
            G.sam_masks(sam, img_rgb, b)

        # ---- measured image iterations ---------------------------------------
        gd_t, sam_t, tot_t, enc_t, dec_t = [], [], [], [], []
        for _ in range(MEASURED):
            b, _, _, g_ms, _ = G.gd_predict(gd, image, prompt)
            _, s_ms, br = G.sam_masks(sam, img_rgb, b, split=True)
            gd_t.append(g_ms); sam_t.append(s_ms); tot_t.append(g_ms + s_ms)
            enc_t.append(br["encode_ms"]); dec_t.append(br["decode_ms"])

        rec["n_detections"] = int(len(boxes))
        rec["measured_iters"] = MEASURED
        rec["warmups"] = WARMUPS
        for k, v in (("gd", gd_t), ("sam", sam_t), ("total", tot_t),
                     ("sam_encode", enc_t), ("sam_decode", dec_t)):
            s = stats(v)
            rec[f"warm_{k}_mean_ms"] = s["mean"]
            rec[f"warm_{k}_median_ms"] = s["median"]
            rec[f"warm_{k}_p95_ms"] = s["p95"]
        rec["warm_total_fps"] = round(1000.0 / rec["warm_total_mean_ms"], 2)
        rec["image_peak_vram_mb"] = G.gpu_mem_mb()["peak_allocated_mb"]

        # ---- video: PER-FRAME detect + segment (no propagation exists) --------
        # keep the models loaded; this is the real cost of "tracking" with SAM v1.
        G.reset_gpu_mem()
        frames = sorted(frames_dir.glob("*.jpg"))[:n_frames]
        per_frame_ms, det_frames = [], 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prev = t0
        for fp in frames:
            b, s, _, _, _ = G.gd_predict(gd, fp, track_prompt)
            if b is not None and len(b):
                det_frames += 1
                top = int(np.argmax(s))
                rgb = cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB)
                G.sam_masks(sam, rgb, np.array([[float(v) for v in b[top]]]))
            torch.cuda.synchronize()
            now = time.perf_counter()
            per_frame_ms.append((now - prev) * 1000.0)
            prev = now
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        rec["track_prompt"] = track_prompt
        rec["track_frames"] = len(frames)
        rec["track_detection_rate"] = round(det_frames / len(frames), 4) if frames else 0.0
        rec["track_mean_ms_per_frame"] = round(float(np.mean(per_frame_ms)), 1)
        rec["track_p95_ms_per_frame"] = pct(per_frame_ms, 95)
        rec["track_fps"] = round(len(frames) / wall, 2) if wall else 0.0
        rec["track_peak_vram_mb"] = G.gpu_mem_mb()["peak_allocated_mb"]
        rec["peak_cpu_rss_mb"] = cpu_rss_mb()
        rec["cpu_rss_delta_mb"] = round(cpu_rss_mb() - rss0, 1)

        G.assert_no_cloud_imports()
        rec["success"] = True

        del gd, sam
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:  # record, do not mask
        rec["error"] = f"{type(e).__name__}: {e}"
        print(f"  !! {sam_size} FAILED: {rec['error']}")

    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path,
                    default=G.REPO / "datasets/refdrone/images/all_image/0000001_02999_d_0000005.jpg")
    ap.add_argument("--prompt", default="car")
    ap.add_argument("--video", type=lambda x: Path(x).resolve(),
                    default=G.GSAM2 / "assets/zebra.mp4")
    ap.add_argument("--track-prompt", default="zebra")
    ap.add_argument("--track-start", type=int, default=20)
    ap.add_argument("--track-frames", type=int, default=TRACK_FRAMES)
    ap.add_argument("--only", choices=("vit_b", "vit_h"),
                    help="benchmark a single config in its OWN process (used internally)")
    ap.add_argument("--frames-dir", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()

    import torch

    G.device()
    print(f"GPU: {torch.cuda.get_device_name(0)}   torch {torch.__version__}")

    # ---- child mode: benchmark exactly one config, emit its JSON, exit -------
    if args.only:
        import cv2
        cap = cv2.VideoCapture(str(args.video))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        n = len(list(args.frames_dir.glob("*.jpg")))
        r = bench_config(args.only, args.image, args.prompt, args.frames_dir, n,
                         args.track_prompt)
        r["track_resolution"] = f"{W}x{H}"
        r["gpu"] = torch.cuda.get_device_name(0)
        G.save_json(r, RES / f"_bench_{args.only}.json")
        return 0 if r["success"] else 1

    # ---- parent mode ---------------------------------------------------------
    from run_video_demo import extract_frames
    frames_dir = OUT / "videos" / f"bench_frames_{args.video.stem}"
    n, W, H, fps = extract_frames(args.video, frames_dir, args.track_frames, args.track_start)
    print(f"video input: {args.video.name} {W}x{H}, {n} frames "
          f"(from frame {args.track_start})\n")

    import json as _json
    import shutil
    import subprocess
    rows = []
    for i, size in enumerate(("vit_b", "vit_h")):
        print(f"--- config {'AB'[i]}: GD Swin-T + SAM v1 {size}  (fresh process) ---")
        cmd = [sys.executable, __file__, "--only", size, "--frames-dir", str(frames_dir),
               "--image", str(args.image), "--prompt", args.prompt,
               "--video", str(args.video), "--track-prompt", args.track_prompt]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        out = RES / f"_bench_{size}.json"
        if not out.exists():
            rows.append({"config": f"GD-SwinT + SAM-v1-{size}", "sam_size": size,
                         "success": False,
                         "error": (cp.stderr.strip().splitlines() or ["no output"])[-1]})
            print(f"  !! {size} produced no result\n{cp.stderr[-800:]}")
            continue
        r = _json.loads(out.read_text())
        out.unlink()
        rows.append(r)
        if r["success"]:
            print(f"  load {r['model_load_total_ms']:.0f} ms | first-inf "
                  f"{r['first_inference_total_ms']:.0f} ms | warm "
                  f"{r['warm_total_mean_ms']:.0f} ms ({r['warm_total_fps']:.1f} FPS)"
                  f"  [GD {r['warm_gd_mean_ms']:.0f} + SAM {r['warm_sam_mean_ms']:.0f}"
                  f" = enc {r['warm_sam_encode_mean_ms']:.0f} + dec"
                  f" {r['warm_sam_decode_mean_ms']:.0f}] | per-frame video {r['track_fps']:.1f} FPS"
                  f" | VRAM {r['track_peak_vram_mb']:.0f} MB\n")

    shutil.rmtree(frames_dir, ignore_errors=True)

    cols = sorted({k for r in rows for k in r})
    lead = ["config", "sam_size", "sam_params_m", "success", "model_load_total_ms",
            "first_inference_total_ms", "warm_total_mean_ms", "warm_total_fps",
            "warm_gd_mean_ms", "warm_sam_mean_ms", "track_fps", "track_peak_vram_mb"]
    cols = lead + [c for c in cols if c not in lead]
    csv_path = RES / "benchmark.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    G.save_json({
        "protocol": {
            "warmups_discarded": WARMUPS, "measured_image_iters": MEASURED,
            "video_frames": n,
            "timing": "torch.cuda.synchronize() on both sides of every timed GPU region",
            "first_inference": "reported separately; includes lazy CUDA/cuDNN init and is "
                               "NOT averaged into the warm numbers",
            "video_protocol": "PER-FRAME detect+segment. SAM v1 has no propagation, so the "
                              "detector runs on every frame -- there is no cheap steady state. "
                              "Directly contrast this with the SAM 2 pipeline, which pays the "
                              "detector once and then propagates.",
            "process_isolation": "each config runs in a FRESH process, so cold model-load time "
                                 "is not contaminated by the previous config's warm CUDA "
                                 "context and page cache",
            "sam_image_latency": "end-to-end = ViT image encode (set_image) + mask decode "
                                 "(predict_torch). SAM v1 runs at standard fp32.",
        },
        "image_input": {"path": str(args.image), "prompt": args.prompt},
        "video_input": {"path": str(args.video.relative_to(G.REPO)),
                        "prompt": args.track_prompt, "start_frame": args.track_start,
                        "frames": n, "resolution": [W, H]},
        "results": rows,
    }, RES / "benchmark.json")
    return 0 if all(r["success"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
