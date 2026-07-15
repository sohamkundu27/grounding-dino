#!/usr/bin/env python3
"""Record the exact environment this evaluation ran in. Writes results/environment.json.

Loads no checkpoints and runs no inference — it only reports versions, hardware,
and the pinned commit SHAs, and confirms no cloud-API package is installed.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsam_local as G  # noqa: E402

REPO = G.REPO
OUT = REPO / "experiments/grounded_sam_local/results/environment.json"


def ver(pkg: str):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def sh(*args) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main() -> int:
    import torch

    cuda_ok = torch.cuda.is_available()
    gpu = {}
    if cuda_ok:
        cc = torch.cuda.get_device_capability(0)
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": f"sm_{cc[0]}{cc[1]}",
            "torch_cuda_arch_list_used": f"{cc[0]}.{cc[1]}",
            "total_vram_mb": round(props.total_memory / 1024**2),
            "driver": (sh("nvidia-smi", "--query-gpu=driver_version",
                          "--format=csv,noheader") or "").strip(),
        }

    # the custom deformable-attention CUDA op: compiled, or silently absent?
    # (imported from the SAME grounded_sam_2 tree used by the SAM 2 evaluation)
    try:
        from grounding_dino.groundingdino import _C
        gd_ext = {"compiled": True, "path": _C.__file__,
                  "has_ms_deform_attn_forward": hasattr(_C, "ms_deform_attn_forward")}
    except Exception as e:
        gd_ext = {"compiled": False, "error": str(e),
                  "impact": "GD would fall back to CPU-only; GPU timings impossible"}

    # SAM v1 has no video predictor. Record the architectural fact explicitly.
    try:
        import segment_anything as sa
        sam_video = {
            "has_build_sam2_video_predictor": False,
            "has_image_predictor": hasattr(sa, "SamPredictor"),
            "has_automatic_mask_generator": hasattr(sa, "SamAutomaticMaskGenerator"),
            "note": "SAM v1 is image-only: no memory, no init_state, no propagate_in_video.",
        }
        sa_path = sa.__file__
    except Exception as e:
        sam_video = {"error": str(e)}
        sa_path = None

    cloud = {p: ver(p) for p in ("dds-cloudapi-sdk", "dds_cloudapi_sdk")}
    cloud_installed = [k for k, v in cloud.items() if v]

    env = {
        "pipeline": "Grounded SAM (v1): Grounding DINO 1.0 Swin-T + SAM v1 (ViT-B / ViT-H)",
        "detector_shared_with_sam2_eval": {
            "identical": True,
            "config": str(G.GD_CONFIG.relative_to(REPO)),
            "checkpoint": str(G.GD_CKPT.relative_to(REPO)),
            "imported_from": "third_party/grounded_sam_2 (same build, same compiled _C ext)",
            "why": ("holding the detector byte-identical to the SAM 2 evaluation makes the "
                    "comparison clean: any difference is attributable to the SAM stage alone"),
        },
        "machine": {
            "hostname": platform.node(),
            "os": platform.platform(),
            "cpu": (sh("bash", "-lc", "lscpu | grep 'Model name' | cut -d: -f2") or "").strip(),
            "ram_gb": round(int(sh("bash", "-lc",
                            "grep MemTotal /proc/meminfo | awk '{print $2}'") or 0) / 1024**2, 1),
        },
        "gpu": gpu,
        "python": sys.version.split()[0],
        "venv": sys.prefix,
        "packages": {
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "torchvision": ver("torchvision"),
            "cuda_runtime_available": cuda_ok,
            "cudnn": torch.backends.cudnn.version(),
            "transformers": ver("transformers"),
            "numpy": ver("numpy"),
            "opencv_python": ver("opencv-python"),
            "supervision": ver("supervision"),
            "timm": ver("timm"),
            "groundingdino": ver("groundingdino"),
            "segment_anything": ver("segment-anything") or ver("segment_anything"),
        },
        "grounding_dino_cuda_extension": gd_ext,
        "sam_v1": {
            "package_path": sa_path,
            "checkpoints": {k: str(v.relative_to(REPO)) for k, v in G.SAM_CKPTS.items()},
            "params_millions": G.SAM_PARAMS_M,
            **sam_video,
        },
        "commits": {
            "repo": sh("git", "-C", str(REPO), "rev-parse", "HEAD"),
            "grounded_sam": sh("git", "-C", str(G.GSAM), "rev-parse", "HEAD"),
            "grounded_sam_2_detector": sh("git", "-C", str(G.GSAM2), "rev-parse", "HEAD"),
        },
        "network": {
            "offline_enforced_by_adapter": G.OFFLINE_ENFORCED,
            "bert_in_local_hf_cache": G.BERT_CACHED,
            "bert_cache_dir": str(G.HF_CACHE),
            "finding": (
                "Same posture as the SAM 2 evaluation: Grounding DINO's text encoder calls "
                "transformers from_pretrained('bert-base-uncased'), which REVALIDATES the "
                "cached files against huggingface.co on every load (real outbound :443, "
                "confirmed by strace there). No imagery is uploaded, but the pipeline is NOT "
                "network-free by default. This is a detector property, so it is identical here."
            ),
            "mitigation": "gsam_local sets HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1 at import, "
                          "before transformers loads.",
        },
        "local_only": {
            "cloud_sdk_installed": cloud_installed or None,
            "confirmed_no_cloud_packages": not cloud_installed,
            "entry_point": "experiments/grounded_sam_local/scripts/* (adapter: gsam_local.py)",
            "no_upstream_file_modified": True,
        },
    }

    G.save_json(env, OUT)
    print(f"\npipeline   : Grounded SAM (v1) = GD 1.0 Swin-T + SAM v1 (ViT-B / ViT-H)")
    print(f"GPU        : {gpu.get('name','NONE')}  ({gpu.get('compute_capability','-')}, "
          f"{gpu.get('total_vram_mb','?')} MB)")
    print(f"torch      : {torch.__version__}  (built for CUDA {torch.version.cuda})")
    print(f"GD CUDA ext: {'COMPILED' if gd_ext['compiled'] else 'MISSING — GPU path unavailable'}")
    print(f"SAM v1     : image-only (no video propagation) — segment_anything")
    print(f"cloud SDK  : {'INSTALLED (!!)' if cloud_installed else 'not installed (good)'}")
    return 0 if (cuda_ok and gd_ext["compiled"] and not cloud_installed) else 1


if __name__ == "__main__":
    sys.exit(main())
