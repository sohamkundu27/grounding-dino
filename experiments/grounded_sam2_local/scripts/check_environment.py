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
import gsam2_local as G  # noqa: E402

REPO = G.REPO
OUT = REPO / "experiments/grounded_sam2_local/results/environment.json"


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
    try:
        from grounding_dino.groundingdino import _C
        gd_ext = {"compiled": True, "path": _C.__file__,
                  "has_ms_deform_attn_forward": hasattr(_C, "ms_deform_attn_forward")}
    except Exception as e:
        gd_ext = {"compiled": False, "error": str(e),
                  "impact": "GD would fall back to CPU-only; GPU timings impossible"}

    cloud = {p: ver(p) for p in ("dds-cloudapi-sdk", "dds_cloudapi_sdk")}
    cloud_installed = [k for k, v in cloud.items() if v]

    env = {
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
            "hydra_core": ver("hydra-core"),
            "timm": ver("timm"),
            "groundingdino": ver("groundingdino"),
            "SAM_2": ver("SAM-2"),
        },
        "nvcc_used_for_build": sh(str(REPO / ".venv-gsam2-cuda/bin/nvcc"), "--version"),
        "grounding_dino_cuda_extension": gd_ext,
        "checkpoints": {
            "grounding_dino": str(G.GD_CKPT.relative_to(REPO)),
            "sam2_tiny": str(G.SAM2_CKPTS["tiny"].relative_to(REPO)),
            "sam2_small": str(G.SAM2_CKPTS["small"].relative_to(REPO)),
        },
        "commits": {
            "repo": sh("git", "-C", str(REPO), "rev-parse", "HEAD"),
            "grounded_sam_2": sh("git", "-C", str(G.GSAM2), "rev-parse", "HEAD"),
        },
        "network": {
            "offline_enforced_by_adapter": G.OFFLINE_ENFORCED,
            "bert_in_local_hf_cache": G.BERT_CACHED,
            "bert_cache_dir": str(G.HF_CACHE),
            "finding": (
                "Grounding DINO's text encoder calls transformers from_pretrained("
                "'bert-base-uncased'), which REVALIDATES the cached files against "
                "huggingface.co on every load. strace -e trace=connect on an unguarded "
                "run showed real outbound :443 connections to the HF CDN (CloudFront). "
                "No imagery is uploaded -- it is a metadata fetch -- but the pipeline is "
                "NOT network-free by default."
            ),
            "mitigation": "gsam2_local sets HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1 at import, "
                          "before transformers loads. Verified: zero AF_INET connect() "
                          "syscalls, identical detections.",
            "verification_method": "strace -f -e trace=connect on a full inference run",
        },
        "local_only": {
            "cloud_sdk_installed": cloud_installed or None,
            "confirmed_no_cloud_packages": not cloud_installed,
            "entry_point": "experiments/grounded_sam2_local/scripts/* (adapter: gsam2_local.py)",
            "upstream_reference": "grounded_sam2_local_demo.py (GD 1.0 local + SAM 2 local)",
            "avoided_demos": [
                "grounded_sam2_gd1.5_demo.py", "grounded_sam2_dinox_demo.py",
                "grounded_sam2_tracking_demo_with_gd1.5.py",
                "grounded_sam2_tracking_demo_with_continuous_id_gd1.5.py",
                "grounded_sam2_tracking_demo_custom_video_input_gd1.5.py",
                "grounded_sam2_tracking_demo_custom_video_input_dinox.py",
            ],
        },
    }

    G.save_json(env, OUT)
    print(f"\nGPU        : {gpu.get('name','NONE')}  ({gpu.get('compute_capability','-')}, "
          f"{gpu.get('total_vram_mb','?')} MB)")
    print(f"torch      : {torch.__version__}  (built for CUDA {torch.version.cuda})")
    print(f"GD CUDA ext: {'COMPILED' if gd_ext['compiled'] else 'MISSING — GPU path unavailable'}")
    print(f"cloud SDK  : {'INSTALLED (!!)' if cloud_installed else 'not installed (good)'}")
    return 0 if (cuda_ok and gd_ext["compiled"] and not cloud_installed) else 1


if __name__ == "__main__":
    sys.exit(main())
