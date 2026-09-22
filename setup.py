"""
Setup script for the Hunyuan3D-2mv Modly extension.

Mirrors the platform-detection pattern documented for
modly-hunyuan3d-mini-extension: Modly passes platform info (gpu_sm,
cuda_version, os/arch) into this script at install time, which then
creates an isolated venv and installs the right PyTorch build plus the
hy3dgen package needed to load tencent/Hunyuan3D-2mv.

Reconstructed from the mini extension's documented behavior; adjust the
exact CLI args / hook names to whatever Modly's installer actually calls
if they differ (e.g. some setups expect a `main(platform_info)` entrypoint
instead of argv parsing).
"""

import json
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path

EXTENSION_DIR = Path(__file__).parent.resolve()
VENV_DIR = EXTENSION_DIR / "venv"


def _venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _venv_pip() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


def _run(cmd, **kwargs):
    print(f"[setup] $ {' '.join(map(str, cmd))}")
    subprocess.run(cmd, check=True, **kwargs)


def _torch_index_for(cuda_version: int | None, is_arm64: bool, is_linux: bool) -> str:
    # CUDA 12.8+ path (mirrors the ARM64 handling called out in the mini
    # extension's README); falls back to a generic recent CUDA wheel index
    # otherwise, or CPU if no GPU was detected at all.
    if cuda_version is None:
        return "https://download.pytorch.org/whl/cpu"
    if cuda_version >= 128:
        return "https://download.pytorch.org/whl/cu128"
    return "https://download.pytorch.org/whl/cu121"


def setup(platform_info: dict):
    """
    platform_info is expected to look like:
        {
          "gpu_sm": "86",
          "cuda_version": 128,
          "os": "windows" | "linux" | "darwin",
          "arch": "x86_64" | "aarch64" | "arm64",
        }
    """
    gpu_sm = platform_info.get("gpu_sm")
    cuda_version = platform_info.get("cuda_version")
    os_name = platform_info.get("os", platform.system().lower())
    arch = platform_info.get("arch", platform.machine().lower())

    is_linux = os_name == "linux"
    is_arm64 = arch in ("aarch64", "arm64")

    print(f"[setup] gpu_sm={gpu_sm} cuda_version={cuda_version} os={os_name} arch={arch}")

    # 1. Create venv
    if not VENV_DIR.exists():
        venv.EnvBuilder(with_pip=True).create(str(VENV_DIR))

    pip = str(_venv_pip())

    # 2. Install PyTorch stack
    torch_index = _torch_index_for(cuda_version, is_arm64, is_linux)
    _run([pip, "install", "--timeout", "120", "--retries", "5",
          "torch", "torchvision", "--index-url", torch_index])

    # 3. Install Hunyuan3D-2 (hy3dgen) + shape/texture deps
    _run([pip, "install", "--timeout", "120", "--retries", "5",
          "git+https://github.com/Tencent/Hunyuan3D-2.git"])

    # 4. Background removal -- GPU build where available, ARM64 falls back
    #    to CPU onnxruntime per the mini extension's platform notes.
    if is_linux and is_arm64:
        _run([pip, "install", "rembg", "onnxruntime"])
    else:
        _run([pip, "install", "rembg[gpu]"])

    # 5. Pre-fetch model weights (multiview variant) so first generation
    #    isn't slowed by a cold download.
    _run([pip, "install", "huggingface_hub"])
    _run([str(_venv_python()), "-c",
          "from huggingface_hub import snapshot_download; "
          "snapshot_download('tencent/Hunyuan3D-2mv')"])

    print("[setup] hunyuan3d-mv extension environment ready")


if __name__ == "__main__":
    # Allow Modly to call this as `python setup.py '<json platform info>'`
    if len(sys.argv) > 1:
        info = json.loads(sys.argv[1])
    else:
        info = {}
    setup(info)
