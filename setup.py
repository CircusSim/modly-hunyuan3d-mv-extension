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
import shutil
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


def _fix_venv_shared_libs():
    """
    On Linux/macOS, the venv's python binary is a symlink to the base
    interpreter and relies on a relative RPATH (typically $ORIGIN/../lib)
    to find libpython*.so at runtime. venv.EnvBuilder does NOT copy that
    shared library into the new venv, so the binary fails with
    "error while loading shared libraries" the moment you try to run it --
    it's looking for <venv>/lib/libpython3.11.so.1.0 and finding nothing.

    Fix: copy (symlink where possible) the actual .so file from the base
    interpreter's lib dir into <venv>/lib, so it's sitting exactly where
    the binary already expects it.
    """
    if platform.system() == "Windows":
        return

    base_python = Path(sys.executable).resolve()
    base_lib_dir = base_python.parent.parent / "lib"
    if not base_lib_dir.exists():
        return

    venv_lib_dir = VENV_DIR / "lib"
    venv_lib_dir.mkdir(parents=True, exist_ok=True)

    for so_file in base_lib_dir.glob("libpython*.so*"):
        target = venv_lib_dir / so_file.name
        if target.exists():
            continue
        try:
            target.symlink_to(so_file)
        except OSError:
            shutil.copy2(so_file, target)


def _create_venv():
    """
    Create the venv WITHOUT letting venv.EnvBuilder try to bootstrap pip
    itself. Some embedded/minimal Python builds (e.g. stripped-down
    python-build-standalone distributions, which is what Modly bundles)
    don't ship ensurepip, so `venv.EnvBuilder(with_pip=True)` fails with
    a nested "command not found" (exit 127) when it tries to invoke it.

    Instead: create a pip-less venv, fix up the shared-library path issue
    (see _fix_venv_shared_libs), then bootstrap pip manually via
    get-pip.py.
    """
    if VENV_DIR.exists():
        return

    venv.EnvBuilder(with_pip=False).create(str(VENV_DIR))
    _fix_venv_shared_libs()

    python = str(_venv_python())
    get_pip_path = EXTENSION_DIR / "get-pip.py"

    print("[setup] ensurepip unavailable in embedded Python -- bootstrapping pip via get-pip.py")
    import urllib.request
    urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", str(get_pip_path))

    try:
        _run([python, str(get_pip_path), "--no-warn-script-location"])
    finally:
        get_pip_path.unlink(missing_ok=True)


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

    # 1. Create venv (pip bootstrapped manually -- see _create_venv)
    _create_venv()

    pip = str(_venv_pip())

    # 2. Install PyTorch stack
    torch_index = _torch_index_for(cuda_version, is_arm64, is_linux)
    _run([pip, "install", "--timeout", "120", "--retries", "5",
          "torch", "torchvision", "--index-url", torch_index])

    # 3. Install Hunyuan3D-2 (hy3dgen) + shape/texture deps.
    #    Installed from a downloaded zip rather than `pip install git+...`
    #    so this doesn't depend on a system git binary being present/on
    #    PATH (hit on at least one Windows machine during testing).
    import urllib.request
    import zipfile
    import tempfile

    zip_url = "https://github.com/Tencent/Hunyuan3D-2/archive/refs/heads/main.zip"
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "hunyuan3d-2.zip"
        print(f"[setup] downloading {zip_url}")
        urllib.request.urlretrieve(zip_url, str(zip_path))
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        extracted = next(Path(tmp).glob("Hunyuan3D-2-*"))
        _run([pip, "install", "--timeout", "120", "--retries", "5", str(extracted)])

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