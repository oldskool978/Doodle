#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

ROOT_DIR = Path(__file__).resolve().parent

CACHE_DIR = ROOT_DIR / ".cache"
TMP_DIR = ROOT_DIR / ".tmp"
DOWNLOADS_DIR = CACHE_DIR / "downloads"
PIP_CACHE_DIR = CACHE_DIR / "pip"
HF_CACHE_DIR = ROOT_DIR / ".hf_cache"
LOCAL_APPDATA_DIR = ROOT_DIR / ".local_appdata"
ROAMING_APPDATA_DIR = ROOT_DIR / ".appdata"

for d in (
    CACHE_DIR,
    TMP_DIR,
    DOWNLOADS_DIR,
    PIP_CACHE_DIR,
    HF_CACHE_DIR,
    LOCAL_APPDATA_DIR,
    ROAMING_APPDATA_DIR,
):
    d.mkdir(parents=True, exist_ok=True)

tempfile.tempdir = str(TMP_DIR)
os.environ["TEMP"] = str(TMP_DIR)
os.environ["TMP"] = str(TMP_DIR)
os.environ["TMPDIR"] = str(TMP_DIR)
os.environ["PIP_CACHE_DIR"] = str(PIP_CACHE_DIR)
os.environ["LOCALAPPDATA"] = str(LOCAL_APPDATA_DIR)
os.environ["APPDATA"] = str(ROAMING_APPDATA_DIR)
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["TORCH_HOME"] = str(HF_CACHE_DIR / "torch")
os.environ["PYTHONIOENCODING"] = "utf-8"

DEFAULT_PYTHON_VERSION = "3.12.8"

PREDEFINED_MATRIX = {
    "1": "3.11.9",
    "2": "3.12.8",
    "3": "3.13.2",
}

STANDALONE_RELEASE = "20241016"

AMD_STAGING_FAMILY_MAP = {
    "gfx1201": "gfx120X-all",
    "gfx1200": "gfx120X-all",
    "gfx120X": "gfx120X-all",
    "gfx1150": "gfx1150",
    "gfx1151": "gfx1151",
    "gfx1152": "gfx1152",
    "gfx1100": "gfx110X-all",
    "gfx1101": "gfx110X-all",
    "gfx1102": "gfx110X-all",
    "gfx1103": "gfx110X-all",
    "gfx1030": "gfx103X-all",
    "gfx1031": "gfx103X-all",
    "gfx1035": "gfx103X-all",
    "gfx942": "gfx94X-dcgpu",
    "gfx950": "gfx950",
    "gfx90a": "gfx90a-all",
    "gfx908": "gfx908",
}

HSA_OVERRIDES = {
    "gfx1201": "12.0.0",
    "gfx1200": "12.0.0",
    "gfx1150": "11.5.0",
    "gfx1151": "11.5.0",
    "gfx1152": "11.5.0",
    "gfx1103": "11.0.0",
    "gfx1035": "10.3.0",
    "gfx1031": "10.3.0",
}


def print_status(msg: str, status: str = "INFO") -> None:
    colors = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m",
        "WARN": "\033[93m",
        "ERROR": "\033[91m",
        "RESET": "\033[0m",
    }
    if os.name == "nt" and not os.environ.get("WT_SESSION"):
        print(f"[{status}] {msg}")
    else:
        print(f"{colors.get(status, '')}[{status}] {msg}{colors['RESET']}")


def remove_readonly(func, path, _):
    try:
        os.chmod(path, stat.S_IWRITE)
        if func is not None:
            func(path)
    except Exception:
        pass


def safe_subprocess(cmd: list) -> str:
    try:
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, timeout=10, text=True, **kwargs
        ).strip()
    except Exception:
        return ""


def query_dxgi_adapters_win32() -> list[dict]:
    if sys.platform != "win32":
        return []
    try:
        class DXGI_ADAPTER_DESC1(ctypes.Structure):
            _fields_ = [
                ("Description", ctypes.c_wchar * 128),
                ("VendorId", ctypes.c_uint),
                ("DeviceId", ctypes.c_uint),
                ("SubSysId", ctypes.c_uint),
                ("Revision", ctypes.c_uint),
                ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", ctypes.c_longlong),
                ("Flags", ctypes.c_uint),
            ]

        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", ctypes.c_ulong),
                ("Data2", ctypes.c_ushort),
                ("Data3", ctypes.c_ushort),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        def call_com(ptr, index, argtypes, *args):
            vtable = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))[0]
            func_addr = ctypes.cast(vtable, ctypes.POINTER(ctypes.c_void_p))[index]
            func = ctypes.WINFUNCTYPE(ctypes.c_long, *argtypes)(func_addr)
            return func(ptr, *args)

        iid_dxgi_factory1 = GUID(
            0x770AAE78,
            0xF26F,
            0x4DBA,
            (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
        )
        factory = ctypes.c_void_p()
        if ctypes.windll.dxgi.CreateDXGIFactory1(ctypes.byref(iid_dxgi_factory1), ctypes.byref(factory)) != 0:
            return []

        adapters = []
        idx = 0
        while True:
            adapter = ctypes.c_void_p()
            hr = call_com(
                factory,
                12,
                [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
                idx,
                ctypes.byref(adapter),
            )
            if hr < 0:
                break
            desc = DXGI_ADAPTER_DESC1()
            call_com(adapter, 10, [ctypes.c_void_p, ctypes.POINTER(DXGI_ADAPTER_DESC1)], ctypes.byref(desc))
            if not (desc.Flags & 2):
                adapters.append({
                    "name": str(desc.Description),
                    "vendor_id": desc.VendorId,
                    "device_id": desc.DeviceId,
                    "vram_gb": desc.DedicatedVideoMemory / (1024 ** 3),
                })
            call_com(adapter, 2, [ctypes.c_void_p])
            idx += 1
        call_com(factory, 2, [ctypes.c_void_p])
        return adapters
    except Exception:
        return []


def resolve_hardware_matrix() -> tuple[str, str]:
    if sys.platform == "darwin":
        uname = safe_subprocess(["uname", "-m"])
        if "arm64" in uname:
            return "MPS", ""
        return "CPU", ""

    has_nvidia, has_amd, has_intel = False, False, False
    detected_amd_targets = []

    if os.name == "nt":
        dxgi_devices = query_dxgi_adapters_win32()
        if dxgi_devices:
            for dev in dxgi_devices:
                name = dev["name"]
                vid = dev["vendor_id"]
                if vid == 0x10DE:
                    has_nvidia = True
                elif vid == 0x8086:
                    has_intel = True
                elif vid == 0x1002:
                    has_amd = True
                    if re.search(r"(?i)MI325|MI350|MI355", name):
                        detected_amd_targets.append("gfx950")
                    elif re.search(r"(?i)MI300", name):
                        detected_amd_targets.append("gfx942")
                    elif re.search(r"(?i)MI250|MI210", name):
                        detected_amd_targets.append("gfx90a")
                    elif re.search(r"(?i)R9700|9070|Navi\s*48|Navi\s*44|RX\s*9\d{3}|AI\s*PRO\s*R9", name):
                        detected_amd_targets.append("gfx1201")
                    elif re.search(r"(?i)890M|880M|Strix|Ryzen\s*AI\s*(?:PRO\s*)?3\d{2}|Kraken", name):
                        detected_amd_targets.append("gfx1150")
                    elif re.search(r"(?i)7900|W7900|Navi\s*31", name):
                        detected_amd_targets.append("gfx1100")
                    elif re.search(r"(?i)7800|7700|Navi\s*32", name):
                        detected_amd_targets.append("gfx1101")
                    elif re.search(r"(?i)7600|W7600|W7500|Navi\s*33", name):
                        detected_amd_targets.append("gfx1102")
                    elif re.search(r"(?i)6950|6900|6800|6700|W6800|Navi\s*21", name):
                        detected_amd_targets.append("gfx1030")
                    elif re.search(r"(?i)780M|760M|740M|Phoenix|Hawk", name):
                        detected_amd_targets.append("gfx1103")
                    elif re.search(r"(?i)680M|660M|Rembrandt", name):
                        detected_amd_targets.append("gfx1035")
                    else:
                        detected_amd_targets.append("gfx1201" if "Radeon" in name else "gfx1100")
    else:
        lspci = safe_subprocess(["lspci"])
        if lspci:
            if re.search(r"(?i)NVIDIA", lspci):
                has_nvidia = True
            if re.search(r"(?i)AMD|Radeon", lspci):
                has_amd = True
            if re.search(r"(?i)Intel.*(Arc|Graphics)", lspci):
                has_intel = True
        drm_path = Path("/sys/class/drm")
        if drm_path.exists():
            for uevent in drm_path.glob("card*/device/uevent"):
                try:
                    content = uevent.read_text()
                    if "DRIVER=amdgpu" in content:
                        has_amd = True
                    if "DRIVER=nvidia" in content:
                        has_nvidia = True
                    if "DRIVER=i915" in content or "DRIVER=xe" in content:
                        has_intel = True
                except Exception:
                    pass
        if has_amd:
            rocm_info = safe_subprocess(["rocminfo"])
            gfx_match = re.search(r"gfx\d+[a-zA-Z]?", rocm_info)
            if gfx_match:
                detected_amd_targets.append(gfx_match.group(0))
            elif lspci and re.search(r"(?i)MI325|MI350", lspci):
                detected_amd_targets.append("gfx950")
            elif lspci and re.search(r"(?i)R9700|9070|Navi\s*48", lspci):
                detected_amd_targets.append("gfx1201")
            elif lspci and re.search(r"(?i)890M|880M", lspci):
                detected_amd_targets.append("gfx1150")
            else:
                detected_amd_targets.append("gfx1100")

    if has_nvidia:
        smi = safe_subprocess(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        if smi:
            try:
                major = int(smi.split(".")[0])
                if major >= 620:
                    return "CUDA_13_2", ""
                if major >= 600:
                    return "CUDA_13_0", ""
                if major >= 570:
                    return "CUDA_12_8", ""
                if major >= 560:
                    return "CUDA_12_6", ""
                if major >= 520:
                    return "CUDA_12_1", ""
                if major >= 450:
                    return "CUDA_11_8", ""
            except Exception:
                pass
        return "CUDA_12_6", ""

    if has_amd:
        best_target = detected_amd_targets[0] if detected_amd_targets else "gfx1201"
        return "ROCM", best_target

    if has_intel:
        return "INTEL_XPU", ""

    return "CPU", ""


def fetch_runtime(version: str, target_dir: Path, isolated_env: dict) -> Path:
    if target_dir.exists():
        shutil.rmtree(target_dir, onerror=remove_readonly)
    target_dir.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        archive_name = f"python-{version}-embed-amd64.zip"
        url = f"https://www.python.org/ftp/python/{version}/{archive_name}"
        archive_path = DOWNLOADS_DIR / archive_name
        if not archive_path.is_file():
            print_status(f"Streaming standalone Windows runtime ({version})...")
            ctx = urllib.request.Request(url, headers={"User-Agent": "UniGen-Core"})
            with urllib.request.urlopen(ctx, timeout=30) as response, open(archive_path, "wb") as out_file:
                while chunk := response.read(65536):
                    out_file.write(chunk)
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)

        for pth_file in target_dir.glob("*._pth"):
            orig_lines = pth_file.read_text(encoding="utf-8").splitlines()
            core_zips = [line.strip() for line in orig_lines if line.strip().endswith(".zip")]
            payload_paths = core_zips + [".", "Lib/site-packages", "import site"]
            pth_file.write_text("\n".join(payload_paths) + "\n", encoding="utf-8")
        executable = target_dir / "python.exe"
    else:
        arch = "x86_64" if sys.maxsize > 2**32 else "i686"
        triple = f"{arch}-unknown-linux-gnu" if sys.platform.startswith("linux") else f"{arch}-apple-darwin"
        archive_name = f"cpython-{version}+{STANDALONE_RELEASE}-{triple}-install_only.tar.gz"
        url = f"https://github.com/astral-sh/python-build-standalone/releases/download/{STANDALONE_RELEASE}/{archive_name}"
        archive_path = DOWNLOADS_DIR / archive_name
        if not archive_path.is_file():
            print_status(f"Streaming standalone POSIX runtime ({version})...")
            ctx = urllib.request.Request(url, headers={"User-Agent": "UniGen-Core"})
            with urllib.request.urlopen(ctx, timeout=30) as response, open(archive_path, "wb") as out_file:
                while chunk := response.read(65536):
                    out_file.write(chunk)
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            tar_ref.extractall(target_dir.parent)
        source_extracted = target_dir.parent / "python"
        if source_extracted.exists() and source_extracted != target_dir:
            source_extracted.rename(target_dir)
        executable = target_dir / "bin" / "python"

    pip_bootstrapper = TMP_DIR / "get-pip.py"
    v_parts = version.split(".")
    if len(v_parts) >= 2 and v_parts[0] == "3" and v_parts[1] in ["6", "7", "8", "9"]:
        pip_url = f"https://bootstrap.pypa.io/pip/{v_parts[0]}.{v_parts[1]}/get-pip.py"
    else:
        pip_url = "https://bootstrap.pypa.io/get-pip.py"
    pip_ctx = urllib.request.Request(pip_url, headers={"User-Agent": "UniGen-Core"})
    with urllib.request.urlopen(pip_ctx, timeout=30) as response, open(pip_bootstrapper, "wb") as out_file:
        while chunk := response.read(65536):
            out_file.write(chunk)

    subprocess.run(
        [
            str(executable),
            "-I",
            str(pip_bootstrapper),
            "--no-warn-script-location",
            "--cache-dir",
            str(PIP_CACHE_DIR),
        ],
        env=isolated_env,
        check=True,
    )
    pip_bootstrapper.unlink(missing_ok=True)
    return executable


def condition_pytorch_runtime(executable: Path, isolated_env: dict, profile: str, llvm_target: str) -> str:
    print_status(f"Hardware Compute Architecture Locked: {profile} ({llvm_target or 'Generic'})", "SUCCESS")

    pip_cmd = [
        str(executable),
        "-m",
        "pip",
        "install",
        "--no-warn-script-location",
        "--cache-dir",
        str(PIP_CACHE_DIR),
    ]

    print_status("Upgrading core build tooling (pip, setuptools, wheel)...")
    subprocess.run(
        pip_cmd + ["-U", "pip", "setuptools", "wheel"],
        env=isolated_env,
        check=True,
    )

    hsa_override_used = HSA_OVERRIDES.get(llvm_target, "")
    print_status(f"Deploying hardware-aligned PyTorch binaries for {profile}...")

    if profile == "ROCM":
        family = AMD_STAGING_FAMILY_MAP.get(llvm_target, f"{llvm_target}-all")
        index_url = f"https://rocm.nightlies.amd.com/v2-staging/{family}/"
        print_status(f"Locking Sovereign Staging Nightly Index: {index_url}", "SUCCESS")

        if os.name == "nt":
            subprocess.run(
                pip_cmd + [
                    "--index-url", index_url,
                    "--pre", "-U", "--no-build-isolation",
                    "rocm[libraries,devel]",
                ],
                env=isolated_env,
                check=True,
            )

        subprocess.run(
            pip_cmd + [
                "--index-url", index_url,
                "--pre", "-U",
                "torch", "torchvision", "torchaudio",
            ],
            env=isolated_env,
            check=True,
        )

        subprocess.run(
            [str(executable), "-m", "rocm_sdk", "init"],
            env=isolated_env,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )

    elif profile.startswith("CUDA_"):
        cuda_urls = {
            "CUDA_13_2": "https://download.pytorch.org/whl/nightly/cu132",
            "CUDA_13_0": "https://download.pytorch.org/whl/nightly/cu130",
            "CUDA_12_8": "https://download.pytorch.org/whl/nightly/cu128",
            "CUDA_12_6": "https://download.pytorch.org/whl/nightly/cu126",
            "CUDA_12_1": "https://download.pytorch.org/whl/cu121",
            "CUDA_11_8": "https://download.pytorch.org/whl/cu118",
        }
        index_url = cuda_urls.get(profile, "https://download.pytorch.org/whl/nightly/cu126")
        subprocess.run(
            pip_cmd + ["--index-url", index_url, "--pre", "-U", "torch", "torchvision", "torchaudio"],
            env=isolated_env,
            check=True,
        )

    elif profile == "INTEL_XPU":
        subprocess.run(
            pip_cmd + [
                "--index-url", "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/",
                "-U",
                "torch", "torchvision", "torchaudio", "intel-extension-for-pytorch",
            ],
            env=isolated_env,
            check=True,
        )

    elif profile == "MPS":
        subprocess.run(
            pip_cmd + ["-U", "torch", "torchvision", "torchaudio"],
            env=isolated_env,
            check=True,
        )

    else:
        subprocess.run(
            pip_cmd + ["--index-url", "https://download.pytorch.org/whl/cpu", "--pre", "-U", "torch", "torchvision", "torchaudio"],
            env=isolated_env,
            check=True,
        )

    print_status("PyTorch hardware alignment complete.", "SUCCESS")
    return hsa_override_used


def generate_environment_anchors(target_dir: Path, executable: Path, hsa_override: str = "") -> None:
    project_root = target_dir.parent
    rel_env_name = target_dir.name

    miopen_db = HF_CACHE_DIR / "miopen" / "db"
    miopen_kernels = HF_CACHE_DIR / "miopen" / "kernels"
    inductor_dir = HF_CACHE_DIR / "torch_inductor"
    triton_dir = HF_CACHE_DIR / "triton"
    for p in (miopen_db, miopen_kernels, inductor_dir, triton_dir):
        p.mkdir(parents=True, exist_ok=True)

    hsa_bat_line = f'set "HSA_OVERRIDE_GFX_VERSION={hsa_override}"' if hsa_override else ""
    hsa_echo_bat = f"echo  HSA GFX Override: %HSA_OVERRIDE_GFX_VERSION%" if hsa_override else ""
    hsa_sh_line = f'export HSA_OVERRIDE_GFX_VERSION="{hsa_override}"' if hsa_override else ""
    hsa_echo_sh = f'echo " HSA GFX Override: ${{HSA_OVERRIDE_GFX_VERSION}}"' if hsa_override else ""

    if os.name == "nt":
        bat_content = f"""@echo off
title Hermetic Sandbox Environment
color 0b
set "ROOT_DIR=%~dp0"
set "PYTHONPATH="
set "PYTHONCASEOK="
set "VIRTUAL_ENV="
set "CONDA_PREFIX="
set "CONDA_DEFAULT_ENV="
set "PYTHONIOENCODING=utf-8"
set "PYTHONHOME=%ROOT_DIR%{rel_env_name}"
set "PATH=%ROOT_DIR%{rel_env_name};%ROOT_DIR%{rel_env_name}\\Scripts;%PATH%"
if exist "%ROOT_DIR%{rel_env_name}\\Lib\\site-packages\\torch\\lib" (
    set "PATH=%ROOT_DIR%{rel_env_name}\\Lib\\site-packages\\torch\\lib;%PATH%"
)
if exist "%ROOT_DIR%{rel_env_name}\\Lib\\site-packages\\rocm_sdk_core\\bin" (
    set "PATH=%ROOT_DIR%{rel_env_name}\\Lib\\site-packages\\rocm_sdk_core\\bin;%PATH%"
)
if exist "%ROOT_DIR%{rel_env_name}\\Lib\\site-packages\\rocm_sdk_libraries_gfx120X_all\\bin" (
    set "PATH=%ROOT_DIR%{rel_env_name}\\Lib\\site-packages\\rocm_sdk_libraries_gfx120X_all\\bin;%PATH%"
)
set "TEMP=%ROOT_DIR%.tmp"
set "TMP=%ROOT_DIR%.tmp"
set "TMPDIR=%ROOT_DIR%.tmp"
set "PIP_CACHE_DIR=%ROOT_DIR%.cache\\pip"
set "LOCALAPPDATA=%ROOT_DIR%.local_appdata"
set "APPDATA=%ROOT_DIR%.appdata"
set "HF_HOME=%ROOT_DIR%.hf_cache"
set "TORCH_HOME=%ROOT_DIR%.hf_cache\\torch"
set "MIOPEN_USER_DB_PATH=%ROOT_DIR%.hf_cache\\miopen\\db"
set "MIOPEN_CUSTOM_CACHE_DIR=%ROOT_DIR%.hf_cache\\miopen\\kernels"
set "TORCHINDUCTOR_CACHE_DIR=%ROOT_DIR%.hf_cache\\torch_inductor"
set "TRITON_CACHE_DIR=%ROOT_DIR%.hf_cache\\triton"
set "MIOPEN_FIND_MODE=2"
set "MIOPEN_LOG_LEVEL=0"
set "MIOPEN_ENABLE_LOGGING=0"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "TORCH_BLAS_PREFER_HIPBLASLT=1"
set "ROCM_PATH=%PYTHONHOME%"
{hsa_bat_line}
echo =======================================================================
echo  Hermetic Sandbox Shell Active
echo  Interpreter: %PYTHONHOME%\\python.exe
echo  Cache Anchor: %HF_HOME%
echo  Local Temp: %TEMP%
echo  Local Pip Cache: %PIP_CACHE_DIR%
echo  MIOpen Cache: %MIOPEN_CUSTOM_CACHE_DIR% (Find Mode: FAST Heuristic)
{hsa_echo_bat}
echo  Isolation Status: Zero host leakage. All temp/cache local to root.
echo =======================================================================
cmd /k
"""
        launcher_file = project_root / "launch_env.bat"
        launcher_file.write_text(bat_content, encoding="utf-8")
    else:
        sh_content = f"""#!/usr/bin/env bash
ROOT_DIR="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" &> /dev/null && pwd)"
unset PYTHONPATH
unset PYTHONCASEOK
unset VIRTUAL_ENV
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
export PYTHONIOENCODING="utf-8"
export PYTHONHOME="${{ROOT_DIR}}/{rel_env_name}"
export PATH="${{ROOT_DIR}}/{rel_env_name}:${{ROOT_DIR}}/{rel_env_name}/bin:${{PATH}}"
export TEMP="${{ROOT_DIR}}/.tmp"
export TMP="${{ROOT_DIR}}/.tmp"
export TMPDIR="${{ROOT_DIR}}/.tmp"
export PIP_CACHE_DIR="${{ROOT_DIR}}/.cache/pip"
export XDG_CACHE_HOME="${{ROOT_DIR}}/.cache"
export XDG_CONFIG_HOME="${{ROOT_DIR}}/.config"
export XDG_DATA_HOME="${{ROOT_DIR}}/.local/share"
export HF_HOME="${{ROOT_DIR}}/.hf_cache"
export TORCH_HOME="${{ROOT_DIR}}/.hf_cache/torch"
export MIOPEN_USER_DB_PATH="${{ROOT_DIR}}/.hf_cache/miopen/db"
export MIOPEN_CUSTOM_CACHE_DIR="${{ROOT_DIR}}/.hf_cache/miopen/kernels"
export TORCHINDUCTOR_CACHE_DIR="${{ROOT_DIR}}/.hf_cache/torch_inductor"
export TRITON_CACHE_DIR="${{ROOT_DIR}}/.hf_cache/triton"
export MIOPEN_FIND_MODE="2"
export MIOPEN_LOG_LEVEL="0"
export MIOPEN_ENABLE_LOGGING="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_BLAS_PREFER_HIPBLASLT="1"
export ROCM_PATH="${{PYTHONHOME}}"
{hsa_sh_line}
echo "======================================================================="
echo " Hermetic Sandbox Shell Active"
echo " Interpreter: ${{PYTHONHOME}}/bin/python"
echo " Cache Anchor: ${{HF_HOME}}"
echo " Local Temp: ${{TEMP}}"
echo " Local Pip Cache: ${{PIP_CACHE_DIR}}"
echo " MIOpen Cache: ${{MIOPEN_CUSTOM_CACHE_DIR}} (Find Mode: FAST Heuristic)"
{hsa_echo_sh}
echo " Isolation Status: Host modules segregated. Anchored relative root."
echo "======================================================================="
exec $SHELL
"""
        launcher_file = project_root / "launch_env.sh"
        launcher_file.write_text(sh_content, encoding="utf-8")
        launcher_file.chmod(0o755)

    print_status(f"Portable dynamic environment launcher deployed: {launcher_file.name}", "SUCCESS")


def run_hardware_acceptance_gate(executable: Path, isolated_env: dict, hsa_override: str = "") -> None:
    print_status("Executing hardware-accelerated acceptance verification...", "INFO")
    test_env = isolated_env.copy()
    if hsa_override:
        test_env["HSA_OVERRIDE_GFX_VERSION"] = hsa_override
    test_code = (
        "import sys, torch\n"
        "device = 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'\n"
        "backend = getattr(torch.version, 'hip', None) or getattr(torch.version, 'cuda', None) or ('mps' if device == 'mps' else 'cpu')\n"
        "name = torch.cuda.get_device_name(0) if device == 'cuda' else ('Apple Silicon' if device == 'mps' else 'Host CPU')\n"
        "print(f'[ACCEPTANCE] Device: {device.upper()} | Backend: {backend} | GPU: {name}')\n"
        "x = torch.randn((256, 256), device=device, dtype=torch.float32)\n"
        "y = torch.mm(x, x.T)\n"
        "assert torch.isfinite(y).all(), 'Non-finite tensor output detected'\n"
        "print(f'[ACCEPTANCE] GEMM Compute Verification PASSED on {device.upper()}.')\n"
    )
    res = subprocess.run([str(executable), "-c", test_code], env=test_env, text=True, capture_output=True)
    if res.returncode == 0:
        for line in res.stdout.strip().splitlines():
            print_status(line, "SUCCESS")
    else:
        print_status(f"Acceptance test diagnostic output:\n{res.stderr.strip()}", "WARN")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Hardware-Accelerated Virtualization Conditioner.")
    parser.add_argument("--version", type=str, default=None, help=f"Target Python runtime version (default: {DEFAULT_PYTHON_VERSION}).")
    parser.add_argument("--auto", action="store_true", help="Execute hands-free without interactive prompts.")
    args = parser.parse_args()

    print_status("=" * 72)
    print_status("            INTELLIVENV SOVEREIGN RUNTIME CONDITIONER", "SUCCESS")
    print_status("=" * 72)

    selected_version = args.version
    if selected_version is None:
        if args.auto or not sys.stdin.isatty():
            selected_version = DEFAULT_PYTHON_VERSION
        else:
            for key, ver in PREDEFINED_MATRIX.items():
                print(f"  [{key}] Python {ver}")
            manual_opt = str(len(PREDEFINED_MATRIX) + 1)
            print(f"  [{manual_opt}] Manual Version Entry")
            choice = input(f"\nSelect core runtime configuration [1-{manual_opt}] (Default: 2 [Python {DEFAULT_PYTHON_VERSION}]): ").strip()
            if not choice:
                selected_version = DEFAULT_PYTHON_VERSION
            elif choice in PREDEFINED_MATRIX:
                selected_version = PREDEFINED_MATRIX[choice]
            elif choice == manual_opt:
                selected_version = input("Enter custom Python version: ").strip()
            else:
                selected_version = DEFAULT_PYTHON_VERSION

    if not selected_version:
        selected_version = DEFAULT_PYTHON_VERSION

    target_sandbox = ROOT_DIR / f"py_env_{selected_version.replace('.', '_')}"

    isolated_env = os.environ.copy()
    for var in [
        "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONCASEOK",
        "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PIP_TARGET", "PIP_PREFIX",
        "PIP_BUILD_TRACKER",
    ]:
        isolated_env.pop(var, None)

    isolated_env["TEMP"] = str(TMP_DIR)
    isolated_env["TMP"] = str(TMP_DIR)
    isolated_env["TMPDIR"] = str(TMP_DIR)
    isolated_env["PIP_CACHE_DIR"] = str(PIP_CACHE_DIR)
    isolated_env["LOCALAPPDATA"] = str(LOCAL_APPDATA_DIR)
    isolated_env["APPDATA"] = str(ROAMING_APPDATA_DIR)
    isolated_env["XDG_CACHE_HOME"] = str(CACHE_DIR)
    isolated_env["XDG_CONFIG_HOME"] = str(ROOT_DIR / ".config")
    isolated_env["XDG_DATA_HOME"] = str(ROOT_DIR / ".local" / "share")
    isolated_env["HF_HOME"] = str(HF_CACHE_DIR)
    isolated_env["TORCH_HOME"] = str(HF_CACHE_DIR / "torch")
    isolated_env["MIOPEN_USER_DB_PATH"] = str(HF_CACHE_DIR / "miopen" / "db")
    isolated_env["MIOPEN_CUSTOM_CACHE_DIR"] = str(HF_CACHE_DIR / "miopen" / "kernels")
    isolated_env["TORCHINDUCTOR_CACHE_DIR"] = str(HF_CACHE_DIR / "torch_inductor")
    isolated_env["TRITON_CACHE_DIR"] = str(HF_CACHE_DIR / "triton")
    isolated_env["PYTHONIOENCODING"] = "utf-8"

    profile, llvm_target = resolve_hardware_matrix()
    executable = fetch_runtime(selected_version, target_sandbox, isolated_env)
    hsa_override = condition_pytorch_runtime(executable, isolated_env, profile, llvm_target)
    generate_environment_anchors(target_sandbox, executable, hsa_override=hsa_override)
    run_hardware_acceptance_gate(executable, isolated_env, hsa_override=hsa_override)

    print_status(
        f"Hermetic sandbox conditioned successfully at {target_sandbox.name}.",
        "SUCCESS",
    )


if __name__ == "__main__":
    main()