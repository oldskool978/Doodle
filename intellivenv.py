#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import ctypes
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile

if os.name == "nt":
    import _winapi

ROOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT_DIR / ".cache"
TMP_DIR = ROOT_DIR / ".tmp"
DOWNLOADS_DIR = CACHE_DIR / "downloads"
WHEELS_DIR = DOWNLOADS_DIR / "wheels"
DEPS_WHEELS_DIR = WHEELS_DIR / "deps"
PIP_CACHE_DIR = CACHE_DIR / "pip"
UV_CACHE_DIR = CACHE_DIR / "uv"
CCACHE_DIR = CACHE_DIR / "ccache"
SCCACHE_DIR = CACHE_DIR / "sccache"
CUDA_CACHE_DIR = CACHE_DIR / "cuda"
CUPY_CACHE_DIR = CACHE_DIR / "cupy"
NUMBA_CACHE_DIR = CACHE_DIR / "numba"
MPL_CACHE_DIR = CACHE_DIR / "matplotlib"
HF_CACHE_DIR = ROOT_DIR / ".hf_cache"
LOCAL_APPDATA_DIR = ROOT_DIR / ".local_appdata"
ROAMING_APPDATA_DIR = ROOT_DIR / ".appdata"
XDG_CONFIG_DIR = ROOT_DIR / ".config"
XDG_DATA_DIR = ROOT_DIR / ".local" / "share"
XDG_STATE_DIR = ROOT_DIR / ".local" / "state"

for _d in (
    CACHE_DIR,
    TMP_DIR,
    DOWNLOADS_DIR,
    WHEELS_DIR,
    DEPS_WHEELS_DIR,
    PIP_CACHE_DIR,
    UV_CACHE_DIR,
    CCACHE_DIR,
    SCCACHE_DIR,
    CUDA_CACHE_DIR,
    CUPY_CACHE_DIR,
    NUMBA_CACHE_DIR,
    MPL_CACHE_DIR,
    HF_CACHE_DIR,
    HF_CACHE_DIR / "hub",
    HF_CACHE_DIR / "torch",
    HF_CACHE_DIR / "torch_extensions",
    HF_CACHE_DIR / "torch_inductor",
    HF_CACHE_DIR / "triton",
    HF_CACHE_DIR / "miopen" / "db",
    HF_CACHE_DIR / "miopen" / "kernels",
    HF_CACHE_DIR / "miopen" / "comgr",
    LOCAL_APPDATA_DIR,
    ROAMING_APPDATA_DIR,
    XDG_CONFIG_DIR,
    XDG_DATA_DIR,
    XDG_STATE_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

tempfile.tempdir = str(TMP_DIR)

USER_AGENT = "IntelliVenv/6.0"
ALLOC_CONF_POLICY = "garbage_collection_threshold:0.8"
DEFAULT_PYTHON_VERSION = "3.14.10"
FALLBACK_SERIES_VERSIONS: dict[tuple[int, int], str] = {
    (3, 11): "3.11.9",
    (3, 12): "3.12.10",
    (3, 13): "3.13.17",
    (3, 14): "3.14.10",
}
FALLBACK_PBS_TAG = "20261021"
PBS_LATEST_JSON_URL = (
    "https://raw.githubusercontent.com/astral-sh/python-build-standalone/"
    "latest-release/latest-release.json"
)
NUGET_FLATCONTAINER_BASE = "https://api.nuget.org/v3-flatcontainer"

PYPI_BUILD_TOOLING = ["pip", "setuptools", "wheel", "packaging", "ninja", "cmake"]
PYPI_RUNTIME_DEPS = [
    "filelock",
    "typing-extensions",
    "sympy",
    "mpmath",
    "networkx",
    "jinja2",
    "MarkupSafe",
    "fsspec",
    "numpy",
    "pillow",
]
PYPI_ROCM_RUNTIME_DEPS = ["rocm-bootstrap"]
PYPI_BINARY_GATE_PACKAGES = ("numpy", "pillow", "MarkupSafe")

COMPUTE_DIST_PREFIXES = (
    "torch",
    "amd_torch",
    "rocm",
    "_rocm",
    "amd_rocm",
    "triton",
    "pytorch_triton",
    "nvidia_",
    "cuda_",
    "intel_extension",
    "oneccl",
)
NON_COMPUTE_DIST_NAMES = frozenset({"rocm_bootstrap"})

AMD_STAGING_FAMILY_MAP = {
    "gfx1250": "gfx1250",
    "gfx1201": "gfx120X-all",
    "gfx1200": "gfx120X-all",
    "gfx120X": "gfx120X-all",
    "gfx1171": "gfx1170",
    "gfx1170": "gfx1170",
    "gfx1153": "gfx1150",
    "gfx1152": "gfx1152",
    "gfx1151": "gfx1151",
    "gfx1150": "gfx1150",
    "gfx1103": "gfx110X-all",
    "gfx1102": "gfx110X-all",
    "gfx1101": "gfx110X-all",
    "gfx1100": "gfx110X-all",
    "gfx1036": "gfx103X-all",
    "gfx1035": "gfx103X-all",
    "gfx1034": "gfx103X-all",
    "gfx1033": "gfx103X-all",
    "gfx1032": "gfx103X-all",
    "gfx1031": "gfx103X-all",
    "gfx1030": "gfx103X-all",
    "gfx950": "gfx950",
    "gfx942": "gfx94X-dcgpu",
    "gfx90a": "gfx90a-all",
    "gfx908": "gfx908",
    "gfx906": "gfx906",
}

LEGACY_KPACK_FALLBACK_MAP = {
    "gfx1171": "gfx1170",
    "gfx1153": "gfx1150",
    "gfx1152": "gfx1150",
    "gfx1103": "gfx1100",
    "gfx1036": "gfx1030",
    "gfx1035": "gfx1030",
    "gfx1034": "gfx1030",
    "gfx1033": "gfx1030",
    "gfx1032": "gfx1030",
    "gfx1031": "gfx1030",
}

LEGACY_HSA_OVERRIDES = {
    "gfx1153": "11.5.0",
    "gfx1152": "11.5.0",
    "gfx1151": "11.5.1",
    "gfx1150": "11.5.0",
    "gfx1103": "11.0.0",
    "gfx1036": "10.3.0",
    "gfx1035": "10.3.0",
    "gfx1034": "10.3.0",
    "gfx1033": "10.3.0",
    "gfx1032": "10.3.0",
    "gfx1031": "10.3.0",
}

DETERMINISTIC_PIP_ERROR_MARKERS = (
    "metadata-generation-failed",
    "subprocess-exited-with-error",
    "ResolutionImpossible",
    "No matching distribution found",
    "PackageRequiresPython",
    "Requested ROCM_SDK_TARGET_FAMILY",
    "Conflicting dependencies",
)

_VERIFIED_ARCHIVE_SIGNATURES: set[tuple[str, int, int]] = set()
_PBS_RELEASE_CACHE: tuple[str, str, dict[str, tuple[str, str]]] | None = None
_NUGET_VERSIONS_CACHE: list[str] | None = None


def print_status(msg: str, status: str = "INFO") -> None:
    colors = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m",
        "WARN": "\033[93m",
        "ERROR": "\033[91m",
        "RESET": "\033[0m",
    }
    if os.name == "nt" and not os.environ.get("WT_SESSION"):
        print(f"[{status}] {msg}", flush=True)
    else:
        print(f"{colors.get(status, '')}[{status}] {msg}{colors['RESET']}", flush=True)


def parse_semver(ver_str: str) -> tuple[int, ...]:
    nums = [int(n) for n in re.findall(r"\d+", ver_str)]
    return tuple(nums) if nums else (0,)


def _pad_tuple(vals: tuple[int, ...], length: int = 3) -> tuple[int, ...]:
    if len(vals) >= length:
        return vals[:length]
    return vals + (0,) * (length - len(vals))


def _pep440_public_key(ver_str: str) -> tuple[int, tuple[int, ...], int, int, int]:
    pub = ver_str.split("+", 1)[0].strip().lower()
    epoch = 0
    if "!" in pub:
        ep_s, pub = pub.split("!", 1)
        if ep_s.isdigit():
            epoch = int(ep_s)
    m_rel = re.match(r"^(\d+(?:\.\d+)*)", pub)
    rel_tuple = (
        _pad_tuple(tuple(int(x) for x in m_rel.group(1).split(".")), 4)
        if m_rel
        else (0, 0, 0, 0)
    )
    rest = pub[m_rel.end() :] if m_rel else pub

    dev_m = re.search(r"\.?dev(\d*)", rest)
    post_m = re.search(r"(?:\.?post|-rev|\.?r)(\d*)", rest)
    pre_m = re.search(r"(a|alpha|b|beta|rc|c|pre|preview)(\d*)", rest)

    if dev_m and not pre_m and not post_m:
        phase = 0
    elif pre_m:
        tag = pre_m.group(1)
        if tag in ("a", "alpha"):
            phase = 1 if not dev_m else 0
        elif tag in ("b", "beta"):
            phase = 2
        else:
            phase = 3
    elif post_m:
        phase = 5
    else:
        phase = 4

    pre_num = int(pre_m.group(2)) if (pre_m and pre_m.group(2)) else 0
    if post_m and post_m.group(1):
        pre_num = int(post_m.group(1))
    dev_num = (
        int(dev_m.group(1))
        if (dev_m and dev_m.group(1))
        else (0 if not dev_m else -1)
    )
    return (epoch, rel_tuple, phase, pre_num, dev_num)


def _rocm_stamp_sort_key(stamp: str) -> tuple[tuple[int, ...], int, int, int]:
    s = stamp.strip().lower()
    if s.startswith("rocm"):
        s = s[4:].lstrip(".-_")
    m_rel = re.match(r"^(\d+(?:\.\d+)*)", s)
    rel_tuple = (
        _pad_tuple(tuple(int(x) for x in m_rel.group(1).split(".")), 3)
        if m_rel
        else (0, 0, 0)
    )
    rest = s[m_rel.end() :] if m_rel else s
    m_date = re.search(r"(20\d{6})", s)
    date_int = int(m_date.group(1)) if m_date else 0

    if "dev" in rest:
        phase = 0
    elif re.search(r"(?:^|[._-])?(?:a|alpha)\d*", rest):
        phase = 1
    elif re.search(r"(?:^|[._-])?(?:b|beta)\d*", rest):
        phase = 2
    elif re.search(r"(?:^|[._-])?(?:rc|c)\d*", rest):
        phase = 3
    elif "post" in rest:
        phase = 5
    else:
        phase = 4

    nums = [int(n) for n in re.findall(r"\d+", rest)]
    tail_num = nums[-1] if nums else 0
    return (rel_tuple, phase, date_int, tail_num)


def _norm_stamp(stamp: str) -> str:
    s = stamp.strip().lower()
    if s.startswith("rocm"):
        s = s[4:]
    return re.sub(r"[+_-]+", ".", s).strip(".")


def remove_readonly(func, path, _) -> None:
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        if func is not None:
            func(path)
    except Exception:
        pass


def _is_reparse_or_link(path: Path) -> bool:
    try:
        st = os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return True
    if os.name == "nt" and getattr(st, "st_file_attributes", 0) & getattr(
        stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
    ):
        return True
    return False


def _remove_reparse_point(path: Path) -> None:
    if os.name == "nt":
        try:
            os.rmdir(path)
            return
        except OSError:
            pass
    path.unlink(missing_ok=True)


def _detach_nested_reparse_points(root: Path) -> None:
    if _is_reparse_or_link(root):
        _remove_reparse_point(root)
        return
    if not root.is_dir():
        return
    try:
        entries = list(os.scandir(root))
    except OSError:
        return
    for entry in entries:
        p = Path(entry.path)
        if _is_reparse_or_link(p):
            _remove_reparse_point(p)
        elif entry.is_dir(follow_symlinks=False):
            _detach_nested_reparse_points(p)


def safe_rmtree(dir_path: Path) -> None:
    if _is_reparse_or_link(dir_path):
        _remove_reparse_point(dir_path)
        return
    if not dir_path.exists():
        return
    _detach_nested_reparse_points(dir_path)
    if not dir_path.exists():
        return
    for attempt in range(8):
        try:
            if sys.version_info >= (3, 12):
                shutil.rmtree(
                    dir_path,
                    onexc=lambda func, path, _: remove_readonly(func, path, None),
                )
            else:
                shutil.rmtree(dir_path, onerror=remove_readonly)
            return
        except Exception:
            gc.collect()
            time.sleep(0.15 * (attempt + 1))


def _link_or_junction(src: Path, dst: Path) -> None:
    if _is_reparse_or_link(dst) or dst.exists():
        if dst.is_dir() and not _is_reparse_or_link(dst):
            safe_rmtree(dst)
        else:
            _remove_reparse_point(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_resolved = src.resolve()
    if src_resolved.is_dir():
        if os.name == "nt":
            try:
                _winapi.CreateJunction(str(src_resolved), str(dst))
                return
            except Exception:
                pass
            try:
                os.symlink(str(src_resolved), str(dst), target_is_directory=True)
                return
            except OSError:
                shutil.copytree(src_resolved, dst, dirs_exist_ok=True)
                return
        os.symlink(str(src_resolved), str(dst))
    elif src_resolved.is_file():
        try:
            os.link(src_resolved, dst)
        except OSError:
            try:
                os.symlink(str(src_resolved), dst)
            except OSError:
                shutil.copy2(src_resolved, dst)


def _has_nonempty_dir(path: Path) -> bool:
    try:
        return path.is_dir() and any(path.iterdir())
    except OSError:
        return False


def _site_packages_dir(target_sandbox: Path) -> Path | None:
    if os.name == "nt":
        return target_sandbox / "Lib" / "site-packages"
    matches = sorted(target_sandbox.glob("lib/python*/site-packages"))
    return matches[0] if matches else None


def _windows_arch_tags() -> tuple[str, str, str]:
    mach = platform.machine().lower()
    if mach in ("arm64", "aarch64"):
        return ("arm64", "pythonarm64", "win_arm64")
    return ("amd64", "python", "win_amd64")


def _posix_target_triple() -> str:
    mach = platform.machine().lower()
    if mach in ("aarch64", "arm64"):
        arch = "aarch64"
    elif mach in ("x86_64", "amd64"):
        arch = "x86_64"
    else:
        arch = mach
    if sys.platform.startswith("linux"):
        return f"{arch}-unknown-linux-gnu"
    return f"{arch}-apple-darwin"


def _is_freethreaded_runtime() -> bool:
    try:
        return bool(sysconfig.get_config_var("Py_GIL_Disabled"))
    except Exception:
        return False


def _probe_url_head(url: str, timeout: int = 5) -> bool:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT},
        method="HEAD",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


def _stream_download(url: str, dest: Path, expected_sha256: str = "") -> None:
    part_path = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    hasher = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=120) as response, open(
        part_path, "wb"
    ) as out_file:
        while chunk := response.read(65536):
            out_file.write(chunk)
            hasher.update(chunk)
    if expected_sha256 and hasher.hexdigest().lower() != expected_sha256.lower():
        part_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {dest.name}: expected {expected_sha256}, "
            f"got {hasher.hexdigest()}"
        )
    part_path.replace(dest)


def _extract_tarball_trusted(archive_path: Path, dest_dir: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tar_ref:
        sig = inspect.signature(tar_ref.extractall)
        if "filter" in sig.parameters:
            tar_ref.extractall(dest_dir, filter="fully_trusted")
        else:
            tar_ref.extractall(dest_dir)


def verify_archive_integrity(archive_path: Path) -> bool:
    if not archive_path.is_file() or archive_path.stat().st_size < 512:
        return False
    stat_res = archive_path.stat()
    sig = (str(archive_path.resolve()), stat_res.st_size, stat_res.st_mtime_ns)
    if sig in _VERIFIED_ARCHIVE_SIGNATURES:
        return True
    name_lower = archive_path.name.lower()
    try:
        if name_lower.endswith((".whl", ".zip", ".nupkg")):
            with zipfile.ZipFile(archive_path, "r") as zf:
                bad_member = zf.testzip()
                if bad_member is not None:
                    print_status(
                        f"Corrupt CRC-32 stream in {archive_path.name} ({bad_member}). Evicting.",
                        "WARN",
                    )
                    return False
        elif name_lower.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:gz") as tf:
                for member in tf.getmembers():
                    if member.isfile():
                        f_obj = tf.extractfile(member)
                        if f_obj is not None:
                            while f_obj.read(65536):
                                pass
        _VERIFIED_ARCHIVE_SIGNATURES.add(sig)
        return True
    except Exception as exc:
        print_status(
            f"Corrupt package archive {archive_path.name} ({exc}). Evicting.",
            "WARN",
        )
        return False


def safe_subprocess(
    cmd: list[str], env: dict[str, str] | None = None, timeout: int = 15
) -> str:
    try:
        kwargs = (
            {"creationflags": subprocess.CREATE_NO_WINDOW}
            if os.name == "nt"
            else {}
        )
        return subprocess.check_output(
            cmd,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            text=True,
            env=env,
            **kwargs,
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

        class DXGI_QUERY_VIDEO_MEMORY_INFO(ctypes.Structure):
            _fields_ = [
                ("Budget", ctypes.c_ulonglong),
                ("CurrentUsage", ctypes.c_ulonglong),
                ("AvailableForReservation", ctypes.c_ulonglong),
                ("CurrentReservation", ctypes.c_ulonglong),
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
        iid_dxgi_adapter3 = GUID(
            0x645967A4,
            0x1392,
            0x4310,
            (ctypes.c_ubyte * 8)(0xA7, 0x98, 0x80, 0x53, 0xCE, 0x3E, 0x93, 0xFD),
        )
        factory = ctypes.c_void_p()
        if (
            ctypes.windll.dxgi.CreateDXGIFactory1(
                ctypes.byref(iid_dxgi_factory1), ctypes.byref(factory)
            )
            != 0
        ):
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
            call_com(
                adapter,
                10,
                [ctypes.c_void_p, ctypes.POINTER(DXGI_ADAPTER_DESC1)],
                ctypes.byref(desc),
            )
            if not (desc.Flags & 2):
                vram_total_gb = desc.DedicatedVideoMemory / (1024**3)
                vram_used_gb = 0.0
                svram_budget_gb = desc.SharedSystemMemory / (1024**3)
                svram_used_gb = 0.0
                adapter3 = ctypes.c_void_p()
                if (
                    call_com(
                        adapter,
                        0,
                        [
                            ctypes.c_void_p,
                            ctypes.POINTER(GUID),
                            ctypes.POINTER(ctypes.c_void_p),
                        ],
                        ctypes.byref(iid_dxgi_adapter3),
                        ctypes.byref(adapter3),
                    )
                    >= 0
                    and adapter3.value
                ):
                    local_info = DXGI_QUERY_VIDEO_MEMORY_INFO()
                    non_local_info = DXGI_QUERY_VIDEO_MEMORY_INFO()
                    if (
                        call_com(
                            adapter3,
                            14,
                            [
                                ctypes.c_void_p,
                                ctypes.c_uint,
                                ctypes.c_int,
                                ctypes.POINTER(DXGI_QUERY_VIDEO_MEMORY_INFO),
                            ],
                            0,
                            0,
                            ctypes.byref(local_info),
                        )
                        >= 0
                    ):
                        vram_used_gb = local_info.CurrentUsage / (1024**3)
                    if (
                        call_com(
                            adapter3,
                            14,
                            [
                                ctypes.c_void_p,
                                ctypes.c_uint,
                                ctypes.c_int,
                                ctypes.POINTER(DXGI_QUERY_VIDEO_MEMORY_INFO),
                            ],
                            0,
                            1,
                            ctypes.byref(non_local_info),
                        )
                        >= 0
                    ):
                        svram_used_gb = non_local_info.CurrentUsage / (1024**3)
                        if non_local_info.Budget > 0:
                            svram_budget_gb = non_local_info.Budget / (1024**3)
                    call_com(adapter3, 2, [ctypes.c_void_p])
                adapters.append(
                    {
                        "name": str(desc.Description),
                        "vendor_id": desc.VendorId,
                        "device_id": desc.DeviceId,
                        "vram_gb": vram_total_gb,
                        "vram_used_gb": vram_used_gb,
                        "vram_avail_gb": max(0.0, vram_total_gb - vram_used_gb),
                        "svram_budget_gb": svram_budget_gb,
                        "svram_used_gb": svram_used_gb,
                    }
                )
            call_com(adapter, 2, [ctypes.c_void_p])
            idx += 1
        call_com(factory, 2, [ctypes.c_void_p])
        return adapters
    except Exception:
        return []


def _classify_amd_gfx_from_name(name: str) -> str:
    if re.search(r"(?i)MI400|MI450|UDNA", name):
        return "gfx1250"
    if re.search(r"(?i)MI325|MI350|MI355", name):
        return "gfx950"
    if re.search(r"(?i)MI300|MI308", name):
        return "gfx942"
    if re.search(r"(?i)MI250|MI210", name):
        return "gfx90a"
    if re.search(r"(?i)MI100", name):
        return "gfx908"
    if re.search(r"(?i)Radeon\s*VII|MI50|MI60", name):
        return "gfx906"
    if re.search(r"(?i)Medusa|gfx1171", name):
        return "gfx1171"
    if re.search(r"(?i)gfx1170", name):
        return "gfx1170"
    if re.search(r"(?i)9060|Navi\s*44", name):
        return "gfx1200"
    if re.search(r"(?i)R9700|9070|Navi\s*48|RX\s*9\d{3}|AI\s*PRO\s*R9", name):
        return "gfx1201"
    if re.search(r"(?i)8060S|8050S|8040S|Strix\s*Halo|Ryzen\s*AI\s*Max", name):
        return "gfx1151"
    if re.search(r"(?i)820M", name):
        return "gfx1153"
    if re.search(r"(?i)860M|840M|Kraken|Ryzen\s*AI\s*[57]\s*3[45]0", name):
        return "gfx1152"
    if re.search(r"(?i)890M|880M|Strix|Ryzen\s*AI\s*(?:9|PRO\s*3\d{2}|3\d{2})", name):
        return "gfx1150"
    if re.search(r"(?i)7900|W7900|W7800|Navi\s*31", name):
        return "gfx1100"
    if re.search(r"(?i)7800|7700|W7700|Navi\s*32", name):
        return "gfx1101"
    if re.search(r"(?i)7600|7650|W7600|W7500|Navi\s*33", name):
        return "gfx1102"
    if re.search(r"(?i)780M|760M|740M|Phoenix|Hawk", name):
        return "gfx1103"
    if re.search(r"(?i)6950|6900|6800|W6900|W6800|V620|Navi\s*21", name):
        return "gfx1030"
    if re.search(r"(?i)6750|6700|W6700|Navi\s*22", name):
        return "gfx1031"
    if re.search(r"(?i)6650|6600|W6600|Navi\s*23", name):
        return "gfx1032"
    if re.search(r"(?i)6500|6400|W6400|Navi\s*24", name):
        return "gfx1034"
    if re.search(r"(?i)680M|660M|Rembrandt", name):
        return "gfx1035"
    return "gfx1201" if "Radeon" in name else "gfx1100"


def _query_nvidia_driver_cuda_ceiling() -> tuple[int, int]:
    lib_names = (
        ("nvcuda.dll",)
        if os.name == "nt"
        else ("libcuda.so.1", "libcuda.so")
    )
    for lib_name in lib_names:
        try:
            lib = (
                ctypes.WinDLL(lib_name)
                if os.name == "nt"
                else ctypes.CDLL(lib_name)
            )
            ver = ctypes.c_int(0)
            if lib.cuDriverGetVersion(ctypes.byref(ver)) == 0 and ver.value > 0:
                return (ver.value // 1000, (ver.value % 1000) // 10)
        except Exception:
            continue

    smi_xml = safe_subprocess(["nvidia-smi", "-q"])
    if smi_xml:
        m = re.search(r"CUDA Version\s*:\s*(\d+)\.(\d+)", smi_xml)
        if m:
            return (int(m.group(1)), int(m.group(2)))

    smi_drv = safe_subprocess(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    if smi_drv:
        try:
            major = int(smi_drv.splitlines()[0].strip().split(".")[0])
            if major >= 615:
                return (13, 4)
            if major >= 580:
                return (13, 2)
            if major >= 570:
                return (12, 8)
            if major >= 550:
                return (12, 6)
            if major >= 525:
                return (12, 1)
            if major >= 450:
                return (11, 8)
        except Exception:
            pass
    return (0, 0)


def resolve_hardware_matrix() -> tuple[str, str]:
    forced_arch = os.environ.get("PYTORCH_ROCM_ARCH", "").strip()
    if sys.platform == "darwin":
        if platform.machine().lower() in ("arm64", "aarch64"):
            return ("MPS", "")
        return ("CPU", "")

    has_nvidia, has_amd, has_intel = False, False, False
    detected_amd_targets: list[tuple[float, str]] = []

    if os.name == "nt":
        dxgi_devices = sorted(
            query_dxgi_adapters_win32(),
            key=lambda d: (d.get("vram_gb", 0.0), d.get("svram_budget_gb", 0.0)),
            reverse=True,
        )
        for dev in dxgi_devices:
            vid = dev["vendor_id"]
            if vid == 0x10DE:
                has_nvidia = True
            elif vid == 0x8086 and dev["vram_gb"] >= 2.0:
                has_intel = True
            elif vid == 0x1002:
                has_amd = True
                detected_amd_targets.append(
                    (dev["vram_gb"], _classify_amd_gfx_from_name(dev["name"]))
                )
        if not has_intel and any(d["vendor_id"] == 0x8086 for d in dxgi_devices):
            has_intel = not (has_nvidia or has_amd)
    else:
        lspci = safe_subprocess(["lspci"])
        if lspci:
            if re.search(r"(?i)NVIDIA", lspci):
                has_nvidia = True
            if re.search(r"(?i)AMD|ATI|Radeon|Instinct", lspci):
                has_amd = True
            if re.search(r"(?i)Intel.*(Arc|Data Center GPU|Graphics)", lspci):
                has_intel = True
        drm_path = Path("/sys/class/drm")
        if drm_path.exists():
            for uevent in sorted(drm_path.glob("card*/device/uevent")):
                try:
                    content = uevent.read_text(encoding="utf-8", errors="ignore")
                    if "DRIVER=amdgpu" in content:
                        has_amd = True
                        vram_file = uevent.parent / "mem_info_vram_total"
                        vram_bytes = (
                            int(vram_file.read_text().strip())
                            if vram_file.is_file()
                            else 0
                        )
                        detected_amd_targets.append((vram_bytes / (1024**3), ""))
                    elif "DRIVER=nvidia" in content:
                        has_nvidia = True
                    elif "DRIVER=xe" in content or "DRIVER=i915" in content:
                        has_intel = True
                except Exception:
                    pass
        if has_amd:
            rocm_info = safe_subprocess(["rocminfo"])
            gfx_matches = [
                g
                for g in re.findall(r"\bgfx\d+[0-9a-zA-Z]*\b", rocm_info)
                if g != "gfx000"
            ]
            if gfx_matches:
                non_apu = [
                    g
                    for g in gfx_matches
                    if g not in ("gfx1036", "gfx1035", "gfx1103")
                ]
                chosen = non_apu[0] if non_apu else gfx_matches[0]
                detected_amd_targets = [(999.0, chosen)]
            elif lspci:
                lines = [
                    ln
                    for ln in lspci.splitlines()
                    if re.search(r"(?i)VGA|3D|Display", ln)
                    and re.search(r"(?i)AMD|ATI|Radeon|Instinct", ln)
                ]
                parsed = [_classify_amd_gfx_from_name(ln) for ln in lines]
                non_apu = [
                    g for g in parsed if g not in ("gfx1036", "gfx1035", "gfx1103")
                ]
                chosen = non_apu[0] if non_apu else (parsed[0] if parsed else "gfx1100")
                detected_amd_targets = [(1.0, chosen)]

    if has_nvidia:
        cu_maj, cu_min = _query_nvidia_driver_cuda_ceiling()
        ceiling_str = f"{cu_maj}.{cu_min}" if cu_maj > 0 else ""
        return ("CUDA", ceiling_str)

    if has_amd:
        if forced_arch:
            primary_arch = forced_arch.split(";")[0].split(",")[0].strip()
            return ("ROCM", primary_arch)
        valid_targets = [
            t
            for _, t in sorted(detected_amd_targets, key=lambda x: x[0], reverse=True)
            if t
        ]
        return ("ROCM", valid_targets[0] if valid_targets else "gfx1201")

    if has_intel:
        return ("INTEL_XPU", "")

    return ("CPU", "")


def _query_pbs_release_metadata() -> tuple[str, str, dict[str, tuple[str, str]]]:
    global _PBS_RELEASE_CACHE
    if _PBS_RELEASE_CACHE is not None:
        return _PBS_RELEASE_CACHE

    tag = FALLBACK_PBS_TAG
    prefix = (
        f"https://github.com/astral-sh/python-build-standalone/releases/download/{tag}"
    )
    assets: dict[str, tuple[str, str]] = {}

    try:
        req = urllib.request.Request(
            PBS_LATEST_JSON_URL,
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            meta = json.loads(resp.read().decode("utf-8", errors="ignore"))
        tag = str(meta.get("tag") or tag).strip()
        prefix = str(meta.get("asset_url_prefix") or prefix).rstrip("/")
    except Exception:
        pass

    try:
        api_url = (
            "https://api.github.com/repos/astral-sh/python-build-standalone/"
            f"releases/tags/{tag}"
        )
        api_req = urllib.request.Request(
            api_url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/vnd.github+json",
            },
        )
        with urllib.request.urlopen(api_req, timeout=12) as resp:
            rel_data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        for item in rel_data.get("assets", []):
            name = str(item.get("name") or "")
            dl_url = str(item.get("browser_download_url") or "")
            digest_raw = str(item.get("digest") or "")
            sha256_hex = (
                digest_raw.split("sha256:", 1)[1].strip()
                if digest_raw.startswith("sha256:")
                else ""
            )
            if name and dl_url:
                assets[name] = (dl_url, sha256_hex)
    except Exception:
        pass

    _PBS_RELEASE_CACHE = (tag, prefix, assets)
    return _PBS_RELEASE_CACHE


def _query_nuget_python_versions() -> list[str]:
    global _NUGET_VERSIONS_CACHE
    if _NUGET_VERSIONS_CACHE is not None:
        return _NUGET_VERSIONS_CACHE

    _, nuget_pkg, _ = _windows_arch_tags()
    idx_url = f"{NUGET_FLATCONTAINER_BASE}/{nuget_pkg}/index.json"
    versions: list[str] = []
    try:
        req = urllib.request.Request(idx_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
        for v in payload.get("versions", []):
            v_str = str(v).strip()
            if re.match(r"^\d+\.\d+\.\d+$", v_str):
                sv = parse_semver(v_str)
                if len(sv) >= 2 and (sv[0], sv[1]) >= (3, 11):
                    versions.append(v_str)
    except Exception:
        pass

    _NUGET_VERSIONS_CACHE = versions
    return versions


def resolve_dynamic_python_matrix() -> dict[str, str]:
    by_minor: dict[tuple[int, int], list[str]] = {
        pair: [ver] for pair, ver in FALLBACK_SERIES_VERSIONS.items()
    }

    if os.name == "nt":
        nuget_vers = _query_nuget_python_versions()
        if nuget_vers:
            for ver in nuget_vers:
                sv = parse_semver(ver)
                by_minor.setdefault((sv[0], sv[1]), []).append(ver)
        else:
            embed_arch, _, _ = _windows_arch_tags()
            try:
                req = urllib.request.Request(
                    "https://www.python.org/ftp/python/",
                    headers={"User-Agent": USER_AGENT},
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")
                raw_vers = [
                    v
                    for v in re.findall(r'href="(\d+\.\d+\.\d+)/"', html)
                    if parse_semver(v)[:2] >= (3, 11)
                ]
                grouped_ftp: dict[tuple[int, int], list[str]] = {}
                for v in raw_vers:
                    sv = parse_semver(v)
                    grouped_ftp.setdefault((sv[0], sv[1]), []).append(v)
                for pair, cands in grouped_ftp.items():
                    for cand in sorted(set(cands), key=parse_semver, reverse=True)[:4]:
                        arc = f"python-{cand}-embed-{embed_arch}.zip"
                        if _probe_url_head(
                            f"https://www.python.org/ftp/python/{cand}/{arc}",
                            timeout=4,
                        ):
                            by_minor.setdefault(pair, []).append(cand)
                            break
            except Exception:
                pass
    else:
        triple = _posix_target_triple()
        tag, _, assets = _query_pbs_release_metadata()
        if assets:
            pat = re.compile(
                rf"^cpython-(\d+\.\d+\.\d+)\+{re.escape(tag)}-{re.escape(triple)}"
                r"-install_only(?:_stripped)?\.tar\.gz$"
            )
            for name in assets:
                m = pat.match(name)
                if m:
                    ver = m.group(1)
                    sv = parse_semver(ver)
                    if len(sv) >= 2 and (sv[0], sv[1]) >= (3, 11):
                        by_minor.setdefault((sv[0], sv[1]), []).append(ver)

    matrix: dict[str, str] = {}
    for idx, minor_pair in enumerate(sorted(by_minor.keys()), start=1):
        matrix[str(idx)] = max(set(by_minor[minor_pair]), key=parse_semver)
    return matrix


def _resolve_windows_embed_url(version: str) -> str:
    embed_arch, _, _ = _windows_arch_tags()
    archive_name = f"python-{version}-embed-{embed_arch}.zip"
    candidates = (
        f"https://www.python.org/ftp/python/{version}/{archive_name}",
        f"https://www.python.org/ftp/python/{version}/{embed_arch}/{archive_name}",
    )
    for url in candidates:
        if _probe_url_head(url, timeout=6):
            return url
    return candidates[0]


def _sandbox_has_c_api(target_dir: Path, version: str = "") -> bool:
    if os.name == "nt":
        if not (target_dir / "python.exe").is_file():
            return False
        has_header = (target_dir / "include" / "Python.h").is_file() or (
            target_dir / "Include" / "Python.h"
        ).is_file()
        if not has_header or not (target_dir / "Lib" / "os.py").is_file():
            return False
        if version:
            sv = parse_semver(version)
            if len(sv) >= 2:
                return (target_dir / "libs" / f"python{sv[0]}{sv[1]}.lib").is_file()
        return any((target_dir / "libs").glob("python3*.lib"))

    exe = (
        target_dir / "bin" / "python3"
        if (target_dir / "bin" / "python3").is_file()
        else target_dir / "bin" / "python"
    )
    return exe.is_file() and any(target_dir.glob("include/python*/Python.h"))


def _prune_legacy_embed_root_binaries(target_dir: Path) -> None:
    dlls_dir = target_dir / "DLLs"
    if not dlls_dir.is_dir():
        return
    for pyd_in_dlls in dlls_dir.glob("*.pyd"):
        root_pyd = target_dir / pyd_in_dlls.name
        if root_pyd.is_file():
            try:
                root_pyd.unlink(missing_ok=True)
            except OSError:
                pass
    for dll_in_dlls in dlls_dir.glob("*.dll"):
        low = dll_in_dlls.name.lower()
        if low.startswith(("python", "vcruntime")):
            continue
        root_dll = target_dir / dll_in_dlls.name
        if root_dll.is_file():
            try:
                root_dll.unlink(missing_ok=True)
            except OSError:
                pass


def _hydrate_windows_nuget_runtime(version: str, target_dir: Path) -> bool:
    _, nuget_pkg, _ = _windows_arch_tags()
    nupkg_name = f"{nuget_pkg}.{version}.nupkg"
    nupkg_path = DOWNLOADS_DIR / nupkg_name
    if not verify_archive_integrity(nupkg_path):
        nupkg_path.unlink(missing_ok=True)
        url = f"{NUGET_FLATCONTAINER_BASE}/{nuget_pkg}/{version}/{nupkg_name}"
        try:
            print_status(
                f"Streaming full C-API Windows runtime ({nuget_pkg} {version})..."
            )
            _stream_download(url, nupkg_path)
        except Exception as exc:
            print_status(
                f"PSF NuGet archive unavailable for {version} ({exc}); falling back to embed runtime.",
                "WARN",
            )
            nupkg_path.unlink(missing_ok=True)
            return False

    if not verify_archive_integrity(nupkg_path):
        nupkg_path.unlink(missing_ok=True)
        return False

    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(nupkg_path, "r") as zf:
        for member in zf.infolist():
            norm_name = member.filename.replace("\\", "/")
            if not norm_name.startswith("tools/"):
                continue
            rel_str = norm_name[len("tools/") :].strip("/")
            if not rel_str:
                continue
            if rel_str.lower().startswith("lib/site-packages/") and (
                target_dir / "Lib" / "site-packages"
            ).is_dir():
                continue
            dest_path = target_dir / Path(rel_str)
            if member.is_dir() or member.filename.endswith("/"):
                dest_path.mkdir(parents=True, exist_ok=True)
                continue
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with zf.open(member, "r") as src_f, open(dest_path, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f, length=65536)
            except PermissionError:
                if not dest_path.is_file():
                    raise

    (target_dir / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    (target_dir / "Scripts").mkdir(parents=True, exist_ok=True)
    if (target_dir / "Lib" / "os.py").is_file():
        for pth_file in target_dir.glob("*._pth"):
            pth_file.unlink(missing_ok=True)
        for std_zip in target_dir.glob("python*.zip"):
            try:
                std_zip.unlink(missing_ok=True)
            except OSError:
                pass
        _prune_legacy_embed_root_binaries(target_dir)
    return _sandbox_has_c_api(target_dir, version)


def fetch_runtime(version: str, target_dir: Path) -> Path:
    executable = (
        target_dir / "python.exe"
        if os.name == "nt"
        else target_dir / "bin" / "python3"
    )
    if not executable.is_file() and os.name != "nt":
        executable = target_dir / "bin" / "python"

    site_packages = _site_packages_dir(target_dir)
    if (
        executable.is_file()
        and _sandbox_has_c_api(target_dir, version)
        and site_packages is not None
        and (site_packages / "pip").is_dir()
    ):
        if os.name == "nt" and (target_dir / "Lib" / "os.py").is_file():
            for pth_file in target_dir.glob("*._pth"):
                pth_file.unlink(missing_ok=True)
            _prune_legacy_embed_root_binaries(target_dir)
        print_status(
            f"Reusing C-API complete hermetic base interpreter at {target_dir.name}.",
            "INFO",
        )
        return executable

    if os.name == "nt":
        hydrated = _hydrate_windows_nuget_runtime(version, target_dir)
        if not hydrated:
            if target_dir.exists() and not executable.is_file():
                safe_rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            embed_arch, _, _ = _windows_arch_tags()
            archive_name = f"python-{version}-embed-{embed_arch}.zip"
            archive_path = DOWNLOADS_DIR / archive_name
            if not verify_archive_integrity(archive_path):
                archive_path.unlink(missing_ok=True)
                url = _resolve_windows_embed_url(version)
                print_status(f"Streaming fallback Windows embed runtime ({version})...")
                _stream_download(url, archive_path)
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(target_dir)
            (target_dir / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
            for pth_file in target_dir.glob("*._pth"):
                orig_lines = pth_file.read_text(encoding="utf-8").splitlines()
                core_zips = [
                    line.strip()
                    for line in orig_lines
                    if line.strip().endswith(".zip")
                ]
                payload_paths = core_zips + [
                    "Lib/site-packages",
                    ".",
                    "..",
                    "import site",
                ]
                pth_file.write_text("\n".join(payload_paths) + "\n", encoding="utf-8")
        executable = target_dir / "python.exe"
    else:
        if target_dir.exists():
            safe_rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        triple = _posix_target_triple()
        tag, prefix, assets = _query_pbs_release_metadata()
        stripped_name = (
            f"cpython-{version}+{tag}-{triple}-install_only_stripped.tar.gz"
        )
        std_name = f"cpython-{version}+{tag}-{triple}-install_only.tar.gz"
        if stripped_name in assets:
            archive_name = stripped_name
            url, sha256_hex = assets[stripped_name]
        elif std_name in assets:
            archive_name = std_name
            url, sha256_hex = assets[std_name]
        else:
            archive_name = std_name
            url = f"{prefix}/{std_name}"
            sha256_hex = ""

        archive_path = DOWNLOADS_DIR / archive_name
        if not verify_archive_integrity(archive_path):
            archive_path.unlink(missing_ok=True)
            print_status(
                f"Streaming standalone POSIX runtime ({version} [{tag}])..."
            )
            _stream_download(url, archive_path, expected_sha256=sha256_hex)

        staging_extract = TMP_DIR / f"pbs_extract_{os.getpid()}"
        safe_rmtree(staging_extract)
        staging_extract.mkdir(parents=True, exist_ok=True)
        _extract_tarball_trusted(archive_path, staging_extract)
        source_extracted = staging_extract / "python"
        if target_dir.exists():
            safe_rmtree(target_dir)
        shutil.move(str(source_extracted), str(target_dir))
        safe_rmtree(staging_extract)
        executable = (
            target_dir / "bin" / "python3"
            if (target_dir / "bin" / "python3").is_file()
            else target_dir / "bin" / "python"
        )

    site_packages = _site_packages_dir(target_dir)
    if site_packages is None or not (site_packages / "pip").is_dir():
        hermetic_env = build_hermetic_env(target_dir)
        pip_bootstrapper = TMP_DIR / "get-pip.py"
        v_parts = version.split(".")
        if (
            len(v_parts) >= 2
            and v_parts[0] == "3"
            and v_parts[1] in ("6", "7", "8", "9")
        ):
            pip_url = (
                f"https://bootstrap.pypa.io/pip/{v_parts[0]}.{v_parts[1]}/get-pip.py"
            )
        else:
            pip_url = "https://bootstrap.pypa.io/get-pip.py"
        _stream_download(pip_url, pip_bootstrapper)
        subprocess.run(
            [
                str(executable),
                "-I",
                "-S",
                str(pip_bootstrapper),
                "--no-warn-script-location",
                "--cache-dir",
                str(PIP_CACHE_DIR),
            ],
            env=hermetic_env,
            stdin=subprocess.DEVNULL,
            check=True,
        )
        pip_bootstrapper.unlink(missing_ok=True)

    return executable


_SIMPLE_INDEX_CACHE: dict[tuple[str, str], tuple[bool, list[str]]] = {}


def _norm_dist_name(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).strip("_").lower()


def _wheel_slot_key(filename: str) -> tuple[str, str, str, str] | None:
    lower = filename.lower()
    if lower.endswith(".whl"):
        parts = filename[:-4].split("-")
        if len(parts) >= 5:
            return (
                _norm_dist_name(parts[0]),
                parts[-3].lower(),
                parts[-2].lower(),
                parts[-1].lower(),
            )
    elif lower.endswith(".tar.gz"):
        m = re.match(r"^([A-Za-z0-9_.-]+?)-(\d.*)$", filename[:-7])
        if m:
            return (_norm_dist_name(m.group(1)), "sdist", "none", "any")
    return None


def _is_compute_dist_name(dist_name: str) -> bool:
    norm = _norm_dist_name(dist_name)
    if norm in NON_COMPUTE_DIST_NAMES:
        return False
    return norm.startswith(COMPUTE_DIST_PREFIXES)


def _is_compute_archive_name(filename: str) -> bool:
    slot = _wheel_slot_key(filename)
    if slot is not None:
        return _is_compute_dist_name(slot[0])
    norm = filename.lower().replace("-", "_")
    if any(norm.startswith(nc + "_") for nc in NON_COMPUTE_DIST_NAMES):
        return False
    return norm.startswith(COMPUTE_DIST_PREFIXES)


def audit_and_prune_dir(pool_dir: Path) -> int:
    pool_dir.mkdir(parents=True, exist_ok=True)
    evicted = 0
    for part_file in pool_dir.glob("*.part"):
        part_file.unlink(missing_ok=True)
    for pkg_file in sorted(pool_dir.iterdir()):
        if not pkg_file.is_file():
            continue
        if not verify_archive_integrity(pkg_file):
            try:
                remove_readonly(None, str(pkg_file), None)
                pkg_file.unlink(missing_ok=True)
            except Exception:
                pass
            evicted += 1
    return evicted


def audit_and_prune_wheels_dir() -> int:
    evicted = audit_and_prune_dir(WHEELS_DIR)
    evicted += audit_and_prune_dir(DEPS_WHEELS_DIR)
    for stray in sorted(DEPS_WHEELS_DIR.iterdir()):
        if stray.is_file() and _is_compute_archive_name(stray.name):
            dest = WHEELS_DIR / stray.name
            if not dest.exists():
                shutil.move(str(stray), str(dest))
            else:
                stray.unlink(missing_ok=True)
    for non_comp in sorted(WHEELS_DIR.iterdir()):
        if non_comp.is_file() and not _is_compute_archive_name(non_comp.name):
            dest = DEPS_WHEELS_DIR / non_comp.name
            if not dest.exists():
                shutil.move(str(non_comp), str(dest))
            else:
                non_comp.unlink(missing_ok=True)
    return evicted


def salvage_cached_wheels() -> None:
    audit_and_prune_wheels_dir()

    for whl in PIP_CACHE_DIR.rglob("*.whl"):
        target_pool = (
            WHEELS_DIR if _is_compute_archive_name(whl.name) else DEPS_WHEELS_DIR
        )
        dest = target_pool / whl.name
        if not dest.exists():
            try:
                shutil.copy2(whl, dest)
                if not verify_archive_integrity(dest):
                    dest.unlink(missing_ok=True)
                    whl.unlink(missing_ok=True)
            except Exception:
                dest.unlink(missing_ok=True)

    http_cache = PIP_CACHE_DIR / "http-v2"
    if not http_cache.exists():
        return
    for body_file in http_cache.rglob("*"):
        if not body_file.is_file() or body_file.stat().st_size < 4096:
            continue
        try:
            with open(body_file, "rb") as f:
                magic = f.read(4)
            if magic != b"PK\x03\x04":
                continue
            with zipfile.ZipFile(body_file, "r") as zf:
                bad_member = zf.testzip()
                if bad_member is not None:
                    print_status(
                        f"Evicting corrupt pip HTTP cache entry ({bad_member}).",
                        "WARN",
                    )
                    remove_readonly(None, str(body_file), None)
                    body_file.unlink(missing_ok=True)
                    continue
                dist_infos = [
                    n for n in zf.namelist() if n.endswith(".dist-info/WHEEL")
                ]
                if not dist_infos:
                    continue
                pkg_prefix = dist_infos[0].split(".dist-info/")[0]
                wheel_meta = zf.read(dist_infos[0]).decode("utf-8", errors="ignore")
                tags = re.findall(r"Tag:\s*(\S+)", wheel_meta)
                tag_str = tags[0] if tags else "py3-none-any"
                wheel_name = f"{pkg_prefix}-{tag_str}.whl"
                target_pool = (
                    WHEELS_DIR
                    if _is_compute_archive_name(wheel_name)
                    else DEPS_WHEELS_DIR
                )
                dest = target_pool / wheel_name
                if not dest.exists():
                    shutil.copy2(body_file, dest)
                    st = dest.stat()
                    _VERIFIED_ARCHIVE_SIGNATURES.add(
                        (str(dest.resolve()), st.st_size, st.st_mtime_ns)
                    )
                    if st.st_size >= 10 * 1024 * 1024:
                        print_status(
                            f"Recovered verified cached wheel: {wheel_name} "
                            f"({st.st_size / (1024**2):.1f} MB)",
                            "SUCCESS",
                        )
        except zipfile.BadZipFile:
            remove_readonly(None, str(body_file), None)
            body_file.unlink(missing_ok=True)
        except Exception:
            pass


def quarantine_sandbox_sitecustomize(target_sandbox: Path) -> None:
    sp = _site_packages_dir(target_sandbox)
    if sp is None or not sp.is_dir():
        return
    sc = sp / "sitecustomize.py"
    if sc.is_file():
        sc.unlink(missing_ok=True)
    pycache = sp / "__pycache__"
    if pycache.is_dir():
        for pyc in pycache.glob("sitecustomize.*.pyc"):
            pyc.unlink(missing_ok=True)


def _installed_distributions(target_sandbox: Path) -> dict[str, str]:
    sp = _site_packages_dir(target_sandbox)
    if sp is None or not sp.is_dir():
        return {}
    installed: dict[str, str] = {}
    for dist_info in sp.glob("*.dist-info"):
        meta = dist_info / "METADATA"
        if not meta.is_file():
            continue
        pkg_name = ""
        pkg_ver = ""
        try:
            with open(meta, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line == "\n" or line == "\r\n":
                        break
                    if line.startswith("Name:"):
                        pkg_name = line.split(":", 1)[1].strip()
                    elif line.startswith("Version:"):
                        pkg_ver = line.split(":", 1)[1].strip()
                    if pkg_name and pkg_ver:
                        break
        except OSError:
            continue
        if pkg_name and pkg_ver:
            installed[_norm_dist_name(pkg_name)] = pkg_ver
    return installed


def _extra_required_dists(
    target_sandbox: Path, dist_norm: str, extra_name: str
) -> set[str]:
    sp = _site_packages_dir(target_sandbox)
    if sp is None or not sp.is_dir() or not extra_name:
        return set()

    extra_pat = re.compile(
        rf"""extra\s*==\s*['"]{re.escape(extra_name)}['"]""", flags=re.I
    )
    required: set[str] = set()
    for dist_info in sp.glob("*.dist-info"):
        prefix = dist_info.name[:-10].split("-", 1)[0]
        if _norm_dist_name(prefix) != _norm_dist_name(dist_norm):
            continue
        meta = dist_info / "METADATA"
        if not meta.is_file():
            continue
        try:
            with open(meta, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line in ("\n", "\r\n"):
                        break
                    if not line.startswith("Requires-Dist:"):
                        continue
                    body = line.split(":", 1)[1].strip()
                    if ";" not in body:
                        continue
                    req_part, marker_part = body.split(";", 1)
                    if not extra_pat.search(marker_part):
                        continue
                    m_low = marker_part.lower()
                    if "linux" in m_low and not sys.platform.startswith("linux"):
                        continue
                    if ("win32" in m_low or "windows" in m_low) and os.name != "nt":
                        continue
                    m_pkg = re.match(r"^([A-Za-z0-9_.-]+)", req_part.strip())
                    if m_pkg:
                        required.add(_norm_dist_name(m_pkg.group(1)))
        except OSError:
            continue
    return required


def _read_hw_prefix_manifest(target_sandbox: Path) -> dict:
    manifest_file = target_sandbox / "hw_prefix" / "manifest.json"
    if not manifest_file.is_file():
        return {}
    try:
        return json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_rocm_stamp(ver: str) -> str:
    m = re.search(r"(?:\+|\.|^|-)rocm([0-9][0-9A-Za-z._-]*)", ver, flags=re.I)
    return m.group(1) if m else ""


def _is_closure_converged(
    target_sandbox: Path,
    profile: str,
    expected_versions: dict[str, str],
    required_extras: dict[str, str] | None = None,
    expected_target_family: str = "",
    expected_stamp: str = "",
) -> bool:
    if not expected_versions:
        return False
    installed = _installed_distributions(target_sandbox)
    for norm_pkg, exp_ver in expected_versions.items():
        if installed.get(norm_pkg) != exp_ver:
            return False

    permitted_transitive: set[str] = set()
    if required_extras:
        for base_pkg, extra_name in required_extras.items():
            extra_reqs = _extra_required_dists(
                target_sandbox, base_pkg, extra_name
            )
            for req_norm in extra_reqs:
                if req_norm not in installed:
                    return False
                permitted_transitive.add(req_norm)

    norm_exp_stamp = _norm_stamp(expected_stamp) if expected_stamp else ""
    for inst_name, inst_ver in installed.items():
        if inst_name.startswith(("rocm_sdk_libraries", "rocm_sdk_device")):
            if (
                inst_name not in expected_versions
                and inst_name not in permitted_transitive
            ):
                return False
        elif inst_name.startswith(
            ("amd_torch_device", "amd_torchvision_device", "amd_torchaudio_device")
        ):
            if (
                inst_name not in expected_versions
                and inst_name not in permitted_transitive
            ):
                return False
            if norm_exp_stamp:
                inst_st = _norm_stamp(_extract_rocm_stamp(inst_ver) or inst_ver)
                if inst_st != norm_exp_stamp:
                    return False

    if profile == "ROCM" and "rocm_sdk_devel" in expected_versions:
        sp = _site_packages_dir(target_sandbox)
        if sp is None or not (sp / "_rocm_sdk_devel" / "bin").is_dir():
            return False
        manifest = _read_hw_prefix_manifest(target_sandbox)
        if expected_target_family and manifest.get("target_family") not in (
            "",
            expected_target_family,
        ):
            return False
    return True


def reconcile_compute_closure(
    executable: Path,
    isolated_env: dict[str, str],
    target_sandbox: Path,
    keep_versions: dict[str, str] | None = None,
    required_extras: dict[str, str] | None = None,
) -> None:
    keep = keep_versions or {}
    installed = _installed_distributions(target_sandbox)
    permitted_transitive: set[str] = set()
    if required_extras:
        for base_pkg, extra_name in required_extras.items():
            if base_pkg in installed and installed.get(base_pkg) == keep.get(base_pkg):
                permitted_transitive.update(
                    _extra_required_dists(target_sandbox, base_pkg, extra_name)
                )

    stale_pkgs: list[str] = []
    rocm_mutated = False

    for dist_name, dist_ver in sorted(installed.items()):
        if not _is_compute_dist_name(dist_name):
            continue
        if dist_name in ("triton", "pytorch_triton_rocm") and dist_name not in keep:
            if "torch" in keep and installed.get("torch") == keep.get("torch"):
                continue
        if dist_name in permitted_transitive and dist_name not in keep:
            parent_ver = keep.get("torch") or ""
            if parent_ver and (
                dist_ver == parent_ver
                or _norm_stamp(_extract_rocm_stamp(dist_ver) or dist_ver)
                == _norm_stamp(_extract_rocm_stamp(parent_ver) or parent_ver)
            ):
                continue
        if keep.get(dist_name) != dist_ver:
            stale_pkgs.append(dist_name.replace("_", "-"))
            if dist_name.startswith(("rocm", "_rocm", "amd_rocm")):
                rocm_mutated = True

    if stale_pkgs:
        print_status(
            f"Purging divergent compute distributions: {', '.join(stale_pkgs)}",
            "INFO",
        )
        subprocess.run(
            [
                str(executable),
                "-I",
                "-m",
                "pip",
                "uninstall",
                "-y",
                *stale_pkgs,
            ],
            env=isolated_env,
            stdin=subprocess.DEVNULL,
            check=False,
        )

    sp = _site_packages_dir(target_sandbox)
    if rocm_mutated and sp is not None:
        safe_rmtree(sp / "_rocm_sdk_devel")
        safe_rmtree(target_sandbox / "hw_prefix")


def _seed_exact_cached_wheels(
    tx_dir: Path,
    exact_filenames: list[str],
    stamp_hint: str = "",
) -> None:
    wanted = set(exact_filenames)
    norm_hint = _norm_stamp(stamp_hint) if stamp_hint else ""
    for cached_file in WHEELS_DIR.iterdir():
        if not cached_file.is_file():
            continue
        match = cached_file.name in wanted
        if (
            not match
            and norm_hint
            and cached_file.name.lower().endswith((".whl", ".tar.gz"))
            and _is_compute_archive_name(cached_file.name)
        ):
            if norm_hint in _norm_stamp(cached_file.name):
                match = True
        if match and verify_archive_integrity(cached_file):
            dest = tx_dir / cached_file.name
            if not dest.exists():
                try:
                    os.link(cached_file, dest)
                except OSError:
                    shutil.copy2(cached_file, dest)


def _promote_verified_transaction(tx_dir: Path, dest_pool: Path) -> None:
    dest_pool.mkdir(parents=True, exist_ok=True)
    for staged_file in sorted(tx_dir.iterdir()):
        if not staged_file.is_file():
            continue
        target_pool = (
            WHEELS_DIR
            if (dest_pool == WHEELS_DIR and _is_compute_archive_name(staged_file.name))
            else DEPS_WHEELS_DIR
        )
        slot = _wheel_slot_key(staged_file.name)
        if slot is not None:
            for existing in list(target_pool.iterdir()):
                if (
                    existing.is_file()
                    and existing.name != staged_file.name
                    and _wheel_slot_key(existing.name) == slot
                ):
                    try:
                        remove_readonly(None, str(existing), None)
                        existing.unlink(missing_ok=True)
                    except Exception:
                        pass
        final_dest = target_pool / staged_file.name
        if not final_dest.exists():
            shutil.copy2(staged_file, final_dest)
            st = final_dest.stat()
            _VERIFIED_ARCHIVE_SIGNATURES.add(
                (str(final_dest.resolve()), st.st_size, st.st_mtime_ns)
            )


def _run_pip_streaming(cmd: list[str], env: dict[str, str]) -> tuple[int, str]:
    captured: list[str] = []
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)
    rc = proc.wait()
    return (rc, "".join(captured))


def _has_deterministic_pip_fault(output: str) -> bool:
    return any(marker in output for marker in DETERMINISTIC_PIP_ERROR_MARKERS)


def stage_and_install_packages(
    executable: Path,
    isolated_env: dict[str, str],
    packages: list[str],
    index_url: str | None = None,
    extra_index_url: str | None = None,
    fallback_pypi: bool = False,
    use_pre: bool = True,
    no_build_isolation: bool = False,
    is_deps_stage: bool = False,
    exact_filenames: list[str] | None = None,
    stamp_hint: str = "",
    max_attempts: int = 3,
) -> None:
    active_extra_url = extra_index_url
    pre_args: list[str] = ["--pre"] if use_pre else []
    iso_args: list[str] = ["--no-build-isolation"] if no_build_isolation else []
    dest_pool = DEPS_WHEELS_DIR if is_deps_stage else WHEELS_DIR

    for attempt in range(1, max_attempts + 1):
        index_args: list[str] = ["--index-url", index_url] if index_url else []
        if active_extra_url and active_extra_url != index_url:
            index_args.extend(["--extra-index-url", active_extra_url])

        audit_and_prune_wheels_dir()
        tx_dir = (
            TMP_DIR / f"wheel_tx_{os.getpid()}_{int(time.time() * 1000)}_{attempt}"
        )
        safe_rmtree(tx_dir)
        tx_dir.mkdir(parents=True, exist_ok=True)

        try:
            if not is_deps_stage and (exact_filenames or stamp_hint):
                _seed_exact_cached_wheels(
                    tx_dir, exact_filenames or [], stamp_hint=stamp_hint
                )

            find_links_args = [
                "--find-links",
                str(tx_dir),
                "--find-links",
                str(DEPS_WHEELS_DIR),
            ]
            dl_cmd = (
                [
                    str(executable),
                    "-I",
                    "-m",
                    "pip",
                    "download",
                    "--dest",
                    str(tx_dir),
                    "--cache-dir",
                    str(PIP_CACHE_DIR),
                    "--progress-bar",
                    "off",
                    "--retries",
                    "8",
                    "--timeout",
                    "120",
                ]
                + find_links_args
                + index_args
                + pre_args
                + iso_args
                + packages
            )
            dl_rc, dl_out = _run_pip_streaming(dl_cmd, isolated_env)
            evicted_tx = audit_and_prune_dir(tx_dir)
            if dl_rc != 0 or evicted_tx > 0:
                if (
                    dl_rc != 0
                    and evicted_tx == 0
                    and fallback_pypi
                    and active_extra_url is None
                    and "No matching distribution found" in dl_out
                    and attempt < max_attempts
                ):
                    active_extra_url = "https://pypi.org/simple"
                    print_status(
                        "Transitive pure-Python dependency not in local pool; enabling PyPI fallback...",
                        "INFO",
                    )
                    continue
                if (
                    dl_rc != 0
                    and evicted_tx == 0
                    and _has_deterministic_pip_fault(dl_out)
                ):
                    raise subprocess.CalledProcessError(dl_rc, dl_cmd)
                if attempt < max_attempts:
                    print_status(
                        f"Wheel staging attempt {attempt}/{max_attempts} encountered "
                        f"transient/CRC fault ({evicted_tx} evicted). Retrying...",
                        "WARN",
                    )
                    time.sleep(2.0 * attempt)
                    continue
                raise subprocess.CalledProcessError(dl_rc or 1, dl_cmd)

            inst_cmd = (
                [
                    str(executable),
                    "-I",
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--find-links",
                    str(tx_dir),
                    "--find-links",
                    str(DEPS_WHEELS_DIR),
                    "--no-warn-script-location",
                    "--no-compile",
                    "--progress-bar",
                    "off",
                    "--no-cache-dir",
                    "-U",
                ]
                + pre_args
                + iso_args
                + packages
            )
            inst_rc, inst_out = _run_pip_streaming(inst_cmd, isolated_env)
            if inst_rc == 0:
                _promote_verified_transaction(tx_dir, dest_pool)
                return

            if _has_deterministic_pip_fault(inst_out) or attempt >= max_attempts:
                raise subprocess.CalledProcessError(inst_rc, inst_cmd)

            print_status(
                f"Package installation attempt {attempt}/{max_attempts} failed. Retrying...",
                "WARN",
            )
            time.sleep(2.0 * attempt)
        finally:
            safe_rmtree(tx_dir)


def _fetch_simple_index_filenames(
    base_url: str, project: str
) -> tuple[bool, list[str]]:
    base = base_url.rstrip("/")
    cache_key = (base.lower(), _norm_dist_name(project))
    cached = _SIMPLE_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    raw_hyph = re.sub(r"[-_.]+", "-", project)
    slug = raw_hyph.lower()
    alt_slug = slug.replace("-", "_")
    urls = [f"{base}/{slug}/", f"{base}/{slug}/index.html"]
    if alt_slug != slug:
        urls.append(f"{base}/{alt_slug}/")
    if raw_hyph != slug:
        urls.extend([f"{base}/{raw_hyph}/", f"{base}/{raw_hyph}/index.html"])

    for url in urls:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if not (200 <= resp.status < 400):
                    continue
                html = resp.read().decode("utf-8", errors="ignore")
            raw_hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I)
            names: list[str] = []
            seen: set[str] = set()
            for href in raw_hrefs:
                clean = urllib.parse.unquote(href.split("#", 1)[0].split("?", 1)[0])
                fname = posixpath.basename(clean.rstrip("/"))
                if fname.lower().endswith((".whl", ".tar.gz")) and fname not in seen:
                    seen.add(fname)
                    names.append(fname)
            res = (True, names)
            _SIMPLE_INDEX_CACHE[cache_key] = res
            return res
        except Exception:
            continue

    res_fail: tuple[bool, list[str]] = (False, [])
    _SIMPLE_INDEX_CACHE[cache_key] = res_fail
    return res_fail


def _fetch_index_subdirs(base_url: str) -> list[str]:
    base = base_url.rstrip("/") + "/"
    req = urllib.request.Request(base, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            if not (200 <= resp.status < 400):
                return []
            html = resp.read().decode("utf-8", errors="ignore")
        hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I)
        out: list[str] = []
        seen: set[str] = set()
        for h in hrefs:
            clean = h.split("#", 1)[0].split("?", 1)[0].strip("/")
            seg = posixpath.basename(clean)
            if seg and seg not in seen:
                seen.add(seg)
                out.append(seg)
        return out
    except Exception:
        return []


def _platform_wheel_ok(plat_tags: list[str]) -> bool:
    if "any" in plat_tags:
        return True
    if os.name == "nt":
        _, _, target_plat = _windows_arch_tags()
        return target_plat in plat_tags
    if sys.platform.startswith("linux"):
        mach = platform.machine().lower()
        target_arch = "aarch64" if mach in ("aarch64", "arm64") else "x86_64"
        return any(
            pt.endswith(target_arch)
            and ("linux" in pt or "manylinux" in pt)
            and "musllinux" not in pt
            for pt in plat_tags
        )
    if sys.platform == "darwin":
        mach = platform.machine().lower()
        valid_archs = (
            ("arm64", "universal2")
            if mach in ("arm64", "aarch64")
            else ("x86_64", "universal2")
        )
        return any(
            pt.startswith("macosx_") and pt.endswith(valid_archs) for pt in plat_tags
        )
    return False


def _is_wheel_compatible(
    filename: str,
    expected_dist: str,
    py_major: int,
    py_minor: int,
    freethreaded: bool | None = None,
    require_platform_binary: bool = False,
) -> str | None:
    if not filename.lower().endswith(".whl"):
        return None
    parts = filename[:-4].split("-")
    if len(parts) < 5:
        return None
    if _norm_dist_name(parts[0]) != _norm_dist_name(expected_dist):
        return None

    if freethreaded is None:
        freethreaded = _is_freethreaded_runtime()

    version = parts[1]
    py_tags = parts[-3].lower().split(".")
    abi_tags = parts[-2].lower().split(".")
    plat_tags = parts[-1].lower().split(".")

    if require_platform_binary and "any" in plat_tags:
        return None

    cp_tag = f"cp{py_major}{py_minor}"
    abi_ok = False
    if "none" in abi_tags and (
        f"py{py_major}" in py_tags
        or f"py{py_major}{py_minor}" in py_tags
        or cp_tag in py_tags
    ):
        abi_ok = True
    elif "abi3" in abi_tags and not freethreaded:
        for pt in py_tags:
            if pt.startswith(f"cp{py_major}") and pt[3:].isdigit():
                if int(pt[3:]) <= py_minor:
                    abi_ok = True
                    break
    elif cp_tag in py_tags:
        if freethreaded:
            abi_ok = f"{cp_tag}t" in abi_tags
        else:
            abi_ok = cp_tag in abi_tags or f"{cp_tag}m" in abi_tags

    if not abi_ok or not _platform_wheel_ok(plat_tags):
        return None
    return version


def _probe_compatible_minors_in_files(
    filenames: list[str],
    expected_dist: str,
    candidate_minors: list[tuple[int, int]],
    require_platform_binary: bool = False,
) -> set[tuple[int, int]]:
    supported: set[tuple[int, int]] = set()
    for maj, min_v in candidate_minors:
        if any(
            _is_wheel_compatible(
                fn,
                expected_dist,
                maj,
                min_v,
                freethreaded=False,
                require_platform_binary=require_platform_binary,
            )
            is not None
            for fn in filenames
        ):
            supported.add((maj, min_v))
    return supported


def _extract_compatible_wheel_map(
    filenames: list[str],
    expected_dist: str,
    py_major: int,
    py_minor: int,
    freethreaded: bool | None = None,
) -> dict[str, str]:
    out: dict[str, str] = {}
    for fn in filenames:
        ver = _is_wheel_compatible(
            fn, expected_dist, py_major, py_minor, freethreaded=freethreaded
        )
        if ver is not None:
            out[ver] = fn
    return out


def _extract_rocm_meta_map(
    filenames: list[str],
    py_major: int,
    py_minor: int,
    freethreaded: bool | None = None,
) -> dict[str, str]:
    out: dict[str, str] = {}
    for fn in filenames:
        if fn.lower().endswith(".tar.gz"):
            m = re.match(r"^rocm-(\d[^\s/]+?)\.tar\.gz$", fn, flags=re.I)
            if m:
                out[m.group(1)] = fn
        elif fn.lower().endswith(".whl"):
            ver = _is_wheel_compatible(
                fn, "rocm", py_major, py_minor, freethreaded=freethreaded
            )
            if ver is not None:
                out[ver] = fn
    return out


def _build_rocm_cascade_tiers(llvm_target: str) -> list[tuple[str, str]]:
    family = AMD_STAGING_FAMILY_MAP.get(llvm_target, f"{llvm_target}-all")
    default_tiers = [
        ("whl-next-nightly", "https://nightly.repo.amd.com/rocm/whl-next/"),
        ("whl-next-rc", "https://rc.repo.amd.com/rocm/whl-next/"),
        ("whl-next-stable", "https://stable.repo.amd.com/rocm/whl-next/"),
        ("whl-multi-arch", "https://rocm.nightlies.amd.com/whl-multi-arch/"),
        (
            "v2-staging-family",
            f"https://rocm.nightlies.amd.com/v2-staging/{family}/",
        ),
    ]
    custom_url = os.environ.get("ROCM_INDEX_URL", "").strip()
    if not custom_url:
        return default_tiers

    tiers: list[tuple[str, str]] = [("custom-rocm-index", custom_url)]
    norm_custom = custom_url.rstrip("/").lower()
    for name, url in default_tiers:
        if url.rstrip("/").lower() != norm_custom:
            tiers.append((name, url))
    return tiers


def _discover_cuda_feed_candidates(ceiling_str: str) -> list[tuple[str, str, bool]]:
    max_pair = (99, 9)
    if ceiling_str and "." in ceiling_str:
        parts = ceiling_str.split(".", 1)
        if parts[0].isdigit() and parts[1].isdigit():
            max_pair = (int(parts[0]), int(parts[1]))

    discovered_tags: dict[tuple[int, int], str] = {
        (13, 4): "cu134",
        (13, 2): "cu132",
        (13, 0): "cu130",
        (12, 9): "cu129",
        (12, 8): "cu128",
        (12, 6): "cu126",
        (12, 4): "cu124",
        (12, 1): "cu121",
        (11, 8): "cu118",
    }
    for sub in _fetch_index_subdirs("https://download.pytorch.org/whl/nightly"):
        m = re.match(r"^cu(\d{2})(\d+)$", sub.lower())
        if m:
            discovered_tags[(int(m.group(1)), int(m.group(2)))] = sub.lower()

    eligible = sorted(
        (pair for pair in discovered_tags if pair <= max_pair),
        reverse=True,
    )
    feeds: list[tuple[str, str, bool]] = []
    for pair in eligible[:5]:
        tag = discovered_tags[pair]
        feeds.append(
            (f"{tag}-nightly", f"https://download.pytorch.org/whl/nightly/{tag}", True)
        )
        feeds.append(
            (f"{tag}-stable", f"https://download.pytorch.org/whl/{tag}", False)
        )
    return feeds


def probe_rocm_tier_synchronized_plan(
    tier_name: str,
    tier_url: str,
    llvm_target: str,
    py_maj: int | None = None,
    py_min: int | None = None,
    freethreaded: bool | None = None,
) -> dict | None:
    if py_maj is None or py_min is None:
        py_maj, py_min = sys.version_info.major, sys.version_info.minor
    if freethreaded is None:
        freethreaded = _is_freethreaded_runtime()

    family = (
        AMD_STAGING_FAMILY_MAP.get(llvm_target, f"{llvm_target}-all")
        if llvm_target
        else ""
    )
    pf_libs_pkg = f"rocm-sdk-libraries-{family}" if family else ""
    dev_primary_pkg = f"rocm-sdk-device-{llvm_target}" if llvm_target else ""
    fb_target = LEGACY_KPACK_FALLBACK_MAP.get(llvm_target, "")
    dev_fb_pkg = (
        f"rocm-sdk-device-{fb_target}"
        if (fb_target and fb_target != llvm_target)
        else ""
    )

    atd_prim_pkg = (
        f"amd-torch-device-{llvm_target}"
        if llvm_target
        else "amd-torch-device-all"
    )
    atd_fb_pkg = f"amd-torch-device-{fb_target}" if dev_fb_pkg else ""
    atvd_prim_pkg = (
        f"amd-torchvision-device-{llvm_target}"
        if llvm_target
        else "amd-torchvision-device-all"
    )
    atvd_fb_pkg = f"amd-torchvision-device-{fb_target}" if dev_fb_pkg else ""
    atad_prim_pkg = (
        f"amd-torchaudio-device-{llvm_target}"
        if llvm_target
        else "amd-torchaudio-device-all"
    )
    atad_fb_pkg = f"amd-torchaudio-device-{fb_target}" if dev_fb_pkg else ""

    projects = [
        "rocm",
        "rocm-sdk-core",
        "rocm-sdk-devel",
        "rocm-sdk-libraries",
        "torch",
        "torchvision",
        "torchaudio",
        "triton",
        "pytorch-triton-rocm",
    ]
    for opt_pkg in (
        pf_libs_pkg,
        dev_primary_pkg,
        dev_fb_pkg,
        atd_prim_pkg,
        atd_fb_pkg,
        atvd_prim_pkg,
        atvd_fb_pkg,
        atad_prim_pkg,
        atad_fb_pkg,
    ):
        if opt_pkg and opt_pkg not in projects:
            projects.append(opt_pkg)

    fetched: dict[str, tuple[bool, list[str]]] = {}
    with ThreadPoolExecutor(max_workers=min(14, len(projects))) as pool:
        futs = {
            proj: pool.submit(_fetch_simple_index_filenames, tier_url, proj)
            for proj in projects
        }
        for proj, fut in futs.items():
            fetched[proj] = fut.result()

    torch_ok, torch_files = fetched["torch"]
    if not torch_ok:
        return {"status": "unreachable", "tier_name": tier_name, "tier_url": tier_url}

    torch_map = _extract_compatible_wheel_map(
        torch_files, "torch", py_maj, py_min, freethreaded=freethreaded
    )
    tv_map = _extract_compatible_wheel_map(
        fetched["torchvision"][1],
        "torchvision",
        py_maj,
        py_min,
        freethreaded=freethreaded,
    )
    ta_map = _extract_compatible_wheel_map(
        fetched["torchaudio"][1],
        "torchaudio",
        py_maj,
        py_min,
        freethreaded=freethreaded,
    )
    if not torch_map or not tv_map or not ta_map:
        return {"status": "incompatible", "tier_name": tier_name, "tier_url": tier_url}

    rocm_map = _extract_rocm_meta_map(
        fetched["rocm"][1], py_maj, py_min, freethreaded=freethreaded
    )
    core_map = _extract_compatible_wheel_map(
        fetched["rocm-sdk-core"][1],
        "rocm_sdk_core",
        py_maj,
        py_min,
        freethreaded=freethreaded,
    )
    devel_map = _extract_compatible_wheel_map(
        fetched["rocm-sdk-devel"][1],
        "rocm_sdk_devel",
        py_maj,
        py_min,
        freethreaded=freethreaded,
    )
    ma_libs_map = _extract_compatible_wheel_map(
        fetched["rocm-sdk-libraries"][1],
        "rocm_sdk_libraries",
        py_maj,
        py_min,
        freethreaded=freethreaded,
    )
    pf_libs_map = (
        _extract_compatible_wheel_map(
            fetched[pf_libs_pkg][1],
            pf_libs_pkg,
            py_maj,
            py_min,
            freethreaded=freethreaded,
        )
        if pf_libs_pkg
        else {}
    )
    dev_primary_map = (
        _extract_compatible_wheel_map(
            fetched[dev_primary_pkg][1],
            dev_primary_pkg,
            py_maj,
            py_min,
            freethreaded=freethreaded,
        )
        if dev_primary_pkg
        else {}
    )
    dev_fb_map = (
        _extract_compatible_wheel_map(
            fetched[dev_fb_pkg][1],
            dev_fb_pkg,
            py_maj,
            py_min,
            freethreaded=freethreaded,
        )
        if dev_fb_pkg
        else {}
    )

    atd_prim_map = _extract_compatible_wheel_map(
        fetched[atd_prim_pkg][1],
        atd_prim_pkg,
        py_maj,
        py_min,
        freethreaded=freethreaded,
    )
    atd_fb_map = (
        _extract_compatible_wheel_map(
            fetched[atd_fb_pkg][1],
            atd_fb_pkg,
            py_maj,
            py_min,
            freethreaded=freethreaded,
        )
        if atd_fb_pkg
        else {}
    )
    atvd_prim_map = _extract_compatible_wheel_map(
        fetched[atvd_prim_pkg][1],
        atvd_prim_pkg,
        py_maj,
        py_min,
        freethreaded=freethreaded,
    )
    atvd_fb_map = (
        _extract_compatible_wheel_map(
            fetched[atvd_fb_pkg][1],
            atvd_fb_pkg,
            py_maj,
            py_min,
            freethreaded=freethreaded,
        )
        if atvd_fb_pkg
        else {}
    )
    atad_prim_map = _extract_compatible_wheel_map(
        fetched[atad_prim_pkg][1],
        atad_prim_pkg,
        py_maj,
        py_min,
        freethreaded=freethreaded,
    )
    atad_fb_map = (
        _extract_compatible_wheel_map(
            fetched[atad_fb_pkg][1],
            atad_fb_pkg,
            py_maj,
            py_min,
            freethreaded=freethreaded,
        )
        if atad_fb_pkg
        else {}
    )

    triton_map = _extract_compatible_wheel_map(
        fetched["triton"][1],
        "triton",
        py_maj,
        py_min,
        freethreaded=freethreaded,
    )
    pt_triton_map = _extract_compatible_wheel_map(
        fetched["pytorch-triton-rocm"][1],
        "pytorch_triton_rocm",
        py_maj,
        py_min,
        freethreaded=freethreaded,
    )

    has_any_rocm_sdk = bool(rocm_map or core_map or devel_map)
    index_uses_torch_kpacks = bool(
        fetched[atd_prim_pkg][1] or (atd_fb_pkg and fetched[atd_fb_pkg][1])
    )
    index_uses_tv_kpacks = bool(
        fetched[atvd_prim_pkg][1] or (atvd_fb_pkg and fetched[atvd_fb_pkg][1])
    )
    index_uses_ta_kpacks = bool(
        fetched[atad_prim_pkg][1] or (atad_fb_pkg and fetched[atad_fb_pkg][1])
    )

    def _by_stamp(
        vmap: dict[str, str], require_rocm_tag: bool = False
    ) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for ver in vmap:
            extracted = _extract_rocm_stamp(ver)
            raw_s = extracted if extracted else ("" if require_rocm_tag else ver)
            if raw_s:
                grouped.setdefault(_norm_stamp(raw_s), []).append(ver)
        return grouped

    rocm_by_s = _by_stamp(rocm_map)
    core_by_s = _by_stamp(core_map)
    devel_by_s = _by_stamp(devel_map)
    ma_libs_by_s = _by_stamp(ma_libs_map)
    pf_libs_by_s = _by_stamp(pf_libs_map)
    dev_prim_by_s = _by_stamp(dev_primary_map)
    dev_fb_by_s = _by_stamp(dev_fb_map)
    atd_prim_by_s = _by_stamp(atd_prim_map)
    atd_fb_by_s = _by_stamp(atd_fb_map)
    atvd_prim_by_s = _by_stamp(atvd_prim_map)
    atvd_fb_by_s = _by_stamp(atvd_fb_map)
    atad_prim_by_s = _by_stamp(atad_prim_map)
    atad_fb_by_s = _by_stamp(atad_fb_map)

    torch_by_s = _by_stamp(torch_map, require_rocm_tag=True)
    tv_by_s = _by_stamp(tv_map, require_rocm_tag=True)
    ta_by_s = _by_stamp(ta_map, require_rocm_tag=True)
    triton_by_s = _by_stamp(triton_map, require_rocm_tag=True)
    pt_triton_by_s = _by_stamp(pt_triton_map, require_rocm_tag=True)

    tv_uses_stamp = bool(tv_by_s)
    ta_uses_stamp = bool(ta_by_s)

    if has_any_rocm_sdk:
        candidates: list[tuple[tuple, dict]] = []
        for ns, raw_t_vers in torch_by_s.items():
            if ns not in rocm_by_s or ns not in core_by_s or ns not in devel_by_s:
                continue
            if tv_uses_stamp and ns not in tv_by_s:
                continue
            if ta_uses_stamp and ns not in ta_by_s:
                continue

            schema = ""
            target_family = ""
            sdk_kpack_target = ""
            libs_dist_norm = ""
            libs_ver = ""
            libs_whl = ""
            dev_dist_norm = ""
            dev_ver = ""
            dev_whl = ""
            has_native_isa = False

            if ns in ma_libs_by_s and (
                not llvm_target or ns in dev_prim_by_s or ns in dev_fb_by_s
            ):
                schema = "multi-arch"
                libs_dist_norm = "rocm_sdk_libraries"
                libs_ver = max(ma_libs_by_s[ns], key=_rocm_stamp_sort_key)
                libs_whl = ma_libs_map[libs_ver]
                if not llvm_target:
                    rocm_extras = "libraries,devel,device-all"
                    has_native_isa = True
                elif ns in dev_prim_by_s:
                    sdk_kpack_target = llvm_target
                    target_family = llvm_target
                    dev_dist_norm = _norm_dist_name(dev_primary_pkg)
                    dev_ver = max(dev_prim_by_s[ns], key=_rocm_stamp_sort_key)
                    dev_whl = dev_primary_map[dev_ver]
                    rocm_extras = f"libraries,devel,device-{sdk_kpack_target}"
                    has_native_isa = True
                else:
                    sdk_kpack_target = fb_target
                    target_family = fb_target
                    dev_dist_norm = _norm_dist_name(dev_fb_pkg)
                    dev_ver = max(dev_fb_by_s[ns], key=_rocm_stamp_sort_key)
                    dev_whl = dev_fb_map[dev_ver]
                    rocm_extras = f"libraries,devel,device-{sdk_kpack_target}"
                    has_native_isa = False
            elif ns in pf_libs_by_s:
                schema = "per-family"
                target_family = family
                libs_dist_norm = _norm_dist_name(pf_libs_pkg)
                libs_ver = max(pf_libs_by_s[ns], key=_rocm_stamp_sort_key)
                libs_whl = pf_libs_map[libs_ver]
                rocm_extras = "libraries,devel"
                has_native_isa = llvm_target not in LEGACY_HSA_OVERRIDES
            else:
                continue

            t_vers = list(raw_t_vers)
            torch_extra = ""
            atd_dist_norm = ""
            atd_ver = ""
            atd_whl = ""
            if schema == "multi-arch" and (
                index_uses_torch_kpacks or ns in atd_prim_by_s or ns in atd_fb_by_s
            ):
                if ns in atd_prim_by_s:
                    atd_cands = atd_prim_by_s[ns]
                    atd_map_sel = atd_prim_map
                    atd_dist_norm = _norm_dist_name(atd_prim_pkg)
                    torch_extra = (
                        f"device-{llvm_target}" if llvm_target else "device-all"
                    )
                elif ns in atd_fb_by_s:
                    atd_cands = atd_fb_by_s[ns]
                    atd_map_sel = atd_fb_map
                    atd_dist_norm = _norm_dist_name(atd_fb_pkg)
                    torch_extra = f"device-{fb_target}"
                    has_native_isa = False
                else:
                    continue
                overlap_t = sorted(set(t_vers) & set(atd_cands))
                if overlap_t:
                    t_vers = overlap_t
                best_t = max(t_vers, key=_pep440_public_key)
                atd_ver = (
                    best_t
                    if best_t in atd_map_sel
                    else max(atd_cands, key=_pep440_public_key)
                )
                atd_whl = atd_map_sel[atd_ver]
            else:
                best_t = max(t_vers, key=_pep440_public_key)

            tv_vers = list(tv_by_s[ns]) if tv_uses_stamp else list(tv_map.keys())
            tv_extra = ""
            atvd_dist_norm = ""
            atvd_ver = ""
            atvd_whl = ""
            if schema == "multi-arch" and (
                index_uses_tv_kpacks or ns in atvd_prim_by_s or ns in atvd_fb_by_s
            ):
                if ns in atvd_prim_by_s:
                    atvd_cands = atvd_prim_by_s[ns]
                    atvd_map_sel = atvd_prim_map
                    atvd_dist_norm = _norm_dist_name(atvd_prim_pkg)
                    tv_extra = (
                        f"device-{llvm_target}" if llvm_target else "device-all"
                    )
                elif ns in atvd_fb_by_s:
                    atvd_cands = atvd_fb_by_s[ns]
                    atvd_map_sel = atvd_fb_map
                    atvd_dist_norm = _norm_dist_name(atvd_fb_pkg)
                    tv_extra = f"device-{fb_target}"
                else:
                    continue
                overlap_tv = sorted(set(tv_vers) & set(atvd_cands))
                if overlap_tv:
                    tv_vers = overlap_tv
                best_tv = max(tv_vers, key=_pep440_public_key)
                atvd_ver = (
                    best_tv
                    if best_tv in atvd_map_sel
                    else max(atvd_cands, key=_pep440_public_key)
                )
                atvd_whl = atvd_map_sel[atvd_ver]
            else:
                best_tv = max(tv_vers, key=_pep440_public_key)

            ta_vers = list(ta_by_s[ns]) if ta_uses_stamp else list(ta_map.keys())
            ta_extra = ""
            atad_dist_norm = ""
            atad_ver = ""
            atad_whl = ""
            if schema == "multi-arch" and (
                index_uses_ta_kpacks or ns in atad_prim_by_s or ns in atad_fb_by_s
            ):
                if ns in atad_prim_by_s:
                    atad_cands = atad_prim_by_s[ns]
                    atad_map_sel = atad_prim_map
                    atad_dist_norm = _norm_dist_name(atad_prim_pkg)
                    ta_extra = (
                        f"device-{llvm_target}" if llvm_target else "device-all"
                    )
                elif ns in atad_fb_by_s:
                    atad_cands = atad_fb_by_s[ns]
                    atad_map_sel = atad_fb_map
                    atad_dist_norm = _norm_dist_name(atad_fb_pkg)
                    ta_extra = f"device-{fb_target}"
                else:
                    continue
                overlap_ta = sorted(set(ta_vers) & set(atad_cands))
                if overlap_ta:
                    ta_vers = overlap_ta
                best_ta = max(ta_vers, key=_pep440_public_key)
                atad_ver = (
                    best_ta
                    if best_ta in atad_map_sel
                    else max(atad_cands, key=_pep440_public_key)
                )
                atad_whl = atad_map_sel[atad_ver]
            else:
                best_ta = max(ta_vers, key=_pep440_public_key)

            best_r = max(rocm_by_s[ns], key=_rocm_stamp_sort_key)
            best_core = max(core_by_s[ns], key=_rocm_stamp_sort_key)
            best_devel = max(devel_by_s[ns], key=_rocm_stamp_sort_key)

            torch_spec = (
                f"torch[{torch_extra}]=={best_t}"
                if torch_extra
                else f"torch=={best_t}"
            )
            tv_spec = (
                f"torchvision[{tv_extra}]=={best_tv}"
                if tv_extra
                else f"torchvision=={best_tv}"
            )
            ta_spec = (
                f"torchaudio[{ta_extra}]=={best_ta}"
                if ta_extra
                else f"torchaudio=={best_ta}"
            )

            pkgs = [
                f"rocm[{rocm_extras}]=={best_r}",
                torch_spec,
                tv_spec,
                ta_spec,
            ]
            required_extras: dict[str, str] = {}
            if torch_extra:
                required_extras["torch"] = torch_extra
            if tv_extra:
                required_extras["torchvision"] = tv_extra
            if ta_extra:
                required_extras["torchaudio"] = ta_extra

            expected_vers: dict[str, str] = {
                "rocm": best_r,
                "rocm_sdk_core": best_core,
                "rocm_sdk_devel": best_devel,
                libs_dist_norm: libs_ver,
                "torch": best_t,
                "torchvision": best_tv,
                "torchaudio": best_ta,
            }
            exact_files = [
                rocm_map[best_r],
                core_map[best_core],
                devel_map[best_devel],
                libs_whl,
                torch_map[best_t],
                tv_map[best_tv],
                ta_map[best_ta],
            ]
            for opt_norm, opt_ver, opt_whl in (
                (dev_dist_norm, dev_ver, dev_whl),
                (atd_dist_norm, atd_ver, atd_whl),
                (atvd_dist_norm, atvd_ver, atvd_whl),
                (atad_dist_norm, atad_ver, atad_whl),
            ):
                if opt_norm and opt_ver and opt_whl:
                    expected_vers[opt_norm] = opt_ver
                    exact_files.append(opt_whl)

            companion_triton: tuple[str, str] | None = None
            if ns in triton_by_s:
                tr_v = max(triton_by_s[ns], key=_pep440_public_key)
                companion_triton = (f"triton=={tr_v}", triton_map[tr_v])
            elif ns in pt_triton_by_s:
                ptr_v = max(pt_triton_by_s[ns], key=_pep440_public_key)
                companion_triton = (
                    f"pytorch-triton-rocm=={ptr_v}",
                    pt_triton_map[ptr_v],
                )
            elif triton_map:
                tr_v = max(triton_map, key=_pep440_public_key)
                companion_triton = (f"triton=={tr_v}", triton_map[tr_v])

            rank_key = (
                1 if schema == "multi-arch" else 0,
                1 if has_native_isa else 0,
                _rocm_stamp_sort_key(best_r),
                _pep440_public_key(best_t),
            )
            candidates.append(
                (
                    rank_key,
                    {
                        "status": "locked",
                        "tier_name": tier_name,
                        "tier_url": tier_url,
                        "schema": schema,
                        "stamp": best_r,
                        "target_family": target_family,
                        "kpack_target": sdk_kpack_target,
                        "has_native_isa": has_native_isa,
                        "packages": pkgs,
                        "required_extras": required_extras,
                        "expected_versions": expected_vers,
                        "exact_filenames": exact_files,
                        "companion_triton": companion_triton,
                        "no_build_isolation": True,
                    },
                )
            )

        if not candidates:
            return {
                "status": "unsynchronized",
                "tier_name": tier_name,
                "tier_url": tier_url,
            }
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    if index_uses_torch_kpacks:
        return {
            "status": "incomplete-sdk",
            "tier_name": tier_name,
            "tier_url": tier_url,
        }

    best_t_only = max(
        torch_map,
        key=lambda v: (
            _rocm_stamp_sort_key(_extract_rocm_stamp(v)),
            _pep440_public_key(v),
        ),
    )
    best_tv_only = max(tv_map, key=_pep440_public_key)
    best_ta_only = max(ta_map, key=_pep440_public_key)
    return {
        "status": "locked",
        "tier_name": tier_name,
        "tier_url": tier_url,
        "schema": "torch-only",
        "stamp": best_t_only,
        "target_family": "",
        "kpack_target": "",
        "has_native_isa": llvm_target not in LEGACY_HSA_OVERRIDES,
        "packages": [
            f"torch=={best_t_only}",
            f"torchvision=={best_tv_only}",
            f"torchaudio=={best_ta_only}",
        ],
        "required_extras": {},
        "expected_versions": {
            "torch": best_t_only,
            "torchvision": best_tv_only,
            "torchaudio": best_ta_only,
        },
        "exact_filenames": [
            torch_map[best_t_only],
            tv_map[best_tv_only],
            ta_map[best_ta_only],
        ],
        "companion_triton": None,
        "no_build_isolation": False,
    }


def probe_generic_torch_plan(
    feed_name: str,
    feed_url: str,
    use_pre: bool,
    py_maj: int | None = None,
    py_min: int | None = None,
    freethreaded: bool | None = None,
) -> dict | None:
    if py_maj is None or py_min is None:
        py_maj, py_min = sys.version_info.major, sys.version_info.minor
    if freethreaded is None:
        freethreaded = _is_freethreaded_runtime()

    projects = ["torch", "torchvision", "torchaudio"]
    fetched: dict[str, tuple[bool, list[str]]] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {
            proj: pool.submit(_fetch_simple_index_filenames, feed_url, proj)
            for proj in projects
        }
        for proj, fut in futs.items():
            fetched[proj] = fut.result()

    if not fetched["torch"][0]:
        return None
    torch_map = _extract_compatible_wheel_map(
        fetched["torch"][1], "torch", py_maj, py_min, freethreaded=freethreaded
    )
    tv_map = _extract_compatible_wheel_map(
        fetched["torchvision"][1],
        "torchvision",
        py_maj,
        py_min,
        freethreaded=freethreaded,
    )
    ta_map = _extract_compatible_wheel_map(
        fetched["torchaudio"][1],
        "torchaudio",
        py_maj,
        py_min,
        freethreaded=freethreaded,
    )
    if not torch_map or not tv_map or not ta_map:
        return None

    def _date_key(ver: str) -> tuple[int, tuple]:
        m = re.search(r"(20\d{6})", ver)
        return (int(m.group(1)) if m else 0, _pep440_public_key(ver))

    best_t = max(torch_map, key=_date_key)
    m_date = re.search(r"(20\d{6})", best_t)
    t_date = m_date.group(1) if m_date else ""

    tv_cands = [v for v in tv_map if t_date and t_date in v] or list(tv_map.keys())
    ta_cands = [v for v in ta_map if t_date and t_date in v] or list(ta_map.keys())
    best_tv = max(tv_cands, key=_date_key)
    best_ta = max(ta_cands, key=_date_key)

    return {
        "feed_name": feed_name,
        "feed_url": feed_url,
        "use_pre": use_pre,
        "stamp": best_t,
        "packages": [
            f"torch=={best_t}",
            f"torchvision=={best_tv}",
            f"torchaudio=={best_ta}",
        ],
        "expected_versions": {
            "torch": best_t,
            "torchvision": best_tv,
            "torchaudio": best_ta,
        },
        "exact_filenames": [
            torch_map[best_t],
            tv_map[best_tv],
            ta_map[best_ta],
        ],
    }


def select_sota_python_version(
    matrix: dict[str, str], profile: str, llvm_target: str
) -> tuple[str, str]:
    minor_to_slot: dict[tuple[int, int], tuple[str, str]] = {}
    for key, ver in matrix.items():
        sv = parse_semver(ver)
        if len(sv) >= 2:
            minor_to_slot[(sv[0], sv[1])] = (key, ver)

    candidate_minors = sorted(minor_to_slot.keys(), reverse=True)

    pypi_fetched: dict[str, tuple[bool, list[str]]] = {}
    with ThreadPoolExecutor(max_workers=len(PYPI_BINARY_GATE_PACKAGES)) as pool:
        futs = {
            pkg: pool.submit(
                _fetch_simple_index_filenames, "https://pypi.org/simple", pkg
            )
            for pkg in PYPI_BINARY_GATE_PACKAGES
        }
        for pkg, fut in futs.items():
            pypi_fetched[pkg] = fut.result()

    eligible_minors = set(candidate_minors)
    if all(ok and files for ok, files in pypi_fetched.values()):
        for pkg, (_, files) in pypi_fetched.items():
            eligible_minors &= _probe_compatible_minors_in_files(
                files,
                pkg,
                candidate_minors,
                require_platform_binary=True,
            )

    ordered_eligible = sorted(eligible_minors, reverse=True) or candidate_minors

    if profile == "ROCM":
        for tier_name, tier_url in _build_rocm_cascade_tiers(llvm_target)[:3]:
            for maj, min_v in ordered_eligible:
                plan = probe_rocm_tier_synchronized_plan(
                    tier_name,
                    tier_url,
                    llvm_target,
                    py_maj=maj,
                    py_min=min_v,
                    freethreaded=False,
                )
                if plan is not None and plan["status"] == "locked":
                    return minor_to_slot[(maj, min_v)]
    elif profile == "CUDA":
        for feed_name, feed_url, use_pre in _discover_cuda_feed_candidates(
            llvm_target
        )[:4]:
            for maj, min_v in ordered_eligible:
                c_plan = probe_generic_torch_plan(
                    feed_name,
                    feed_url,
                    use_pre,
                    py_maj=maj,
                    py_min=min_v,
                    freethreaded=False,
                )
                if c_plan is not None:
                    return minor_to_slot[(maj, min_v)]
    else:
        feed_url = (
            "https://download.pytorch.org/whl/nightly/xpu"
            if profile == "INTEL_XPU"
            else "https://download.pytorch.org/whl/nightly/cpu"
        )
        for maj, min_v in ordered_eligible:
            g_plan = probe_generic_torch_plan(
                profile.lower(),
                feed_url,
                True,
                py_maj=maj,
                py_min=min_v,
                freethreaded=False,
            )
            if g_plan is not None:
                return minor_to_slot[(maj, min_v)]

    for fallback_minor in [(3, 14), (3, 13), (3, 12)]:
        if fallback_minor in minor_to_slot:
            return minor_to_slot[fallback_minor]
    return minor_to_slot[candidate_minors[0]]


def condition_pytorch_runtime(
    executable: Path,
    isolated_env: dict[str, str],
    profile: str,
    llvm_target: str,
    target_sandbox: Path,
) -> dict[str, str]:
    quarantine_sandbox_sitecustomize(target_sandbox)

    arch_label = llvm_target or "Generic"
    if profile == "CUDA" and llvm_target:
        arch_label = f"Driver CUDA <= {llvm_target}"
    print_status(
        f"Hardware Compute Architecture Locked: {profile} ({arch_label})",
        "SUCCESS",
    )
    salvage_cached_wheels()

    print_status(
        "Upgrading core build tooling (pip, setuptools, wheel, packaging, ninja, cmake)..."
    )
    stage_and_install_packages(
        executable=executable,
        isolated_env=isolated_env,
        packages=PYPI_BUILD_TOOLING,
        index_url="https://pypi.org/simple",
        use_pre=False,
        is_deps_stage=True,
    )

    runtime_deps = list(PYPI_RUNTIME_DEPS)
    if profile == "ROCM":
        runtime_deps.extend(PYPI_ROCM_RUNTIME_DEPS)

    print_status("Pre-staging verified PyPI runtime dependency pool...")
    stage_and_install_packages(
        executable=executable,
        isolated_env=isolated_env,
        packages=runtime_deps,
        index_url="https://pypi.org/simple",
        use_pre=False,
        is_deps_stage=True,
    )

    hsa_override_used = ""
    locked_family = ""
    locked_schema = ""
    locked_stamp = ""

    print_status(f"Deploying hardware-aligned PyTorch binaries for {profile}...")

    if profile == "ROCM":
        tiers = _build_rocm_cascade_tiers(llvm_target)
        deployed_rocm = False
        last_err: Exception | None = None

        for tier_name, tier_url in tiers:
            plan = probe_rocm_tier_synchronized_plan(
                tier_name, tier_url, llvm_target
            )
            if plan is None or plan["status"] != "locked":
                reason = plan["status"] if plan else "empty"
                print_status(
                    f"Skipping ROCm tier {tier_name} ({reason} for Python {sys.version.split()[0]}).",
                    "WARN",
                )
                continue

            locked_family = plan["target_family"]
            locked_schema = plan["schema"]
            locked_stamp = plan["stamp"]

            tier_env = dict(isolated_env)
            if locked_family:
                tier_env["ROCM_SDK_TARGET_FAMILY"] = locked_family
            else:
                tier_env.pop("ROCM_SDK_TARGET_FAMILY", None)
            if llvm_target:
                tier_env["PYTORCH_ROCM_ARCH"] = llvm_target

            extras_summary = (
                ",".join(f"{k}[{v}]" for k, v in plan["required_extras"].items())
                or "monolithic"
            )
            print_status(
                f"Locked ROCm Tier [{plan['tier_name']}] ({locked_schema} | "
                f"family={locked_family or 'n/a'} | kpacks={extras_summary}) "
                f"@ {locked_stamp} -> {plan['tier_url']}",
                "SUCCESS",
            )

            if _is_closure_converged(
                target_sandbox,
                profile,
                plan["expected_versions"],
                required_extras=plan["required_extras"],
                expected_target_family=locked_family,
                expected_stamp=locked_stamp,
            ):
                print_status(
                    f"Compute & kpack closure already converged at {locked_stamp}; skipping redundant extraction.",
                    "SUCCESS",
                )
                deployed_rocm = True
            else:
                reconcile_compute_closure(
                    executable,
                    tier_env,
                    target_sandbox,
                    keep_versions=plan["expected_versions"],
                    required_extras=plan["required_extras"],
                )
                try:
                    stage_and_install_packages(
                        executable=executable,
                        isolated_env=tier_env,
                        packages=plan["packages"],
                        index_url=plan["tier_url"],
                        fallback_pypi=True,
                        use_pre=True,
                        no_build_isolation=plan["no_build_isolation"],
                        is_deps_stage=False,
                        exact_filenames=plan["exact_filenames"],
                        stamp_hint=locked_stamp,
                        max_attempts=2,
                    )
                    deployed_rocm = True
                except subprocess.CalledProcessError as exc:
                    last_err = exc
                    print_status(
                        f"Tier {tier_name} staging faulted; advancing to next ROCm cascade tier...",
                        "WARN",
                    )
                    continue

            companion = plan.get("companion_triton")
            if companion is not None:
                inst_now = _installed_distributions(target_sandbox)
                if "triton" not in inst_now and "pytorch_triton_rocm" not in inst_now:
                    tr_req, tr_whl = companion
                    try:
                        print_status(
                            f"Deploying synchronized ROCm Triton compiler ({tr_req})...",
                            "INFO",
                        )
                        stage_and_install_packages(
                            executable=executable,
                            isolated_env=tier_env,
                            packages=[tr_req],
                            index_url=plan["tier_url"],
                            fallback_pypi=True,
                            use_pre=True,
                            is_deps_stage=False,
                            exact_filenames=[tr_whl],
                            stamp_hint=locked_stamp,
                            max_attempts=2,
                        )
                    except Exception as tr_exc:
                        print_status(
                            f"Optional Triton wheel skipped ({tr_exc}).",
                            "WARN",
                        )

            if not plan["has_native_isa"]:
                hsa_override_used = LEGACY_HSA_OVERRIDES.get(llvm_target, "")
            isolated_env.update(tier_env)
            break

        if not deployed_rocm:
            if last_err is not None:
                raise last_err
            raise RuntimeError(
                f"All ROCm cascade tiers exhausted without synchronized wheels for Python {sys.version.split()[0]}."
            )

    elif profile == "CUDA":
        cuda_feeds = _discover_cuda_feed_candidates(llvm_target)
        deployed_cuda = False
        last_cuda_err: Exception | None = None

        for feed_name, feed_url, use_pre_flag in cuda_feeds:
            c_plan = probe_generic_torch_plan(feed_name, feed_url, use_pre_flag)
            if c_plan is None:
                continue
            locked_schema = feed_name
            locked_stamp = c_plan["stamp"]
            print_status(
                f"Locked CUDA Feed [{feed_name}] @ {locked_stamp} -> {feed_url}",
                "SUCCESS",
            )
            if _is_closure_converged(
                target_sandbox, profile, c_plan["expected_versions"]
            ):
                deployed_cuda = True
                break
            reconcile_compute_closure(
                executable,
                isolated_env,
                target_sandbox,
                keep_versions=c_plan["expected_versions"],
            )
            try:
                stage_and_install_packages(
                    executable=executable,
                    isolated_env=isolated_env,
                    packages=c_plan["packages"],
                    index_url=feed_url,
                    fallback_pypi=True,
                    use_pre=use_pre_flag,
                    exact_filenames=c_plan["exact_filenames"],
                    max_attempts=2,
                )
                deployed_cuda = True
                break
            except subprocess.CalledProcessError as exc:
                last_cuda_err = exc
                print_status(
                    f"CUDA feed {feed_name} faulted; advancing cascade...",
                    "WARN",
                )

        if not deployed_cuda:
            if last_cuda_err is not None:
                raise last_cuda_err
            raise RuntimeError(
                f"No compatible CUDA PyTorch feed found for Python {sys.version.split()[0]}."
            )

    else:
        feed_url = (
            "https://download.pytorch.org/whl/nightly/xpu"
            if profile == "INTEL_XPU"
            else "https://download.pytorch.org/whl/nightly/cpu"
        )
        g_plan = probe_generic_torch_plan(profile.lower(), feed_url, True)
        if g_plan is not None:
            locked_stamp = g_plan["stamp"]
            if not _is_closure_converged(
                target_sandbox, profile, g_plan["expected_versions"]
            ):
                reconcile_compute_closure(
                    executable,
                    isolated_env,
                    target_sandbox,
                    keep_versions=g_plan["expected_versions"],
                )
                stage_and_install_packages(
                    executable=executable,
                    isolated_env=isolated_env,
                    packages=g_plan["packages"],
                    index_url=feed_url,
                    fallback_pypi=True,
                    use_pre=True,
                    exact_filenames=g_plan["exact_filenames"],
                )
        else:
            stage_and_install_packages(
                executable=executable,
                isolated_env=isolated_env,
                packages=["torch", "torchvision", "torchaudio"],
                index_url=feed_url,
                fallback_pypi=True,
                use_pre=True,
            )

    materialize_hardware_prefix(
        executable=executable,
        isolated_env=isolated_env,
        target_sandbox=target_sandbox,
        profile=profile,
        llvm_target=llvm_target,
        target_family=locked_family,
        schema=locked_schema,
        stamp=locked_stamp,
    )
    print_status("PyTorch & Sovereign Hardware Prefix alignment complete.", "SUCCESS")
    return {
        "hsa_override": hsa_override_used,
        "target_family": locked_family,
        "schema": locked_schema,
        "stamp": locked_stamp,
    }


def _has_installed_device_kpack(target_sandbox: Path, llvm_target: str) -> bool:
    if not llvm_target:
        return False
    sp = _site_packages_dir(target_sandbox)
    if sp is None or not sp.is_dir():
        return False
    norm_t = llvm_target.lower().replace("-", "_")
    if (sp / f"_rocm_sdk_device_{norm_t}").is_dir():
        return True
    if any(sp.glob(f"rocm_sdk_device_{norm_t}-*.dist-info")):
        return True
    return False


def _parse_dist_info_families(dist_info_py: Path) -> tuple[str, list[str]]:
    if not dist_info_py.is_file():
        return ("", [])
    try:
        src = dist_info_py.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ("", [])

    default_fam = ""
    m_def = re.search(
        r"default_target_family\s*=\s*['\"]([^'\"]+)['\"]", src
    )
    if m_def:
        default_fam = m_def.group(1).strip()

    families: list[str] = []
    m_avail = re.search(
        r"available_target_families\s*(?::[^=]+)?=\s*(\[[^\]]*\])",
        src,
        flags=re.S,
    )
    if m_avail:
        try:
            parsed = ast.literal_eval(m_avail.group(1))
            if isinstance(parsed, (list, tuple)):
                families = [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            families = re.findall(
                r"['\"](gfx[0-9A-Za-z_-]+)['\"]", m_avail.group(1)
            )
    if not families:
        families = re.findall(r"['\"](gfx[0-9A-Za-z_-]+)['\"]", src)
    return (default_fam, families)


def _detect_installed_rocm_family(
    target_sandbox: Path, llvm_target: str = ""
) -> str:
    manifest = _read_hw_prefix_manifest(target_sandbox)
    m_fam = str(manifest.get("target_family") or "").strip()
    sp = _site_packages_dir(target_sandbox)
    if sp is None or not sp.is_dir():
        return m_fam

    default_family, available_families = _parse_dist_info_families(
        sp / "rocm_sdk" / "_dist_info.py"
    )
    if available_families:
        avail_set = set(available_families)
        if m_fam and m_fam in avail_set:
            return m_fam
        if llvm_target and llvm_target in avail_set:
            return llvm_target
        mapped = (
            AMD_STAGING_FAMILY_MAP.get(llvm_target, f"{llvm_target}-all")
            if llvm_target
            else ""
        )
        if mapped and mapped in avail_set:
            return mapped
        for dev_dir in sorted(sp.glob("_rocm_sdk_device_*")):
            cand_hyph = dev_dir.name[len("_rocm_sdk_device_") :].replace("_", "-")
            if cand_hyph in avail_set:
                return cand_hyph
            cand_raw = dev_dir.name[len("_rocm_sdk_device_") :]
            if cand_raw in avail_set:
                return cand_raw
        if default_family and default_family in avail_set:
            return default_family
        return available_families[0]

    installed = _installed_distributions(target_sandbox)
    if llvm_target:
        norm_dev = f"rocm_sdk_device_{llvm_target.lower().replace('-', '_')}"
        if norm_dev in installed:
            return llvm_target
    for dist_name in installed:
        if (
            dist_name.startswith("rocm_sdk_libraries_")
            and dist_name != "rocm_sdk_libraries"
        ):
            return (
                AMD_STAGING_FAMILY_MAP.get(llvm_target, f"{llvm_target}-all")
                if llvm_target
                else m_fam
            )
    return m_fam


def _merge_tree_links(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.is_dir():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(src_dir.iterdir()):
        target = dst_dir / item.name
        if item.is_dir():
            if not target.exists() and not _is_reparse_or_link(target):
                _link_or_junction(item, target)
            elif target.is_dir() and not _is_reparse_or_link(target):
                _merge_tree_links(item, target)
        elif item.is_file() and not target.exists():
            _link_or_junction(item, target)


def _synthesize_cuda_prefix(site_packages: Path, cuda_prefix: Path) -> bool:
    nvidia_root = site_packages / "nvidia"
    if not nvidia_root.is_dir():
        return False

    safe_rmtree(cuda_prefix)
    bin_dir = cuda_prefix / "bin"
    inc_dir = cuda_prefix / "include"
    lib_dir = (
        cuda_prefix / "lib" / "x64"
        if os.name == "nt"
        else cuda_prefix / "lib64"
    )
    bin_dir.mkdir(parents=True, exist_ok=True)
    inc_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)

    populated = False
    for pkg_dir in sorted(nvidia_root.iterdir()):
        if not pkg_dir.is_dir():
            continue
        if (pkg_dir / "bin").is_dir():
            _merge_tree_links(pkg_dir / "bin", bin_dir)
            populated = True
        if (pkg_dir / "include").is_dir():
            _merge_tree_links(pkg_dir / "include", inc_dir)
            populated = True
        if (pkg_dir / "lib").is_dir():
            for lib_item in sorted((pkg_dir / "lib").iterdir()):
                if os.name == "nt" and lib_item.suffix.lower() == ".dll":
                    if not (bin_dir / lib_item.name).exists():
                        _link_or_junction(lib_item, bin_dir / lib_item.name)
                elif lib_item.is_file() and not (lib_dir / lib_item.name).exists():
                    _link_or_junction(lib_item, lib_dir / lib_item.name)
                elif lib_item.is_dir() and lib_item.name == "cmake":
                    _merge_tree_links(lib_item, cuda_prefix / "lib" / "cmake")
            populated = True
        if (pkg_dir / "nvvm").is_dir() and not (cuda_prefix / "nvvm").exists():
            _link_or_junction(pkg_dir / "nvvm", cuda_prefix / "nvvm")
            populated = True

    if os.name != "nt" and not (cuda_prefix / "lib").exists():
        _link_or_junction(lib_dir, cuda_prefix / "lib")

    if not populated:
        safe_rmtree(cuda_prefix)
    return populated


def materialize_hardware_prefix(
    executable: Path,
    isolated_env: dict[str, str],
    target_sandbox: Path,
    profile: str,
    llvm_target: str,
    target_family: str = "",
    schema: str = "",
    stamp: str = "",
) -> dict[str, str]:
    site_packages = _site_packages_dir(target_sandbox)
    hw_prefix = target_sandbox / "hw_prefix"
    hw_prefix.mkdir(parents=True, exist_ok=True)

    legacy_py_cycle = hw_prefix / "python"
    if _is_reparse_or_link(legacy_py_cycle) or legacy_py_cycle.exists():
        safe_rmtree(legacy_py_cycle)

    effective_family = target_family or _detect_installed_rocm_family(
        target_sandbox, llvm_target
    )

    if (
        profile == "ROCM"
        and site_packages is not None
        and (site_packages / "rocm_sdk").is_dir()
    ):
        init_env = dict(isolated_env)
        if effective_family:
            init_env["ROCM_SDK_TARGET_FAMILY"] = effective_family
        else:
            init_env.pop("ROCM_SDK_TARGET_FAMILY", None)
        if llvm_target:
            init_env["PYTORCH_ROCM_ARCH"] = llvm_target

        print_status(
            "Materializing TheRock rocm_sdk devel sysroot and linking device kpacks...",
            "INFO",
        )
        init_res = subprocess.run(
            [str(executable), "-I", "-m", "rocm_sdk", "init"],
            env=init_env,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        if init_res.returncode != 0:
            raise RuntimeError(
                f"rocm_sdk init failed with exit code {init_res.returncode} "
                f"(ROCM_SDK_TARGET_FAMILY={effective_family!r})."
            )

        q_root = safe_subprocess(
            [str(executable), "-I", "-m", "rocm_sdk", "path", "--root"],
            env=init_env,
        )
        rocm_src: Path | None = None
        if q_root and Path(q_root).is_dir():
            rocm_src = Path(q_root).resolve()
        elif (site_packages / "_rocm_sdk_devel" / "bin").is_dir():
            rocm_src = (site_packages / "_rocm_sdk_devel").resolve()
        elif (site_packages / "_rocm_sdk_core" / "bin").is_dir():
            rocm_src = (site_packages / "_rocm_sdk_core").resolve()

        if rocm_src is not None:
            _link_or_junction(rocm_src, hw_prefix / "rocm")

    elif profile == "CUDA" and site_packages is not None:
        _synthesize_cuda_prefix(site_packages, hw_prefix / "cuda")

    layout = resolve_hardware_prefix_layout(
        target_sandbox,
        profile=profile,
        llvm_target=llvm_target,
        executable=executable,
        isolated_env=isolated_env,
    )
    kpack_count = 0
    if site_packages is not None:
        torch_kpack_dir = site_packages / "torch" / ".kpack"
        if torch_kpack_dir.is_dir():
            kpack_count = len(list(torch_kpack_dir.glob("*.kpack")))

    manifest_payload = {
        "profile": profile,
        "llvm_target": llvm_target,
        "target_family": layout["target_family"] or effective_family,
        "schema": schema,
        "stamp": stamp,
        "hw_root": layout["rocm_root"] or layout["cuda_root"],
        "cmake_prefix_path": layout["cmake_prefix_path"],
        "torch_kpack_archives": kpack_count,
        "c_api_ready": _sandbox_has_c_api(target_sandbox),
    }
    (hw_prefix / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8"
    )
    return layout


def resolve_hardware_prefix_layout(
    target_sandbox: Path,
    profile: str = "",
    llvm_target: str = "",
    executable: Path | None = None,
    isolated_env: dict[str, str] | None = None,
) -> dict[str, str]:
    site_packages = _site_packages_dir(target_sandbox)
    hw_prefix = target_sandbox / "hw_prefix"
    hw_rocm = hw_prefix / "rocm"
    hw_cuda = hw_prefix / "cuda"

    target_family = (
        _detect_installed_rocm_family(target_sandbox, llvm_target)
        if profile == "ROCM"
        else ""
    )

    rocm_root: Path | None = None
    rocm_bin: Path | None = None
    rocm_cmake: Path | None = None
    llvm_bin: Path | None = None
    device_lib: Path | None = None
    miopen_sys_db: Path | None = None
    cuda_root: Path | None = None

    if (hw_rocm / "bin").is_dir():
        rocm_root = hw_rocm

    if (
        rocm_root is None
        and profile == "ROCM"
        and executable is not None
        and executable.is_file()
        and site_packages is not None
        and (site_packages / "rocm_sdk").is_dir()
    ):
        probe_env = dict(isolated_env or os.environ)
        if target_family:
            probe_env["ROCM_SDK_TARGET_FAMILY"] = target_family
        q_root = safe_subprocess(
            [str(executable), "-I", "-m", "rocm_sdk", "path", "--root"],
            env=probe_env,
        )
        if q_root and Path(q_root).is_dir():
            rocm_root = Path(q_root)

    sdk_roots: list[Path] = []
    if rocm_root is not None:
        sdk_roots.append(rocm_root)

    if site_packages is not None and site_packages.is_dir():
        devel_dir = site_packages / "_rocm_sdk_devel"
        core_dir = site_packages / "_rocm_sdk_core"
        libs_dir = site_packages / "_rocm_sdk_libraries"
        for cand in (devel_dir, core_dir, libs_dir):
            if cand.is_dir() and cand not in sdk_roots:
                sdk_roots.append(cand)
        for pkg_dir in sorted(site_packages.glob("_rocm_sdk_*")):
            if pkg_dir.is_dir() and pkg_dir not in sdk_roots:
                sdk_roots.append(pkg_dir)
        if rocm_root is None:
            if (devel_dir / "bin").is_dir():
                rocm_root = devel_dir
            elif (core_dir / "bin").is_dir():
                rocm_root = core_dir

    if rocm_root is not None and (rocm_root / "bin").is_dir():
        rocm_bin = rocm_root / "bin"

    for root_cand in sdk_roots:
        if rocm_cmake is None and (root_cand / "lib" / "cmake").is_dir():
            rocm_cmake = root_cand / "lib" / "cmake"
        if llvm_bin is None and (root_cand / "lib" / "llvm" / "bin").is_dir():
            llvm_bin = root_cand / "lib" / "llvm" / "bin"
        if device_lib is None and _has_nonempty_dir(
            root_cand / "lib" / "llvm" / "amdgcn" / "bitcode"
        ):
            device_lib = root_cand / "lib" / "llvm" / "amdgcn" / "bitcode"
        if miopen_sys_db is None and _has_nonempty_dir(
            root_cand / "share" / "miopen" / "db"
        ):
            miopen_sys_db = root_cand / "share" / "miopen" / "db"

    if (
        miopen_sys_db is None
        and site_packages is not None
        and _has_nonempty_dir(site_packages / "torch" / "share" / "miopen" / "db")
    ):
        miopen_sys_db = site_packages / "torch" / "share" / "miopen" / "db"

    if _has_nonempty_dir(hw_cuda):
        cuda_root = hw_cuda

    extra_bins: list[str] = []
    extra_libs: list[str] = []

    def _add_unique(lst: list[str], p: Path) -> None:
        if p.is_dir():
            s = str(p)
            if s not in lst:
                lst.append(s)

    if site_packages is not None:
        _add_unique(extra_bins, site_packages / "torch" / "lib")
        _add_unique(extra_libs, site_packages / "torch" / "lib")

    if rocm_bin is not None:
        _add_unique(extra_bins, rocm_bin)
    if llvm_bin is not None:
        _add_unique(extra_bins, llvm_bin)
    for root_cand in sdk_roots:
        _add_unique(extra_bins, root_cand / "bin")
        _add_unique(extra_libs, root_cand / "lib")

    if cuda_root is not None:
        _add_unique(extra_bins, cuda_root / "bin")
        _add_unique(
            extra_libs,
            cuda_root / "lib" / "x64"
            if os.name == "nt"
            else cuda_root / "lib64",
        )
    elif site_packages is not None and (site_packages / "nvidia").is_dir():
        for nv_pkg in sorted((site_packages / "nvidia").iterdir()):
            if nv_pkg.is_dir():
                _add_unique(extra_bins, nv_pkg / "bin")
                _add_unique(extra_libs, nv_pkg / "lib")
                if os.name == "nt":
                    _add_unique(extra_bins, nv_pkg / "lib")

    cmake_prefixes: list[str] = []
    if rocm_cmake is not None:
        _add_unique(cmake_prefixes, rocm_cmake)
    if cuda_root is not None and (cuda_root / "lib" / "cmake").is_dir():
        _add_unique(cmake_prefixes, cuda_root / "lib" / "cmake")
    if (
        site_packages is not None
        and (site_packages / "torch" / "share" / "cmake").is_dir()
    ):
        _add_unique(cmake_prefixes, site_packages / "torch" / "share" / "cmake")
    _add_unique(cmake_prefixes, target_sandbox)

    return {
        "rocm_root": str(rocm_root) if rocm_root is not None else "",
        "rocm_bin": str(rocm_bin) if rocm_bin is not None else "",
        "rocm_cmake": str(rocm_cmake) if rocm_cmake is not None else "",
        "llvm_bin": str(llvm_bin) if llvm_bin is not None else "",
        "device_lib": str(device_lib) if device_lib is not None else "",
        "miopen_sys_db": str(miopen_sys_db) if miopen_sys_db is not None else "",
        "cuda_root": str(cuda_root) if cuda_root is not None else "",
        "target_family": target_family,
        "cmake_prefix_path": os.pathsep.join(cmake_prefixes),
        "extra_bins": os.pathsep.join(extra_bins),
        "extra_libs": os.pathsep.join(extra_libs),
    }


def build_hermetic_env(
    target_sandbox: Path,
    profile: str = "",
    llvm_target: str = "",
    executable: Path | None = None,
) -> dict[str, str]:
    if not profile and not llvm_target:
        profile, llvm_target = resolve_hardware_matrix()

    layout = resolve_hardware_prefix_layout(
        target_sandbox,
        profile=profile,
        llvm_target=llvm_target,
        executable=executable,
    )
    extra_bins = [p for p in layout["extra_bins"].split(os.pathsep) if p]
    extra_libs = [p for p in layout["extra_libs"].split(os.pathsep) if p]

    env: dict[str, str] = {}
    passthrough_keys = (
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
        "OS",
        "USERNAME",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "COMMONPROGRAMFILES",
        "PROGRAMDATA",
        "WT_SESSION",
        "TERM",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "ROCM_INDEX_URL",
        "INTELLIVENV_ALLOW_HOST_PREFIX",
    )
    for k in passthrough_keys:
        if k in os.environ:
            env[k] = os.environ[k]

    if os.name == "nt":
        sys_root = os.environ.get("SystemRoot", r"C:\Windows")
        base_os_paths = (
            [
                str(target_sandbox),
                str(target_sandbox / "Scripts"),
            ]
            + extra_bins
            + [
                os.path.join(sys_root, "System32"),
                sys_root,
                os.path.join(sys_root, "System32", "Wbem"),
                os.path.join(sys_root, "System32", "WindowsPowerShell", "v1.0"),
            ]
        )
        env["PATH"] = ";".join(base_os_paths)
        env["PIP_CONFIG_FILE"] = "NUL"
    else:
        base_os_paths = (
            [
                str(target_sandbox / "bin"),
                str(target_sandbox),
            ]
            + extra_bins
            + [
                "/usr/local/bin",
                "/usr/bin",
                "/bin",
            ]
        )
        env["PATH"] = ":".join(base_os_paths)
        env["PIP_CONFIG_FILE"] = "/dev/null"
        if extra_libs and sys.platform.startswith("linux"):
            env["LD_LIBRARY_PATH"] = ":".join(extra_libs)

    sovereign_vars = {
        "PYTHONHOME": str(target_sandbox),
        "PYTHONNOUSERSITE": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PIP_NO_USER": "1",
        "PIP_USER": "0",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "UV_CACHE_DIR": str(UV_CACHE_DIR),
        "UV_NO_CONFIG": "1",
        "CCACHE_DIR": str(CCACHE_DIR),
        "SCCACHE_DIR": str(SCCACHE_DIR),
        "TEMP": str(TMP_DIR),
        "TMP": str(TMP_DIR),
        "TMPDIR": str(TMP_DIR),
        "PIP_CACHE_DIR": str(PIP_CACHE_DIR),
        "LOCALAPPDATA": str(LOCAL_APPDATA_DIR),
        "APPDATA": str(ROAMING_APPDATA_DIR),
        "XDG_CACHE_HOME": str(CACHE_DIR),
        "XDG_CONFIG_HOME": str(XDG_CONFIG_DIR),
        "XDG_DATA_HOME": str(XDG_DATA_DIR),
        "XDG_STATE_HOME": str(XDG_STATE_DIR),
        "HF_HOME": str(HF_CACHE_DIR),
        "HF_HUB_CACHE": str(HF_CACHE_DIR / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(HF_CACHE_DIR / "hub"),
        "TORCH_HOME": str(HF_CACHE_DIR / "torch"),
        "TORCH_EXTENSIONS_DIR": str(HF_CACHE_DIR / "torch_extensions"),
        "TORCHINDUCTOR_CACHE_DIR": str(HF_CACHE_DIR / "torch_inductor"),
        "TRITON_CACHE_DIR": str(HF_CACHE_DIR / "triton"),
        "CUDA_CACHE_PATH": str(CUDA_CACHE_DIR),
        "CUPY_CACHE_DIR": str(CUPY_CACHE_DIR),
        "NUMBA_CACHE_DIR": str(NUMBA_CACHE_DIR),
        "MPLCONFIGDIR": str(MPL_CACHE_DIR),
        "MIOPEN_USER_DB_PATH": str(HF_CACHE_DIR / "miopen" / "db"),
        "MIOPEN_CUSTOM_CACHE_DIR": str(HF_CACHE_DIR / "miopen" / "kernels"),
        "AMD_COMGR_CACHE": "1",
        "AMD_COMGR_CACHE_DIR": str(HF_CACHE_DIR / "miopen" / "comgr"),
        "MIOPEN_FIND_MODE": "3",
        "MIOPEN_LOG_LEVEL": "0",
        "MIOPEN_ENABLE_LOGGING": "0",
        "AMD_LOG_LEVEL": "0",
        "TORCH_BLAS_PREFER_HIPBLASLT": "1",
        "HSA_ENABLE_SDMA": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTORCH_ALLOC_CONF": ALLOC_CONF_POLICY,
        "CMAKE_PREFIX_PATH": layout["cmake_prefix_path"],
    }

    if profile == "ROCM":
        sovereign_vars["HIP_PLATFORM"] = "amd"
        if llvm_target:
            sovereign_vars["PYTORCH_ROCM_ARCH"] = llvm_target
        if layout["rocm_root"]:
            sovereign_vars["ROCM_PATH"] = layout["rocm_root"]
            sovereign_vars["ROCM_HOME"] = layout["rocm_root"]
            sovereign_vars["HIP_PATH"] = layout["rocm_root"]
        if layout["target_family"]:
            sovereign_vars["ROCM_SDK_TARGET_FAMILY"] = layout["target_family"]
        if layout["llvm_bin"]:
            sovereign_vars["HIP_CLANG_PATH"] = layout["llvm_bin"]
        if layout["device_lib"]:
            sovereign_vars["HIP_DEVICE_LIB_PATH"] = layout["device_lib"]
        if layout["miopen_sys_db"]:
            sovereign_vars["MIOPEN_SYSTEM_DB_PATH"] = layout["miopen_sys_db"]
    elif profile == "CUDA" and layout["cuda_root"]:
        sovereign_vars["CUDA_HOME"] = layout["cuda_root"]
        sovereign_vars["CUDA_PATH"] = layout["cuda_root"]

    allow_host = os.environ.get("INTELLIVENV_ALLOW_HOST_PREFIX") == "1"
    for k, v in sovereign_vars.items():
        if not v:
            continue
        if allow_host and os.environ.get(k, "").strip():
            env[k] = os.environ[k]
        else:
            env[k] = v

    return env


def inject_hermetic_sitecustomize(
    target_dir: Path,
    profile: str,
    llvm_target: str,
    hsa_override: str = "",
) -> None:
    if os.name == "nt":
        site_packages_dirs = [target_dir / "Lib" / "site-packages"]
        if (target_dir / "Lib" / "os.py").is_file():
            for pth_file in target_dir.glob("*._pth"):
                pth_file.unlink(missing_ok=True)
        else:
            for pth_file in target_dir.glob("*._pth"):
                orig_lines = pth_file.read_text(encoding="utf-8").splitlines()
                core_zips = [
                    line.strip()
                    for line in orig_lines
                    if line.strip().endswith(".zip")
                ]
                payload_paths = core_zips + [
                    "Lib/site-packages",
                    ".",
                    "..",
                    "import site",
                ]
                pth_file.write_text(
                    "\n".join(payload_paths) + "\n", encoding="utf-8"
                )
    else:
        site_packages_dirs = list((target_dir / "lib").glob("python*/site-packages"))

    env_parent_depth = 2 if os.name == "nt" else 3
    family_map_lit = repr(AMD_STAGING_FAMILY_MAP)
    is_rocm_lit = repr(profile == "ROCM")
    is_cuda_lit = repr(profile == "CUDA")
    llvm_target_lit = repr(llvm_target if profile == "ROCM" else "")
    hsa_override_lit = repr(hsa_override)

    sitecustomize_code = f"""from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import sys

_HERE = Path(__file__).resolve()
_SP_DIR = _HERE.parent
_ENV_DIR = _HERE.parents[{env_parent_depth}]
_ROOT = _ENV_DIR.parent
_HW_DIR = _ENV_DIR / "hw_prefix"
_HF = _ROOT / ".hf_cache"
_TMP = _ROOT / ".tmp"
_CACHE = _ROOT / ".cache"
_ALLOW_HOST = os.environ.get("INTELLIVENV_ALLOW_HOST_PREFIX") == "1"
_FAMILY_MAP = {family_map_lit}

for _d in (
    _HF,
    _HF / "hub",
    _HF / "torch",
    _HF / "torch_extensions",
    _HF / "torch_inductor",
    _HF / "triton",
    _HF / "miopen" / "db",
    _HF / "miopen" / "kernels",
    _HF / "miopen" / "comgr",
    _TMP,
    _CACHE / "pip",
    _CACHE / "uv",
    _CACHE / "ccache",
    _CACHE / "sccache",
    _CACHE / "cuda",
    _CACHE / "cupy",
    _CACHE / "numba",
    _CACHE / "matplotlib",
    _ROOT / ".config",
    _ROOT / ".local" / "share",
    _ROOT / ".local" / "state",
):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _env_enforce(key: str, val: str) -> None:
    if not val:
        if not _ALLOW_HOST:
            os.environ.pop(key, None)
        return
    if _ALLOW_HOST and os.environ.get(key, "").strip():
        return
    os.environ[key] = val


def _env_default(key: str, val: str) -> None:
    cur = os.environ.get(key)
    if cur is None or not cur.strip():
        os.environ[key] = val


def _has_files(path: Path) -> bool:
    try:
        return path.is_dir() and any(path.iterdir())
    except OSError:
        return False


def _resolve_installed_family(isa_target: str) -> str:
    manifest_fam = ""
    manifest_path = _HW_DIR / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest_fam = str(
                json.loads(manifest_path.read_text(encoding="utf-8")).get("target_family") or ""
            ).strip()
        except Exception:
            pass
    dist_py = _SP_DIR / "rocm_sdk" / "_dist_info.py"
    if dist_py.is_file():
        try:
            txt = dist_py.read_text(encoding="utf-8", errors="ignore")
            avail: list[str] = []
            m_av = re.search(r"available_target_families\\s*(?::[^=]+)?=\\s*(\\[[^\\]]*\\])", txt, flags=re.S)
            if m_av:
                try:
                    parsed = ast.literal_eval(m_av.group(1))
                    if isinstance(parsed, (list, tuple)):
                        avail = [str(x).strip() for x in parsed if str(x).strip()]
                except Exception:
                    avail = re.findall(r"['\\\"](gfx[0-9A-Za-z_-]+)['\\\"]", m_av.group(1))
            if not avail:
                avail = re.findall(r"['\\\"](gfx[0-9A-Za-z_-]+)['\\\"]", txt)
            if avail:
                avail_set = set(avail)
                if manifest_fam and manifest_fam in avail_set:
                    return manifest_fam
                if isa_target and isa_target in avail_set:
                    return isa_target
                mapped = _FAMILY_MAP.get(isa_target, f"{{isa_target}}-all") if isa_target else ""
                if mapped and mapped in avail_set:
                    return mapped
                m_def = re.search(r"default_target_family\\s*=\\s*['\\\"]([^'\\\"]+)['\\\"]", txt)
                if m_def and m_def.group(1) in avail_set:
                    return m_def.group(1)
                return avail[0]
        except Exception:
            pass
    return manifest_fam


_HW_ROCM = _HW_DIR / "rocm"
_ROCM_DEVEL = _SP_DIR / "_rocm_sdk_devel"
_ROCM_CORE = _SP_DIR / "_rocm_sdk_core"
_ROCM_LIBS = _SP_DIR / "_rocm_sdk_libraries"

if (_HW_ROCM / "bin").is_dir():
    _ROCM_ROOT: Path | None = _HW_ROCM
elif (_ROCM_DEVEL / "bin").is_dir():
    _ROCM_ROOT = _ROCM_DEVEL
elif (_ROCM_CORE / "bin").is_dir():
    _ROCM_ROOT = _ROCM_CORE
else:
    _ROCM_ROOT = None

_SDK_ROOTS: list[Path] = []
if _ROCM_ROOT is not None:
    _SDK_ROOTS.append(_ROCM_ROOT)
for _cand_r in [_ROCM_DEVEL, _ROCM_CORE, _ROCM_LIBS] + sorted(_SP_DIR.glob("_rocm_sdk_*")):
    if _cand_r.is_dir() and _cand_r not in _SDK_ROOTS:
        _SDK_ROOTS.append(_cand_r)

_LLVM_BIN: Path | None = None
_DEVICE_LIB: Path | None = None
_ROCM_CMAKE: Path | None = None
_MIOPEN_SYS_DB: Path | None = None

for _r in _SDK_ROOTS:
    if _LLVM_BIN is None and (_r / "lib" / "llvm" / "bin").is_dir():
        _LLVM_BIN = _r / "lib" / "llvm" / "bin"
    if _DEVICE_LIB is None and _has_files(_r / "lib" / "llvm" / "amdgcn" / "bitcode"):
        _DEVICE_LIB = _r / "lib" / "llvm" / "amdgcn" / "bitcode"
    if _ROCM_CMAKE is None and (_r / "lib" / "cmake").is_dir():
        _ROCM_CMAKE = _r / "lib" / "cmake"
    if _MIOPEN_SYS_DB is None and _has_files(_r / "share" / "miopen" / "db"):
        _MIOPEN_SYS_DB = _r / "share" / "miopen" / "db"

if _MIOPEN_SYS_DB is None and _has_files(_SP_DIR / "torch" / "share" / "miopen" / "db"):
    _MIOPEN_SYS_DB = _SP_DIR / "torch" / "share" / "miopen" / "db"

_HW_CUDA = _HW_DIR / "cuda"
_CUDA_ROOT: Path | None = _HW_CUDA if _has_files(_HW_CUDA) else None

_env_enforce("PYTHONHOME", str(_ENV_DIR))
_env_default("PYTHONNOUSERSITE", "1")
_env_default("PYTHONUTF8", "1")
_env_default("PYTHONIOENCODING", "utf-8")
_env_default("PIP_NO_USER", "1")
_env_default("PIP_USER", "0")
_env_default("PIP_DISABLE_PIP_VERSION_CHECK", "1")
_env_enforce("PIP_CACHE_DIR", str(_CACHE / "pip"))
_env_enforce("UV_CACHE_DIR", str(_CACHE / "uv"))
_env_default("UV_NO_CONFIG", "1")
_env_enforce("CCACHE_DIR", str(_CACHE / "ccache"))
_env_enforce("SCCACHE_DIR", str(_CACHE / "sccache"))
_env_enforce("TEMP", str(_TMP))
_env_enforce("TMP", str(_TMP))
_env_enforce("TMPDIR", str(_TMP))
_env_enforce("LOCALAPPDATA", str(_ROOT / ".local_appdata"))
_env_enforce("APPDATA", str(_ROOT / ".appdata"))
_env_enforce("XDG_CACHE_HOME", str(_CACHE))
_env_enforce("XDG_CONFIG_HOME", str(_ROOT / ".config"))
_env_enforce("XDG_DATA_HOME", str(_ROOT / ".local" / "share"))
_env_enforce("XDG_STATE_HOME", str(_ROOT / ".local" / "state"))

_CMAKE_DIRS: list[str] = []
if _ROCM_CMAKE is not None:
    _CMAKE_DIRS.append(str(_ROCM_CMAKE))
if _CUDA_ROOT is not None and (_CUDA_ROOT / "lib" / "cmake").is_dir():
    _CMAKE_DIRS.append(str(_CUDA_ROOT / "lib" / "cmake"))
if (_SP_DIR / "torch" / "share" / "cmake").is_dir():
    _CMAKE_DIRS.append(str(_SP_DIR / "torch" / "share" / "cmake"))
_CMAKE_DIRS.append(str(_ENV_DIR))
_env_enforce("CMAKE_PREFIX_PATH", os.pathsep.join(_CMAKE_DIRS))

if {is_rocm_lit}:
    _env_enforce("HIP_PLATFORM", "amd")
    _isa_t = os.environ.get("PYTORCH_ROCM_ARCH", "").strip() or {llvm_target_lit}
    if _isa_t:
        _env_default("PYTORCH_ROCM_ARCH", _isa_t)
    _env_enforce("ROCM_PATH", str(_ROCM_ROOT) if _ROCM_ROOT is not None else "")
    _env_enforce("ROCM_HOME", str(_ROCM_ROOT) if _ROCM_ROOT is not None else "")
    _env_enforce("HIP_PATH", str(_ROCM_ROOT) if _ROCM_ROOT is not None else "")
    _env_enforce("HIP_CLANG_PATH", str(_LLVM_BIN) if _LLVM_BIN is not None else "")
    _env_enforce("HIP_DEVICE_LIB_PATH", str(_DEVICE_LIB) if _DEVICE_LIB is not None else "")
    _env_enforce("MIOPEN_SYSTEM_DB_PATH", str(_MIOPEN_SYS_DB) if _MIOPEN_SYS_DB is not None else "")
    _det_fam = _resolve_installed_family(_isa_t)
    _env_enforce("ROCM_SDK_TARGET_FAMILY", _det_fam)
elif {is_cuda_lit} and _CUDA_ROOT is not None:
    _env_enforce("CUDA_HOME", str(_CUDA_ROOT))
    _env_enforce("CUDA_PATH", str(_CUDA_ROOT))

os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ.pop("PYTORCH_HIP_ALLOC_CONF", None)
_env_enforce("HF_HOME", str(_HF))
_env_enforce("HF_HUB_CACHE", str(_HF / "hub"))
_env_enforce("HUGGINGFACE_HUB_CACHE", str(_HF / "hub"))
_env_enforce("TORCH_HOME", str(_HF / "torch"))
_env_enforce("TORCH_EXTENSIONS_DIR", str(_HF / "torch_extensions"))
_env_enforce("TORCHINDUCTOR_CACHE_DIR", str(_HF / "torch_inductor"))
_env_enforce("TRITON_CACHE_DIR", str(_HF / "triton"))
_env_enforce("CUDA_CACHE_PATH", str(_CACHE / "cuda"))
_env_enforce("CUPY_CACHE_DIR", str(_CACHE / "cupy"))
_env_enforce("NUMBA_CACHE_DIR", str(_CACHE / "numba"))
_env_enforce("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
_env_enforce("MIOPEN_USER_DB_PATH", str(_HF / "miopen" / "db"))
_env_enforce("MIOPEN_CUSTOM_CACHE_DIR", str(_HF / "miopen" / "kernels"))
_env_default("AMD_COMGR_CACHE", "1")
_env_enforce("AMD_COMGR_CACHE_DIR", str(_HF / "miopen" / "comgr"))
os.environ.pop("MIOPEN_FIND_ENFORCE", None)
if os.environ.get("MIOPEN_FIND_MODE", "").strip().lower() in (
    "",
    "1",
    "2",
    "normal",
    "fast",
):
    os.environ["MIOPEN_FIND_MODE"] = "3"
_env_default("MIOPEN_LOG_LEVEL", "0")
_env_default("MIOPEN_ENABLE_LOGGING", "0")
_env_default("AMD_LOG_LEVEL", "0")
_env_default("TORCH_BLAS_PREFER_HIPBLASLT", "1")
_env_default("HSA_ENABLE_SDMA", "1")
_env_default("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
_env_default("PYTORCH_ALLOC_CONF", "{ALLOC_CONF_POLICY}")

_hsa_ov = {hsa_override_lit}
if _hsa_ov:
    _env_default("HSA_OVERRIDE_GFX_VERSION", _hsa_ov)
elif os.environ.get("HSA_OVERRIDE_GFX_VERSION", "") == "":
    os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)

_TORCH_LIB = _SP_DIR / "torch" / "lib"
_env_bins: list[str] = [
    str(_ENV_DIR if sys.platform == "win32" else _ENV_DIR / "bin"),
]
if sys.platform == "win32":
    _env_bins.append(str(_ENV_DIR / "Scripts"))
if _TORCH_LIB.is_dir():
    _env_bins.append(str(_TORCH_LIB))
if _ROCM_ROOT is not None and (_ROCM_ROOT / "bin").is_dir():
    _env_bins.append(str(_ROCM_ROOT / "bin"))
if _LLVM_BIN is not None and _LLVM_BIN.is_dir():
    _env_bins.append(str(_LLVM_BIN))
for _pkg_b in sorted(_SP_DIR.glob("_rocm_sdk_*/bin")):
    if _pkg_b.is_dir() and str(_pkg_b) not in _env_bins:
        _env_bins.append(str(_pkg_b))
if _CUDA_ROOT is not None and (_CUDA_ROOT / "bin").is_dir():
    _env_bins.append(str(_CUDA_ROOT / "bin"))
elif (_SP_DIR / "nvidia").is_dir():
    for _nv_b in sorted((_SP_DIR / "nvidia").glob("*/bin")):
        if _nv_b.is_dir() and str(_nv_b) not in _env_bins:
            _env_bins.append(str(_nv_b))

_cur_path = os.environ.get("PATH", "")
for _b in reversed(_env_bins):
    if _b not in _cur_path:
        _cur_path = _b + (os.pathsep + _cur_path if _cur_path else "")
os.environ["PATH"] = _cur_path

if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _dll_candidates: list[Path] = [
        _ENV_DIR,
        _ENV_DIR / "Library" / "bin",
        _TORCH_LIB,
    ]
    if _ROCM_ROOT is not None:
        _dll_candidates.append(_ROCM_ROOT / "bin")
    if _LLVM_BIN is not None:
        _dll_candidates.append(_LLVM_BIN)
    _dll_candidates.extend(sorted(_SP_DIR.glob("_rocm_sdk_*/bin")))
    if _CUDA_ROOT is not None:
        _dll_candidates.append(_CUDA_ROOT / "bin")
    elif (_SP_DIR / "nvidia").is_dir():
        _dll_candidates.extend(sorted((_SP_DIR / "nvidia").glob("*/bin")))
        _dll_candidates.extend(sorted((_SP_DIR / "nvidia").glob("*/lib")))
    for _dll_dir in _dll_candidates:
        if _dll_dir.is_dir():
            try:
                os.add_dll_directory(str(_dll_dir))
            except Exception:
                pass
"""
    for sp_dir in site_packages_dirs:
        sp_dir.mkdir(parents=True, exist_ok=True)
        (sp_dir / "sitecustomize.py").write_text(
            sitecustomize_code, encoding="utf-8"
        )


def _portable_launcher_path(raw_path: str, project_root: Path, mode: str) -> str:
    if not raw_path:
        return ""
    p_raw = Path(raw_path)
    rel: Path | None = None
    for cand_p, cand_root in (
        (p_raw, project_root),
        (p_raw.resolve(), project_root.resolve()),
    ):
        try:
            rel = cand_p.relative_to(cand_root)
            break
        except ValueError:
            continue
    if rel is None:
        return raw_path
    rel_parts = rel.parts
    if mode == "bat":
        joined = "\\".join(rel_parts)
        return f"%ROOT_DIR%{joined}" if joined else "%ROOT_DIR%"
    if mode == "ps1":
        joined = "\\".join(rel_parts)
        return f"$ROOT_DIR\\{joined}" if joined else "$ROOT_DIR"
    joined = "/".join(rel_parts)
    return f"${{ROOT_DIR}}/{joined}" if joined else "${ROOT_DIR}"


def _portable_path_list(
    joined_paths: str, project_root: Path, mode: str, sep: str
) -> str:
    items = [p for p in joined_paths.split(os.pathsep) if p]
    return sep.join(
        _portable_launcher_path(p, project_root, mode) for p in items
    )


def _bat_var_line(key: str, val: str) -> str:
    return f'set "{key}={val}"' if val else f'set "{key}="'


def _ps1_var_line(key: str, val: str) -> str:
    if val:
        return f'$env:{key} = "{val}"'
    return f"Remove-Item Env:{key} -ErrorAction SilentlyContinue"


def _sh_var_line(key: str, val: str) -> str:
    return f'export {key}="{val}"' if val else f"unset {key}"


def generate_environment_anchors(
    target_dir: Path,
    executable: Path,
    profile: str,
    llvm_target: str,
    hsa_override: str = "",
) -> None:
    project_root = target_dir.parent
    rel_env_name = target_dir.name

    inject_hermetic_sitecustomize(
        target_dir, profile, llvm_target, hsa_override=hsa_override
    )

    base_env = build_hermetic_env(
        target_dir, profile=profile, llvm_target=llvm_target, executable=executable
    )
    layout = resolve_hardware_prefix_layout(
        target_dir,
        profile=profile,
        llvm_target=llvm_target,
        executable=executable,
        isolated_env=base_env,
    )
    rocm_family = layout["target_family"]
    hip_platform_val = "amd" if profile == "ROCM" else ""
    rocm_arch_val = llvm_target if profile == "ROCM" else ""

    extra_bins = [p for p in layout["extra_bins"].split(os.pathsep) if p]
    extra_libs = [p for p in layout["extra_libs"].split(os.pathsep) if p]

    if os.name == "nt":
        bat_rocm_root = _portable_launcher_path(
            layout["rocm_root"], project_root, "bat"
        )
        bat_cuda_root = _portable_launcher_path(
            layout["cuda_root"], project_root, "bat"
        )
        bat_llvm_bin = _portable_launcher_path(
            layout["llvm_bin"], project_root, "bat"
        )
        bat_device_lib = _portable_launcher_path(
            layout["device_lib"], project_root, "bat"
        )
        bat_miopen_db = _portable_launcher_path(
            layout["miopen_sys_db"], project_root, "bat"
        )
        bat_cmake = _portable_path_list(
            layout["cmake_prefix_path"], project_root, "bat", ";"
        )
        bat_bins = [
            _portable_launcher_path(p, project_root, "bat") for p in extra_bins
        ]
        extra_bat_paths = ";".join(bat_bins) + ";" if bat_bins else ""

        bat_hw_lines = "\n".join(
            [
                _bat_var_line("ROCM_PATH", bat_rocm_root),
                _bat_var_line("ROCM_HOME", bat_rocm_root),
                _bat_var_line("HIP_PATH", bat_rocm_root),
                _bat_var_line("HIP_PLATFORM", hip_platform_val),
                _bat_var_line("PYTORCH_ROCM_ARCH", rocm_arch_val),
                _bat_var_line("ROCM_SDK_TARGET_FAMILY", rocm_family),
                _bat_var_line("HIP_CLANG_PATH", bat_llvm_bin),
                _bat_var_line("HIP_DEVICE_LIB_PATH", bat_device_lib),
                _bat_var_line("MIOPEN_SYSTEM_DB_PATH", bat_miopen_db),
                _bat_var_line("CUDA_HOME", bat_cuda_root),
                _bat_var_line("CUDA_PATH", bat_cuda_root),
                _bat_var_line("CMAKE_PREFIX_PATH", bat_cmake),
                _bat_var_line("HSA_OVERRIDE_GFX_VERSION", hsa_override),
            ]
        )
        hsa_echo_bat = (
            "echo  HSA GFX Override: %HSA_OVERRIDE_GFX_VERSION%"
            if hsa_override
            else ""
        )

        bat_content = f"""@echo off
title Hermetic Sandbox Environment
color 0b
set "ROOT_DIR=%~dp0"
set "PYTHONPATH="
set "PYTHONCASEOK="
set "VIRTUAL_ENV="
set "CONDA_PREFIX="
set "CONDA_DEFAULT_ENV="
set "TRANSFORMERS_CACHE="
set "MIOPEN_FIND_ENFORCE="
set "PYTORCH_CUDA_ALLOC_CONF="
set "PYTORCH_HIP_ALLOC_CONF="
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_NO_USER=1"
set "PIP_USER=0"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PIP_CONFIG_FILE=NUL"
set "UV_NO_CONFIG=1"
set "PYTHONHOME=%ROOT_DIR%{rel_env_name}"
{bat_hw_lines}
set "PATH=%PYTHONHOME%;%PYTHONHOME%\\Scripts;{extra_bat_paths}%SystemRoot%\\System32;%SystemRoot%;%SystemRoot%\\System32\\Wbem;%SystemRoot%\\System32\\WindowsPowerShell\\v1.0"
set "TEMP=%ROOT_DIR%.tmp"
set "TMP=%ROOT_DIR%.tmp"
set "TMPDIR=%ROOT_DIR%.tmp"
set "PIP_CACHE_DIR=%ROOT_DIR%.cache\\pip"
set "UV_CACHE_DIR=%ROOT_DIR%.cache\\uv"
set "CCACHE_DIR=%ROOT_DIR%.cache\\ccache"
set "SCCACHE_DIR=%ROOT_DIR%.cache\\sccache"
set "CUDA_CACHE_PATH=%ROOT_DIR%.cache\\cuda"
set "CUPY_CACHE_DIR=%ROOT_DIR%.cache\\cupy"
set "NUMBA_CACHE_DIR=%ROOT_DIR%.cache\\numba"
set "MPLCONFIGDIR=%ROOT_DIR%.cache\\matplotlib"
set "LOCALAPPDATA=%ROOT_DIR%.local_appdata"
set "APPDATA=%ROOT_DIR%.appdata"
set "XDG_CACHE_HOME=%ROOT_DIR%.cache"
set "XDG_CONFIG_HOME=%ROOT_DIR%.config"
set "XDG_DATA_HOME=%ROOT_DIR%.local\\share"
set "XDG_STATE_HOME=%ROOT_DIR%.local\\state"
set "HF_HOME=%ROOT_DIR%.hf_cache"
set "HF_HUB_CACHE=%ROOT_DIR%.hf_cache\\hub"
set "HUGGINGFACE_HUB_CACHE=%ROOT_DIR%.hf_cache\\hub"
set "TORCH_HOME=%ROOT_DIR%.hf_cache\\torch"
set "TORCH_EXTENSIONS_DIR=%ROOT_DIR%.hf_cache\\torch_extensions"
set "TORCHINDUCTOR_CACHE_DIR=%ROOT_DIR%.hf_cache\\torch_inductor"
set "TRITON_CACHE_DIR=%ROOT_DIR%.hf_cache\\triton"
set "MIOPEN_USER_DB_PATH=%ROOT_DIR%.hf_cache\\miopen\\db"
set "MIOPEN_CUSTOM_CACHE_DIR=%ROOT_DIR%.hf_cache\\miopen\\kernels"
set "AMD_COMGR_CACHE=1"
set "AMD_COMGR_CACHE_DIR=%ROOT_DIR%.hf_cache\\miopen\\comgr"
set "MIOPEN_FIND_MODE=3"
set "MIOPEN_LOG_LEVEL=0"
set "MIOPEN_ENABLE_LOGGING=0"
set "AMD_LOG_LEVEL=0"
set "TORCH_BLAS_PREFER_HIPBLASLT=1"
set "HSA_ENABLE_SDMA=1"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
set "PYTORCH_ALLOC_CONF={ALLOC_CONF_POLICY}"
echo =======================================================================
echo  Hermetic Sandbox Shell Active (CMD)
echo  Interpreter: %PYTHONHOME%\\python.exe
echo  Compute Profile: {profile} [ISA: %PYTORCH_ROCM_ARCH% / Family: %ROCM_SDK_TARGET_FAMILY%]
echo  Hardware Prefix: %PYTHONHOME%\\hw_prefix
echo  Cache Anchor: %HF_HOME%
echo  Allocator Policy: %PYTORCH_ALLOC_CONF%
{hsa_echo_bat}
echo  Isolation Status: Zero host leakage. Sovereign prefix active.
echo =======================================================================
cmd /k
"""
        (project_root / "launch_env.bat").write_text(bat_content, encoding="utf-8")

        ps1_rocm_root = _portable_launcher_path(
            layout["rocm_root"], project_root, "ps1"
        )
        ps1_cuda_root = _portable_launcher_path(
            layout["cuda_root"], project_root, "ps1"
        )
        ps1_llvm_bin = _portable_launcher_path(
            layout["llvm_bin"], project_root, "ps1"
        )
        ps1_device_lib = _portable_launcher_path(
            layout["device_lib"], project_root, "ps1"
        )
        ps1_miopen_db = _portable_launcher_path(
            layout["miopen_sys_db"], project_root, "ps1"
        )
        ps1_cmake = _portable_path_list(
            layout["cmake_prefix_path"], project_root, "ps1", ";"
        )
        ps1_bins = [
            _portable_launcher_path(p, project_root, "ps1") for p in extra_bins
        ]
        extra_ps1_paths = ";".join(ps1_bins) + ";" if ps1_bins else ""

        ps1_hw_lines = "\n".join(
            [
                _ps1_var_line("ROCM_PATH", ps1_rocm_root),
                _ps1_var_line("ROCM_HOME", ps1_rocm_root),
                _ps1_var_line("HIP_PATH", ps1_rocm_root),
                _ps1_var_line("HIP_PLATFORM", hip_platform_val),
                _ps1_var_line("PYTORCH_ROCM_ARCH", rocm_arch_val),
                _ps1_var_line("ROCM_SDK_TARGET_FAMILY", rocm_family),
                _ps1_var_line("HIP_CLANG_PATH", ps1_llvm_bin),
                _ps1_var_line("HIP_DEVICE_LIB_PATH", ps1_device_lib),
                _ps1_var_line("MIOPEN_SYSTEM_DB_PATH", ps1_miopen_db),
                _ps1_var_line("CUDA_HOME", ps1_cuda_root),
                _ps1_var_line("CUDA_PATH", ps1_cuda_root),
                _ps1_var_line("CMAKE_PREFIX_PATH", ps1_cmake),
                _ps1_var_line("HSA_OVERRIDE_GFX_VERSION", hsa_override),
            ]
        )
        hsa_echo_ps1 = (
            'Write-Host " HSA GFX Override: $env:HSA_OVERRIDE_GFX_VERSION"'
            if hsa_override
            else ""
        )

        ps1_content = f"""$ErrorActionPreference = "Continue"
$ROOT_DIR = $PSScriptRoot
Remove-Item Env:PYTHONPATH, Env:PYTHONCASEOK, Env:VIRTUAL_ENV, Env:CONDA_PREFIX, Env:CONDA_DEFAULT_ENV, Env:TRANSFORMERS_CACHE, Env:MIOPEN_FIND_ENFORCE, Env:PYTORCH_CUDA_ALLOC_CONF, Env:PYTORCH_HIP_ALLOC_CONF -ErrorAction SilentlyContinue
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PIP_NO_USER = "1"
$env:PIP_USER = "0"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:PIP_CONFIG_FILE = "NUL"
$env:UV_NO_CONFIG = "1"
$env:PYTHONHOME = "$ROOT_DIR\\{rel_env_name}"
{ps1_hw_lines}
$sysRoot = $env:SystemRoot
$env:PATH = "$env:PYTHONHOME;$env:PYTHONHOME\\Scripts;{extra_ps1_paths}$sysRoot\\System32;$sysRoot;$sysRoot\\System32\\Wbem;$sysRoot\\System32\\WindowsPowerShell\\v1.0"
$env:TEMP = "$ROOT_DIR\\.tmp"
$env:TMP = "$ROOT_DIR\\.tmp"
$env:TMPDIR = "$ROOT_DIR\\.tmp"
$env:PIP_CACHE_DIR = "$ROOT_DIR\\.cache\\pip"
$env:UV_CACHE_DIR = "$ROOT_DIR\\.cache\\uv"
$env:CCACHE_DIR = "$ROOT_DIR\\.cache\\ccache"
$env:SCCACHE_DIR = "$ROOT_DIR\\.cache\\sccache"
$env:CUDA_CACHE_PATH = "$ROOT_DIR\\.cache\\cuda"
$env:CUPY_CACHE_DIR = "$ROOT_DIR\\.cache\\cupy"
$env:NUMBA_CACHE_DIR = "$ROOT_DIR\\.cache\\numba"
$env:MPLCONFIGDIR = "$ROOT_DIR\\.cache\\matplotlib"
$env:LOCALAPPDATA = "$ROOT_DIR\\.local_appdata"
$env:APPDATA = "$ROOT_DIR\\.appdata"
$env:XDG_CACHE_HOME = "$ROOT_DIR\\.cache"
$env:XDG_CONFIG_HOME = "$ROOT_DIR\\.config"
$env:XDG_DATA_HOME = "$ROOT_DIR\\.local\\share"
$env:XDG_STATE_HOME = "$ROOT_DIR\\.local\\state"
$env:HF_HOME = "$ROOT_DIR\\.hf_cache"
$env:HF_HUB_CACHE = "$ROOT_DIR\\.hf_cache\\hub"
$env:HUGGINGFACE_HUB_CACHE = "$ROOT_DIR\\.hf_cache\\hub"
$env:TORCH_HOME = "$ROOT_DIR\\.hf_cache\\torch"
$env:TORCH_EXTENSIONS_DIR = "$ROOT_DIR\\.hf_cache\\torch_extensions"
$env:TORCHINDUCTOR_CACHE_DIR = "$ROOT_DIR\\.hf_cache\\torch_inductor"
$env:TRITON_CACHE_DIR = "$ROOT_DIR\\.hf_cache\\triton"
$env:MIOPEN_USER_DB_PATH = "$ROOT_DIR\\.hf_cache\\miopen\\db"
$env:MIOPEN_CUSTOM_CACHE_DIR = "$ROOT_DIR\\.hf_cache\\miopen\\kernels"
$env:AMD_COMGR_CACHE = "1"
$env:AMD_COMGR_CACHE_DIR = "$ROOT_DIR\\.hf_cache\\miopen\\comgr"
$env:MIOPEN_FIND_MODE = "3"
$env:MIOPEN_LOG_LEVEL = "0"
$env:MIOPEN_ENABLE_LOGGING = "0"
$env:AMD_LOG_LEVEL = "0"
$env:TORCH_BLAS_PREFER_HIPBLASLT = "1"
$env:HSA_ENABLE_SDMA = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:PYTORCH_ALLOC_CONF = "{ALLOC_CONF_POLICY}"
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host " Hermetic Sandbox Active (PowerShell)" -ForegroundColor Cyan
Write-Host " Interpreter: $env:PYTHONHOME\\python.exe"
Write-Host " Compute Profile: {profile} [ISA: $env:PYTORCH_ROCM_ARCH / Family: $env:ROCM_SDK_TARGET_FAMILY]"
Write-Host " Hardware Prefix: $env:PYTHONHOME\\hw_prefix"
Write-Host " Cache Anchor: $env:HF_HOME"
Write-Host " Allocator Policy: $env:PYTORCH_ALLOC_CONF"
{hsa_echo_ps1}
Write-Host " Isolation Status: Zero host leakage. Sovereign prefix active." -ForegroundColor Green
Write-Host "=======================================================================" -ForegroundColor Cyan
"""
        (project_root / "launch_env.ps1").write_text(ps1_content, encoding="utf-8")
    else:
        sh_rocm_root = _portable_launcher_path(
            layout["rocm_root"], project_root, "sh"
        )
        sh_cuda_root = _portable_launcher_path(
            layout["cuda_root"], project_root, "sh"
        )
        sh_llvm_bin = _portable_launcher_path(
            layout["llvm_bin"], project_root, "sh"
        )
        sh_device_lib = _portable_launcher_path(
            layout["device_lib"], project_root, "sh"
        )
        sh_miopen_db = _portable_launcher_path(
            layout["miopen_sys_db"], project_root, "sh"
        )
        sh_cmake = _portable_path_list(
            layout["cmake_prefix_path"], project_root, "sh", ":"
        )
        sh_bins = [
            _portable_launcher_path(p, project_root, "sh") for p in extra_bins
        ]
        sh_libs = [
            _portable_launcher_path(p, project_root, "sh") for p in extra_libs
        ]
        extra_sh_paths = ":".join(sh_bins) + ":" if sh_bins else ""
        ld_lib_line = (
            f'export LD_LIBRARY_PATH="{":".join(sh_libs)}"'
            if (sh_libs and sys.platform.startswith("linux"))
            else "unset LD_LIBRARY_PATH"
        )

        sh_hw_lines = "\n".join(
            [
                _sh_var_line("ROCM_PATH", sh_rocm_root),
                _sh_var_line("ROCM_HOME", sh_rocm_root),
                _sh_var_line("HIP_PATH", sh_rocm_root),
                _sh_var_line("HIP_PLATFORM", hip_platform_val),
                _sh_var_line("PYTORCH_ROCM_ARCH", rocm_arch_val),
                _sh_var_line("ROCM_SDK_TARGET_FAMILY", rocm_family),
                _sh_var_line("HIP_CLANG_PATH", sh_llvm_bin),
                _sh_var_line("HIP_DEVICE_LIB_PATH", sh_device_lib),
                _sh_var_line("MIOPEN_SYSTEM_DB_PATH", sh_miopen_db),
                _sh_var_line("CUDA_HOME", sh_cuda_root),
                _sh_var_line("CUDA_PATH", sh_cuda_root),
                _sh_var_line("CMAKE_PREFIX_PATH", sh_cmake),
                _sh_var_line("HSA_OVERRIDE_GFX_VERSION", hsa_override),
            ]
        )
        hsa_echo_sh = (
            'echo " HSA GFX Override: ${HSA_OVERRIDE_GFX_VERSION}"'
            if hsa_override
            else ""
        )

        sh_content = f"""#!/usr/bin/env bash
ROOT_DIR="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" &> /dev/null && pwd)"
unset PYTHONPATH PYTHONCASEOK VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV TRANSFORMERS_CACHE MIOPEN_FIND_ENFORCE PYTORCH_CUDA_ALLOC_CONF PYTORCH_HIP_ALLOC_CONF
export PYTHONNOUSERSITE="1"
export PYTHONUTF8="1"
export PYTHONIOENCODING="utf-8"
export PIP_NO_USER="1"
export PIP_USER="0"
export PIP_DISABLE_PIP_VERSION_CHECK="1"
export PIP_CONFIG_FILE="/dev/null"
export UV_NO_CONFIG="1"
export PYTHONHOME="${{ROOT_DIR}}/{rel_env_name}"
{sh_hw_lines}
export PATH="${{PYTHONHOME}}/bin:${{PYTHONHOME}}:{extra_sh_paths}/usr/local/bin:/usr/bin:/bin"
{ld_lib_line}
export TEMP="${{ROOT_DIR}}/.tmp"
export TMP="${{ROOT_DIR}}/.tmp"
export TMPDIR="${{ROOT_DIR}}/.tmp"
export PIP_CACHE_DIR="${{ROOT_DIR}}/.cache/pip"
export UV_CACHE_DIR="${{ROOT_DIR}}/.cache/uv"
export CCACHE_DIR="${{ROOT_DIR}}/.cache/ccache"
export SCCACHE_DIR="${{ROOT_DIR}}/.cache/sccache"
export CUDA_CACHE_PATH="${{ROOT_DIR}}/.cache/cuda"
export CUPY_CACHE_DIR="${{ROOT_DIR}}/.cache/cupy"
export NUMBA_CACHE_DIR="${{ROOT_DIR}}/.cache/numba"
export MPLCONFIGDIR="${{ROOT_DIR}}/.cache/matplotlib"
export XDG_CACHE_HOME="${{ROOT_DIR}}/.cache"
export XDG_CONFIG_HOME="${{ROOT_DIR}}/.config"
export XDG_DATA_HOME="${{ROOT_DIR}}/.local/share"
export XDG_STATE_HOME="${{ROOT_DIR}}/.local/state"
export HF_HOME="${{ROOT_DIR}}/.hf_cache"
export HF_HUB_CACHE="${{ROOT_DIR}}/.hf_cache/hub"
export HUGGINGFACE_HUB_CACHE="${{ROOT_DIR}}/.hf_cache/hub"
export TORCH_HOME="${{ROOT_DIR}}/.hf_cache/torch"
export TORCH_EXTENSIONS_DIR="${{ROOT_DIR}}/.hf_cache/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="${{ROOT_DIR}}/.hf_cache/torch_inductor"
export TRITON_CACHE_DIR="${{ROOT_DIR}}/.hf_cache/triton"
export MIOPEN_USER_DB_PATH="${{ROOT_DIR}}/.hf_cache/miopen/db"
export MIOPEN_CUSTOM_CACHE_DIR="${{ROOT_DIR}}/.hf_cache/miopen/kernels"
export AMD_COMGR_CACHE="1"
export AMD_COMGR_CACHE_DIR="${{ROOT_DIR}}/.hf_cache/miopen/comgr"
export MIOPEN_FIND_MODE="3"
export MIOPEN_LOG_LEVEL="0"
export MIOPEN_ENABLE_LOGGING="0"
export AMD_LOG_LEVEL="0"
export TORCH_BLAS_PREFER_HIPBLASLT="1"
export HSA_ENABLE_SDMA="1"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_ALLOC_CONF="{ALLOC_CONF_POLICY}"
echo "======================================================================="
echo " Hermetic Sandbox Shell Active"
echo " Interpreter: ${{PYTHONHOME}}/bin/python"
echo " Compute Profile: {profile} [ISA: ${{PYTORCH_ROCM_ARCH:-n/a}} / Family: ${{ROCM_SDK_TARGET_FAMILY:-n/a}}]"
echo " Hardware Prefix: ${{PYTHONHOME}}/hw_prefix"
echo " Cache Anchor: ${{HF_HOME}}"
echo " Allocator Policy: ${{PYTORCH_ALLOC_CONF}}"
{hsa_echo_sh}
echo " Isolation Status: Zero host leakage. Sovereign prefix active."
echo "======================================================================="
exec "${{SHELL:-/bin/bash}}"
"""
        launcher_sh = project_root / "launch_env.sh"
        launcher_sh.write_text(sh_content, encoding="utf-8")
        launcher_sh.chmod(0o755)

    print_status(
        "Hermetic launch scripts & sovereign sitecustomize deployed.",
        "SUCCESS",
    )


def run_hardware_acceptance_gate(
    executable: Path,
    target_sandbox: Path,
    profile: str,
    llvm_target: str,
    hsa_override: str = "",
) -> None:
    print_status("Executing applied-science hardware acceptance gate...", "INFO")
    test_env = build_hermetic_env(
        target_sandbox,
        profile=profile,
        llvm_target=llvm_target,
        executable=executable,
    )
    if hsa_override:
        test_env["HSA_OVERRIDE_GFX_VERSION"] = hsa_override
    else:
        test_env.pop("HSA_OVERRIDE_GFX_VERSION", None)

    if sys.platform == "win32":
        adapters = query_dxgi_adapters_win32()
        for idx, ad in enumerate(adapters):
            print_status(
                f"[DXGI ADAPTER {idx}] {ad['name']} | Dedicated VRAM: "
                f"{ad['vram_avail_gb']:.2f}/{ad['vram_gb']:.2f} GB Avail | "
                f"Shared SoC Used: {ad['svram_used_gb']:.2f}/{ad['svram_budget_gb']:.2f} GB",
                "SUCCESS",
            )

    test_code = (
        "import os, pathlib, shutil, sys, sysconfig, torch, torch.nn.functional as F, torchaudio, torchvision\n"
        "env_root = pathlib.Path(sys.prefix).resolve()\n"
        "inc_cand = pathlib.Path(sysconfig.get_path('include') or (env_root / 'include'))\n"
        "has_py_h = (inc_cand / 'Python.h').is_file() or (env_root / 'include' / 'Python.h').is_file()\n"
        "has_py_lib = True\n"
        "if os.name == 'nt':\n"
        "    has_py_lib = (env_root / 'libs' / f'python{sys.version_info.major}{sys.version_info.minor}.lib').is_file()\n"
        "ninja_p, cmake_p = shutil.which('ninja'), shutil.which('cmake')\n"
        "assert ninja_p and cmake_p, f'Missing hermetic build tools: ninja={ninja_p}, cmake={cmake_p}'\n"
        "has_xpu = hasattr(torch, 'xpu') and torch.xpu.is_available()\n"
        "has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()\n"
        "device = 'cuda' if torch.cuda.is_available() else ('xpu' if has_xpu else ('mps' if has_mps else 'cpu'))\n"
        "backend = getattr(torch.version, 'hip', None) or getattr(torch.version, 'cuda', None) or getattr(torch.version, 'xpu', None) or device\n"
        "name = torch.cuda.get_device_name(0) if device == 'cuda' else (torch.xpu.get_device_name(0) if device == 'xpu' else ('Apple Silicon' if device == 'mps' else 'Host CPU'))\n"
        "arch = getattr(torch.cuda.get_device_properties(0), 'gcnArchName', 'n/a') if device == 'cuda' else 'n/a'\n"
        "torch_kpacks = len(list((pathlib.Path(torch.__file__).parent / '.kpack').glob('*.kpack'))) if (pathlib.Path(torch.__file__).parent / '.kpack').is_dir() else 0\n"
        "if getattr(torch.version, 'hip', None):\n"
        "    rocm_raw = os.environ.get('ROCM_PATH', '')\n"
        "    assert rocm_raw, 'ROCM_PATH unset inside ROCm runtime'\n"
        "    rocm_res = pathlib.Path(rocm_raw).resolve()\n"
        "    assert env_root in rocm_res.parents or rocm_res == env_root, f'ROCM_PATH escaped sandbox: {rocm_res}'\n"
        "print(f'[ACCEPTANCE] Interpreter: {sys.executable} (Python {sys.version.split()[0]} | C-API={has_py_h and has_py_lib} | torch {torch.__version__} | torchvision {torchvision.__version__} | torchaudio {torchaudio.__version__})')\n"
        "print(f'[ACCEPTANCE] Device: {device.upper()} | Backend: {backend} | GPU: {name} | ISA: {arch} | TorchKpacks: {torch_kpacks}')\n"
        "hw_p = os.environ.get('ROCM_PATH') or os.environ.get('CUDA_HOME') or str(env_root / 'hw_prefix')\n"
        "print(f'[ACCEPTANCE] Sovereign Prefix: {hw_p} | FAMILY={os.environ.get(\"ROCM_SDK_TARGET_FAMILY\", \"n/a\")} | CMAKE={bool(os.environ.get(\"CMAKE_PREFIX_PATH\"))}')\n"
        "print(f'[ACCEPTANCE] Runtime Flags: MIOPEN_FIND_MODE={os.environ.get(\"MIOPEN_FIND_MODE\")} | HIPBLASLT={os.environ.get(\"TORCH_BLAS_PREFER_HIPBLASLT\")} | ALLOC_CONF={os.environ.get(\"PYTORCH_ALLOC_CONF\")}')\n"
        "dtype = torch.bfloat16 if device in ('cuda', 'xpu') else torch.float32\n"
        "x = torch.randn((512, 512), device=device, dtype=dtype, requires_grad=True)\n"
        "y = torch.mm(x, x.T).float()\n"
        "y.mean().backward()\n"
        "q = torch.randn((2, 8, 128, 64), device=device, dtype=dtype, requires_grad=True)\n"
        "k = torch.randn((2, 8, 128, 64), device=device, dtype=dtype, requires_grad=True)\n"
        "v = torch.randn((2, 8, 128, 64), device=device, dtype=dtype, requires_grad=True)\n"
        "attn = F.scaled_dot_product_attention(q, k, v)\n"
        "attn.float().mean().backward()\n"
        "c2_in = torch.randn((1, 64, 32, 32), device=device, dtype=dtype, requires_grad=True)\n"
        "c2_w = torch.randn((64, 64, 3, 3), device=device, dtype=dtype, requires_grad=True)\n"
        "c2_out = F.conv2d(c2_in, c2_w, padding=1)\n"
        "c2_out.float().mean().backward()\n"
        "c1_in = torch.randn((2, 128, 1024), device=device, dtype=dtype, requires_grad=True)\n"
        "c1_w = torch.randn((128, 128, 7), device=device, dtype=dtype, requires_grad=True)\n"
        "c1_out = F.conv1d(c1_in, c1_w, padding=6, dilation=2)\n"
        "c1_out.float().mean().backward()\n"
        "boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]], device=device, dtype=torch.float32)\n"
        "scores = torch.tensor([0.9, 0.8, 0.7], device=device, dtype=torch.float32)\n"
        "nms_keep = torchvision.ops.nms(boxes, scores, 0.5)\n"
        "assert nms_keep.numel() == 2, f'Unexpected torchvision.ops.nms output: {nms_keep}'\n"
        "fp8_tag = 'N/A'\n"
        "if device == 'cuda' and hasattr(torch, '_scaled_mm') and hasattr(torch, 'float8_e4m3fn'):\n"
        "    arch_base = str(arch).split(':', 1)[0].lower()\n"
        "    rocm_fp8 = bool(getattr(torch.version, 'hip', None)) and arch_base.startswith(('gfx120', 'gfx125', 'gfx942', 'gfx950'))\n"
        "    cuda_fp8 = (not getattr(torch.version, 'hip', None)) and torch.cuda.get_device_capability(0) >= (8, 9)\n"
        "    if rocm_fp8 or cuda_fp8:\n"
        "        fp8_dt = torch.float8_e4m3fnuz if arch_base == 'gfx942' and hasattr(torch, 'float8_e4m3fnuz') else torch.float8_e4m3fn\n"
        "        try:\n"
        "            a8 = torch.randn((256, 256), device=device, dtype=torch.bfloat16).to(fp8_dt)\n"
        "            b8 = torch.randn((256, 256), device=device, dtype=torch.bfloat16).to(fp8_dt).t()\n"
        "            one = torch.tensor(1.0, device=device, dtype=torch.float32)\n"
        "            o8 = torch._scaled_mm(a8, b8, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)\n"
        "            t8 = o8[0] if isinstance(o8, tuple) else o8\n"
        "            assert torch.isfinite(t8.float()).all(), 'Non-finite FP8 tensor output'\n"
        "            fp8_tag = f'PASSED ({fp8_dt})'\n"
        "        except (RuntimeError, NotImplementedError) as fp8_exc:\n"
        "            fp8_tag = f'SKIPPED ({type(fp8_exc).__name__})'\n"
        "wav = torch.randn((2, 16384), device=device, dtype=torch.float32)\n"
        "win = torch.hann_window(1024, device=device, dtype=torch.float32)\n"
        "spec = torch.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window=win, return_complex=True)\n"
        "rec = torch.istft(spec, n_fft=1024, hop_length=256, win_length=1024, window=win, length=16384)\n"
        "if device == 'cuda': torch.cuda.synchronize()\n"
        "elif device == 'xpu': torch.xpu.synchronize()\n"
        "assert x.grad is not None and q.grad is not None and c2_w.grad is not None and c1_w.grad is not None\n"
        "assert torch.isfinite(y).all() and torch.isfinite(attn.float()).all() and torch.isfinite(c2_out.float()).all() and torch.isfinite(c1_out.float()).all() and torch.isfinite(rec).all(), 'Non-finite tensor output detected'\n"
        "print(f'[ACCEPTANCE] BF16 GEMM, SDPA Autograd, Conv2d/Conv1d Autograd, Vision NMS, FP8={fp8_tag} & Complex64 STFT/iSTFT PASSED on {device.upper()}.')\n"
    )
    res = subprocess.run(
        [str(executable), "-c", test_code],
        env=test_env,
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
    )
    if res.returncode == 0:
        for line in res.stdout.strip().splitlines():
            print_status(line, "SUCCESS")
    else:
        print_status(
            f"Acceptance test diagnostic failure:\n{res.stderr.strip() or res.stdout.strip()}",
            "ERROR",
        )
        sys.exit(res.returncode)


def _find_existing_sandbox() -> Path | None:
    candidates = sorted(
        [p for p in ROOT_DIR.glob("py_env_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Hardware-Accelerated Sovereign Virtualization Conditioner."
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Target Python runtime version (defaults to highest ABI with SOTA compute wheels).",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Execute hands-free without interactive prompts.",
    )
    parser.add_argument(
        "--launchers-only",
        action="store_true",
        help="Materialize sovereign hw_prefix, regenerate sitecustomize.py and launch scripts for the active py_env_* sandbox, and run acceptance gate.",
    )
    parser.add_argument(
        "--hermetic-internal",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.launchers_only:
        existing_sandbox = _find_existing_sandbox()
        if existing_sandbox is None:
            print_status("No existing py_env_* directory found.", "ERROR")
            sys.exit(1)
        quarantine_sandbox_sitecustomize(existing_sandbox)
        if os.name == "nt":
            exe = existing_sandbox / "python.exe"
        else:
            exe = (
                existing_sandbox / "bin" / "python3"
                if (existing_sandbox / "bin" / "python3").is_file()
                else existing_sandbox / "bin" / "python"
            )
        profile, llvm_target = resolve_hardware_matrix()
        hsa_override = (
            ""
            if _has_installed_device_kpack(existing_sandbox, llvm_target)
            else LEGACY_HSA_OVERRIDES.get(llvm_target, "")
        )
        isolated_env = build_hermetic_env(
            existing_sandbox,
            profile=profile,
            llvm_target=llvm_target,
            executable=exe,
        )
        materialize_hardware_prefix(
            executable=exe,
            isolated_env=isolated_env,
            target_sandbox=existing_sandbox,
            profile=profile,
            llvm_target=llvm_target,
        )
        generate_environment_anchors(
            existing_sandbox,
            exe,
            profile=profile,
            llvm_target=llvm_target,
            hsa_override=hsa_override,
        )
        run_hardware_acceptance_gate(
            exe,
            existing_sandbox,
            profile=profile,
            llvm_target=llvm_target,
            hsa_override=hsa_override,
        )
        return

    if not args.hermetic_internal:
        print_status("=" * 72)
        print_status(
            "            INTELLIVENV SOVEREIGN RUNTIME CONDITIONER", "SUCCESS"
        )
        print_status("=" * 72)

    profile, llvm_target = resolve_hardware_matrix()
    selected_version = args.version
    if selected_version is None:
        matrix = resolve_dynamic_python_matrix()
        default_key, default_ver = select_sota_python_version(
            matrix, profile, llvm_target
        )
        if args.auto or not sys.stdin.isatty():
            selected_version = default_ver
        else:
            for key, ver in matrix.items():
                tag_note = " [SOTA Compute Aligned]" if key == default_key else ""
                print(f"  [{key}] Python {ver}{tag_note}")
            manual_opt = str(len(matrix) + 1)
            print(f"  [{manual_opt}] Manual Version Entry")
            choice = input(
                f"\nSelect core runtime configuration [1-{manual_opt}] "
                f"(Default: {default_key} [Python {default_ver}]): "
            ).strip()
            if not choice:
                selected_version = default_ver
            elif choice in matrix:
                selected_version = matrix[choice]
            elif choice == manual_opt:
                selected_version = input("Enter custom Python version: ").strip()
            else:
                selected_version = default_ver

    if not selected_version:
        selected_version = DEFAULT_PYTHON_VERSION

    target_sandbox = ROOT_DIR / f"py_env_{selected_version.replace('.', '_')}"
    quarantine_sandbox_sitecustomize(target_sandbox)
    executable = fetch_runtime(selected_version, target_sandbox)
    isolated_env = build_hermetic_env(
        target_sandbox,
        profile=profile,
        llvm_target=llvm_target,
        executable=executable,
    )

    current_exe = Path(sys.executable).resolve()
    target_exe = executable.resolve()
    if current_exe != target_exe and not args.hermetic_internal:
        print_status(
            f"Detaching from host system Python ({sys.version.split()[0]} -> {current_exe}).",
            "INFO",
        )
        print_status(
            f"Trampolining into hermetic runtime: {target_exe}",
            "SUCCESS",
        )
        respawn_cmd = [
            str(target_exe),
            "-I",
            str(Path(__file__).resolve()),
            "--version",
            selected_version,
            "--hermetic-internal",
        ]
        res = subprocess.run(respawn_cmd, env=isolated_env)
        sys.exit(res.returncode)

    cond_meta = condition_pytorch_runtime(
        executable, isolated_env, profile, llvm_target, target_sandbox
    )
    hsa_override = cond_meta["hsa_override"]
    generate_environment_anchors(
        target_sandbox,
        executable,
        profile=profile,
        llvm_target=llvm_target,
        hsa_override=hsa_override,
    )
    run_hardware_acceptance_gate(
        executable,
        target_sandbox,
        profile=profile,
        llvm_target=llvm_target,
        hsa_override=hsa_override,
    )
    print_status(
        f"Hermetic sandbox & sovereign prefix conditioned at {target_sandbox.name}.",
        "SUCCESS",
    )


if __name__ == "__main__":
    main()
