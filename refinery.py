from __future__ import annotations

import atexit
import importlib.util
import math
from pathlib import Path
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

_DOODLE_ROOT: Path = Path(__file__).resolve().parent
_LOCAL_CONDUCTOR_DIR: Path = _DOODLE_ROOT / "conductor"

MIN_CONDUCTOR_ABI_VERSION: int = 3

_REQUIRED_CONDUCTOR_METHODS: Tuple[str, ...] = (
    "contender_cast",
    "rms_norm_contender",
    "residual_rms_norm_contender",
    "swiglu_contender",
    "swiglu_packed_contender",
    "qk_norm_rope_contender",
    "qk_norm_rope_packed_contender",
    "qk_norm_rope_packed_strided_contender",
    "snake_beta_f32",
    "analyze_abc",
    "rewrite_abc_tempo",
    "abc_extract_midi",
    "abc_to_midi_smf1",
    "encode_midi_smf1",
    "decode_midi_smf1",
    "estimate_acoustic_bpm",
)

DTYPE_MAP: Dict[str, Optional[torch.dtype]] = {
    "native": None,
    "none": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "e4m3fn": getattr(torch, "float8_e4m3fn", None),
    "e4m3fnuz": getattr(torch, "float8_e4m3fnuz", None),
    "e5m2": getattr(torch, "float8_e5m2", None),
    "e5m2fnuz": getattr(torch, "float8_e5m2fnuz", None),
}

FP8_DTYPES: List[torch.dtype] = [
    dt
    for key in ("e4m3fn", "e4m3fnuz", "e5m2", "e5m2fnuz")
    if (dt := DTYPE_MAP.get(key)) is not None
]

_CONDUCTOR_DST_DTYPES: Set[torch.dtype] = {
    dt
    for dt in (
        torch.float32,
        torch.bfloat16,
        torch.float16,
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dt is not None
}
_CONDUCTOR_SRC_DTYPES: Set[torch.dtype] = _CONDUCTOR_DST_DTYPES | {torch.float64}

_CONDUCTOR_LOCK = threading.Lock()
_CONDUCTOR_INIT_DONE: bool = False
_CONDUCTOR_MOD: Optional[Any] = None
_CONDUCTOR_CTX: Optional[Any] = None
_CONDUCTOR_IS_LOCAL_BUNDLE: bool = False
_CONDUCTOR_GPU_WARMED: Dict[int, bool] = {}

_B4_MATRIX: Tuple[Tuple[int, int, int, int], ...] = (
    (0, 8, 2, 10),
    (12, 4, 14, 6),
    (3, 11, 1, 9),
    (15, 7, 13, 5),
)


class _ConductorAbiError(RuntimeError):
    pass


def _platform_conductor_lib_names() -> Tuple[str, ...]:
    if sys.platform == "win32":
        return ("conductor.dll", "libconductor.dll")
    if sys.platform == "darwin":
        return ("libconductor.dylib", "conductor.dylib", "libconductor.so")
    return ("libconductor.so", "conductor.so")


def _is_stream_capturing(device: torch.device) -> bool:
    if (
        device.type == "cuda"
        and torch.cuda.is_available()
        and hasattr(torch.cuda, "is_current_stream_capturing")
    ):
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            return False
    return False


def _verify_conductor_abi(ctx: Any) -> None:
    abi = int(getattr(ctx, "abi_version", 0))
    if abi < MIN_CONDUCTOR_ABI_VERSION:
        raise _ConductorAbiError(
            f"Conductor runtime at {getattr(ctx, 'lib_path', '<unknown>')} reports ABI v{abi}; "
            f"Doodle requires ABI >= v{MIN_CONDUCTOR_ABI_VERSION}."
        )
    missing = [m for m in _REQUIRED_CONDUCTOR_METHODS if not hasattr(ctx, m)]
    if missing:
        raise _ConductorAbiError(
            f"Conductor runtime ABI v{abi} is missing required methods: {missing}."
        )


def _load_local_conductor_package() -> Tuple[Optional[Any], Optional[Path], Optional[Path]]:
    if not _LOCAL_CONDUCTOR_DIR.is_dir():
        return None, None, None

    init_py = _LOCAL_CONDUCTOR_DIR / "__init__.py"
    mod_py = _LOCAL_CONDUCTOR_DIR / "conductor.py"
    if not (init_py.is_file() or mod_py.is_file()):
        return None, None, None

    local_lib: Optional[Path] = None
    for name in _platform_conductor_lib_names():
        cand = _LOCAL_CONDUCTOR_DIR / name
        if cand.is_file():
            local_lib = cand.resolve()
            break
    if local_lib is None:
        raise _ConductorAbiError(
            f"Local Conductor bundle at {_LOCAL_CONDUCTOR_DIR} is missing native library "
            f"{_platform_conductor_lib_names()}."
        )

    lut_cand = _LOCAL_CONDUCTOR_DIR / "conductor.lut.bin"
    local_lut: Optional[Path] = lut_cand.resolve() if lut_cand.is_file() else None

    pkg_name = "_doodle_conductor"
    existing = sys.modules.get(pkg_name)
    if existing is not None:
        return existing, local_lib, local_lut

    if init_py.is_file():
        spec = importlib.util.spec_from_file_location(
            pkg_name,
            str(init_py),
            submodule_search_locations=[str(_LOCAL_CONDUCTOR_DIR)],
        )
    else:
        spec = importlib.util.spec_from_file_location(pkg_name, str(mod_py))

    if spec is None or spec.loader is None:
        raise _ConductorAbiError(
            f"Unable to construct import spec for local Conductor bundle at {_LOCAL_CONDUCTOR_DIR}."
        )

    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(pkg_name, None)
        raise _ConductorAbiError(
            f"Failed to import local Conductor bundle at {_LOCAL_CONDUCTOR_DIR}: {exc}"
        ) from exc
    return mod, local_lib, local_lut


def _ensure_conductor_loaded() -> None:
    global _CONDUCTOR_INIT_DONE
    global _CONDUCTOR_MOD
    global _CONDUCTOR_CTX
    global _CONDUCTOR_IS_LOCAL_BUNDLE

    if _CONDUCTOR_INIT_DONE:
        return
    with _CONDUCTOR_LOCK:
        if _CONDUCTOR_INIT_DONE:
            return
        ctx: Optional[Any] = None
        try:
            local_mod, local_lib, local_lut = _load_local_conductor_package()
            if local_mod is not None and local_lib is not None:
                try:
                    ctx = local_mod.ConductorContext(
                        lib_path=local_lib,
                        lut_path=local_lut,
                        require_fast_lut=False,
                    )
                except Exception as exc:
                    raise _ConductorAbiError(
                        f"Failed to initialize local ConductorContext ({local_lib}): {exc}"
                    ) from exc
                _verify_conductor_abi(ctx)
                _CONDUCTOR_MOD = local_mod
                _CONDUCTOR_CTX = ctx
                _CONDUCTOR_IS_LOCAL_BUNDLE = True
                _CONDUCTOR_INIT_DONE = True
                return

            try:
                import conductor as ext_cnd  # type: ignore[import-not-found]
            except ImportError:
                _CONDUCTOR_MOD = None
                _CONDUCTOR_CTX = None
                _CONDUCTOR_IS_LOCAL_BUNDLE = False
                _CONDUCTOR_INIT_DONE = True
                return

            if ext_cnd.has_native_conductor():
                try:
                    ctx = ext_cnd.ConductorContext(require_fast_lut=False)
                except Exception as exc:
                    raise _ConductorAbiError(
                        f"Failed to initialize external ConductorContext: {exc}"
                    ) from exc
                _verify_conductor_abi(ctx)
                _CONDUCTOR_MOD = ext_cnd
                _CONDUCTOR_CTX = ctx
                _CONDUCTOR_IS_LOCAL_BUNDLE = (
                    ctx.lib_path.resolve().parent == _LOCAL_CONDUCTOR_DIR.resolve()
                )
            _CONDUCTOR_INIT_DONE = True
        except Exception:
            if ctx is not None:
                try:
                    ctx.close()
                except Exception:
                    pass
            raise


def close_conductor_runtime() -> None:
    global _CONDUCTOR_INIT_DONE
    global _CONDUCTOR_MOD
    global _CONDUCTOR_CTX
    global _CONDUCTOR_IS_LOCAL_BUNDLE

    with _CONDUCTOR_LOCK:
        ctx = _CONDUCTOR_CTX
        _CONDUCTOR_CTX = None
        _CONDUCTOR_MOD = None
        _CONDUCTOR_IS_LOCAL_BUNDLE = False
        _CONDUCTOR_INIT_DONE = False
        _CONDUCTOR_GPU_WARMED.clear()
    if ctx is not None:
        try:
            ctx.close()
        except Exception:
            pass


atexit.register(close_conductor_runtime)


def get_conductor_module() -> Optional[Any]:
    if not _CONDUCTOR_INIT_DONE:
        _ensure_conductor_loaded()
    return _CONDUCTOR_MOD


def get_conductor_context() -> Optional[Any]:
    if not _CONDUCTOR_INIT_DONE:
        _ensure_conductor_loaded()
    return _CONDUCTOR_CTX


def has_conductor_runtime() -> bool:
    return get_conductor_context() is not None


def get_conductor_info() -> Dict[str, Any]:
    ctx = get_conductor_context()
    if ctx is None:
        return {
            "active": False,
            "abi_version": 0,
            "has_fast_lut": False,
            "gpu_backend": "NONE",
            "lib_path": None,
            "lut_path": None,
            "is_local_bundle": False,
        }
    return {
        "active": True,
        "abi_version": int(ctx.abi_version),
        "has_fast_lut": bool(ctx.has_fast_lut),
        "gpu_backend": str(ctx.gpu_backend.name),
        "lib_path": str(ctx.lib_path),
        "lut_path": str(ctx.lut_path) if ctx.lut_path is not None else None,
        "is_local_bundle": bool(_CONDUCTOR_IS_LOCAL_BUNDLE),
    }


def warmup_conductor_gpu(device: Optional[torch.device] = None) -> bool:
    ctx = get_conductor_context()
    if ctx is None:
        return False
    if device is not None and device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False

    dev_idx = (
        int(device.index)
        if (device is not None and device.index is not None)
        else int(torch.cuda.current_device())
    )
    cached = _CONDUCTOR_GPU_WARMED.get(dev_idx)
    if cached is not None:
        return cached

    target_dev = torch.device("cuda", dev_idx)
    if _is_stream_capturing(target_dev):
        return False

    with _CONDUCTOR_LOCK:
        cached = _CONDUCTOR_GPU_WARMED.get(dev_idx)
        if cached is not None:
            return cached
        if _is_stream_capturing(target_dev):
            return False
        try:
            ok = int(ctx.gpu_backend) != 0 and bool(ctx.warmup_gpu(dev_idx))
        except Exception:
            ok = False
        _CONDUCTOR_GPU_WARMED[dev_idx] = ok
        return ok


def can_use_conductor_tensor(
    device: torch.device,
    *src_dtypes: torch.dtype,
    dst_dtype: Optional[torch.dtype] = None,
) -> bool:
    if device.type not in ("cpu", "cuda"):
        return False
    ctx = get_conductor_context()
    if ctx is None:
        return False
    for dt in src_dtypes:
        if dt not in _CONDUCTOR_SRC_DTYPES:
            return False
    if dst_dtype is not None and dst_dtype not in _CONDUCTOR_DST_DTYPES:
        return False
    if device.type == "cuda":
        dev_idx = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        warmed = _CONDUCTOR_GPU_WARMED.get(dev_idx)
        if warmed is None:
            warmed = warmup_conductor_gpu(device)
        return bool(warmed)
    return True


def _validate_out_buffer(
    out: torch.Tensor,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if tuple(out.shape) != shape:
        raise ValueError(f"out shape {tuple(out.shape)} != expected {shape}.")
    if out.dtype != dtype:
        raise ValueError(f"out dtype {out.dtype} != expected {dtype}.")
    if out.device != device:
        raise ValueError(f"out device {out.device} != expected {device}.")
    if not out.is_contiguous():
        raise ValueError("out buffer must be contiguous.")
    return out


def _build_doubly_balanced_bayer_8x8_tuple() -> Tuple[float, ...]:
    grid = [[0.0] * 8 for _ in range(8)]
    for i in range(4):
        even_macro = (i & 1) == 0
        for j in range(4):
            m = float(_B4_MATRIX[i][j])
            r = 2 * i
            c = 2 * j
            if even_macro:
                grid[r][c] = (m - 31.5) / 64.0
                grid[r][c + 1] = (31.5 - m) / 64.0
                grid[r + 1][c] = (15.5 - m) / 64.0
                grid[r + 1][c + 1] = (m - 15.5) / 64.0
            else:
                grid[r][c] = (m - 15.5) / 64.0
                grid[r][c + 1] = (15.5 - m) / 64.0
                grid[r + 1][c] = (31.5 - m) / 64.0
                grid[r + 1][c + 1] = (m - 31.5) / 64.0
    return tuple(val for row in grid for val in row)


_BAYER_8X8_TUPLE: Tuple[float, ...] = _build_doubly_balanced_bayer_8x8_tuple()

_BF16_MAX_ABS_I32 = 0x7F7F0000
_BF16_MAX_F32 = float(torch.finfo(torch.bfloat16).max)
_BF16_RAIL_THR_F32 = _BF16_MAX_F32 * 0.9
_INV_BF16_RAIL_SPAN = 1.0 / max(_BF16_MAX_F32 - _BF16_RAIL_THR_F32, 1e-9)

_FP32_MIN_NORMAL_I64 = 0x3810000000000000
_FP32_SUB_SCALE_UP_F64 = float(2.0**149)
_FP32_SUB_SCALE_DOWN_F64 = float(2.0**-149)
_FP32_MAX_ABS_I64_IN_F64 = 0x47EFFFFFE0000000

_ENGINE_CACHE: Dict[Tuple[str, Optional[int]], Tuple["TheContender", "TheRefinery"]] = {}
_FP8_LUT_CACHE: Dict[Tuple[str, Optional[int], torch.dtype], torch.Tensor] = {}
_BAYER_BASE_CACHE: Dict[Tuple[str, Optional[int], torch.dtype], torch.Tensor] = {}
_BAYER_3D_LUT_CACHE: Dict[Tuple[str, Optional[int], torch.dtype], torch.Tensor] = {}
_RUNTIME_INF_CACHE: Dict[
    Tuple[str, Optional[int], torch.dtype], Tuple[torch.Tensor, torch.Tensor]
] = {}
_CPU_BAYER_CUBE64_F64: Optional[torch.Tensor] = None


def clear_refinery_cache() -> None:
    _ENGINE_CACHE.clear()
    _FP8_LUT_CACHE.clear()
    _BAYER_BASE_CACHE.clear()
    _BAYER_3D_LUT_CACHE.clear()
    _RUNTIME_INF_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _device_key(device: torch.device) -> Tuple[str, Optional[int]]:
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        return ("cuda", torch.cuda.current_device())
    return (device.type, device.index)


def get_doubly_balanced_bayer_8x8(
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    dev_t, dev_i = _device_key(device)
    key = (dev_t, dev_i, dtype)
    cached = _BAYER_BASE_CACHE.get(key)
    if cached is not None and not cached.is_inference():
        return cached
    with torch.inference_mode(False):
        mat = (
            torch.tensor(_BAYER_8X8_TUPLE, dtype=torch.float64, device="cpu")
            .reshape(8, 8)
            .to(device=device, dtype=dtype)
            .contiguous()
        )
    _BAYER_BASE_CACHE[key] = mat
    return mat


def _get_cpu_bayer_cube64_f64() -> torch.Tensor:
    global _CPU_BAYER_CUBE64_F64
    if _CPU_BAYER_CUBE64_F64 is not None and not _CPU_BAYER_CUBE64_F64.is_inference():
        return _CPU_BAYER_CUBE64_F64
    with torch.inference_mode(False):
        base8 = torch.tensor(
            _BAYER_8X8_TUPLE, dtype=torch.float64, device="cpu"
        ).reshape(8, 8)
        idx = torch.arange(64, dtype=torch.int64, device="cpu")
        z0 = (idx & 7).view(64, 1, 1)
        zb = ((idx >> 3) & 7).view(64, 1, 1)
        r0 = (idx & 7).view(1, 64, 1)
        rb = ((idx >> 3) & 7).view(1, 64, 1)
        c0 = (idx & 7).view(1, 1, 64)
        cb = ((idx >> 3) & 7).view(1, 1, 64)
        u = (r0 + rb + cb + 3 * z0 + zb) & 7
        v = (c0 + rb + cb + 2 * z0 + zb) & 7
        cube64 = base8[u, v].contiguous()
    _CPU_BAYER_CUBE64_F64 = cube64
    return cube64


def _get_bayer_3d_lut(device: torch.device, work_dtype: torch.dtype) -> torch.Tensor:
    dev_t, dev_i = _device_key(device)
    key = (dev_t, dev_i, work_dtype)
    cached = _BAYER_3D_LUT_CACHE.get(key)
    if cached is not None and not cached.is_inference():
        return cached
    cube64 = _get_cpu_bayer_cube64_f64()
    with torch.inference_mode(False):
        lut = (
            cube64.repeat(2, 1, 1)
            .to(device=device, dtype=work_dtype)
            .contiguous()
        )
    _BAYER_3D_LUT_CACHE[key] = lut
    return lut


def _axis_intervals(length: int) -> Tuple[Tuple[int, int, int, int], ...]:
    if length <= 64:
        return ((0, length, 1, length),)
    body = (length // 64) * 64
    if body == length:
        return ((0, length, length // 64, 64),)
    return ((0, body, body // 64, 64), (body, length, 1, length - body))


def sample_bayer_lattice(
    shape: Tuple[int, ...],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    *,
    phase_offset: int = 0,
) -> torch.Tensor:
    if len(shape) == 0:
        lut = _get_bayer_3d_lut(device, dtype)
        z0 = int(phase_offset) & 63
        return lut[z0, 0, 0].clone()
    if any(dim == 0 for dim in shape):
        return torch.empty(shape, dtype=dtype, device=device)

    lut = _get_bayer_3d_lut(device, dtype)
    z0 = int(phase_offset) & 63
    if len(shape) == 1:
        z_len, r_len, c_len = 1, 1, int(shape[0])
        r0 = (int(phase_offset) >> 6) & 63
    elif len(shape) == 2:
        z_len, r_len, c_len = 1, int(shape[0]), int(shape[1])
        r0 = 0
    else:
        z_len = int(math.prod(shape[:-2]))
        r_len = int(shape[-2])
        c_len = int(shape[-1])
        r0 = 0

    z_ivs = _axis_intervals(z_len)
    r_ivs = _axis_intervals(r_len)
    c_ivs = _axis_intervals(c_len)

    if len(z_ivs) == 1 and len(r_ivs) == 1 and len(c_ivs) == 1:
        _, _, z_blk, z_sub = z_ivs[0]
        _, _, r_blk, r_sub = r_ivs[0]
        _, _, c_blk, c_sub = c_ivs[0]
        view_6d = lut[z0 : z0 + z_sub, r0 : r0 + r_sub, :c_sub].view(
            1, z_sub, 1, r_sub, 1, c_sub
        )
        return (
            view_6d.expand(z_blk, z_sub, r_blk, r_sub, c_blk, c_sub)
            .contiguous()
            .view(shape)
        )

    out_3d = torch.empty((z_len, r_len, c_len), dtype=dtype, device=device)
    for z_s, z_e, z_blk, z_sub in z_ivs:
        for r_s, r_e, r_blk, r_sub in r_ivs:
            for c_s, c_e, c_blk, c_sub in c_ivs:
                view_6d = lut[z0 : z0 + z_sub, r0 : r0 + r_sub, :c_sub].view(
                    1, z_sub, 1, r_sub, 1, c_sub
                )
                out_3d[z_s:z_e, r_s:r_e, c_s:c_e].copy_(
                    view_6d.expand(z_blk, z_sub, r_blk, r_sub, c_blk, c_sub).reshape(
                        z_e - z_s, r_e - r_s, c_e - c_s
                    )
                )
    return out_3d.view(shape)


def _get_fp8_sorted_lut(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    dev_t, dev_i = _device_key(device)
    key = (dev_t, dev_i, dtype)
    cached = _FP8_LUT_CACHE.get(key)
    if cached is not None and not cached.is_inference():
        return cached
    with torch.inference_mode(False):
        raw_u8 = torch.arange(256, dtype=torch.uint8, device="cpu")
        vals = raw_u8.view(dtype).to(dtype=torch.float32)
        finite = vals[torch.isfinite(vals)]
        unique_sorted = (
            torch.unique(torch.sort(finite).values)
            .to(device=device, dtype=torch.float32)
            .contiguous()
        )
    _FP8_LUT_CACHE[key] = unique_sorted
    return unique_sorted


def _get_runtime_inf_pair(
    device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    dev_t, dev_i = _device_key(device)
    key = (dev_t, dev_i, dtype)
    cached = _RUNTIME_INF_CACHE.get(key)
    if cached is not None and not cached[0].is_inference():
        return cached
    with torch.inference_mode(False):
        pair = (
            torch.tensor(-float("inf"), device=device, dtype=dtype),
            torch.tensor(float("inf"), device=device, dtype=dtype),
        )
    _RUNTIME_INF_CACHE[key] = pair
    return pair


def _resolve_dtype_limits(dtype: torch.dtype) -> Tuple[float, float]:
    try:
        finfo = torch.finfo(dtype)
        return float(finfo.min), float(finfo.max)
    except Exception:
        name = str(dtype).split(".")[-1]
        if name == "float8_e4m3fn":
            return (-448.0, 448.0)
        if name == "float8_e4m3fnuz":
            return (-240.0, 240.0)
        if name in ("float8_e5m2", "float8_e5m2fnuz"):
            return (-57344.0, 57344.0)
        if name == "int8":
            return (-128.0, 127.0)
        return (-65504.0, 65504.0)


def _round_to_odd_f64_to_f32(x_f64: torch.Tensor) -> torch.Tensor:
    raw_bits = x_f64.view(torch.int64)
    abs_bits = torch.clamp(
        raw_bits & 0x7FFFFFFFFFFFFFFF, max=_FP32_MAX_ABS_I64_IN_F64
    )
    is_sub = (abs_bits > 0) & (abs_bits < _FP32_MIN_NORMAL_I64)

    trunc_norm = abs_bits & ~0x1FFFFFFF
    sticky_norm = torch.where((abs_bits & 0x1FFFFFFF) != 0, 0x20000000, 0)
    rto_norm_f64 = (trunc_norm | sticky_norm).view(torch.float64)

    sub_f64 = torch.where(
        is_sub, abs_bits.view(torch.float64), torch.zeros_like(x_f64)
    )
    u_sub = sub_f64 * _FP32_SUB_SCALE_UP_F64
    u_floor = torch.floor(u_sub)
    u_int = u_floor.to(dtype=torch.int64)
    has_frac = (u_sub > u_floor) | (is_sub & (u_int == 0))
    rto_sub_int = torch.where(has_frac, u_int | 1, u_int)
    rto_sub_f64 = rto_sub_int.to(dtype=torch.float64) * _FP32_SUB_SCALE_DOWN_F64

    pos_f64 = torch.where(is_sub, rto_sub_f64, rto_norm_f64)
    return torch.copysign(pos_f64, x_f64).to(dtype=torch.float32)


def _exact_lattice_brackets_fp32(
    x_f32: torch.Tensor, target_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    min_v, max_v = _resolve_dtype_limits(target_dtype)
    x_clamped = torch.clamp(x_f32, min_v, max_v)
    if target_dtype in (torch.bfloat16, torch.float16, torch.float32):
        q = x_clamped.to(dtype=target_dtype)
        z_near = q.to(dtype=torch.float32)
        neg_inf, pos_inf = _get_runtime_inf_pair(x_f32.device, target_dtype)
        z_prev = torch.clamp(
            torch.nextafter(q, neg_inf).to(dtype=torch.float32), min=min_v, max=max_v
        )
        z_next = torch.clamp(
            torch.nextafter(q, pos_inf).to(dtype=torch.float32), min=min_v, max=max_v
        )
        s_low = torch.where(x_clamped >= z_near, z_near, z_prev)
        s_high = torch.where(x_clamped > z_near, z_next, z_near)
        return z_near, s_low, s_high

    lut = _get_fp8_sorted_lut(x_f32.device, target_dtype)
    n_lut = lut.numel()
    idx_hi = torch.searchsorted(lut, x_clamped).clamp(min=0, max=n_lut - 1)
    s_high = lut[idx_hi]
    idx_lo = torch.where(x_clamped < s_high, idx_hi - 1, idx_hi).clamp(min=0)
    s_low = lut[idx_lo]
    z_near = x_clamped.to(dtype=target_dtype).to(dtype=torch.float32)
    return z_near, s_low, s_high


class PhysicsLab:
    PI_BY_2 = math.pi / 2.0

    MANTISSA_MAP: Dict[str, int] = {
        "float64": 52,
        "float32": 23,
        "float16": 10,
        "bfloat16": 7,
        "float8_e4m3fn": 3,
        "float8_e4m3fnuz": 3,
        "float8_e5m2": 2,
        "float8_e5m2fnuz": 2,
        "float8_e8m0fnu": 0,
        "int8": 7,
        "uint8": 0,
    }

    @staticmethod
    def get_mantissa_width(dtype: torch.dtype) -> int:
        key = str(dtype).split(".")[-1]
        return PhysicsLab.MANTISSA_MAP.get(key, 10)

    @staticmethod
    def get_hybrid_grace_strength(error_ratio: torch.Tensor) -> torch.Tensor:
        strength_quad = error_ratio * error_ratio
        strength_cos = 1.0 - torch.cos(error_ratio * PhysicsLab.PI_BY_2)
        return (strength_quad + strength_cos) * 0.5


def _dispatch_nd_bayer_blocks(
    x: torch.Tensor,
    work_dtype: torch.dtype,
    out_dtype: torch.dtype,
    phase_offset: int,
    block_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    orig_shape = x.shape
    lut = _get_bayer_3d_lut(x.device, work_dtype)
    z0 = int(phase_offset) & 63

    if x.ndim == 0:
        b_view = lut[z0 : z0 + 1, 0:1, 0:1]
        return block_fn(x.reshape(1, 1, 1), b_view).reshape(())

    if x.ndim == 1:
        z_len, r_len, c_len = 1, 1, int(orig_shape[0])
        r0 = (int(phase_offset) >> 6) & 63
    elif x.ndim == 2:
        z_len, r_len, c_len = 1, int(orig_shape[0]), int(orig_shape[1])
        r0 = 0
    else:
        z_len = int(math.prod(orig_shape[:-2]))
        r_len = int(orig_shape[-2])
        c_len = int(orig_shape[-1])
        r0 = 0

    x_3d = x.reshape(z_len, r_len, c_len)
    z_ivs = _axis_intervals(z_len)
    r_ivs = _axis_intervals(r_len)
    c_ivs = _axis_intervals(c_len)

    if len(z_ivs) == 1 and len(r_ivs) == 1 and len(c_ivs) == 1:
        _, _, z_blk, z_sub = z_ivs[0]
        _, _, r_blk, r_sub = r_ivs[0]
        _, _, c_blk, c_sub = c_ivs[0]
        x_6d = x_3d.reshape(z_blk, z_sub, r_blk, r_sub, c_blk, c_sub)
        b_6d = lut[z0 : z0 + z_sub, r0 : r0 + r_sub, :c_sub].view(
            1, z_sub, 1, r_sub, 1, c_sub
        )
        return block_fn(x_6d, b_6d).reshape(orig_shape)

    out_3d = torch.empty((z_len, r_len, c_len), dtype=out_dtype, device=x.device)
    for z_s, z_e, z_blk, z_sub in z_ivs:
        for r_s, r_e, r_blk, r_sub in r_ivs:
            for c_s, c_e, c_blk, c_sub in c_ivs:
                sub_6d = x_3d[z_s:z_e, r_s:r_e, c_s:c_e].reshape(
                    z_blk, z_sub, r_blk, r_sub, c_blk, c_sub
                )
                b_6d = lut[z0 : z0 + z_sub, r0 : r0 + r_sub, :c_sub].view(
                    1, z_sub, 1, r_sub, 1, c_sub
                )
                out_3d[z_s:z_e, r_s:r_e, c_s:c_e] = block_fn(sub_6d, b_6d).reshape(
                    z_e - z_s, r_e - r_s, c_e - c_s
                )
    return out_3d.reshape(orig_shape)


def _bitwise_fp32_to_bf16_block(
    x_f32: torch.Tensor, bayer_view: torch.Tensor, unbiased: bool = False
) -> torch.Tensor:
    raw_bits = x_f32.view(torch.int32)
    is_neg = raw_bits < 0
    abs_bits = torch.clamp(raw_bits & 0x7FFFFFFF, max=_BF16_MAX_ABS_I32)

    u_mag_lo = abs_bits & 0x7FFF0000
    rem_i32 = abs_bits & 0xFFFF
    has_rem = rem_i32 != 0
    u_mag_hi = torch.where(has_rem, u_mag_lo + 0x00010000, u_mag_lo)

    alpha = rem_i32.to(dtype=torch.float32) * (1.0 / 65536.0)
    signed_bayer = torch.where(is_neg, -bayer_view, bayer_view)

    if unbiased:
        score = alpha + signed_bayer
    else:
        err_ratio = 1.0 - (2.0 * alpha - 1.0).abs()
        grace = PhysicsLab.get_hybrid_grace_strength(err_ratio)
        abs_f32 = abs_bits.view(torch.float32)
        rail_mask = torch.clamp(
            (_BF16_MAX_F32 - abs_f32) * _INV_BF16_RAIL_SPAN, 0.0, 1.0
        )
        score = alpha + (grace * rail_mask) * signed_bayer

    lsb_odd = ((u_mag_lo >> 16) & 1) != 0
    round_up = torch.where(score == 0.5, lsb_odd, score > 0.5)
    chosen_mag = torch.where(has_rem & round_up, u_mag_hi, u_mag_lo)
    return torch.copysign(chosen_mag.view(torch.float32), x_f32).to(
        dtype=torch.bfloat16
    )


def _bitwise_fp64_to_fp32_block(
    x_f64: torch.Tensor, bayer_view: torch.Tensor, unbiased: bool = False
) -> torch.Tensor:
    raw_bits = x_f64.view(torch.int64)
    is_neg = raw_bits < 0
    abs_bits = torch.clamp(
        raw_bits & 0x7FFFFFFFFFFFFFFF, max=_FP32_MAX_ABS_I64_IN_F64
    )
    is_sub = (abs_bits > 0) & (abs_bits < _FP32_MIN_NORMAL_I64)
    signed_bayer = torch.where(is_neg, -bayer_view, bayer_view)

    u_norm_lo = abs_bits & ~0x1FFFFFFF
    rem_i64 = abs_bits & 0x1FFFFFFF
    has_rem_norm = rem_i64 != 0
    u_norm_hi = torch.where(has_rem_norm, u_norm_lo + 0x20000000, u_norm_lo)
    alpha_norm = rem_i64.to(dtype=torch.float64) * (1.0 / 536870912.0)

    if unbiased:
        score_norm = alpha_norm + signed_bayer
    else:
        err_norm = 1.0 - (2.0 * alpha_norm - 1.0).abs()
        grace_norm = PhysicsLab.get_hybrid_grace_strength(err_norm)
        score_norm = alpha_norm + grace_norm * signed_bayer

    lsb_norm = ((u_norm_lo >> 29) & 1) != 0
    up_norm = torch.where(score_norm == 0.5, lsb_norm, score_norm > 0.5)
    pos_norm_f64 = torch.where(
        has_rem_norm & up_norm, u_norm_hi, u_norm_lo
    ).view(torch.float64)

    sub_f64 = torch.where(
        is_sub, abs_bits.view(torch.float64), torch.zeros_like(x_f64)
    )
    u_sub = sub_f64 * _FP32_SUB_SCALE_UP_F64
    u_sub_lo = torch.floor(u_sub)
    alpha_sub = u_sub - u_sub_lo
    has_rem_sub = is_sub & ((alpha_sub > 0.0) | (u_sub_lo == 0.0))
    u_sub_hi = torch.where(has_rem_sub, u_sub_lo + 1.0, u_sub_lo)

    if unbiased:
        score_sub = alpha_sub + signed_bayer
    else:
        err_sub = 1.0 - (2.0 * alpha_sub - 1.0).abs()
        grace_sub = PhysicsLab.get_hybrid_grace_strength(err_sub)
        score_sub = alpha_sub + grace_sub * signed_bayer

    lsb_sub = (u_sub_lo.to(dtype=torch.int64) & 1) != 0
    up_sub = torch.where(score_sub == 0.5, lsb_sub, score_sub > 0.5)
    pos_sub_f64 = (
        torch.where(has_rem_sub & up_sub, u_sub_hi, u_sub_lo)
        * _FP32_SUB_SCALE_DOWN_F64
    )

    pos_out_f64 = torch.where(is_sub, pos_sub_f64, pos_norm_f64)
    return torch.copysign(pos_out_f64, x_f64).to(dtype=torch.float32)


def _lattice_fp32_to_target_block(
    x_f32: torch.Tensor,
    bayer_view: torch.Tensor,
    target_dtype: torch.dtype,
    min_v: float,
    max_v: float,
    unbiased: bool = False,
) -> torch.Tensor:
    x_clamped = torch.clamp(x_f32, min_v, max_v)
    z_near, s_low, s_high = _exact_lattice_brackets_fp32(x_clamped, target_dtype)
    step = s_high - s_low
    has_span = step > 0.0
    safe_step = torch.where(has_span, step, torch.ones_like(step))
    alpha = torch.where(
        has_span,
        ((x_clamped - s_low) / safe_step).clamp(0.0, 1.0),
        torch.zeros_like(x_clamped),
    )

    if unbiased:
        score = alpha + bayer_view
    else:
        err_ratio = 1.0 - (2.0 * alpha - 1.0).abs()
        grace = PhysicsLab.get_hybrid_grace_strength(err_ratio)
        if max_v > 0.0 and min_v < 0.0:
            rail_thr = max_v * 0.9
            rail_mask = torch.clamp(
                (max_v - x_clamped.abs()) / max(max_v - rail_thr, 1e-9), 0.0, 1.0
            )
            score = alpha + (grace * rail_mask) * bayer_view
        else:
            score = alpha + grace * bayer_view

    use_high = torch.where(score == 0.5, z_near == s_high, score > 0.5)
    chosen = torch.where(has_span & use_high, s_high, s_low)
    if min_v <= 0.0:
        chosen = torch.where(
            chosen == 0.0, torch.copysign(chosen, x_f32), chosen
        )
    return chosen.to(dtype=target_dtype)


@torch.no_grad()
def contender_cast(
    x: torch.Tensor,
    target_dtype: torch.dtype,
    *,
    phase_offset: int = 0,
    unbiased: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is not None:
        _validate_out_buffer(out, tuple(x.shape), target_dtype, x.device)

    if not x.is_floating_point() or x.dtype == target_dtype or x.numel() == 0:
        if out is not None:
            if x.numel() > 0:
                out.copy_(x)
            return out
        return x.to(dtype=target_dtype) if x.dtype != target_dtype else x

    m_src = PhysicsLab.get_mantissa_width(x.dtype)
    m_tgt = PhysicsLab.get_mantissa_width(target_dtype)
    if m_src <= m_tgt:
        if out is not None:
            out.copy_(x)
            return out
        return x.to(dtype=target_dtype)

    if can_use_conductor_tensor(x.device, x.dtype, dst_dtype=target_dtype):
        ctx = get_conductor_context()
        if ctx is not None:
            return ctx.contender_cast(
                x,
                target_dtype,
                phase_offset=int(phase_offset),
                unbiased=bool(unbiased),
                out=out,
            )

    if target_dtype == torch.float32 and x.dtype == torch.float64:
        res = _dispatch_nd_bayer_blocks(
            x,
            torch.float64,
            torch.float32,
            phase_offset,
            lambda blk, b_v: _bitwise_fp64_to_fp32_block(blk, b_v, unbiased=unbiased),
        )
    else:
        x_f32 = (
            _round_to_odd_f64_to_f32(x)
            if x.dtype == torch.float64
            else (x if x.dtype == torch.float32 else x.to(dtype=torch.float32))
        )
        if target_dtype == torch.bfloat16:
            res = _dispatch_nd_bayer_blocks(
                x_f32,
                torch.float32,
                torch.bfloat16,
                phase_offset,
                lambda blk, b_v: _bitwise_fp32_to_bf16_block(
                    blk, b_v, unbiased=unbiased
                ),
            )
        else:
            min_v, max_v = _resolve_dtype_limits(target_dtype)
            res = _dispatch_nd_bayer_blocks(
                x_f32,
                torch.float32,
                target_dtype,
                phase_offset,
                lambda blk, b_v: _lattice_fp32_to_target_block(
                    blk, b_v, target_dtype, min_v, max_v, unbiased=unbiased
                ),
            )

    if out is not None:
        out.copy_(res)
        return out
    return res


class TheContender:
    def __init__(self, device: torch.device):
        self.device = device

    @torch.no_grad()
    def quantize(
        self,
        x_input: torch.Tensor,
        target_dtype: torch.dtype,
        *,
        phase_offset: int = 0,
    ) -> torch.Tensor:
        x_dev = x_input if x_input.device == self.device else x_input.to(self.device)
        return contender_cast(x_dev, target_dtype, phase_offset=phase_offset)


class TheRefinery:
    DTYPE_BOUNDARIES: Dict[torch.dtype, Tuple[float, float]] = {
        torch.float32: _resolve_dtype_limits(torch.float32),
        torch.float16: _resolve_dtype_limits(torch.float16),
        torch.bfloat16: _resolve_dtype_limits(torch.bfloat16),
        torch.int8: (-128.0, 127.0),
    }

    def __init__(self, device: torch.device):
        self.device = device
        self.bayer_8x8 = get_doubly_balanced_bayer_8x8(device, torch.float32)

    @torch.no_grad()
    def quantize(
        self,
        tensor_fp32: torch.Tensor,
        target_dtype: torch.dtype,
        dither_enabled: bool = True,
        max_chunk_channels: int = 2048,
        *,
        phase_offset: int = 0,
    ) -> torch.Tensor:
        if not tensor_fp32.is_floating_point() or tensor_fp32.numel() == 0:
            return tensor_fp32.to(dtype=target_dtype)

        t_dev = (
            tensor_fp32
            if tensor_fp32.device == self.device
            else tensor_fp32.to(device=self.device)
        )

        if not dither_enabled:
            min_v, max_v = TheRefinery.DTYPE_BOUNDARIES.get(
                target_dtype, _resolve_dtype_limits(target_dtype)
            )
            work_t = (
                _round_to_odd_f64_to_f32(t_dev)
                if t_dev.dtype == torch.float64 and target_dtype != torch.float32
                else (
                    t_dev
                    if t_dev.dtype in (torch.float32, torch.float64)
                    else t_dev.to(dtype=torch.float32)
                )
            )
            clamped = torch.clamp(work_t, min_v, max_v)
            if min_v <= 0.0:
                clamped = torch.where(
                    clamped == 0.0, torch.copysign(clamped, t_dev), clamped
                )
            return clamped.to(dtype=target_dtype)

        chunk_step = max(64, ((int(max_chunk_channels) + 63) // 64) * 64)
        orig_shape = t_dev.shape

        if t_dev.ndim <= 1:
            return contender_cast(t_dev, target_dtype, phase_offset=phase_offset)

        if t_dev.ndim == 2:
            r_len, c_len = int(orig_shape[0]), int(orig_shape[1])
            if r_len <= chunk_step:
                return contender_cast(t_dev, target_dtype, phase_offset=phase_offset)
            out_2d = torch.empty(
                (r_len, c_len), dtype=target_dtype, device=self.device
            )
            for r_start in range(0, r_len, chunk_step):
                r_end = min(r_start + chunk_step, r_len)
                contender_cast(
                    t_dev[r_start:r_end],
                    target_dtype,
                    phase_offset=phase_offset,
                    out=out_2d[r_start:r_end],
                )
            return out_2d

        z_len = int(math.prod(orig_shape[:-2]))
        r_len = int(orig_shape[-2])
        c_len = int(orig_shape[-1])
        t_3d = t_dev.reshape(z_len, r_len, c_len)

        if z_len * r_len <= chunk_step:
            return contender_cast(
                t_3d, target_dtype, phase_offset=phase_offset
            ).reshape(orig_shape)

        out_3d = torch.empty(
            (z_len, r_len, c_len), dtype=target_dtype, device=self.device
        )
        z_step = max(64, ((max(1, chunk_step // max(1, r_len)) + 63) // 64) * 64)
        for z_start in range(0, z_len, z_step):
            z_end = min(z_start + z_step, z_len)
            z_phase = phase_offset + z_start
            if r_len <= chunk_step:
                contender_cast(
                    t_3d[z_start:z_end],
                    target_dtype,
                    phase_offset=z_phase,
                    out=out_3d[z_start:z_end],
                )
            else:
                for r_start in range(0, r_len, chunk_step):
                    r_end = min(r_start + chunk_step, r_len)
                    out_slice = out_3d[z_start:z_end, r_start:r_end]
                    if out_slice.is_contiguous():
                        contender_cast(
                            t_3d[z_start:z_end, r_start:r_end],
                            target_dtype,
                            phase_offset=z_phase,
                            out=out_slice,
                        )
                    else:
                        out_slice.copy_(
                            contender_cast(
                                t_3d[z_start:z_end, r_start:r_end],
                                target_dtype,
                                phase_offset=z_phase,
                            )
                        )
        return out_3d.reshape(orig_shape)


def _get_engine_instances(device: torch.device) -> Tuple[TheContender, TheRefinery]:
    key = _device_key(device)
    cached = _ENGINE_CACHE.get(key)
    if cached is not None:
        return cached
    resolved = (
        torch.device(key[0], key[1])
        if key[1] is not None
        else torch.device(key[0])
    )
    pair = (TheContender(resolved), TheRefinery(resolved))
    _ENGINE_CACHE[key] = pair
    return pair


def refineTensor(
    origin_tensor: torch.Tensor,
    target_dtype: torch.dtype,
    target_device: Optional[torch.device] = None,
    *,
    phase_offset: int = 0,
) -> torch.Tensor:
    with torch.inference_mode(False), torch.no_grad():
        if not origin_tensor.is_floating_point() or origin_tensor.dtype == target_dtype:
            return (
                origin_tensor.to(dtype=target_dtype)
                if origin_tensor.dtype != target_dtype
                else origin_tensor
            )

        m_src = PhysicsLab.get_mantissa_width(origin_tensor.dtype)
        m_tgt = PhysicsLab.get_mantissa_width(target_dtype)
        if m_src <= m_tgt:
            return origin_tensor.to(dtype=target_dtype)

        if target_device is None:
            target_device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        if target_dtype not in TheRefinery.DTYPE_BOUNDARIES:
            TheRefinery.DTYPE_BOUNDARIES[target_dtype] = _resolve_dtype_limits(
                target_dtype
            )

        if origin_tensor.dtype == torch.float64:
            dev_in = origin_tensor.to(device=target_device, dtype=torch.float64)
            if target_dtype == torch.float32:
                res_f32 = contender_cast(
                    dev_in, torch.float32, phase_offset=phase_offset
                )
                return res_f32.to(origin_tensor.device, non_blocking=True)
            work_t = _round_to_odd_f64_to_f32(dev_in)
        else:
            work_t = origin_tensor.to(device=target_device, dtype=torch.float32)

        contender, refinery = _get_engine_instances(target_device)
        if target_dtype == torch.bfloat16:
            result = contender.quantize(
                work_t, target_dtype, phase_offset=phase_offset
            )
        else:
            result = refinery.quantize(
                work_t,
                target_dtype,
                dither_enabled=True,
                phase_offset=phase_offset,
            )

        return result.to(origin_tensor.device, non_blocking=True)


__all__ = [
    "DTYPE_MAP",
    "FP8_DTYPES",
    "MIN_CONDUCTOR_ABI_VERSION",
    "PhysicsLab",
    "TheContender",
    "TheRefinery",
    "can_use_conductor_tensor",
    "clear_refinery_cache",
    "close_conductor_runtime",
    "get_conductor_context",
    "get_conductor_info",
    "get_conductor_module",
    "get_doubly_balanced_bayer_8x8",
    "has_conductor_runtime",
    "sample_bayer_lattice",
    "contender_cast",
    "refineTensor",
    "warmup_conductor_gpu",
]