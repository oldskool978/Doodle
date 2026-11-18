from __future__ import annotations

import math
import threading
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.modeling_vae import IEEE754FP32Guard, enforce_full_precision_reductions

enforce_full_precision_reductions()

try:
    from refinery import PhysicsLab, contender_cast

    _HAS_CONTENDER = True
except ImportError:
    _HAS_CONTENDER = False

_FP8_DTYPES = tuple(
    dt
    for dt in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dt is not None
)

_E5M2_DTYPES = tuple(
    dt
    for dt in (
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dt is not None
)

_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", None)
_LM_HEAD_FP32_TILE_ROWS = 4096
_NARROW_TILE_MAX_ROWS = 8192

_UNIT_SCALE_CACHE: Dict[Tuple[str, Optional[int]], torch.Tensor] = {}
_L2_TILE_ROWS_CACHE: Dict[Tuple[str, Optional[int], int, int], int] = {}
_FP32_TILE_WORKSPACE: Dict[Tuple[str, Optional[int], int], torch.Tensor] = {}
_NARROW_TILE_WORKSPACE: Dict[
    Tuple[str, Optional[int], torch.dtype, int], torch.Tensor
] = {}
_MM_FP32_OUT_CACHE: Dict[Tuple[str, Optional[int], torch.dtype], bool] = {}
_MM_FP32_SLICE_OUT_CACHE: Dict[Tuple[str, Optional[int], torch.dtype], bool] = {}
_SCALED_MM_MODE_CACHE: Dict[
    Tuple[str, Optional[int], torch.dtype, torch.dtype], str
] = {}


def _probe_sdpa_gqa_support() -> bool:
    try:
        with torch.inference_mode(False):
            z = torch.zeros((1, 1, 1, 1), dtype=torch.float32, device="cpu")
            F.scaled_dot_product_attention(z, z, z, enable_gqa=False)
        return True
    except TypeError:
        return False


_HAS_SDPA_GQA: bool = _probe_sdpa_gqa_support()


def _tensor_version(tensor: torch.Tensor) -> int:
    return -1 if tensor.is_inference() else tensor._version


def _device_key(device: torch.device) -> Tuple[str, Optional[int]]:
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        return ("cuda", torch.cuda.current_device())
    return (device.type, device.index)


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


def clear_yue2_workspaces(
    device: Optional[Union[str, torch.device]] = None,
) -> None:
    if device is None:
        _FP32_TILE_WORKSPACE.clear()
        _NARROW_TILE_WORKSPACE.clear()
        _UNIT_SCALE_CACHE.clear()
        return
    dev = torch.device(device) if isinstance(device, str) else device
    for ws_key in [
        k
        for k, buf in _FP32_TILE_WORKSPACE.items()
        if buf.device.type == dev.type
        and (dev.index is None or buf.device.index == dev.index)
    ]:
        _FP32_TILE_WORKSPACE.pop(ws_key, None)
    for nws_key in [
        k
        for k, buf in _NARROW_TILE_WORKSPACE.items()
        if buf.device.type == dev.type
        and (dev.index is None or buf.device.index == dev.index)
    ]:
        _NARROW_TILE_WORKSPACE.pop(nws_key, None)
    for sc_key in [
        k
        for k, buf in _UNIT_SCALE_CACHE.items()
        if buf.device.type == dev.type
        and (dev.index is None or buf.device.index == dev.index)
    ]:
        _UNIT_SCALE_CACHE.pop(sc_key, None)


def _get_unit_scale_fp32(device: torch.device) -> torch.Tensor:
    key = _device_key(device)
    cached = _UNIT_SCALE_CACHE.get(key)
    if (
        cached is not None
        and _device_key(cached.device) == key
        and not cached.is_inference()
    ):
        return cached
    with torch.inference_mode(False):
        scale = torch.ones((), dtype=torch.float32, device=device)
    if not _is_stream_capturing(device):
        _UNIT_SCALE_CACHE[_device_key(scale.device)] = scale
    return scale


def _can_use_mm_fp32_out(device: torch.device, dtype: torch.dtype) -> bool:
    if dtype not in (torch.bfloat16, torch.float16):
        return False
    dev_t, dev_i = _device_key(device)
    key = (dev_t, dev_i, dtype)
    cached = _MM_FP32_OUT_CACHE.get(key)
    if cached is not None:
        return cached
    if _is_stream_capturing(device):
        return False
    supported = False
    try:
        with torch.inference_mode(False), torch.no_grad():
            b_mat = torch.zeros((64, 64), dtype=dtype, device=device).t()
            for m_test in (1, 3, 16):
                a_mat = torch.zeros((m_test, 64), dtype=dtype, device=device)
                probe = torch.mm(a_mat, b_mat, out_dtype=torch.float32)
                if probe.dtype != torch.float32:
                    break
            else:
                supported = True
    except (TypeError, RuntimeError, NotImplementedError):
        supported = False
    _MM_FP32_OUT_CACHE[key] = supported
    return supported


def _can_use_mm_fp32_slice_out(device: torch.device, dtype: torch.dtype) -> bool:
    if not _can_use_mm_fp32_out(device, dtype):
        return False
    dev_t, dev_i = _device_key(device)
    key = (dev_t, dev_i, dtype)
    cached = _MM_FP32_SLICE_OUT_CACHE.get(key)
    if cached is not None:
        return cached
    if _is_stream_capturing(device):
        return False
    supported = False
    try:
        with torch.inference_mode(False), torch.no_grad():
            full = torch.empty((3, 64), dtype=torch.float32, device=device)
            a_mat = torch.zeros((3, 64), dtype=dtype, device=device)
            b_mat = torch.zeros((32, 64), dtype=dtype, device=device).t()
            out_s = full[:, :32]
            res = torch.mm(a_mat, b_mat, out_dtype=torch.float32, out=out_s)
            supported = (
                res.dtype == torch.float32 and res.data_ptr() == out_s.data_ptr()
            )
    except (TypeError, RuntimeError, NotImplementedError):
        supported = False
    _MM_FP32_SLICE_OUT_CACHE[key] = supported
    return supported


def _get_scaled_mm_mode(
    device: torch.device, x_dtype: torch.dtype, w_dtype: torch.dtype
) -> str:
    dev_t, dev_i = _device_key(device)
    key = (dev_t, dev_i, x_dtype, w_dtype)
    cached = _SCALED_MM_MODE_CACHE.get(key)
    if cached is not None:
        return cached
    if _is_stream_capturing(device):
        return "none"
    mode = "none"
    try:
        with torch.inference_mode(False), torch.no_grad():
            a = torch.zeros((16, 16), dtype=torch.bfloat16, device=device).to(x_dtype)
            b = (
                torch.zeros((16, 16), dtype=torch.bfloat16, device=device)
                .to(w_dtype)
                .t()
            )
            one = _get_unit_scale_fp32(device)
            try:
                o32 = torch._scaled_mm(
                    a, b, scale_a=one, scale_b=one, out_dtype=torch.float32
                )
                t32 = o32[0] if isinstance(o32, tuple) else o32
                if t32.dtype == torch.float32:
                    mode = "fp32"
            except (TypeError, RuntimeError, NotImplementedError):
                mode = "none"
            if mode == "none":
                o16 = torch._scaled_mm(
                    a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16
                )
                t16 = o16[0] if isinstance(o16, tuple) else o16
                if t16.dtype == torch.bfloat16:
                    mode = "bf16"
    except (TypeError, RuntimeError, NotImplementedError):
        mode = "none"
    _SCALED_MM_MODE_CACHE[key] = mode
    return mode


def _prewarm_linear_capabilities(
    device: torch.device, weight_dtype: torch.dtype
) -> None:
    if device.type == "meta" or _is_stream_capturing(device):
        return
    if weight_dtype in (torch.bfloat16, torch.float16):
        _can_use_mm_fp32_out(device, weight_dtype)
    elif _FP8_DTYPES and weight_dtype in _FP8_DTYPES:
        _can_use_mm_fp32_out(device, torch.bfloat16)
        _can_use_mm_fp32_slice_out(device, torch.bfloat16)
        _can_use_mm_fp32_out(device, torch.float16)
        _can_use_mm_fp32_slice_out(device, torch.float16)
        if device.type == "cuda" and hasattr(torch, "_scaled_mm"):
            _get_unit_scale_fp32(device)
            if weight_dtype != _E8M0_DTYPE:
                _get_scaled_mm_mode(device, weight_dtype, weight_dtype)


def _get_l2_tile_rows(
    device: torch.device, in_features: int, elem_bytes: int = 4
) -> int:
    dev_t, dev_i = _device_key(device)
    eb = max(1, int(elem_bytes))
    key = (dev_t, dev_i, int(in_features), eb)
    cached = _L2_TILE_ROWS_CACHE.get(key)
    if cached is not None:
        return cached
    cap_rows = _LM_HEAD_FP32_TILE_ROWS if eb >= 4 else _NARROW_TILE_MAX_ROWS
    tile_rows = 2048 if eb >= 4 else 4096
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            dev_idx = dev_i if dev_i is not None else torch.cuda.current_device()
            l2_bytes = int(
                getattr(torch.cuda.get_device_properties(dev_idx), "L2_cache_size", 0)
                or 0
            )
            if l2_bytes > 0:
                row_bytes = max(1, int(in_features) * eb)
                raw_rows = (l2_bytes // 2) // row_bytes
                aligned = (raw_rows // 256) * 256
                tile_rows = max(512, min(cap_rows, aligned))
        except Exception:
            pass
    _L2_TILE_ROWS_CACHE[key] = tile_rows
    return tile_rows


def _get_fp32_tile_workspace(
    device: torch.device, rows: int, cols: int
) -> torch.Tensor:
    dev_t, dev_i = _device_key(device)
    key = (dev_t, dev_i, threading.get_ident())
    needed = rows * cols
    cached = _FP32_TILE_WORKSPACE.get(key)
    if (
        cached is not None
        and _device_key(cached.device) == (dev_t, dev_i)
        and cached.numel() >= needed
        and not cached.is_inference()
    ):
        return cached.view(-1)[:needed].view(rows, cols)
    with torch.inference_mode(False):
        buf = torch.empty((rows, cols), dtype=torch.float32, device=device)
    if not _is_stream_capturing(device):
        buf_t, buf_i = _device_key(buf.device)
        _FP32_TILE_WORKSPACE[(buf_t, buf_i, key[2])] = buf
    return buf


def _get_narrow_tile_workspace(
    device: torch.device, dtype: torch.dtype, rows: int, cols: int
) -> torch.Tensor:
    dev_t, dev_i = _device_key(device)
    key = (dev_t, dev_i, dtype, threading.get_ident())
    needed = rows * cols
    cached = _NARROW_TILE_WORKSPACE.get(key)
    if (
        cached is not None
        and cached.dtype == dtype
        and _device_key(cached.device) == (dev_t, dev_i)
        and cached.numel() >= needed
        and not cached.is_inference()
    ):
        return cached.view(-1)[:needed].view(rows, cols)
    with torch.inference_mode(False):
        buf = torch.empty((rows, cols), dtype=dtype, device=device)
    if not _is_stream_capturing(device):
        buf_t, buf_i = _device_key(buf.device)
        _NARROW_TILE_WORKSPACE[(buf_t, buf_i, dtype, key[3])] = buf
    return buf


class _ContenderNarrowSTE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, target_dtype: torch.dtype, phase_offset: int
    ) -> torch.Tensor:
        ctx.src_dtype = x.dtype
        out = contender_cast(x.detach(), target_dtype, phase_offset=phase_offset)
        return out.clone() if out.is_inference() else out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.to(dtype=ctx.src_dtype), None, None


def _project_narrowing(
    tensor: torch.Tensor, target_dtype: torch.dtype, *, phase_offset: int = 0
) -> torch.Tensor:
    if tensor.dtype == target_dtype:
        return tensor
    if _HAS_CONTENDER and (
        PhysicsLab.get_mantissa_width(tensor.dtype)
        > PhysicsLab.get_mantissa_width(target_dtype)
    ):
        if (
            torch.is_grad_enabled()
            and not torch.is_inference_mode_enabled()
            and tensor.requires_grad
        ):
            return _ContenderNarrowSTE.apply(tensor, target_dtype, int(phase_offset))
        return contender_cast(tensor, target_dtype, phase_offset=int(phase_offset))
    return tensor.to(dtype=target_dtype)


class _ColMajorWeightCache:
    __slots__ = ("w_col", "key")

    def __init__(self) -> None:
        self.w_col: Optional[torch.Tensor] = None
        self.key: Optional[Tuple[int, int, torch.device]] = None

    def get(self, w: torch.Tensor) -> torch.Tensor:
        sig = (w.data_ptr(), _tensor_version(w), w.device)
        if (
            self.w_col is not None
            and self.key == sig
            and (torch.is_inference_mode_enabled() or not self.w_col.is_inference())
        ):
            return self.w_col
        with torch.inference_mode(False):
            w_det = w.detach()
            w_base = w_det.clone() if w_det.is_inference() else w_det.contiguous()
            w_col = w_base.t()
        if not _is_stream_capturing(w.device):
            self.w_col = w_col
            self.key = sig
        return w_col

    def clear(self) -> None:
        self.w_col = None
        self.key = None


def _linear_narrow_to_fp32(
    x_c: torch.Tensor,
    w_c: torch.Tensor,
    b: Optional[torch.Tensor],
    grad_active: bool,
) -> torch.Tensor:
    orig_shape = x_c.shape[:-1]
    in_features = x_c.shape[-1]
    out_features = w_c.shape[0]
    if x_c.numel() == 0 or out_features == 0:
        return torch.empty(
            (*orig_shape, out_features), dtype=torch.float32, device=x_c.device
        )

    if not grad_active and _can_use_mm_fp32_out(x_c.device, x_c.dtype):
        x_2d = x_c.reshape(-1, in_features)
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        w_contig = w_c if w_c.is_contiguous() else w_c.contiguous()
        try:
            out_2d = torch.mm(x_2d, w_contig.t(), out_dtype=torch.float32)
            if b is not None:
                out_2d.add_(
                    b if b.dtype == torch.float32 else b.to(dtype=torch.float32)
                )
            return out_2d.view(*orig_shape, out_features)
        except (TypeError, RuntimeError, NotImplementedError):
            pass

    raw = F.linear(x_c, w_c, None)
    out_f32 = raw if raw.dtype == torch.float32 else raw.to(dtype=torch.float32)
    if b is not None:
        b_f32 = b if b.dtype == torch.float32 else b.to(dtype=torch.float32)
        out_f32 = (out_f32 + b_f32) if grad_active else out_f32.add_(b_f32)
    return out_f32


def _linear_fp8_weight_to_fp32(
    x_c: torch.Tensor,
    w_fp8: torch.Tensor,
    b: Optional[torch.Tensor],
    grad_active: bool,
) -> torch.Tensor:
    orig_shape = x_c.shape[:-1]
    in_features = x_c.shape[-1]
    out_features = int(w_fp8.shape[0])
    if x_c.numel() == 0 or out_features == 0:
        return torch.empty(
            (*orig_shape, out_features), dtype=torch.float32, device=x_c.device
        )

    eff_tile = _get_l2_tile_rows(
        x_c.device, in_features, elem_bytes=x_c.element_size()
    )

    if grad_active:
        if out_features <= eff_tile:
            return _linear_narrow_to_fp32(
                x_c, w_fp8.to(dtype=x_c.dtype), b, grad_active=True
            )
        flat_x = x_c.reshape(-1, in_features)
        chunks: List[torch.Tensor] = []
        for offset in range(0, out_features, eff_tile):
            stop = min(offset + eff_tile, out_features)
            w_tile = w_fp8[offset:stop].to(dtype=x_c.dtype)
            b_tile = b[offset:stop] if b is not None else None
            chunks.append(
                _linear_narrow_to_fp32(flat_x, w_tile, b_tile, grad_active=True)
            )
        return torch.cat(chunks, dim=-1).view(*orig_shape, out_features)

    x_2d = x_c.reshape(-1, in_features)
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    num_tokens = x_2d.shape[0]
    use_mm_f32 = _can_use_mm_fp32_out(x_c.device, x_c.dtype)

    if out_features <= eff_tile:
        w_buf = _get_narrow_tile_workspace(
            x_c.device, x_c.dtype, out_features, in_features
        )
        w_buf.copy_(w_fp8)
        if use_mm_f32:
            try:
                out_2d = torch.mm(x_2d, w_buf.t(), out_dtype=torch.float32)
                if b is not None:
                    out_2d.add_(
                        b if b.dtype == torch.float32 else b.to(dtype=torch.float32)
                    )
                return out_2d.view(*orig_shape, out_features)
            except (TypeError, RuntimeError, NotImplementedError):
                pass
        raw = F.linear(x_2d, w_buf, None)
        out_2d = raw if raw.dtype == torch.float32 else raw.to(dtype=torch.float32)
        if b is not None:
            out_2d.add_(
                b if b.dtype == torch.float32 else b.to(dtype=torch.float32)
            )
        return out_2d.view(*orig_shape, out_features)

    out_2d = torch.empty(
        (num_tokens, out_features), dtype=torch.float32, device=x_c.device
    )
    w_buf = _get_narrow_tile_workspace(
        x_c.device, x_c.dtype, eff_tile, in_features
    )
    use_slice_out = use_mm_f32 and _can_use_mm_fp32_slice_out(
        x_c.device, x_c.dtype
    )

    for offset in range(0, out_features, eff_tile):
        stop = min(offset + eff_tile, out_features)
        w_view = w_buf[: stop - offset]
        w_view.copy_(w_fp8[offset:stop])
        out_slice = out_2d[:, offset:stop]
        computed = False
        if use_slice_out:
            try:
                torch.mm(
                    x_2d, w_view.t(), out_dtype=torch.float32, out=out_slice
                )
                computed = True
            except (TypeError, RuntimeError, NotImplementedError):
                use_slice_out = False
        if not computed and use_mm_f32:
            try:
                out_slice.copy_(
                    torch.mm(x_2d, w_view.t(), out_dtype=torch.float32)
                )
                computed = True
            except (TypeError, RuntimeError, NotImplementedError):
                use_mm_f32 = False
        if not computed:
            out_slice.copy_(F.linear(x_2d, w_view, None))
        if b is not None:
            b_tile = b[offset:stop]
            out_slice.add_(
                b_tile
                if b_tile.dtype == torch.float32
                else b_tile.to(dtype=torch.float32)
            )
    return out_2d.view(*orig_shape, out_features)


def _dispatch_quant_safe_linear_fp32(
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    col_cache: _ColMajorWeightCache,
) -> torch.Tensor:
    grad_active = (
        torch.is_grad_enabled()
        and not torch.is_inference_mode_enabled()
        and (
            x.requires_grad
            or w.requires_grad
            or (b is not None and b.requires_grad)
        )
    )

    if x.dtype == torch.float32 or w.dtype == torch.float32:
        x_f32 = x if x.dtype == torch.float32 else x.to(dtype=torch.float32)
        w_f32 = w if w.dtype == torch.float32 else w.to(dtype=torch.float32)
        b_f32 = (
            (b if b.dtype == torch.float32 else b.to(dtype=torch.float32))
            if b is not None
            else None
        )
        with IEEE754FP32Guard(), torch.autocast(
            device_type=x_f32.device.type, enabled=False
        ):
            return F.linear(x_f32, w_f32, b_f32)

    if _FP8_DTYPES and w.dtype in _FP8_DTYPES:
        if (
            not grad_active
            and x.dtype in _FP8_DTYPES
            and x.device.type == "cuda"
            and hasattr(torch, "_scaled_mm")
            and not (x.dtype in _E5M2_DTYPES and w.dtype in _E5M2_DTYPES)
        ):
            orig_shape = x.shape[:-1]
            in_features = x.shape[-1]
            out_features = w.shape[0]
            x_2d = x.reshape(-1, in_features).contiguous()
            m_rows = x_2d.shape[0]
            if (
                m_rows >= 16
                and (m_rows & 15) == 0
                and (in_features & 15) == 0
                and (out_features & 15) == 0
            ):
                smm_mode = _get_scaled_mm_mode(x.device, x.dtype, w.dtype)
                if smm_mode in ("fp32", "bf16"):
                    try:
                        one = _get_unit_scale_fp32(x.device)
                        w_col = col_cache.get(w)
                        req_dt = (
                            torch.float32 if smm_mode == "fp32" else torch.bfloat16
                        )
                        mm_out = torch._scaled_mm(
                            x_2d,
                            w_col,
                            scale_a=one,
                            scale_b=one,
                            out_dtype=req_dt,
                        )
                        res = mm_out[0] if isinstance(mm_out, tuple) else mm_out
                        res_f32 = (
                            res
                            if res.dtype == torch.float32
                            else res.to(dtype=torch.float32)
                        )
                        if b is not None:
                            res_f32.add_(
                                b
                                if b.dtype == torch.float32
                                else b.to(dtype=torch.float32)
                            )
                        return res_f32.view(*orig_shape, out_features)
                    except (RuntimeError, NotImplementedError):
                        pass

        compute_dt = (
            torch.float16
            if (x.dtype == torch.float16 and w.dtype != _E8M0_DTYPE)
            else torch.bfloat16
        )
        x_c = x if x.dtype == compute_dt else x.to(dtype=compute_dt)
        return _linear_fp8_weight_to_fp32(x_c, w, b, grad_active=grad_active)

    if x.dtype == w.dtype and x.dtype in (torch.bfloat16, torch.float16):
        return _linear_narrow_to_fp32(x, w, b, grad_active=grad_active)

    x_f32 = x.to(dtype=torch.float32)
    w_f32 = w.to(dtype=torch.float32)
    b_f32 = (
        (b if b.dtype == torch.float32 else b.to(dtype=torch.float32))
        if b is not None
        else None
    )
    with IEEE754FP32Guard(), torch.autocast(
        device_type=x_f32.device.type, enabled=False
    ):
        return F.linear(x_f32, w_f32, b_f32)


def _dispatch_quant_safe_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    col_cache: _ColMajorWeightCache,
    *,
    phase_offset: int = 0,
) -> torch.Tensor:
    if _FP8_DTYPES and x.dtype in _FP8_DTYPES:
        out_dtype = torch.bfloat16
    elif x.dtype in (torch.float32, torch.bfloat16, torch.float16):
        out_dtype = x.dtype
    else:
        out_dtype = torch.float32

    acc_f32 = _dispatch_quant_safe_linear_fp32(x, w, b, col_cache)
    if out_dtype == torch.float32:
        return acc_f32
    return _project_narrowing(acc_f32, out_dtype, phase_offset=phase_offset)


class QuantSafeLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        *,
        phase_offset: int = 0,
    ) -> None:
        with torch.inference_mode(False):
            super().__init__(
                in_features, out_features, bias=bias, device=device, dtype=dtype
            )
        self._col_cache = _ColMajorWeightCache()
        self.phase_offset: int = int(phase_offset)

    def forward_fp32(self, x: torch.Tensor) -> torch.Tensor:
        return _dispatch_quant_safe_linear_fp32(
            x, self.weight, self.bias, self._col_cache
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _dispatch_quant_safe_linear(
            x,
            self.weight,
            self.bias,
            self._col_cache,
            phase_offset=self.phase_offset,
        )


def _linear_rows_fp32(
    x_f32: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    row_start: int = 0,
    row_end: Optional[int] = None,
    tile_rows: Optional[int] = None,
) -> torch.Tensor:
    x_clean = x_f32 if x_f32.dtype == torch.float32 else x_f32.float()
    total_rows = int(weight.shape[0])
    start = max(0, int(row_start))
    end = total_rows if row_end is None else min(total_rows, int(row_end))
    out_features = max(0, end - start)
    orig_shape = x_clean.shape[:-1]
    in_features = x_clean.shape[-1]

    if out_features == 0:
        return torch.empty(
            (*orig_shape, 0), dtype=torch.float32, device=x_clean.device
        )

    w_slice = weight[start:end]
    b_slice = bias[start:end] if bias is not None else None
    dev_type = x_clean.device.type

    with IEEE754FP32Guard(), torch.autocast(device_type=dev_type, enabled=False):
        if w_slice.dtype == torch.float32 and (
            b_slice is None or b_slice.dtype == torch.float32
        ):
            return F.linear(x_clean, w_slice, b_slice)

        eff_tile = (
            int(tile_rows)
            if tile_rows is not None
            else _get_l2_tile_rows(x_clean.device, in_features, elem_bytes=4)
        )
        grad_active = torch.is_grad_enabled() and (
            x_clean.requires_grad
            or w_slice.requires_grad
            or (b_slice is not None and b_slice.requires_grad)
        )

        flat_x = x_clean.reshape(-1, in_features).contiguous()
        num_tokens = flat_x.shape[0]

        if grad_active:
            if out_features <= eff_tile:
                w_f32 = w_slice.to(dtype=torch.float32)
                b_f32 = (
                    b_slice.to(dtype=torch.float32) if b_slice is not None else None
                )
                return F.linear(flat_x, w_f32, b_f32).view(*orig_shape, out_features)
            chunks: List[torch.Tensor] = []
            for offset in range(0, out_features, eff_tile):
                stop = min(offset + eff_tile, out_features)
                w_tile = w_slice[offset:stop].to(dtype=torch.float32)
                b_tile = (
                    b_slice[offset:stop].to(dtype=torch.float32)
                    if b_slice is not None
                    else None
                )
                chunks.append(F.linear(flat_x, w_tile, b_tile))
            return torch.cat(chunks, dim=-1).view(*orig_shape, out_features)

        out_flat = torch.empty(
            (num_tokens, out_features), dtype=torch.float32, device=x_clean.device
        )
        ws_rows = min(out_features, eff_tile)
        w_buf = _get_fp32_tile_workspace(x_clean.device, ws_rows, in_features)
        for offset in range(0, out_features, eff_tile):
            stop = min(offset + eff_tile, out_features)
            w_view = w_buf[: stop - offset]
            w_view.copy_(w_slice[offset:stop])
            torch.mm(flat_x, w_view.t(), out=out_flat[:, offset:stop])
            if b_slice is not None:
                b_tile = b_slice[offset:stop]
                out_flat[:, offset:stop].add_(
                    b_tile
                    if b_tile.dtype == torch.float32
                    else b_tile.to(dtype=torch.float32)
                )
        return out_flat.view(*orig_shape, out_features)


def sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    grouped = query.shape[1] != key.shape[1]
    use_native_gqa = (
        _HAS_SDPA_GQA
        and grouped
        and query.device.type == "cuda"
        and query.dtype in (torch.float16, torch.bfloat16)
    )
    if grouped and not use_native_gqa:
        groups = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    if query.dtype == torch.float32:
        with IEEE754FP32Guard(), torch.autocast(
            device_type=query.device.type, enabled=False
        ):
            if use_native_gqa:
                return F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                    enable_gqa=True,
                )
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
    if use_native_gqa:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            is_causal=is_causal,
            enable_gqa=True,
        )
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        is_causal=is_causal,
    )


def _single_step_2d_mask(
    attention_mask: torch.Tensor,
    key_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    mask = (
        attention_mask
        if attention_mask.device == device
        else attention_mask.to(device=device)
    )
    if mask.shape[0] != batch_size or mask.shape[1] > key_length:
        raise ValueError(
            "2D attention mask must span current batch and populated cache sequences."
        )
    if mask.shape[1] < key_length:
        mask = F.pad(mask, (0, key_length - mask.shape[1]), value=0)
    return (mask if mask.dtype == torch.bool else mask.bool())[:, None, None, :]


def _causal_mask(
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    key_length: int,
    batch_size: int,
) -> torch.Tensor:
    device = cache_position.device
    visible = (
        torch.arange(key_length, device=device)[None, :] <= cache_position[:, None]
    )
    visible = visible[None, None].expand(batch_size, 1, -1, -1)
    if attention_mask is None:
        return visible
    mask = attention_mask.to(device=device)
    if mask.ndim == 2:
        if mask.shape[0] != batch_size or mask.shape[1] > key_length:
            raise ValueError(
                "2D attention mask must span current batch and populated cache sequences."
            )
        if mask.shape[1] < key_length:
            mask = F.pad(mask, (0, key_length - mask.shape[1]), value=0)
        return visible & mask[:, None, None, :].bool()
    if mask.ndim != 4 or mask.shape[-2:] != visible.shape[-2:]:
        raise ValueError(
            "Dimensional layout of attention mask fails 4D causal verification."
        )
    if mask.dtype == torch.bool:
        return visible & mask
    return mask.masked_fill(~visible, float("-inf"))


class YuE2Config(PretrainedConfig):
    model_type = "yue2"
    _hf_fields = frozenset(
        {
            "model_type",
            "architectures",
            "auto_map",
            "transformers_version",
            "dtype",
            "torch_dtype",
            "return_dict",
            "output_hidden_states",
            "output_attentions",
            "use_cache",
            "tie_word_embeddings",
            "torchscript",
            "is_decoder",
            "is_encoder_decoder",
            "add_cross_attention",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "decoder_start_token_id",
            "attn_implementation",
        }
    )
    _inference_fields = frozenset(
        {
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "intermediate_size",
            "vocab_size",
            "rms_norm_eps",
            "rope_theta",
            "max_position_embeddings",
            "tie_word_embeddings",
            "latent_type",
            "latent_dim",
            "max_latent_frames",
            "timestep_shift",
        }
    )

    def to_dict(self):
        return {
            key: value
            for key, value in super().to_dict().items()
            if key in self._hf_fields or key in self._inference_fields
        }

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 6144,
        vocab_size: int = 184704,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        max_position_embeddings: int = 24576,
        tie_word_embeddings: bool = False,
        latent_type: str = "vae",
        latent_dim: int = 64,
        max_latent_frames: int = 24576,
        timestep_shift: float = 1.0,
        return_dict: bool = True,
        **kwargs,
    ):
        if latent_type != "vae":
            raise ValueError(
                "YuE2 specification requires continuous latent_type='vae'."
            )
        clean_kwargs = {
            key: value for key, value in kwargs.items() if key in self._hf_fields
        }
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            return_dict=return_dict,
            **clean_kwargs,
        )
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.latent_type = latent_type
        self.latent_dim = latent_dim
        self.max_latent_frames = max_latent_frames
        self.timestep_shift = timestep_shift


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, *, phase_offset: int = 0):
        super().__init__()
        with torch.inference_mode(False):
            self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.phase_offset: int = int(phase_offset)

    def forward_fp32(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x if x.dtype == torch.float32 else x.float()
        w_f32 = (
            self.weight if self.weight.dtype == torch.float32 else self.weight.float()
        )
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        return x_f32 * torch.rsqrt(variance + self.eps) * w_f32

    def forward_to_dtype(
        self,
        x: torch.Tensor,
        target_dtype: torch.dtype,
        *,
        phase_offset: Optional[int] = None,
    ) -> torch.Tensor:
        eff_phase = self.phase_offset if phase_offset is None else int(phase_offset)
        return _project_narrowing(
            self.forward_fp32(x), target_dtype, phase_offset=eff_phase
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_to_dtype(x, x.dtype)


def _project_unit_circle_contenders_cpu(
    cos_64: torch.Tensor, sin_64: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    neg_inf = torch.tensor(-float("inf"), dtype=torch.float32)
    pos_inf = torch.tensor(float("inf"), dtype=torch.float32)

    c_rtn = cos_64.to(dtype=torch.float32)
    c_rtn64 = c_rtn.to(dtype=torch.float64)
    c_lo = torch.where(cos_64 >= c_rtn64, c_rtn, torch.nextafter(c_rtn, neg_inf))
    c_hi = torch.where(cos_64 > c_rtn64, torch.nextafter(c_rtn, pos_inf), c_rtn)

    s_rtn = sin_64.to(dtype=torch.float32)
    s_rtn64 = s_rtn.to(dtype=torch.float64)
    s_lo = torch.where(sin_64 >= s_rtn64, s_rtn, torch.nextafter(s_rtn, neg_inf))
    s_hi = torch.where(sin_64 > s_rtn64, torch.nextafter(s_rtn, pos_inf), s_rtn)

    c_lo64, c_hi64 = c_lo.to(dtype=torch.float64), c_hi.to(dtype=torch.float64)
    s_lo64, s_hi64 = s_lo.to(dtype=torch.float64), s_hi.to(dtype=torch.float64)

    def _s1_loss(ca64: torch.Tensor, sb64: torch.Tensor) -> torch.Tensor:
        dc = ca64 - cos_64
        ds = sb64 - sin_64
        dr = (ca64 * ca64 + sb64 * sb64) - 1.0
        return dc * dc + ds * ds + dr * dr

    best_c = c_lo
    best_s = s_lo
    best_loss = _s1_loss(c_lo64, s_lo64)

    for cand_c, cand_s, cand_c64, cand_s64 in (
        (c_lo, s_hi, c_lo64, s_hi64),
        (c_hi, s_lo, c_hi64, s_lo64),
        (c_hi, s_hi, c_hi64, s_hi64),
    ):
        cand_loss = _s1_loss(cand_c64, cand_s64)
        better = cand_loss < best_loss
        best_c = torch.where(better, cand_c, best_c)
        best_s = torch.where(better, cand_s, best_s)
        best_loss = torch.where(better, cand_loss, best_loss)

    best_c = torch.where(best_c == 0.0, torch.copysign(best_c, c_rtn), best_c)
    best_s = torch.where(best_s == 0.0, torch.copysign(best_s, s_rtn), best_s)
    return best_c, best_s


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: float = 1000000.0,
        max_cached_positions: int = 24576,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.max_cached_positions = max_cached_positions
        self._inv_freq: Optional[torch.Tensor] = None
        self._inv_freq_f64: Optional[torch.Tensor] = None
        self._cos_cache: Optional[torch.Tensor] = None
        self._sin_cache: Optional[torch.Tensor] = None
        self._cached_seq_len: int = 0

    def _clear_tables(self) -> None:
        self._inv_freq = None
        self._inv_freq_f64 = None
        self._cos_cache = None
        self._sin_cache = None
        self._cached_seq_len = 0

    def _apply(self, fn, recurse: bool = True):
        with torch.inference_mode(False):
            res = super()._apply(fn, recurse=recurse)
            ref_buf = (
                self._cos_cache if self._cos_cache is not None else self._inv_freq
            )
            if ref_buf is not None:
                probe = fn(torch.empty(0, dtype=torch.float32, device=ref_buf.device))
                if probe.is_meta:
                    self._clear_tables()
                elif probe.device != ref_buf.device:
                    if self._inv_freq is not None:
                        inv = self._inv_freq.detach().to(
                            device=probe.device, dtype=torch.float32
                        )
                        self._inv_freq = inv.clone() if inv.is_inference() else inv
                    if self._inv_freq_f64 is not None:
                        if probe.device.type != "mps":
                            inv64 = self._inv_freq_f64.detach().to(
                                device=probe.device, dtype=torch.float64
                            )
                            self._inv_freq_f64 = (
                                inv64.clone() if inv64.is_inference() else inv64
                            )
                        else:
                            self._inv_freq_f64 = None
                    if self._cos_cache is not None and self._sin_cache is not None:
                        cc = (
                            self._cos_cache.detach()
                            .to(device=probe.device, dtype=torch.float32)
                            .contiguous()
                        )
                        sc = (
                            self._sin_cache.detach()
                            .to(device=probe.device, dtype=torch.float32)
                            .contiguous()
                        )
                        self._cos_cache = cc.clone() if cc.is_inference() else cc
                        self._sin_cache = sc.clone() if sc.is_inference() else sc
            return res

    def _compute_inv_freq_cpu_f64(self) -> torch.Tensor:
        with torch.inference_mode(False):
            exponents = (
                torch.arange(0, self.head_dim, 2, dtype=torch.float64, device="cpu")
                / float(self.head_dim)
            )
            return 1.0 / (float(self.base) ** exponents)

    def _ensure_inv_freq(self, device: torch.device) -> None:
        if (
            self._inv_freq is None
            or self._inv_freq.device != device
            or self._inv_freq.is_inference()
            or (
                device.type != "mps"
                and (
                    self._inv_freq_f64 is None
                    or self._inv_freq_f64.device != device
                    or self._inv_freq_f64.is_inference()
                )
            )
        ):
            inv_cpu_f64 = self._compute_inv_freq_cpu_f64()
            with torch.inference_mode(False):
                self._inv_freq = inv_cpu_f64.to(
                    dtype=torch.float32, device=device
                ).contiguous()
                self._inv_freq_f64 = (
                    inv_cpu_f64.to(dtype=torch.float64, device=device).contiguous()
                    if device.type != "mps"
                    else None
                )

    def _ensure_tables(self, device: torch.device, required_pos: int) -> None:
        if (
            self._cos_cache is not None
            and self._sin_cache is not None
            and not self._cos_cache.is_inference()
            and not self._sin_cache.is_inference()
            and self._cos_cache.device == device
            and required_pos <= self._cached_seq_len
        ):
            return
        with torch.inference_mode(False):
            target_len = max(self.max_cached_positions, required_pos)
            inv_cpu_f64 = self._compute_inv_freq_cpu_f64()
            t_cpu_f64 = torch.arange(
                target_len, dtype=torch.float64, device="cpu"
            ).unsqueeze(-1)
            angles_cpu_f64 = t_cpu_f64 * inv_cpu_f64
            cos_cpu_f32, sin_cpu_f32 = _project_unit_circle_contenders_cpu(
                angles_cpu_f64.cos(), angles_cpu_f64.sin()
            )
            self._inv_freq = inv_cpu_f64.to(
                dtype=torch.float32, device=device
            ).contiguous()
            self._inv_freq_f64 = (
                inv_cpu_f64.to(dtype=torch.float64, device=device).contiguous()
                if device.type != "mps"
                else None
            )
            self._cos_cache = cos_cpu_f32.to(device=device).contiguous()
            self._sin_cache = sin_cpu_f32.to(device=device).contiguous()
            self._cached_seq_len = target_len

    def _eval_trig_direct(
        self, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = position_ids.device
        self._ensure_inv_freq(device)
        if device.type != "mps" and self._inv_freq_f64 is not None:
            angles_64 = (
                position_ids.to(dtype=torch.float64).unsqueeze(-1)
                * self._inv_freq_f64
            )
            return (
                angles_64.cos().to(dtype=torch.float32),
                angles_64.sin().to(dtype=torch.float32),
            )
        assert self._inv_freq is not None
        angles_32 = (
            position_ids.to(dtype=torch.float32).unsqueeze(-1) * self._inv_freq
        )
        return angles_32.cos(), angles_32.sin()

    def forward(
        self,
        position_ids: torch.Tensor,
        seq_len_hint: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = position_ids.device
        if position_ids.numel() == 0:
            empty = torch.empty(
                (*position_ids.shape, self.head_dim // 2),
                dtype=torch.float32,
                device=device,
            )
            return empty, empty

        if position_ids.is_meta:
            meta_out = torch.empty(
                (*position_ids.shape, self.head_dim // 2),
                dtype=torch.float32,
                device=device,
            )
            return meta_out, meta_out

        if position_ids.dtype in (torch.int32, torch.int64):
            if (
                seq_len_hint is not None
                and seq_len_hint <= self.max_cached_positions * 2
            ):
                self._ensure_tables(device, max(1, int(seq_len_hint)))
                assert self._cos_cache is not None and self._sin_cache is not None
                clamped = position_ids.clamp(0, self._cached_seq_len - 1)
                return self._cos_cache[clamped], self._sin_cache[clamped]

            if (
                self._cos_cache is None
                or self._sin_cache is None
                or self._cos_cache.is_inference()
                or self._cos_cache.device != device
            ) and not _is_stream_capturing(device):
                self._ensure_tables(device, self.max_cached_positions)

            if (
                self._cos_cache is not None
                and self._sin_cache is not None
                and not self._cos_cache.is_inference()
                and self._cos_cache.device == device
                and self._cached_seq_len > 0
            ):
                safe_idx = position_ids.clamp(0, self._cached_seq_len - 1)
                c_tab = self._cos_cache[safe_idx]
                s_tab = self._sin_cache[safe_idx]
                in_bounds = (
                    (position_ids >= 0) & (position_ids < self._cached_seq_len)
                ).unsqueeze(-1)
                c_dir, s_dir = self._eval_trig_direct(position_ids)
                return (
                    torch.where(in_bounds, c_tab, c_dir),
                    torch.where(in_bounds, s_tab, s_dir),
                )

        return self._eval_trig_direct(position_ids)


def _apply_rotary_fp32(
    x_f32: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    half = x_f32.shape[-1] // 2
    x1, x2 = x_f32[..., :half], x_f32[..., half:]
    cos_f = cos if cos.dtype == torch.float32 else cos.float()
    sin_f = sin if sin.dtype == torch.float32 else sin.float()
    if not torch.is_grad_enabled():
        out = torch.empty_like(x_f32)
        o1, o2 = out[..., :half], out[..., half:]
        torch.mul(x1, cos_f, out=o1)
        torch.addcmul(o1, x2, sin_f, value=-1.0, out=o1)
        torch.mul(x2, cos_f, out=o2)
        torch.addcmul(o2, x1, sin_f, value=1.0, out=o2)
        return out
    rot_1 = x1 * cos_f - x2 * sin_f
    rot_2 = x2 * cos_f + x1 * sin_f
    return torch.cat([rot_1, rot_2], dim=-1)


def _apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    phase_offset: int = 0,
) -> torch.Tensor:
    x_f32 = x if x.dtype == torch.float32 else x.float()
    out_f32 = _apply_rotary_fp32(x_f32, cos, sin)
    return _project_narrowing(out_f32, x.dtype, phase_offset=phase_offset)


class Attention(nn.Module):
    def __init__(self, config: YuE2Config, *, phase_base: int = 1):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self._phase_q = int(phase_base) & 63
        self._phase_k = (int(phase_base) + 1) & 63
        self._phase_v = (int(phase_base) + 2) & 63
        self._phase_o = (int(phase_base) + 3) & 63
        self.q_proj = QuantSafeLinear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            phase_offset=self._phase_q,
        )
        self.k_proj = QuantSafeLinear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            phase_offset=self._phase_k,
        )
        self.v_proj = QuantSafeLinear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            phase_offset=self._phase_v,
        )
        self.o_proj = QuantSafeLinear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            phase_offset=self._phase_o,
        )
        self.q_norm = RMSNorm(
            self.head_dim, config.rms_norm_eps, phase_offset=self._phase_q
        )
        self.k_norm = RMSNorm(
            self.head_dim, config.rms_norm_eps, phase_offset=self._phase_k
        )
        self._qkv_weight: Optional[torch.Tensor] = None
        self._qkv_sig: Optional[Tuple[int, int, int, int, int, int]] = None
        self._qkv_col_cache = _ColMajorWeightCache()
        self._pending_rebind: bool = False
        self._register_state_dict_hook(self._detach_shared_storage_hook)

    @staticmethod
    def _detach_shared_storage_hook(module, destination, prefix, local_metadata):
        if module._qkv_weight is None or module._qkv_weight.is_meta:
            return destination
        shared_ptr = module._qkv_weight.untyped_storage().data_ptr()
        for suffix in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
            key = prefix + suffix
            t = destination.get(key)
            if (
                t is not None
                and not t.is_meta
                and t.untyped_storage().data_ptr() == shared_ptr
            ):
                with torch.inference_mode(False):
                    cloned = t.detach().clone()
                    destination[key] = (
                        nn.Parameter(cloned, requires_grad=t.requires_grad)
                        if isinstance(t, nn.Parameter)
                        else cloned
                    )
        return destination

    def _clear_fused_qkv(self) -> None:
        self._qkv_weight = None
        self._qkv_sig = None
        self._qkv_col_cache.clear()

    def unbind_fused_qkv(self) -> None:
        self._pending_rebind = False
        if self._qkv_weight is not None:
            if not self._qkv_weight.is_meta:
                with torch.inference_mode(False), torch.no_grad():
                    shared_ptr = self._qkv_weight.untyped_storage().data_ptr()
                    for proj in (self.q_proj, self.k_proj, self.v_proj):
                        w = proj.weight
                        if (
                            not w.is_meta
                            and w.untyped_storage().data_ptr() == shared_ptr
                        ):
                            proj.weight = nn.Parameter(
                                w.detach().clone(), requires_grad=w.requires_grad
                            )
            self._clear_fused_qkv()

    def bind_fused_qkv(self) -> None:
        self._pending_rebind = False
        if not self.o_proj.weight.is_meta:
            _prewarm_linear_capabilities(
                self.o_proj.weight.device, self.o_proj.weight.dtype
            )
        if (
            self.q_proj.bias is not None
            or self.k_proj.bias is not None
            or self.v_proj.bias is not None
        ):
            self._clear_fused_qkv()
            return
        wq = self.q_proj.weight
        wk = self.k_proj.weight
        wv = self.v_proj.weight
        if (
            wq.is_meta
            or wk.is_meta
            or wv.is_meta
            or wq.requires_grad
            or wk.requires_grad
            or wv.requires_grad
            or wq.dtype != wk.dtype
            or wq.dtype != wv.dtype
            or wq.device != wk.device
            or wq.device != wv.device
        ):
            self._clear_fused_qkv()
            return
        cur_sig = (
            wq.data_ptr(),
            wk.data_ptr(),
            wv.data_ptr(),
            _tensor_version(wq),
            _tensor_version(wk),
            _tensor_version(wv),
        )
        if self._qkv_weight is not None and self._qkv_sig == cur_sig:
            return
        q_out, in_dim = wq.shape
        k_out = wk.shape[0]
        v_out = wv.shape[0]
        with torch.inference_mode(False), torch.no_grad():
            fused = torch.empty(
                (q_out + k_out + v_out, in_dim), dtype=wq.dtype, device=wq.device
            )
            fused[:q_out].copy_(wq.detach())
            fused[q_out : q_out + k_out].copy_(wk.detach())
            fused[q_out + k_out :].copy_(wv.detach())
            self.q_proj.weight = nn.Parameter(fused[:q_out], requires_grad=False)
            self.k_proj.weight = nn.Parameter(
                fused[q_out : q_out + k_out], requires_grad=False
            )
            self.v_proj.weight = nn.Parameter(
                fused[q_out + k_out :], requires_grad=False
            )
            self._qkv_weight = fused
            self._qkv_col_cache.clear()
            self._qkv_sig = (
                self.q_proj.weight.data_ptr(),
                self.k_proj.weight.data_ptr(),
                self.v_proj.weight.data_ptr(),
                _tensor_version(self.q_proj.weight),
                _tensor_version(self.k_proj.weight),
                _tensor_version(self.v_proj.weight),
            )
            _prewarm_linear_capabilities(wq.device, wq.dtype)

    def train(self, mode: bool = True):
        was_bound = self._qkv_weight is not None or self._pending_rebind
        super().train(mode)
        if mode:
            self.unbind_fused_qkv()
            self._pending_rebind = was_bound
        elif was_bound:
            self.bind_fused_qkv()
        return self

    def _apply(self, fn, recurse: bool = True):
        was_bound = self._qkv_weight is not None or self._pending_rebind
        self._clear_fused_qkv()
        with torch.inference_mode(False):
            res = super()._apply(fn, recurse=recurse)
        if was_bound and not self.training:
            if recurse:
                self.bind_fused_qkv()
            else:
                self._pending_rebind = True
        return res

    def _can_use_fused_qkv(self) -> bool:
        if (
            self._pending_rebind
            and not self.training
            and not torch.is_grad_enabled()
        ):
            self.bind_fused_qkv()
        if (
            self._qkv_weight is None
            or self.q_proj.bias is not None
            or self.k_proj.bias is not None
            or self.v_proj.bias is not None
        ):
            return False
        wq = self.q_proj.weight
        wk = self.k_proj.weight
        wv = self.v_proj.weight
        if torch.is_grad_enabled() and (
            wq.requires_grad or wk.requires_grad or wv.requires_grad
        ):
            return False
        return self._qkv_sig == (
            wq.data_ptr(),
            wk.data_ptr(),
            wv.data_ptr(),
            _tensor_version(wq),
            _tensor_version(wk),
            _tensor_version(wv),
        )

    def project_qkv(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        target_dt = x.dtype
        if self._can_use_fused_qkv():
            assert self._qkv_weight is not None
            q_out = self.num_heads * self.head_dim
            kv_out = self.num_kv_heads * self.head_dim
            qkv_f32 = _dispatch_quant_safe_linear_fp32(
                x, self._qkv_weight, None, self._qkv_col_cache
            )
            q_raw_f32 = qkv_f32[..., :q_out].view(b, t, self.num_heads, self.head_dim)
            k_raw_f32 = qkv_f32[..., q_out : q_out + kv_out].view(
                b, t, self.num_kv_heads, self.head_dim
            )
            v_raw_f32 = qkv_f32[..., q_out + kv_out :].view(
                b, t, self.num_kv_heads, self.head_dim
            )
        else:
            q_raw_f32 = self.q_proj.forward_fp32(x).view(
                b, t, self.num_heads, self.head_dim
            )
            k_raw_f32 = self.k_proj.forward_fp32(x).view(
                b, t, self.num_kv_heads, self.head_dim
            )
            v_raw_f32 = self.v_proj.forward_fp32(x).view(
                b, t, self.num_kv_heads, self.head_dim
            )

        q_f32 = self.q_norm.forward_fp32(q_raw_f32)
        k_f32 = self.k_norm.forward_fp32(k_raw_f32)
        rc, rs = cos.unsqueeze(2), sin.unsqueeze(2)
        q_rot_f32 = _apply_rotary_fp32(q_f32, rc, rs)
        k_rot_f32 = _apply_rotary_fp32(k_f32, rc, rs)

        q = _project_narrowing(q_rot_f32, target_dt, phase_offset=self._phase_q)
        k = _project_narrowing(k_rot_f32, target_dt, phase_offset=self._phase_k)
        v_proj = _project_narrowing(v_raw_f32, target_dt, phase_offset=self._phase_v)
        v = v_proj if v_proj.is_contiguous() else v_proj.contiguous()
        return q, k, v

    def forward_fp32(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: Optional[DynamicCache] = None,
        layer_idx: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.project_qkv(x, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if past_key_value is not None:
            k, v = past_key_value.update(
                k, v, layer_idx, {"cache_position": cache_position}
            )
        if attention_mask is not None:
            mask_slice = attention_mask[..., : k.shape[2]]
            if mask_slice.dtype != torch.bool and mask_slice.dtype != q.dtype:
                mask_slice = mask_slice.to(dtype=q.dtype)
            out = sdpa(q, k, v, attn_mask=mask_slice)
        else:
            out = sdpa(q, k, v, is_causal=(t > 1 and k.shape[2] == t))
        return self.o_proj.forward_fp32(out.transpose(1, 2).reshape(b, t, -1))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: Optional[DynamicCache] = None,
        layer_idx: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out_f32 = self.forward_fp32(
            x,
            cos,
            sin,
            past_key_value=past_key_value,
            layer_idx=layer_idx,
            attention_mask=attention_mask,
            cache_position=cache_position,
        )
        return _project_narrowing(out_f32, x.dtype, phase_offset=self._phase_o)


class MLP(nn.Module):
    def __init__(self, config: YuE2Config, *, phase_base: int = 6):
        super().__init__()
        self._phase_act = int(phase_base) & 63
        self._phase_out = (int(phase_base) + 1) & 63
        self.gate_proj = QuantSafeLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            phase_offset=self._phase_act,
        )
        self.up_proj = QuantSafeLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            phase_offset=(self._phase_act + 1) & 63,
        )
        self.down_proj = QuantSafeLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            phase_offset=self._phase_out,
        )
        self._gate_up_weight: Optional[torch.Tensor] = None
        self._gate_up_sig: Optional[Tuple[int, int, int, int]] = None
        self._gate_up_col_cache = _ColMajorWeightCache()
        self._pending_rebind: bool = False
        self._register_state_dict_hook(self._detach_shared_storage_hook)

    @staticmethod
    def _detach_shared_storage_hook(module, destination, prefix, local_metadata):
        if module._gate_up_weight is None or module._gate_up_weight.is_meta:
            return destination
        shared_ptr = module._gate_up_weight.untyped_storage().data_ptr()
        for suffix in ("gate_proj.weight", "up_proj.weight"):
            key = prefix + suffix
            t = destination.get(key)
            if (
                t is not None
                and not t.is_meta
                and t.untyped_storage().data_ptr() == shared_ptr
            ):
                with torch.inference_mode(False):
                    cloned = t.detach().clone()
                    destination[key] = (
                        nn.Parameter(cloned, requires_grad=t.requires_grad)
                        if isinstance(t, nn.Parameter)
                        else cloned
                    )
        return destination

    def _clear_fused_gate_up(self) -> None:
        self._gate_up_weight = None
        self._gate_up_sig = None
        self._gate_up_col_cache.clear()

    def unbind_fused_gate_up(self) -> None:
        self._pending_rebind = False
        if self._gate_up_weight is not None:
            if not self._gate_up_weight.is_meta:
                with torch.inference_mode(False), torch.no_grad():
                    shared_ptr = self._gate_up_weight.untyped_storage().data_ptr()
                    for proj in (self.gate_proj, self.up_proj):
                        w = proj.weight
                        if (
                            not w.is_meta
                            and w.untyped_storage().data_ptr() == shared_ptr
                        ):
                            proj.weight = nn.Parameter(
                                w.detach().clone(), requires_grad=w.requires_grad
                            )
            self._clear_fused_gate_up()

    def bind_fused_gate_up(self) -> None:
        self._pending_rebind = False
        if not self.down_proj.weight.is_meta:
            _prewarm_linear_capabilities(
                self.down_proj.weight.device, self.down_proj.weight.dtype
            )
        if self.gate_proj.bias is not None or self.up_proj.bias is not None:
            self._clear_fused_gate_up()
            return
        wg = self.gate_proj.weight
        wu = self.up_proj.weight
        if (
            wg.is_meta
            or wu.is_meta
            or wg.requires_grad
            or wu.requires_grad
            or wg.dtype != wu.dtype
            or wg.device != wu.device
        ):
            self._clear_fused_gate_up()
            return
        cur_sig = (
            wg.data_ptr(),
            wu.data_ptr(),
            _tensor_version(wg),
            _tensor_version(wu),
        )
        if self._gate_up_weight is not None and self._gate_up_sig == cur_sig:
            return
        inter, in_dim = wg.shape
        with torch.inference_mode(False), torch.no_grad():
            fused = torch.empty(
                (inter * 2, in_dim), dtype=wg.dtype, device=wg.device
            )
            fused[:inter].copy_(wg.detach())
            fused[inter:].copy_(wu.detach())
            self.gate_proj.weight = nn.Parameter(fused[:inter], requires_grad=False)
            self.up_proj.weight = nn.Parameter(fused[inter:], requires_grad=False)
            self._gate_up_weight = fused
            self._gate_up_col_cache.clear()
            self._gate_up_sig = (
                self.gate_proj.weight.data_ptr(),
                self.up_proj.weight.data_ptr(),
                _tensor_version(self.gate_proj.weight),
                _tensor_version(self.up_proj.weight),
            )
            _prewarm_linear_capabilities(wg.device, wg.dtype)

    def train(self, mode: bool = True):
        was_bound = self._gate_up_weight is not None or self._pending_rebind
        super().train(mode)
        if mode:
            self.unbind_fused_gate_up()
            self._pending_rebind = was_bound
        elif was_bound:
            self.bind_fused_gate_up()
        return self

    def _apply(self, fn, recurse: bool = True):
        was_bound = self._gate_up_weight is not None or self._pending_rebind
        self._clear_fused_gate_up()
        with torch.inference_mode(False):
            res = super()._apply(fn, recurse=recurse)
        if was_bound and not self.training:
            if recurse:
                self.bind_fused_gate_up()
            else:
                self._pending_rebind = True
        return res

    def _can_use_fused_gate_up(self) -> bool:
        if (
            self._pending_rebind
            and not self.training
            and not torch.is_grad_enabled()
        ):
            self.bind_fused_gate_up()
        if (
            self._gate_up_weight is None
            or self.gate_proj.bias is not None
            or self.up_proj.bias is not None
        ):
            return False
        wg = self.gate_proj.weight
        wu = self.up_proj.weight
        if torch.is_grad_enabled() and (wg.requires_grad or wu.requires_grad):
            return False
        return self._gate_up_sig == (
            wg.data_ptr(),
            wu.data_ptr(),
            _tensor_version(wg),
            _tensor_version(wu),
        )

    def forward_fp32(self, x: torch.Tensor) -> torch.Tensor:
        if self._can_use_fused_gate_up():
            assert self._gate_up_weight is not None
            gu_f32 = _dispatch_quant_safe_linear_fp32(
                x, self._gate_up_weight, None, self._gate_up_col_cache
            )
            inter = gu_f32.shape[-1] // 2
            gate_f32 = gu_f32[..., :inter]
            up_f32 = gu_f32[..., inter:]
            if not torch.is_grad_enabled():
                F.silu(gate_f32, inplace=True)
                gate_f32.mul_(up_f32)
                act = _project_narrowing(
                    gate_f32, x.dtype, phase_offset=self._phase_act
                )
                return self.down_proj.forward_fp32(act)
            prod = F.silu(gate_f32) * up_f32
            return self.down_proj.forward_fp32(
                _project_narrowing(prod, x.dtype, phase_offset=self._phase_act)
            )

        gate_f32 = self.gate_proj.forward_fp32(x)
        up_f32 = self.up_proj.forward_fp32(x)
        if not torch.is_grad_enabled():
            F.silu(gate_f32, inplace=True)
            gate_f32.mul_(up_f32)
            act = _project_narrowing(
                gate_f32, x.dtype, phase_offset=self._phase_act
            )
            return self.down_proj.forward_fp32(act)

        prod = F.silu(gate_f32) * up_f32
        return self.down_proj.forward_fp32(
            _project_narrowing(prod, x.dtype, phase_offset=self._phase_act)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_f32 = self.forward_fp32(x)
        return _project_narrowing(out_f32, x.dtype, phase_offset=self._phase_out)


class DecoderLayer(nn.Module):
    def __init__(self, config: YuE2Config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = int(layer_idx)
        base = (self.layer_idx * 17) & 63
        self._phase_res = (base + 16) & 63
        self.input_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, phase_offset=base
        )
        self.self_attn = Attention(config, phase_base=(base + 1) & 63)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, phase_offset=(base + 5) & 63
        )
        self.mlp = MLP(config, phase_base=(base + 6) & 63)
        self.nar_input_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, phase_offset=(base + 8) & 63
        )
        self.nar_self_attn = Attention(config, phase_base=(base + 9) & 63)
        self.nar_pre_mlp_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, phase_offset=(base + 13) & 63
        )
        self.nar_mlp = MLP(config, phase_base=(base + 14) & 63)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: Optional[DynamicCache] = None,
        layer_idx: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        ar_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        compute_dtype: Optional[torch.dtype] = None,
        _inplace_residual: bool = False,
    ) -> torch.Tensor:
        act_dt = compute_dtype if compute_dtype is not None else x.dtype
        can_inplace = (not torch.is_grad_enabled()) and (
            _inplace_residual or x.dtype != torch.float32
        )
        x_res_f32 = x if x.dtype == torch.float32 else x.float()

        if ar_mask is not None:
            mask_3d = ar_mask.unsqueeze(-1)
            mask_4d = ar_mask.unsqueeze(-1).unsqueeze(-1)
            ln_ar = self.input_layernorm.forward_to_dtype(x_res_f32, act_dt)
            ln_nar = self.nar_input_layernorm.forward_to_dtype(x_res_f32, act_dt)
            q_ar, k_ar, v_ar = self.self_attn.project_qkv(ln_ar, cos, sin)
            q_nar, k_nar, v_nar = self.nar_self_attn.project_qkv(ln_nar, cos, sin)
            query = torch.where(mask_4d, q_ar, q_nar)
            key = torch.where(mask_4d, k_ar, k_nar)
            value = torch.where(mask_4d, v_ar, v_nar)
            b, s = x_res_f32.shape[:2]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            if (
                attention_mask is not None
                and attention_mask.dtype != torch.bool
                and attention_mask.dtype != query.dtype
            ):
                attention_mask = attention_mask.to(dtype=query.dtype)
            core_out = sdpa(query, key, value, attn_mask=attention_mask)
            core_out = core_out.transpose(1, 2).reshape(b, s, -1)
            o_ar_f32 = self.self_attn.o_proj.forward_fp32(core_out)
            o_nar_f32 = self.nar_self_attn.o_proj.forward_fp32(core_out)
            h_f32 = torch.where(mask_3d, o_ar_f32, o_nar_f32)
            x_res_f32 = x_res_f32.add_(h_f32) if can_inplace else (x_res_f32 + h_f32)
            ar_out_f32 = self.mlp.forward_fp32(
                self.post_attention_layernorm.forward_to_dtype(x_res_f32, act_dt)
            )
            nar_out_f32 = self.nar_mlp.forward_fp32(
                self.nar_pre_mlp_layernorm.forward_to_dtype(x_res_f32, act_dt)
            )
            m_f32 = torch.where(mask_3d, ar_out_f32, nar_out_f32)
        else:
            h_f32 = self.self_attn.forward_fp32(
                self.input_layernorm.forward_to_dtype(x_res_f32, act_dt),
                cos,
                sin,
                past_key_value,
                layer_idx,
                attention_mask,
                cache_position,
            )
            x_res_f32 = x_res_f32.add_(h_f32) if can_inplace else (x_res_f32 + h_f32)
            m_f32 = self.mlp.forward_fp32(
                self.post_attention_layernorm.forward_to_dtype(x_res_f32, act_dt)
            )

        x_res_f32 = x_res_f32.add_(m_f32) if can_inplace else (x_res_f32 + m_f32)
        return _project_narrowing(x_res_f32, x.dtype, phase_offset=self._phase_res)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            QuantSafeLinear(
                frequency_embedding_size, hidden_size, phase_offset=59
            ),
            nn.SiLU(),
            QuantSafeLinear(hidden_size, hidden_size, phase_offset=60),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self._freqs_cache: Optional[torch.Tensor] = None
        self._freqs_cache_f64: Optional[torch.Tensor] = None

    def _apply(self, fn, recurse: bool = True):
        with torch.inference_mode(False):
            res = super()._apply(fn, recurse=recurse)
            if self._freqs_cache is not None:
                probe = fn(
                    torch.empty(
                        0, dtype=torch.float32, device=self._freqs_cache.device
                    )
                )
                if probe.is_meta:
                    self._freqs_cache = None
                    self._freqs_cache_f64 = None
                elif probe.device != self._freqs_cache.device:
                    fc = self._freqs_cache.detach().to(
                        device=probe.device, dtype=torch.float32
                    )
                    self._freqs_cache = fc.clone() if fc.is_inference() else fc
                    if self._freqs_cache_f64 is not None:
                        if probe.device.type != "mps":
                            fc64 = self._freqs_cache_f64.detach().to(
                                device=probe.device, dtype=torch.float64
                            )
                            self._freqs_cache_f64 = (
                                fc64.clone() if fc64.is_inference() else fc64
                            )
                        else:
                            self._freqs_cache_f64 = None
            return res

    def forward_fp32(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        if (
            self._freqs_cache is None
            or self._freqs_cache.device != t.device
            or self._freqs_cache.is_inference()
            or (
                t.device.type != "mps"
                and (
                    self._freqs_cache_f64 is None
                    or self._freqs_cache_f64.device != t.device
                    or self._freqs_cache_f64.is_inference()
                )
            )
        ):
            with torch.inference_mode(False):
                freqs_cpu_64 = torch.exp(
                    -math.log(10000.0)
                    * torch.arange(half, device="cpu", dtype=torch.float64)
                    / float(half)
                )
                self._freqs_cache = freqs_cpu_64.to(
                    dtype=torch.float32
                ).to(device=t.device).contiguous()
                self._freqs_cache_f64 = (
                    freqs_cpu_64.to(device=t.device, dtype=torch.float64).contiguous()
                    if t.device.type != "mps"
                    else None
                )

        if t.device.type != "mps" and self._freqs_cache_f64 is not None:
            t_f64 = t if t.dtype == torch.float64 else t.to(dtype=torch.float64)
            args_64 = t_f64.unsqueeze(-1) * self._freqs_cache_f64.unsqueeze(0)
            emb = torch.cat(
                [
                    args_64.cos().to(dtype=torch.float32),
                    args_64.sin().to(dtype=torch.float32),
                ],
                dim=-1,
            )
        else:
            assert self._freqs_cache is not None
            t_f32 = t if t.dtype == torch.float32 else t.float()
            args_32 = t_f32.unsqueeze(-1) * self._freqs_cache.unsqueeze(0)
            emb = torch.cat([torch.cos(args_32), torch.sin(args_32)], dim=-1)

        lin0: QuantSafeLinear = self.mlp[0]  # type: ignore[assignment]
        lin2: QuantSafeLinear = self.mlp[2]  # type: ignore[assignment]
        h = lin0.forward_fp32(emb)
        if not torch.is_grad_enabled():
            F.silu(h, inplace=True)
        else:
            h = F.silu(h)
        return lin2.forward_fp32(h)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        out_f32 = self.forward_fp32(t)
        target_dtype = self.mlp[0].weight.dtype
        if target_dtype in (torch.float16, torch.bfloat16):
            return _project_narrowing(out_f32, target_dtype, phase_offset=60)
        return out_f32


def synthesize_audio_pe_fp32(
    max_frames: int,
    hidden_size: int,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    with torch.inference_mode(False):
        pe_64 = torch.zeros(max_frames, hidden_size, dtype=torch.float64, device="cpu")
        position = torch.arange(
            0, max_frames, dtype=torch.float64, device="cpu"
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float64, device="cpu")
            * (-math.log(10000.0) / float(hidden_size))
        )
        pe_64[:, 0::2] = torch.sin(position * div_term)
        pe_64[:, 1::2] = torch.cos(position * div_term)
        return pe_64.to(dtype=torch.float32).to(device=device).contiguous()


class AudioPositionEmbedding(nn.Module):
    def __init__(self, max_frames: int, hidden_size: int):
        super().__init__()
        self.max_frames = max_frames
        self.hidden_size = hidden_size
        with torch.inference_mode(False):
            if torch.zeros(1).is_meta:
                pe_buf = torch.empty(max_frames, hidden_size, dtype=torch.float32)
            else:
                pe_buf = synthesize_audio_pe_fp32(
                    max_frames, hidden_size, device="cpu"
                )
            self.register_buffer("pe", pe_buf)

    def forward_fp32(self, position_ids: torch.Tensor) -> torch.Tensor:
        out = self.pe[position_ids]
        return out if out.dtype == torch.float32 else out.float()

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.pe[position_ids]


class StaticKVCache:
    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self._seen_tokens = 0
        self._cached_pos_key: Optional[Tuple[int, int]] = None
        self._cached_pos_bounds: Tuple[int, int] = (0, 0)
        with torch.inference_mode(False):
            self.key_cache: List[torch.Tensor] = [
                torch.empty(
                    (batch_size, num_kv_heads, max_seq_len, head_dim),
                    dtype=dtype,
                    device=device,
                )
                for _ in range(num_layers)
            ]
            self.value_cache: List[torch.Tensor] = [
                torch.empty(
                    (batch_size, num_kv_heads, max_seq_len, head_dim),
                    dtype=dtype,
                    device=device,
                )
                for _ in range(num_layers)
            ]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen_tokens

    def ensure_capacity(
        self, required_seq_len: int, keep_tokens: Optional[int] = None
    ) -> None:
        req = int(required_seq_len)
        retained = (
            self._seen_tokens
            if keep_tokens is None
            else max(0, min(int(keep_tokens), self._seen_tokens))
        )
        if req <= self.max_seq_len:
            self._seen_tokens = retained
            self._cached_pos_key = None
            return

        with torch.inference_mode(False):
            for idx in range(self.num_layers):
                old_k = self.key_cache[idx]
                new_k = torch.empty(
                    (self.batch_size, self.num_kv_heads, req, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
                if retained > 0:
                    new_k[:, :, :retained, :].copy_(old_k[:, :, :retained, :].detach())
                self.key_cache[idx] = new_k
                del old_k

                old_v = self.value_cache[idx]
                new_v = torch.empty(
                    (self.batch_size, self.num_kv_heads, req, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
                if retained > 0:
                    new_v[:, :, :retained, :].copy_(old_v[:, :, :retained, :].detach())
                self.value_cache[idx] = new_v
                del old_v

        self.max_seq_len = req
        self._seen_tokens = retained
        self._cached_pos_key = None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = key_states.shape[2]
        cache_position = (
            cache_kwargs.get("cache_position")
            if isinstance(cache_kwargs, dict)
            else None
        )
        base_phase = (int(layer_idx) * 17) & 63
        k_cast = _project_narrowing(
            key_states, self.dtype, phase_offset=(base_phase + 2) & 63
        )
        v_cast = _project_narrowing(
            value_states, self.dtype, phase_offset=(base_phase + 3) & 63
        )

        if cache_position is not None and cache_position.numel() == t and t > 0:
            pos_key = (cache_position.data_ptr(), t)
            if self._cached_pos_key != pos_key:
                bounds_cpu = torch.stack(
                    (cache_position[0], cache_position[-1])
                ).to(device="cpu", dtype=torch.int64)
                self._cached_pos_bounds = (
                    int(bounds_cpu[0]),
                    int(bounds_cpu[1]),
                )
                self._cached_pos_key = pos_key
            pos0, pos_last = self._cached_pos_bounds
            if pos_last - pos0 == t - 1 and pos0 >= 0:
                end = pos_last + 1
                if end > self.max_seq_len:
                    raise ValueError(
                        f"StaticKVCache capacity limit {self.max_seq_len} exceeded by request ({end})."
                    )
                if pos0 > self._seen_tokens:
                    self.key_cache[layer_idx][:, :, self._seen_tokens : pos0].zero_()
                    self.value_cache[layer_idx][:, :, self._seen_tokens : pos0].zero_()
                self.key_cache[layer_idx][:, :, pos0:end] = k_cast
                self.value_cache[layer_idx][:, :, pos0:end] = v_cast
            else:
                end = pos_last + 1
                if end > self.max_seq_len:
                    raise ValueError(
                        f"StaticKVCache capacity limit {self.max_seq_len} exceeded by request ({end})."
                    )
                if end > self._seen_tokens:
                    self.key_cache[layer_idx][:, :, self._seen_tokens : end].zero_()
                    self.value_cache[layer_idx][:, :, self._seen_tokens : end].zero_()
                idx = cache_position.to(device=self.device, dtype=torch.long)
                self.key_cache[layer_idx].index_copy_(2, idx, k_cast)
                self.value_cache[layer_idx].index_copy_(2, idx, v_cast)
            effective_end = max(self._seen_tokens, end)
            if layer_idx == self.num_layers - 1:
                self._seen_tokens = effective_end
                self._cached_pos_key = None
            return (
                self.key_cache[layer_idx][:, :, :effective_end],
                self.value_cache[layer_idx][:, :, :effective_end],
            )

        pos = self._seen_tokens
        end = pos + t
        if end > self.max_seq_len:
            raise ValueError(
                f"StaticKVCache capacity limit {self.max_seq_len} exceeded by request ({end})."
            )
        self.key_cache[layer_idx][:, :, pos:end] = k_cast
        self.value_cache[layer_idx][:, :, pos:end] = v_cast
        if layer_idx == self.num_layers - 1:
            self._seen_tokens = end
        return (
            self.key_cache[layer_idx][:, :, :end],
            self.value_cache[layer_idx][:, :, :end],
        )

    def truncate(self, length: int) -> None:
        self._seen_tokens = max(0, min(int(length), self._seen_tokens))
        self._cached_pos_key = None

    def reset(self) -> None:
        self._seen_tokens = 0
        self._cached_pos_key = None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        with torch.inference_mode(False):
            idx = beam_idx.detach().to(self.device)
            self.key_cache = [v.index_select(0, idx) for v in self.key_cache]
            self.value_cache = [v.index_select(0, idx) for v in self.value_cache]


class Backbone(nn.Module):
    def __init__(self, config: YuE2Config):
        super().__init__()
        with torch.inference_mode(False):
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, phase_offset=63
        )
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            max_cached_positions=config.max_position_embeddings,
        )

    @property
    def trunk_dtype(self) -> torch.dtype:
        emb_dt = self.embed_tokens.weight.dtype
        if emb_dt in (torch.bfloat16, torch.float16, torch.float32):
            return emb_dt
        return torch.bfloat16

    def has_fused_projections(self) -> bool:
        for mod in self.layers.modules():
            if isinstance(mod, Attention) and (
                mod._qkv_weight is not None or mod._pending_rebind
            ):
                return True
            if isinstance(mod, MLP) and (
                mod._gate_up_weight is not None or mod._pending_rebind
            ):
                return True
        return False

    def bind_fused_projections(self) -> None:
        for mod in self.layers.modules():
            if isinstance(mod, Attention):
                mod.bind_fused_qkv()
            elif isinstance(mod, MLP):
                mod.bind_fused_gate_up()

    def unbind_fused_projections(self) -> None:
        for mod in self.layers.modules():
            if isinstance(mod, Attention):
                mod.unbind_fused_qkv()
            elif isinstance(mod, MLP):
                mod.unbind_fused_gate_up()

    def forward_prenorm(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        ar_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        seq_len_hint: Optional[int] = None,
    ) -> Tuple[torch.Tensor, ...]:
        raw_x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        act_dt = (
            raw_x.dtype
            if raw_x.dtype in (torch.bfloat16, torch.float16)
            else self.trunk_dtype
        )
        inplace_res = not torch.is_grad_enabled()
        if raw_x.dtype != torch.float32:
            x_f32 = raw_x.float()
        elif inputs_embeds is not None and inplace_res:
            x_f32 = raw_x.clone()
        else:
            x_f32 = raw_x
        cos, sin = self.rotary_emb(position_ids, seq_len_hint=seq_len_hint)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        with IEEE754FP32Guard():
            for i, layer in enumerate(self.layers):
                x_f32 = layer(
                    x_f32,
                    cos,
                    sin,
                    past_key_values if use_cache else None,
                    layer_idx=i,
                    attention_mask=attention_mask,
                    ar_mask=ar_mask,
                    cache_position=cache_position,
                    compute_dtype=act_dt,
                    _inplace_residual=inplace_res,
                )
        return x_f32, past_key_values

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        ar_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        x_f32, past_key_values = self.forward_prenorm(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
            ar_mask=ar_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
        )
        out_f32 = self.norm.forward_fp32(x_f32)
        target_dt = (
            inputs_embeds.dtype
            if (inputs_embeds is not None and inputs_embeds.dtype != torch.float32)
            else self.trunk_dtype
        )
        return _project_narrowing(out_f32, target_dt, phase_offset=63), past_key_values


class YuE2PreTrainedModel(PreTrainedModel):
    config_class = YuE2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]
    _supports_sdpa = True

    def _apply(self, fn, recurse: bool = True):
        with torch.inference_mode(False):
            return super()._apply(fn, recurse=recurse)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.01)

    def save_pretrained(self, *args, **kwargs):
        backbone = getattr(self, "model", None)
        was_fused = (
            isinstance(backbone, Backbone) and backbone.has_fused_projections()
        )
        if was_fused:
            backbone.unbind_fused_projections()
        try:
            return super().save_pretrained(*args, **kwargs)
        finally:
            if was_fused and not self.training:
                backbone.bind_fused_projections()


class YuE2ForCausalLM(YuE2PreTrainedModel, GenerationMixin):
    def __init__(self, config: YuE2Config):
        with torch.inference_mode(False):
            super().__init__(config)
            self.model = Backbone(config)
            self.lm_head = QuantSafeLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                phase_offset=61,
            )
            self.llm2vae = QuantSafeLinear(
                config.hidden_size, config.latent_dim, phase_offset=62
            )
            self.vae2llm = QuantSafeLinear(
                config.latent_dim, config.hidden_size, phase_offset=58
            )
            self.time_embedder = TimestepEmbedder(config.hidden_size)
            self.latent_pos_embed = AudioPositionEmbedding(
                config.max_latent_frames, config.hidden_size
            )
            self.post_init()

    @property
    def trunk_dtype(self) -> torch.dtype:
        return self.model.trunk_dtype

    def bind_fused_projections(self) -> None:
        self.model.bind_fused_projections()
        for head_mod in (
            self.lm_head,
            self.llm2vae,
            self.vae2llm,
            self.time_embedder.mlp[0],
            self.time_embedder.mlp[2],
        ):
            if isinstance(head_mod, nn.Linear) and not head_mod.weight.is_meta:
                _prewarm_linear_capabilities(
                    head_mod.weight.device, head_mod.weight.dtype
                )

    def unbind_fused_projections(self) -> None:
        self.model.unbind_fused_projections()

    def requires_grad_(self, requires_grad: bool = True):
        with torch.inference_mode(False):
            if requires_grad and self.model.has_fused_projections():
                self.model.unbind_fused_projections()
            super().requires_grad_(requires_grad)
            if not requires_grad and not self.training:
                self.bind_fused_projections()
            return self

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def _shift_t_value(
        self,
        t_value: float,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        t_val = float(t_value)
        if t_val >= 0.0:
            t_sig = 1.0 / (1.0 + math.exp(-t_val))
        else:
            et = math.exp(t_val)
            t_sig = et / (1.0 + et)
        shift = float(self.config.timestep_shift)
        shifted = shift * t_sig / (1.0 + (shift - 1.0) * t_sig)
        return torch.full((), shifted, dtype=dtype, device=device)

    def project_latent_to_trunk_fp32(
        self, x_nar_f32: torch.Tensor, raw_t: float, pos_emb_f32: torch.Tensor
    ) -> torch.Tensor:
        device = x_nar_f32.device
        t_dt = torch.float64 if device.type != "mps" else torch.float32
        t_shifted = self._shift_t_value(raw_t, device, t_dt).reshape(1)
        x_in = x_nar_f32 if x_nar_f32.dtype == torch.float32 else x_nar_f32.float()
        with IEEE754FP32Guard(), torch.autocast(
            device_type=device.type, enabled=False
        ):
            h_f32 = self.vae2llm.forward_fp32(x_in)
            time_vec = self.time_embedder.forward_fp32(t_shifted).unsqueeze(0)
            if not torch.is_grad_enabled():
                h_f32.add_(time_vec)
                h_f32.add_(pos_emb_f32)
            else:
                h_f32 = h_f32 + time_vec + pos_emb_f32
        return h_f32

    def project_latent_to_trunk(
        self, x_nar_f32: torch.Tensor, raw_t: float, pos_emb_f32: torch.Tensor
    ) -> torch.Tensor:
        return _project_narrowing(
            self.project_latent_to_trunk_fp32(x_nar_f32, raw_t, pos_emb_f32),
            self.trunk_dtype,
            phase_offset=58,
        )

    def project_trunk_to_velocity_fp32(
        self, prenorm_hidden: torch.Tensor
    ) -> torch.Tensor:
        with IEEE754FP32Guard(), torch.autocast(
            device_type=prenorm_hidden.device.type, enabled=False
        ):
            norm_f32 = self.model.norm.forward_fp32(prenorm_hidden)
            return self.llm2vae.forward_fp32(norm_f32)

    def forward_lm_head_fp32(
        self,
        prenorm_selected: torch.Tensor,
        vocab_slice: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        norm_f32 = self.model.norm.forward_fp32(prenorm_selected)
        if vocab_slice is not None:
            row_start, row_end = vocab_slice
        else:
            row_start, row_end = 0, int(self.lm_head.weight.shape[0])
        return _linear_rows_fp32(
            norm_f32,
            self.lm_head.weight,
            self.lm_head.bias,
            row_start=row_start,
            row_end=row_end,
        )

    def forward_vocab_slice_fp32(
        self,
        input_ids: torch.LongTensor,
        vocab_slice: Tuple[int, int],
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: bool = True,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 1,
        seq_len_hint: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape[:2]
        device = input_ids.device
        past_len = (
            past_key_values.get_seq_length()
            if past_key_values is not None and use_cache
            else 0
        )
        auto_cache_pos = cache_position is None
        is_static_kv = isinstance(past_key_values, StaticKVCache)

        key_length = past_len + seq_len
        if (
            use_cache
            and past_key_values is not None
            and hasattr(past_key_values, "get_max_cache_shape")
        ):
            cap = past_key_values.get_max_cache_shape()
            if cap is not None and cap > 0:
                key_length = cap

        needs_mask = (
            attention_mask is not None
            or key_length != past_len + seq_len
            or (past_len > 0 and seq_len > 1)
        )

        has_2d_mask = attention_mask is not None and attention_mask.ndim == 2
        fast_1step_2d = (
            seq_len == 1
            and auto_cache_pos
            and key_length == past_len + 1
            and has_2d_mask
        )
        if not auto_cache_pos:
            cache_position = cache_position.to(device=device, dtype=torch.long)
        elif (
            (needs_mask and not fast_1step_2d)
            or (position_ids is None and not has_2d_mask)
            or (use_cache and not is_static_kv)
        ):
            if seq_len == 1:
                cache_position = torch.full(
                    (1,), past_len, dtype=torch.long, device=device
                )
            else:
                cache_position = torch.arange(
                    past_len, past_len + seq_len, device=device
                )

        effective_hint = seq_len_hint
        if position_ids is None:
            if effective_hint is None:
                effective_hint = key_length
            if has_2d_mask:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)
                position_ids = position_ids[:, -seq_len:].to(device)
            else:
                position_ids = cache_position[None]
        else:
            position_ids = position_ids.to(device=device, dtype=torch.long)

        if fast_1step_2d:
            causal_mask: Optional[torch.Tensor] = _single_step_2d_mask(
                attention_mask, key_length, batch_size, device
            )
        elif needs_mask:
            causal_mask = _causal_mask(
                attention_mask, cache_position, key_length, batch_size
            )
        else:
            causal_mask = None

        kv_cache_pos = None if (auto_cache_pos and is_static_kv) else cache_position

        prenorm_states, _ = self.model.forward_prenorm(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=causal_mask,
            cache_position=kv_cache_pos,
            seq_len_hint=effective_hint,
        )
        selected_pre = (
            prenorm_states[:, -logits_to_keep:, :]
            if logits_to_keep > 0
            else prenorm_states
        )
        return self.forward_lm_head_fp32(selected_pre, vocab_slice=vocab_slice)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        return_dict: Optional[bool] = None,
        vocab_slice: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Supply exactly one of input_ids or inputs_embeds.")

        use_cache = (
            use_cache
            if use_cache is not None
            else getattr(self.config, "use_cache", True)
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else getattr(self.config, "return_dict", True)
        )

        tensor = input_ids if input_ids is not None else inputs_embeds
        batch_size, seq_len = tensor.shape[:2]
        if not seq_len:
            raise ValueError("Input must contain at least one token.")
        device = tensor.device

        past_len = (
            past_key_values.get_seq_length()
            if past_key_values is not None and use_cache
            else 0
        )
        auto_cache_pos = cache_position is None
        is_static_kv = isinstance(past_key_values, StaticKVCache)

        key_length = past_len + seq_len
        if (
            use_cache
            and past_key_values is not None
            and hasattr(past_key_values, "get_max_cache_shape")
        ):
            cap = past_key_values.get_max_cache_shape()
            if cap is not None and cap > 0:
                key_length = cap

        needs_mask = (
            attention_mask is not None
            or key_length != past_len + seq_len
            or (past_len > 0 and seq_len > 1)
        )

        has_2d_mask = attention_mask is not None and attention_mask.ndim == 2
        fast_1step_2d = (
            seq_len == 1
            and auto_cache_pos
            and key_length == past_len + 1
            and has_2d_mask
        )
        if not auto_cache_pos:
            cache_position = cache_position.to(device=device, dtype=torch.long)
            if cache_position.ndim != 1 or cache_position.numel() != seq_len:
                raise ValueError(
                    "cache_position must identify each current token's physical cache slot."
                )
        elif (
            (needs_mask and not fast_1step_2d)
            or (position_ids is None and not has_2d_mask)
            or (use_cache and not is_static_kv)
        ):
            if seq_len == 1:
                cache_position = torch.full(
                    (1,), past_len, dtype=torch.long, device=device
                )
            else:
                cache_position = torch.arange(
                    past_len, past_len + seq_len, device=device
                )

        seq_len_hint: Optional[int] = None
        if position_ids is None:
            seq_len_hint = key_length
            if has_2d_mask:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)
                position_ids = position_ids[:, -seq_len:].to(device)
            else:
                position_ids = cache_position[None]
        else:
            position_ids = position_ids.to(device=device, dtype=torch.long)
        if position_ids.shape[-1] != seq_len:
            raise ValueError("position_ids must cover the current input tokens.")

        if fast_1step_2d:
            causal_mask: Optional[torch.Tensor] = _single_step_2d_mask(
                attention_mask, key_length, batch_size, device
            )
        elif needs_mask:
            causal_mask = _causal_mask(
                attention_mask, cache_position, key_length, batch_size
            )
        else:
            causal_mask = None

        kv_cache_pos = None if (auto_cache_pos and is_static_kv) else cache_position

        prenorm_states, past_key_values = self.model.forward_prenorm(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=causal_mask,
            inputs_embeds=inputs_embeds,
            cache_position=kv_cache_pos,
            seq_len_hint=seq_len_hint,
        )

        if isinstance(logits_to_keep, int):
            if logits_to_keep < 0:
                raise ValueError("logits_to_keep must be nonnegative.")
            selected_pre = (
                prenorm_states[:, -logits_to_keep:, :]
                if logits_to_keep
                else prenorm_states
            )
        else:
            selected_pre = prenorm_states[:, logits_to_keep.to(device), :]

        if labels is not None and (
            selected_pre.shape[1] != prenorm_states.shape[1] or vocab_slice is not None
        ):
            raise ValueError(
                "Loss computation requires full vocabulary and logits_to_keep=0."
            )

        logits = self.forward_lm_head_fp32(selected_pre, vocab_slice=vocab_slice)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits, past_key_values) if use_cache else (logits,)
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        past_len = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        if cache_position is None:
            total = (
                inputs_embeds.shape[1]
                if inputs_embeds is not None and past_len == 0
                else input_ids.shape[1]
            )
            count = max(total - past_len, 1) if past_len else total
            cache_position = torch.arange(
                past_len, past_len + count, device=input_ids.device
            )
        count = cache_position.numel()
        use_embeds = inputs_embeds is not None and past_len == 0
        if use_embeds:
            current_ids, current_embeds = None, inputs_embeds[:, -count:]
        else:
            current_ids, current_embeds = input_ids[:, -count:].contiguous(), None

        if (
            position_ids is None
            and attention_mask is not None
            and attention_mask.ndim == 2
        ):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
        if position_ids is not None:
            position_ids = position_ids[:, -count:].contiguous()

        return {
            "input_ids": current_ids,
            "inputs_embeds": current_embeds,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "cache_position": cache_position,
            "use_cache": kwargs.get("use_cache", True),
            "logits_to_keep": kwargs.get("logits_to_keep", 1),
        }

    @torch.no_grad()
    def nar_velocity(
        self,
        tokens: torch.LongTensor,
        ar_mask: torch.BoolTensor,
        nar_mask: torch.BoolTensor,
        nar_content_mask: torch.BoolTensor,
        x_t: torch.Tensor,
        t_value: float,
        nar_cond_end: int = 0,
    ) -> torch.Tensor:
        device = tokens.device
        _, s = tokens.shape
        token_emb_f32 = self.model.embed_tokens(tokens).float()
        t_lat = x_t.shape[0]
        nar_indices = nar_mask[0].nonzero(as_tuple=True)[0]
        content_indices = nar_content_mask[0].nonzero(as_tuple=True)[0]
        n_nar = nar_indices.shape[0]

        x_nar_f32 = torch.zeros(
            (1, n_nar, x_t.shape[1]), device=device, dtype=torch.float32
        )
        x_nar_f32[0, 1 : 1 + t_lat] = _project_narrowing(
            x_t.to(device=device), torch.float32, phase_offset=57
        )

        pos_ids = torch.arange(n_nar, device=device).clamp(
            max=self.config.max_latent_frames - 1
        )
        pos_emb_f32 = self.latent_pos_embed.forward_fp32(pos_ids).unsqueeze(0)
        latent_hidden_nar_f32 = self.project_latent_to_trunk_fp32(
            x_nar_f32, t_value, pos_emb_f32
        )
        token_emb_f32[0, nar_indices] = latent_hidden_nar_f32[0]

        ar_q = ar_mask.unsqueeze(2)
        ar_k = ar_mask.unsqueeze(1)
        nar_q = nar_mask.unsqueeze(2)
        nar_k = nar_mask.unsqueeze(1)
        idx_s = torch.arange(s, device=device)
        causal_bool = idx_s[:, None] >= idx_s[None, :]

        if nar_cond_end > 0:
            cond_k = (idx_s < nar_cond_end)[None, None, :]
            mask_bool = (ar_q & ar_k & causal_bool) | (nar_q & cond_k) | (nar_q & nar_k)
        else:
            mask_bool = (ar_q & ar_k & causal_bool) | (nar_q & ar_k) | (nar_q & nar_k)

        act_dt = self.trunk_dtype
        zero_act = torch.zeros((), dtype=act_dt, device=device)
        neg_inf_act = torch.full((), float("-inf"), dtype=act_dt, device=device)
        attn_mask = torch.where(mask_bool.unsqueeze(1), zero_act, neg_inf_act)

        position_ids = idx_s.unsqueeze(0)
        prenorm_states, _ = self.model.forward_prenorm(
            inputs_embeds=token_emb_f32,
            position_ids=position_ids,
            use_cache=False,
            attention_mask=attn_mask,
            ar_mask=ar_mask,
            seq_len_hint=s,
        )
        nar_pred_f32 = self.project_trunk_to_velocity_fp32(
            prenorm_states[:, content_indices, :]
        )
        return nar_pred_f32[0]


YuE2Config.register_for_auto_class()
YuE2ForCausalLM.register_for_auto_class("AutoModelForCausalLM")

__all__ = [
    "Attention",
    "AudioPositionEmbedding",
    "Backbone",
    "DecoderLayer",
    "MLP",
    "QuantSafeLinear",
    "RMSNorm",
    "RotaryEmbedding",
    "StaticKVCache",
    "TimestepEmbedder",
    "YuE2Config",
    "YuE2ForCausalLM",
    "YuE2PreTrainedModel",
    "clear_yue2_workspaces",
    "sdpa",
    "synthesize_audio_pe_fp32",
]
