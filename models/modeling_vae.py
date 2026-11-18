from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel

HAS_NEW_FP32_PRECISION_API: bool = bool(
    hasattr(torch.backends, "fp32_precision")
    or (
        hasattr(torch.backends, "cuda")
        and hasattr(torch.backends.cuda, "matmul")
        and hasattr(torch.backends.cuda.matmul, "fp32_precision")
    )
)

_MAX_CONV_WORKSPACE_BYTES: int = 16 * 1024 * 1024
_MAX_GEMM_SPAN: int = 262144
_MATMUL_REDUCTION_LOCK_ATTRS: Tuple[str, ...] = (
    "allow_bf16_reduced_precision_reduction",
    "allow_fp16_reduced_precision_reduction",
    "allow_fp16_accumulation",
)


def _tensor_version(tensor: torch.Tensor) -> int:
    return -1 if tensor.is_inference() else tensor._version


def enforce_full_precision_reductions() -> None:
    cuda_mod = getattr(torch.backends, "cuda", None)
    mm_mod = getattr(cuda_mod, "matmul", None) if cuda_mod is not None else None
    if mm_mod is None:
        return
    for attr in _MATMUL_REDUCTION_LOCK_ATTRS:
        if hasattr(mm_mod, attr):
            try:
                if getattr(mm_mod, attr) is not False:
                    setattr(mm_mod, attr, False)
            except Exception:
                pass


enforce_full_precision_reductions()


def _discover_fp32_precision_targets() -> Tuple[object, ...]:
    if not HAS_NEW_FP32_PRECISION_API:
        return ()
    candidates: List[object] = [torch.backends]
    mkldnn_mod = getattr(torch.backends, "mkldnn", None)
    cudnn_mod = getattr(torch.backends, "cudnn", None)
    xpu_mod = getattr(torch.backends, "xpu", None)
    cuda_mod = getattr(torch.backends, "cuda", None)

    for backend_mod in (mkldnn_mod, cudnn_mod, xpu_mod):
        if backend_mod is not None:
            candidates.append(backend_mod)

    if mkldnn_mod is not None:
        for sub in ("matmul", "conv", "rnn"):
            sub_mod = getattr(mkldnn_mod, sub, None)
            if sub_mod is not None:
                candidates.append(sub_mod)
    if cuda_mod is not None:
        mm_mod = getattr(cuda_mod, "matmul", None)
        if mm_mod is not None:
            candidates.append(mm_mod)
    if cudnn_mod is not None:
        for sub in ("conv", "rnn"):
            sub_mod = getattr(cudnn_mod, sub, None)
            if sub_mod is not None:
                candidates.append(sub_mod)
    if xpu_mod is not None:
        for sub in ("matmul", "conv", "rnn"):
            sub_mod = getattr(xpu_mod, sub, None)
            if sub_mod is not None:
                candidates.append(sub_mod)

    valid: List[object] = []
    for mod in candidates:
        try:
            _ = mod.fp32_precision
            valid.append(mod)
        except Exception:
            continue
    return tuple(valid)


_FP32_PRECISION_TARGETS: Tuple[object, ...] = _discover_fp32_precision_targets()


class IEEE754FP32Guard:
    __slots__ = (
        "_saved_targets",
        "_saved_reduction_locks",
        "_prev_fp32_prec",
        "_prev_matmul_tf32",
        "_prev_cudnn_tf32",
    )

    def __init__(self) -> None:
        self._saved_targets: Optional[List[Tuple[object, str]]] = None
        self._saved_reduction_locks: Optional[List[Tuple[object, str, bool]]] = None
        self._prev_fp32_prec: Optional[str] = None
        self._prev_matmul_tf32: Optional[bool] = None
        self._prev_cudnn_tf32: Optional[bool] = None

    def __enter__(self) -> IEEE754FP32Guard:
        cuda_mod = getattr(torch.backends, "cuda", None)
        mm_mod = getattr(cuda_mod, "matmul", None) if cuda_mod is not None else None
        if mm_mod is not None:
            locks: List[Tuple[object, str, bool]] = []
            for attr in _MATMUL_REDUCTION_LOCK_ATTRS:
                if hasattr(mm_mod, attr):
                    try:
                        prev_val = bool(getattr(mm_mod, attr))
                        locks.append((mm_mod, attr, prev_val))
                        if prev_val:
                            setattr(mm_mod, attr, False)
                    except Exception:
                        pass
            self._saved_reduction_locks = locks

        if _FP32_PRECISION_TARGETS:
            saved: List[Tuple[object, str]] = [
                (mod, str(mod.fp32_precision)) for mod in _FP32_PRECISION_TARGETS
            ]
            self._saved_targets = saved
            for mod, _ in saved:
                if mod.fp32_precision != "ieee":
                    mod.fp32_precision = "ieee"
            return self

        if hasattr(torch, "get_float32_matmul_precision"):
            self._prev_fp32_prec = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("highest")
        if mm_mod is not None and hasattr(mm_mod, "allow_tf32"):
            self._prev_matmul_tf32 = bool(mm_mod.allow_tf32)
            mm_mod.allow_tf32 = False
        if hasattr(torch.backends, "cudnn") and hasattr(
            torch.backends.cudnn, "allow_tf32"
        ):
            cudnn_backend = torch.backends.cudnn
            self._prev_cudnn_tf32 = bool(cudnn_backend.allow_tf32)
            cudnn_backend.allow_tf32 = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._saved_targets is not None:
            for mod, prev_mode in self._saved_targets:
                if mod.fp32_precision != prev_mode:
                    mod.fp32_precision = prev_mode
            self._saved_targets = None
        else:
            if self._prev_fp32_prec is not None and hasattr(
                torch, "set_float32_matmul_precision"
            ):
                torch.set_float32_matmul_precision(self._prev_fp32_prec)
            if self._prev_cudnn_tf32 is not None and hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = self._prev_cudnn_tf32
            if (
                self._prev_matmul_tf32 is not None
                and hasattr(torch.backends, "cuda")
                and hasattr(torch.backends.cuda, "matmul")
            ):
                torch.backends.cuda.matmul.allow_tf32 = self._prev_matmul_tf32

        if self._saved_reduction_locks is not None:
            for mod, attr, prev_val in reversed(self._saved_reduction_locks):
                try:
                    if bool(getattr(mod, attr)) != prev_val:
                        setattr(mod, attr, prev_val)
                except Exception:
                    pass
            self._saved_reduction_locks = None


def checkpoint(function, *args, **kwargs):
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    kwargs.setdefault("use_reentrant", False)
    return torch_checkpoint(function, *args, **kwargs)


def _ensure_blas_2d(mat: torch.Tensor) -> torch.Tensor:
    r, c = mat.shape
    s0, s1 = mat.stride()
    if s1 == 1 and s0 >= max(1, c):
        return mat
    if s0 == 1 and s1 >= max(1, r):
        return mat
    return mat.contiguous()


def _autocast_active_for_device(dev_type: str) -> bool:
    if dev_type == "cuda":
        return bool(torch.is_autocast_enabled())
    if dev_type == "cpu":
        return bool(torch.is_autocast_cpu_enabled())
    if hasattr(torch, "is_autocast_enabled"):
        try:
            return bool(torch.is_autocast_enabled(dev_type))
        except Exception:
            return False
    return False


def _pack_conv_transpose1d_polyphase_weight(
    w_w: torch.Tensor,
    s: int,
    d: int,
    groups: int,
) -> Tuple[torch.Tensor, int]:
    c_in, c_out_g, k_size = w_w.shape
    c_in_g = c_in // groups
    max_kd = (k_size - 1) * d
    m_phases = (max_kd // s) + 1
    poly_span = m_phases * s

    w_4d = w_w.reshape(groups, c_in_g, c_out_g, k_size)
    if d == 1:
        w_span = (
            w_4d
            if poly_span == k_size
            else F.pad(w_4d, (0, poly_span - k_size))
        )
    else:
        kd_len = k_size * d
        w_dil = F.pad(w_4d.unsqueeze(-1), (0, d - 1)).reshape(
            groups, c_in_g, c_out_g, kd_len
        )
        if kd_len == poly_span:
            w_span = w_dil
        elif kd_len < poly_span:
            w_span = F.pad(w_dil, (0, poly_span - kd_len))
        else:
            w_span = w_dil[:, :, :, :poly_span]

    w_poly_g = (
        w_span.reshape(groups, c_in_g, c_out_g, m_phases, s)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(groups, c_in_g, m_phases * c_out_g * s)
    )
    return w_poly_g, m_phases


def _sovereign_conv1d_impl(
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    stride: Tuple[int, ...],
    padding: Union[str, Tuple[int, ...]],
    dilation: Tuple[int, ...],
    groups: int,
    padding_mode: str,
) -> torch.Tensor:
    if x.ndim == 2:
        unbatched = True
        x = x.unsqueeze(0)
    elif x.ndim == 3:
        unbatched = False
    else:
        raise ValueError(
            f"Expected 2D or 3D input to SovereignConv1d, got shape {tuple(x.shape)}"
        )

    batch, c_in, t_in = x.shape
    c_out, c_in_g, k_size = w.shape
    if groups < 1 or c_in != c_in_g * groups or c_out % groups != 0:
        raise RuntimeError(
            f"Channel/group mismatch in SovereignConv1d: x={tuple(x.shape)}, w={tuple(w.shape)}, groups={groups}"
        )

    s = int(stride[0])
    d = int(dilation[0])
    if isinstance(padding, str):
        if padding == "valid":
            pad_l = 0
            pad_r = 0
        elif padding == "same":
            if s != 1:
                raise RuntimeError(
                    "padding='same' is not supported for strided convolutions"
                )
            total_p = d * (k_size - 1)
            pad_l = total_p // 2
            pad_r = total_p - pad_l
        else:
            raise ValueError(f"Unsupported padding specification: {padding!r}")
    else:
        pad_l = int(padding[0])
        pad_r = int(padding[0])

    work_dtype = (
        torch.float64
        if (x.dtype == torch.float64 or w.dtype == torch.float64)
        else torch.float32
    )
    out_dtype = x.dtype if x.dtype == w.dtype else work_dtype

    x_w = x if x.dtype == work_dtype else x.to(dtype=work_dtype)
    w_w = w if w.dtype == work_dtype else w.to(dtype=work_dtype)
    b_w = (
        (b if b.dtype == work_dtype else b.to(dtype=work_dtype))
        if b is not None
        else None
    )

    if padding_mode != "zeros" and (pad_l > 0 or pad_r > 0):
        x_w = F.pad(x_w, (pad_l, pad_r), mode=padding_mode)
        t_in = x_w.shape[2]
        pad_l = 0
        pad_r = 0

    k_eff = d * (k_size - 1) + 1
    t_pad = t_in + pad_l + pad_r
    if t_pad < k_eff:
        raise RuntimeError(
            f"Calculated padded input size per channel: ({t_pad}). Kernel size: ({k_eff})."
        )

    t_out = (t_pad - k_eff) // s + 1
    if batch == 0 or t_out == 0:
        empty_out = x_w.new_empty((batch, c_out, t_out)).to(dtype=out_dtype)
        return empty_out.squeeze(0) if unbatched else empty_out

    grad_active = (
        torch.is_grad_enabled()
        and not torch.is_inference_mode_enabled()
        and (
            x_w.requires_grad
            or w_w.requires_grad
            or (b_w is not None and b_w.requires_grad)
        )
    )

    if k_size == 1 and s == 1 and pad_l == 0 and pad_r == 0 and groups == 1:
        w_2d = (
            w_w.view(c_out, c_in)
            if w_w.is_contiguous()
            else _ensure_blas_2d(w_w.reshape(c_out, c_in))
        )
        if batch == 1 and t_out <= _MAX_GEMM_SPAN:
            out = torch.mm(w_2d, _ensure_blas_2d(x_w[0])).unsqueeze(0)
            if b_w is not None:
                b_view = b_w.view(1, c_out, 1)
                out = (out + b_view) if grad_active else out.add_(b_view)
        elif not grad_active:
            out = torch.empty(
                (batch, c_out, t_out), dtype=work_dtype, device=x_w.device
            )
            for b_idx in range(batch):
                xb = x_w[b_idx]
                for t0 in range(0, t_out, _MAX_GEMM_SPAN):
                    t1 = min(t0 + _MAX_GEMM_SPAN, t_out)
                    torch.mm(
                        w_2d,
                        _ensure_blas_2d(xb[:, t0:t1]),
                        out=out[b_idx, :, t0:t1],
                    )
            if b_w is not None:
                out.add_(b_w.view(1, c_out, 1))
        else:
            b_outs: List[torch.Tensor] = []
            for b_idx in range(batch):
                xb = x_w[b_idx]
                if t_out <= _MAX_GEMM_SPAN:
                    b_outs.append(torch.mm(w_2d, _ensure_blas_2d(xb)))
                else:
                    t_chunks = [
                        torch.mm(
                            w_2d,
                            _ensure_blas_2d(
                                xb[:, t0 : min(t0 + _MAX_GEMM_SPAN, t_out)]
                            ),
                        )
                        for t0 in range(0, t_out, _MAX_GEMM_SPAN)
                    ]
                    b_outs.append(torch.cat(t_chunks, dim=1))
            out = torch.stack(b_outs, dim=0)
            if b_w is not None:
                out = out + b_w.view(1, c_out, 1)

        if out.dtype != out_dtype:
            out = out.to(dtype=out_dtype)
        return out.squeeze(0) if unbatched else out

    if min(x_w.stride()) < 0:
        x_w = x_w.contiguous()

    c_out_g = c_out // groups
    w_contig = w_w if w_w.is_contiguous() else w_w.contiguous()
    w_grouped = w_contig.view(groups, c_out_g, c_in_g * k_size)

    elem_bytes = x_w.element_size()
    eff_b = 1 if (not grad_active and groups == 1) else batch
    bytes_per_frame = max(1, eff_b * c_in * k_size * elem_bytes)
    max_tile_frames = max(1, _MAX_GEMM_SPAN // max(1, eff_b))
    raw_tile = min(
        max_tile_frames, max(1, _MAX_CONV_WORKSPACE_BYTES // bytes_per_frame)
    )
    l_tile = (raw_tile // 64) * 64 if raw_tile >= 64 else raw_tile

    t_int_start = min(t_out, (pad_l + s - 1) // s) if pad_l > 0 else 0
    r_num = t_in - 1 + pad_l - d * (k_size - 1)
    t_int_end = (
        max(t_int_start, min(t_out, (r_num // s) + 1)) if r_num >= 0 else t_int_start
    )

    def _make_boundary_buffer(a: int, b_end: int) -> torch.Tensor:
        u_min = a * s - pad_l
        u_max = (b_end - 1) * s - pad_l + d * (k_size - 1)
        src_0 = max(0, u_min)
        src_1 = min(t_in, u_max + 1)
        if src_0 < src_1:
            return F.pad(
                x_w[:, :, src_0:src_1],
                (src_0 - u_min, (u_max + 1) - src_1),
            )
        return x_w.new_zeros((batch, c_in, u_max - u_min + 1))

    if not grad_active:
        out = torch.empty((batch, c_out, t_out), dtype=work_dtype, device=x_w.device)

        def _contract_into(src_x: torch.Tensor, t_base: int, pos: int, cnt: int) -> None:
            sb, sc, st = src_x.stride()
            base_off = src_x.storage_offset() + t_base * st
            if groups == 1:
                w_2d = w_grouped[0]
                for b_idx in range(batch):
                    patches = torch.as_strided(
                        src_x,
                        size=(c_in, k_size, cnt),
                        stride=(sc, d * st, s * st),
                        storage_offset=base_off + b_idx * sb,
                    )
                    col_2d = patches.contiguous().view(c_in * k_size, cnt)
                    torch.mm(w_2d, col_2d, out=out[b_idx, :, pos : pos + cnt])
            else:
                patches = torch.as_strided(
                    src_x,
                    size=(groups, c_in_g, k_size, batch, cnt),
                    stride=(c_in_g * sc, sc, d * st, sb, s * st),
                    storage_offset=base_off,
                )
                col_g = patches.contiguous().view(
                    groups, c_in_g * k_size, batch * cnt
                )
                prod = (
                    torch.bmm(w_grouped, col_g)
                    .view(groups, c_out_g, batch, cnt)
                    .permute(2, 0, 1, 3)
                    .contiguous()
                    .view(batch, c_out, cnt)
                )
                out[:, :, pos : pos + cnt].copy_(prod)

        for pos in range(0, t_int_start, l_tile):
            end_pos = min(pos + l_tile, t_int_start)
            x_sub = _make_boundary_buffer(pos, end_pos)
            _contract_into(x_sub, 0, pos, end_pos - pos)

        for pos in range(t_int_start, t_int_end, l_tile):
            end_pos = min(pos + l_tile, t_int_end)
            _contract_into(x_w, pos * s - pad_l, pos, end_pos - pos)

        for pos in range(t_int_end, t_out, l_tile):
            end_pos = min(pos + l_tile, t_out)
            x_sub = _make_boundary_buffer(pos, end_pos)
            _contract_into(x_sub, 0, pos, end_pos - pos)

        if b_w is not None:
            out.add_(b_w.view(1, c_out, 1))
    else:
        chunks: List[torch.Tensor] = []

        def _contract_grad(src_x: torch.Tensor, t_base: int, cnt: int) -> torch.Tensor:
            span = (cnt - 1) * s + 1
            patches = torch.stack(
                [
                    src_x[:, :, t_base + k * d : t_base + k * d + span : s]
                    for k in range(k_size)
                ],
                dim=2,
            )
            if groups == 1 and batch == 1:
                col_2d = patches[0].contiguous().view(c_in * k_size, cnt)
                return torch.mm(w_grouped[0], col_2d).unsqueeze(0)
            col_g = (
                patches.view(batch, groups, c_in_g, k_size, cnt)
                .permute(1, 2, 3, 0, 4)
                .contiguous()
                .view(groups, c_in_g * k_size, batch * cnt)
            )
            if groups == 1:
                return (
                    torch.mm(w_grouped[0], col_g[0])
                    .view(c_out, batch, cnt)
                    .permute(1, 0, 2)
                    .contiguous()
                )
            return (
                torch.bmm(w_grouped, col_g)
                .view(groups, c_out_g, batch, cnt)
                .permute(2, 0, 1, 3)
                .contiguous()
                .view(batch, c_out, cnt)
            )

        for pos in range(0, t_int_start, l_tile):
            end_pos = min(pos + l_tile, t_int_start)
            x_sub = _make_boundary_buffer(pos, end_pos)
            chunks.append(_contract_grad(x_sub, 0, end_pos - pos))

        for pos in range(t_int_start, t_int_end, l_tile):
            end_pos = min(pos + l_tile, t_int_end)
            chunks.append(_contract_grad(x_w, pos * s - pad_l, end_pos - pos))

        for pos in range(t_int_end, t_out, l_tile):
            end_pos = min(pos + l_tile, t_out)
            x_sub = _make_boundary_buffer(pos, end_pos)
            chunks.append(_contract_grad(x_sub, 0, end_pos - pos))

        out = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=2)
        if b_w is not None:
            out = out + b_w.view(1, c_out, 1)

    if out.dtype != out_dtype:
        out = out.to(dtype=out_dtype)
    return out.squeeze(0) if unbatched else out


def _sovereign_conv_transpose1d_impl(
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    output_padding: Tuple[int, ...],
    groups: int,
    dilation: Tuple[int, ...],
) -> torch.Tensor:
    if x.ndim == 2:
        unbatched = True
        x = x.unsqueeze(0)
    elif x.ndim == 3:
        unbatched = False
    else:
        raise ValueError(
            f"Expected 2D or 3D input to SovereignConvTranspose1d, got shape {tuple(x.shape)}"
        )

    batch, c_in, t_in = x.shape
    w_in, c_out_g, k_size = w.shape
    if groups < 1 or w_in != c_in or c_in % groups != 0 or t_in < 1:
        raise RuntimeError(
            f"Invalid geometry in SovereignConvTranspose1d: x={tuple(x.shape)}, w={tuple(w.shape)}, groups={groups}"
        )

    c_in_g = c_in // groups
    c_out = c_out_g * groups
    s = int(stride[0])
    p = int(padding[0])
    d = int(dilation[0])
    op = int(output_padding[0])

    t_out = (t_in - 1) * s - 2 * p + d * (k_size - 1) + op + 1
    if t_out < 1:
        raise RuntimeError(
            f"Calculated output size per channel ({t_out}) is too small in SovereignConvTranspose1d."
        )

    work_dtype = (
        torch.float64
        if (x.dtype == torch.float64 or w.dtype == torch.float64)
        else torch.float32
    )
    out_dtype = x.dtype if x.dtype == w.dtype else work_dtype

    if batch == 0:
        empty_out = x.new_empty((0, c_out, t_out)).to(dtype=out_dtype)
        return empty_out.squeeze(0) if unbatched else empty_out

    x_w = x if x.dtype == work_dtype else x.to(dtype=work_dtype)
    w_w = w if w.dtype == work_dtype else w.to(dtype=work_dtype)
    b_w = (
        (b if b.dtype == work_dtype else b.to(dtype=work_dtype))
        if b is not None
        else None
    )

    if min(x_w.stride()) < 0:
        x_w = x_w.contiguous()

    grad_active = (
        torch.is_grad_enabled()
        and not torch.is_inference_mode_enabled()
        and (
            x_w.requires_grad
            or w_w.requires_grad
            or (b_w is not None and b_w.requires_grad)
        )
    )

    if k_size == 2 * s and d == 1 and groups == 1:
        w_contig = w_w if w_w.is_contiguous() else w_w.contiguous()
        w_view = w_contig.view(c_in, c_out * 2 * s)

        elem_bytes = x_w.element_size()
        eff_b = 1 if not grad_active else batch
        bytes_per_frame = max(1, eff_b * c_out * 2 * s * elem_bytes)
        max_tile_frames = max(1, _MAX_GEMM_SPAN // max(1, eff_b))
        raw_tile = min(
            max_tile_frames, max(1, _MAX_CONV_WORKSPACE_BYTES // bytes_per_frame)
        )
        l_tile = (raw_tile // 64) * 64 if raw_tile >= 64 else raw_tile

        if not grad_active:
            out = torch.empty(
                (batch, c_out, t_out), dtype=work_dtype, device=x_w.device
            )
            tail_u0 = max(p, (t_in + 1) * s)
            if p + t_out > tail_u0:
                out[:, :, (tail_u0 - p) : t_out].zero_()

            for b_idx in range(batch):
                carry: Optional[torch.Tensor] = None
                x_b = x_w[b_idx]
                for t0 in range(0, t_in, l_tile):
                    cnt = min(l_tile, t_in - t0)
                    x_tile_t = _ensure_blas_2d(x_b[:, t0 : t0 + cnt].t())
                    p_mat = torch.mm(x_tile_t, w_view).view(cnt, c_out, 2, s)
                    acc = p_mat[:, :, 0, :].contiguous()
                    if carry is not None:
                        acc[0].add_(carry)
                    if cnt > 1:
                        acc[1:].add_(p_mat[:-1, :, 1, :])
                    carry = p_mat[-1, :, 1, :].clone()

                    seg_u0 = t0 * s
                    seg_u1 = (t0 + cnt) * s
                    u_lo = max(p, seg_u0)
                    u_hi = min(p + t_out, seg_u1)
                    if u_hi > u_lo:
                        seg_2d = (
                            acc.permute(1, 0, 2)
                            .contiguous()
                            .view(c_out, cnt * s)
                        )
                        out[b_idx, :, (u_lo - p) : (u_hi - p)].copy_(
                            seg_2d[:, (u_lo - seg_u0) : (u_hi - seg_u0)]
                        )

                seg_u0 = t_in * s
                seg_u1 = (t_in + 1) * s
                u_lo = max(p, seg_u0)
                u_hi = min(p + t_out, seg_u1)
                if carry is not None and u_hi > u_lo:
                    out[b_idx, :, (u_lo - p) : (u_hi - p)].copy_(
                        carry[:, (u_lo - seg_u0) : (u_hi - seg_u0)]
                    )

            if b_w is not None:
                out.add_(b_w.view(1, c_out, 1))
        else:
            out_chunks: List[torch.Tensor] = []
            carry_b: Optional[torch.Tensor] = None
            for t0 in range(0, t_in, l_tile):
                cnt = min(l_tile, t_in - t0)
                if batch == 1:
                    x_2d = _ensure_blas_2d(x_w[0, :, t0 : t0 + cnt].t())
                    p_5d = torch.mm(x_2d, w_view).view(1, cnt, c_out, 2, s)
                else:
                    x_2d = (
                        x_w[:, :, t0 : t0 + cnt]
                        .transpose(1, 2)
                        .contiguous()
                        .view(batch * cnt, c_in)
                    )
                    p_5d = torch.mm(x_2d, w_view).view(batch, cnt, c_out, 2, s)

                p0 = p_5d[:, :, :, 0, :]
                p1 = p_5d[:, :, :, 1, :]
                head = (p0[:, 0] + carry_b) if carry_b is not None else p0[:, 0]
                if cnt > 1:
                    acc_b = torch.cat(
                        [head.unsqueeze(1), p0[:, 1:] + p1[:, :-1]], dim=1
                    )
                else:
                    acc_b = head.unsqueeze(1)
                carry_b = p1[:, -1]

                seg_u0 = t0 * s
                seg_u1 = (t0 + cnt) * s
                u_lo = max(p, seg_u0)
                u_hi = min(p + t_out, seg_u1)
                if u_hi > u_lo:
                    seg_3d = (
                        acc_b.permute(0, 2, 1, 3)
                        .contiguous()
                        .view(batch, c_out, cnt * s)
                    )
                    out_chunks.append(
                        seg_3d[:, :, (u_lo - seg_u0) : (u_hi - seg_u0)]
                    )

            seg_u0 = t_in * s
            seg_u1 = (t_in + 1) * s
            u_lo = max(p, seg_u0)
            u_hi = min(p + t_out, seg_u1)
            if carry_b is not None and u_hi > u_lo:
                out_chunks.append(
                    carry_b[:, :, (u_lo - seg_u0) : (u_hi - seg_u0)]
                )

            tail_u0 = max(p, (t_in + 1) * s)
            tail_u1 = p + t_out
            if tail_u1 > tail_u0:
                out_chunks.append(x_w.new_zeros((batch, c_out, tail_u1 - tail_u0)))

            out = (
                out_chunks[0] if len(out_chunks) == 1 else torch.cat(out_chunks, dim=2)
            )
            if b_w is not None:
                out = out + b_w.view(1, c_out, 1)

        if out.dtype != out_dtype:
            out = out.to(dtype=out_dtype)
        return out.squeeze(0) if unbatched else out

    w_poly_g, m_phases = _pack_conv_transpose1d_polyphase_weight(
        w_w, s=s, d=d, groups=groups
    )

    elem_bytes = x_w.element_size()
    bytes_per_frame = max(1, batch * m_phases * c_out * s * elem_bytes)
    max_tile_frames = max(1, _MAX_GEMM_SPAN // max(1, batch))
    raw_tile = min(
        max_tile_frames, max(1, _MAX_CONV_WORKSPACE_BYTES // bytes_per_frame)
    )
    l_tile = (raw_tile // 64) * 64 if raw_tile >= 64 else raw_tile

    def _contract_polyphase_tile(t0: int, cnt: int) -> torch.Tensor:
        x_slice = x_w[:, :, t0 : t0 + cnt]
        if groups == 1:
            if batch == 1:
                x_2d = _ensure_blas_2d(x_slice[0].t())
            else:
                x_2d = (
                    x_slice.transpose(1, 2)
                    .contiguous()
                    .view(batch * cnt, c_in)
                )
            p_5d = torch.mm(x_2d, w_poly_g[0]).view(
                batch, cnt, m_phases, c_out, s
            )
        else:
            x_g = (
                x_slice.reshape(batch, groups, c_in_g, cnt)
                .permute(1, 0, 3, 2)
                .contiguous()
                .view(groups, batch * cnt, c_in_g)
            )
            p_5d = (
                torch.bmm(x_g, w_poly_g)
                .view(groups, batch, cnt, m_phases, c_out_g, s)
                .permute(1, 2, 3, 0, 4, 5)
                .contiguous()
                .view(batch, cnt, m_phases, c_out, s)
            )
        return p_5d.permute(2, 1, 0, 3, 4).contiguous()

    total_poly_u = (t_in + m_phases - 1) * s

    if not grad_active:
        out = torch.empty((batch, c_out, t_out), dtype=work_dtype, device=x_w.device)
        tail_u0 = max(p, total_poly_u)
        if p + t_out > tail_u0:
            out[:, :, (tail_u0 - p) : t_out].zero_()

        carry_m: Optional[torch.Tensor] = None
        for t0 in range(0, t_in, l_tile):
            cnt = min(l_tile, t_in - t0)
            p_by_m = _contract_polyphase_tile(t0, cnt)

            if m_phases == 1:
                ready = p_by_m[0]
            else:
                work = x_w.new_zeros((cnt + m_phases - 1, batch, c_out, s))
                if carry_m is not None:
                    work[: m_phases - 1].copy_(carry_m)
                for m in range(m_phases):
                    work[m : m + cnt].add_(p_by_m[m])
                ready = work[:cnt]
                carry_m = work[cnt:].clone()

            seg_u0 = t0 * s
            seg_u1 = (t0 + cnt) * s
            u_lo = max(p, seg_u0)
            u_hi = min(p + t_out, seg_u1)
            if u_hi > u_lo:
                seg_3d = (
                    ready.permute(1, 2, 0, 3)
                    .contiguous()
                    .view(batch, c_out, cnt * s)
                )
                out[:, :, (u_lo - p) : (u_hi - p)].copy_(
                    seg_3d[:, :, (u_lo - seg_u0) : (u_hi - seg_u0)]
                )

        if m_phases > 1 and carry_m is not None:
            seg_u0 = t_in * s
            seg_u1 = total_poly_u
            u_lo = max(p, seg_u0)
            u_hi = min(p + t_out, seg_u1)
            if u_hi > u_lo:
                carry_3d = (
                    carry_m.permute(1, 2, 0, 3)
                    .contiguous()
                    .view(batch, c_out, (m_phases - 1) * s)
                )
                out[:, :, (u_lo - p) : (u_hi - p)].copy_(
                    carry_3d[:, :, (u_lo - seg_u0) : (u_hi - seg_u0)]
                )

        if b_w is not None:
            out.add_(b_w.view(1, c_out, 1))
    else:
        poly_chunks: List[torch.Tensor] = []
        carry_g: Optional[torch.Tensor] = None

        for t0 in range(0, t_in, l_tile):
            cnt = min(l_tile, t_in - t0)
            p_by_m = _contract_polyphase_tile(t0, cnt)

            if m_phases == 1:
                ready = p_by_m[0]
            else:
                if carry_g is not None:
                    work = F.pad(carry_g, (0, 0, 0, 0, 0, 0, 0, cnt))
                else:
                    work = F.pad(p_by_m[0], (0, 0, 0, 0, 0, 0, 0, m_phases - 1))
                start_m = 0 if carry_g is not None else 1
                for m in range(start_m, m_phases):
                    work = work + F.pad(
                        p_by_m[m], (0, 0, 0, 0, 0, 0, m, (m_phases - 1) - m)
                    )
                ready = work[:cnt]
                carry_g = work[cnt:]

            seg_u0 = t0 * s
            seg_u1 = (t0 + cnt) * s
            u_lo = max(p, seg_u0)
            u_hi = min(p + t_out, seg_u1)
            if u_hi > u_lo:
                seg_3d = (
                    ready.permute(1, 2, 0, 3)
                    .contiguous()
                    .view(batch, c_out, cnt * s)
                )
                poly_chunks.append(
                    seg_3d[:, :, (u_lo - seg_u0) : (u_hi - seg_u0)]
                )

        if m_phases > 1 and carry_g is not None:
            seg_u0 = t_in * s
            seg_u1 = total_poly_u
            u_lo = max(p, seg_u0)
            u_hi = min(p + t_out, seg_u1)
            if u_hi > u_lo:
                carry_3d = (
                    carry_g.permute(1, 2, 0, 3)
                    .contiguous()
                    .view(batch, c_out, (m_phases - 1) * s)
                )
                poly_chunks.append(
                    carry_3d[:, :, (u_lo - seg_u0) : (u_hi - seg_u0)]
                )

        tail_u0 = max(p, total_poly_u)
        tail_u1 = p + t_out
        if tail_u1 > tail_u0:
            poly_chunks.append(x_w.new_zeros((batch, c_out, tail_u1 - tail_u0)))

        out = (
            poly_chunks[0]
            if len(poly_chunks) == 1
            else torch.cat(poly_chunks, dim=2)
        )
        if b_w is not None:
            out = out + b_w.view(1, c_out, 1)

    if out.dtype != out_dtype:
        out = out.to(dtype=out_dtype)
    return out.squeeze(0) if unbatched else out


class SovereignConv1d(nn.Conv1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        dev_type = input.device.type
        with IEEE754FP32Guard():
            if _autocast_active_for_device(dev_type):
                with torch.autocast(device_type=dev_type, enabled=False):
                    return _sovereign_conv1d_impl(
                        input,
                        self.weight,
                        self.bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                        self.padding_mode,
                    )
            return _sovereign_conv1d_impl(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                self.padding_mode,
            )


class SovereignConvTranspose1d(nn.ConvTranspose1d):
    def forward(
        self, input: torch.Tensor, output_size: Optional[List[int]] = None
    ) -> torch.Tensor:
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only 'zeros' padding mode is supported for SovereignConvTranspose1d"
            )
        out_pad = self._output_padding(
            input,
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            1,
            self.dilation,  # type: ignore[arg-type]
        )
        dev_type = input.device.type
        with IEEE754FP32Guard():
            if _autocast_active_for_device(dev_type):
                with torch.autocast(device_type=dev_type, enabled=False):
                    return _sovereign_conv_transpose1d_impl(
                        input,
                        self.weight,
                        self.bias,
                        self.stride,
                        self.padding,
                        out_pad,
                        self.groups,
                        self.dilation,
                    )
            return _sovereign_conv_transpose1d_impl(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                out_pad,
                self.groups,
                self.dilation,
            )


def WNConv1d(*args, **kwargs) -> SovereignConv1d:
    with torch.inference_mode(False):
        return SovereignConv1d(*args, **kwargs)


def WNConvTranspose1d(*args, **kwargs) -> SovereignConvTranspose1d:
    with torch.inference_mode(False):
        return SovereignConvTranspose1d(*args, **kwargs)


def fold_weight_norm_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    with torch.inference_mode(False), torch.no_grad():
        fused_dict: Dict[str, torch.Tensor] = {}
        consumed_keys = set()

        for key in list(state_dict.keys()):
            if key in consumed_keys:
                continue
            if key.endswith(".weight_g"):
                base_key = key[:-9]
                v_key = f"{base_key}.weight_v"
                if v_key not in state_dict:
                    raise KeyError(
                        f"Corrupt state dict: located {key} but missing counterpart {v_key}"
                    )
                g = state_dict[key].detach().to(dtype=torch.float64)
                v = state_dict[v_key].detach().to(dtype=torch.float64)
                reduce_dims = tuple(range(1, v.ndim))
                norm = torch.linalg.vector_norm(v, ord=2, dim=reduce_dims, keepdim=True)
                w_fused = (
                    (v * (g / torch.clamp_min(norm, 1e-15)))
                    .to(dtype=torch.float32)
                    .contiguous()
                )
                if w_fused.is_inference():
                    w_fused = w_fused.clone()
                fused_dict[f"{base_key}.weight"] = w_fused
                consumed_keys.update([key, v_key])
            elif key.endswith(".weight_v"):
                base_key = key[:-9]
                g_key = f"{base_key}.weight_g"
                if g_key not in state_dict:
                    raise KeyError(
                        f"Corrupt state dict: located {key} but missing counterpart {g_key}"
                    )
            else:
                val = state_dict[key].detach()
                if val.is_floating_point():
                    out_val = val.to(dtype=torch.float32).contiguous()
                else:
                    out_val = val
                if out_val.is_inference():
                    out_val = out_val.clone()
                fused_dict[key] = out_val
                consumed_keys.add(key)

        return fused_dict


def snake_beta_fused(
    x: torch.Tensor, alpha_val: torch.Tensor, inv_beta_val: torch.Tensor
) -> torch.Tensor:
    grad_active = torch.is_grad_enabled() and (
        x.requires_grad or alpha_val.requires_grad or inv_beta_val.requires_grad
    )
    if x.dtype in (torch.float32, torch.float64):
        a_c = alpha_val if alpha_val.dtype == x.dtype else alpha_val.to(dtype=x.dtype)
        ib_c = (
            inv_beta_val
            if inv_beta_val.dtype == x.dtype
            else inv_beta_val.to(dtype=x.dtype)
        )
        if grad_active:
            s = torch.sin(x * a_c)
            return torch.addcmul(x, s * s, ib_c)
        buf = x * a_c
        buf.sin_()
        buf.mul_(buf)
        return torch.addcmul(x, buf, ib_c, out=buf)

    x_f32 = x.float()
    a_f32 = alpha_val if alpha_val.dtype == torch.float32 else alpha_val.float()
    ib_f32 = (
        inv_beta_val if inv_beta_val.dtype == torch.float32 else inv_beta_val.float()
    )
    if grad_active:
        s = torch.sin(x_f32 * a_f32)
        return torch.addcmul(x_f32, s * s, ib_f32).to(dtype=x.dtype)
    buf = x_f32 * a_f32
    buf.sin_()
    buf.mul_(buf)
    return torch.addcmul(x_f32, buf, ib_f32, out=buf).to(dtype=x.dtype)


class SnakeBeta(nn.Module):
    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale

        with torch.inference_mode(False):
            if self.alpha_logscale:
                a_init = torch.zeros(in_features, dtype=torch.float32) * alpha
                b_init = torch.zeros(in_features, dtype=torch.float32) * alpha
            else:
                a_init = torch.ones(in_features, dtype=torch.float32) * alpha
                b_init = torch.ones(in_features, dtype=torch.float32) * alpha
            self.alpha = nn.Parameter(a_init, requires_grad=alpha_trainable)
            self.beta = nn.Parameter(b_init, requires_grad=alpha_trainable)

        self._cached_alpha: Optional[torch.Tensor] = None
        self._cached_inv_beta: Optional[torch.Tensor] = None
        self._cache_sig: Optional[
            Tuple[int, int, int, int, torch.device, torch.dtype]
        ] = None

    def invalidate_cache(self) -> None:
        self._cached_alpha = None
        self._cached_inv_beta = None
        self._cache_sig = None

    def _compute_coeffs(
        self, target_device: torch.device, calc_dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        comp_dev = (
            torch.device("cpu")
            if target_device.type == "mps"
            else self.alpha.device
        )
        alpha_64 = (
            self.alpha.to(device=comp_dev, dtype=torch.float64)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        beta_64 = (
            self.beta.to(device=comp_dev, dtype=torch.float64)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        if self.alpha_logscale:
            alpha_64 = torch.exp(alpha_64)
            beta_64 = torch.exp(beta_64)
        inv_beta_64 = 1.0 / (beta_64 + 1e-9)
        alpha_c = (
            alpha_64.to(dtype=calc_dtype).to(device=target_device).contiguous()
        )
        inv_beta_c = (
            inv_beta_64.to(dtype=calc_dtype).to(device=target_device).contiguous()
        )
        return alpha_c, inv_beta_c

    def materialize_eval_cache(self) -> None:
        if self.alpha.is_meta or self.beta.is_meta:
            self.invalidate_cache()
            return
        with torch.inference_mode(False), torch.no_grad():
            target_dtype = (
                self.alpha.dtype
                if self.alpha.dtype in (torch.float32, torch.float64)
                else torch.float32
            )
            comp_dev = (
                torch.device("cpu")
                if self.alpha.device.type == "mps"
                else self.alpha.device
            )
            a = (
                self.alpha.detach()
                .to(device=comp_dev, dtype=torch.float64)
                .unsqueeze(0)
                .unsqueeze(-1)
            )
            b = (
                self.beta.detach()
                .to(device=comp_dev, dtype=torch.float64)
                .unsqueeze(0)
                .unsqueeze(-1)
            )
            if self.alpha_logscale:
                a = torch.exp(a)
                b = torch.exp(b)
            inv_b = 1.0 / (b + 1e-9)
            cached_a = (
                a.to(dtype=target_dtype).to(device=self.alpha.device).contiguous()
            )
            cached_ib = (
                inv_b.to(dtype=target_dtype)
                .to(device=self.beta.device)
                .contiguous()
            )
            if cached_a.is_inference():
                cached_a = cached_a.clone()
            if cached_ib.is_inference():
                cached_ib = cached_ib.clone()
            self._cached_alpha = cached_a
            self._cached_inv_beta = cached_ib
            self._cache_sig = (
                self.alpha.data_ptr(),
                _tensor_version(self.alpha),
                self.beta.data_ptr(),
                _tensor_version(self.beta),
                self.alpha.device,
                target_dtype,
            )

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.invalidate_cache()
        else:
            self.materialize_eval_cache()
        return self

    def _apply(self, fn, recurse: bool = True):
        with torch.inference_mode(False):
            super()._apply(fn, recurse=recurse)
        if not self.training:
            self.materialize_eval_cache()
        else:
            self.invalidate_cache()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        calc_dtype = (
            x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
        )
        param_grad_needed = torch.is_grad_enabled() and (
            self.alpha.requires_grad or self.beta.requires_grad
        )
        if param_grad_needed:
            alpha_c, inv_beta_c = self._compute_coeffs(x.device, calc_dtype)
            return snake_beta_fused(x, alpha_c, inv_beta_c)

        sig = (
            self.alpha.data_ptr(),
            _tensor_version(self.alpha),
            self.beta.data_ptr(),
            _tensor_version(self.beta),
            x.device,
            calc_dtype,
        )
        if (
            not self.training
            and self._cached_alpha is not None
            and self._cached_inv_beta is not None
            and self._cache_sig == sig
        ):
            return snake_beta_fused(x, self._cached_alpha, self._cached_inv_beta)

        if not x.is_meta:
            with torch.inference_mode(False), torch.no_grad():
                comp_dev = (
                    torch.device("cpu")
                    if self.alpha.device.type == "mps"
                    else self.alpha.device
                )
                a64 = (
                    self.alpha.detach()
                    .to(device=comp_dev, dtype=torch.float64)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                )
                b64 = (
                    self.beta.detach()
                    .to(device=comp_dev, dtype=torch.float64)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                )
                if self.alpha_logscale:
                    a64 = torch.exp(a64)
                    b64 = torch.exp(b64)
                ib64 = 1.0 / (b64 + 1e-9)
                alpha_c = (
                    a64.to(dtype=calc_dtype).to(device=x.device).contiguous()
                )
                inv_beta_c = (
                    ib64.to(dtype=calc_dtype).to(device=x.device).contiguous()
                )
                if alpha_c.is_inference():
                    alpha_c = alpha_c.clone()
                if inv_beta_c.is_inference():
                    inv_beta_c = inv_beta_c.clone()
                if not self.training:
                    self._cached_alpha = alpha_c
                    self._cached_inv_beta = inv_beta_c
                    self._cache_sig = sig
        else:
            alpha_c, inv_beta_c = self._compute_coeffs(x.device, calc_dtype)
        return snake_beta_fused(x, alpha_c, inv_beta_c)


def get_activation(
    activation: Literal["elu", "snake", "none"], channels: Optional[int] = None
) -> nn.Module:
    if activation == "elu":
        return nn.ELU()
    if activation == "snake":
        if channels is None:
            raise ValueError("SnakeBeta activation requires explicit channel count.")
        return SnakeBeta(channels)
    if activation == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation specifier: {activation}")


class ResidualUnit(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, dilation: int, act_type: str
    ):
        super().__init__()
        self.dilation = dilation
        padding = (dilation * (7 - 1)) // 2
        self.layers = nn.Sequential(
            get_activation(act_type, channels=out_channels),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                dilation=dilation,
                padding=padding,
            ),
            get_activation(act_type, channels=out_channels),
            WNConv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.training:
            out = checkpoint(self.layers, x)
            return out + residual
        out = self.layers(x)
        if not torch.is_grad_enabled():
            return out.add_(residual)
        return out + residual


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, act_type: str):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualUnit(in_channels, in_channels, 1, act_type),
            ResidualUnit(in_channels, in_channels, 3, act_type),
            ResidualUnit(in_channels, in_channels, 9, act_type),
            get_activation(act_type, channels=in_channels),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, act_type: str):
        super().__init__()
        upsample_layer = WNConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
        )
        self.layers = nn.Sequential(
            get_activation(act_type, channels=in_channels),
            upsample_layer,
            ResidualUnit(out_channels, out_channels, 1, act_type),
            ResidualUnit(out_channels, out_channels, 3, act_type),
            ResidualUnit(out_channels, out_channels, 9, act_type),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class OobleckEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        channels: int = 64,
        latent_dim: int = 128,
        c_mults: Tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        strides: Tuple[int, ...] = (2, 2, 4, 4, 5, 6),
        use_snake: bool = True,
        antialias_activation: bool = False,
    ):
        super().__init__()
        if antialias_activation:
            raise ValueError(
                "Upstream Oobleck encoder specification rejects antialias_activation."
            )
        self.in_channels = in_channels
        c_mults_list = [1] + list(c_mults)
        self.depth = len(c_mults_list)

        layers = [
            WNConv1d(
                in_channels=in_channels,
                out_channels=c_mults_list[0] * channels,
                kernel_size=7,
                padding=3,
            )
        ]
        act_type = "snake" if use_snake else "elu"
        for i in range(self.depth - 1):
            layers.append(
                EncoderBlock(
                    in_channels=c_mults_list[i] * channels,
                    out_channels=c_mults_list[i + 1] * channels,
                    stride=strides[i],
                    act_type=act_type,
                )
            )
        layers.extend(
            [
                get_activation(act_type, channels=c_mults_list[-1] * channels),
                WNConv1d(
                    in_channels=c_mults_list[-1] * channels,
                    out_channels=latent_dim,
                    kernel_size=3,
                    padding=1,
                ),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 2,
        channels: int = 64,
        latent_dim: int = 64,
        c_mults: Tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        strides: Tuple[int, ...] = (2, 2, 4, 4, 5, 6),
        use_snake: bool = True,
        snake_type: str = "vanilla",
        antialias_activation: bool = False,
        use_nearest_upsample: bool = False,
        use_filter: bool = False,
        final_tanh: bool = False,
    ):
        super().__init__()
        if antialias_activation or use_nearest_upsample or use_filter:
            raise ValueError(
                "Unsupported operational modes enabled for Oobleck decoder."
            )
        if use_snake and snake_type != "vanilla":
            raise ValueError(
                "Oobleck specification requires vanilla SnakeBeta activation."
            )
        self.out_channels = out_channels
        c_mults_list = [1] + list(c_mults)
        self.depth = len(c_mults_list)

        layers = [
            WNConv1d(
                in_channels=latent_dim,
                out_channels=c_mults_list[-1] * channels,
                kernel_size=7,
                padding=3,
            )
        ]
        act_type = "snake" if use_snake else "elu"
        for i in range(self.depth - 1, 0, -1):
            layers.append(
                DecoderBlock(
                    in_channels=c_mults_list[i] * channels,
                    out_channels=c_mults_list[i - 1] * channels,
                    stride=strides[i - 1],
                    act_type=act_type,
                )
            )
        layers.extend(
            [
                get_activation(act_type, channels=c_mults_list[0] * channels),
                WNConv1d(
                    in_channels=c_mults_list[0] * channels,
                    out_channels=out_channels,
                    kernel_size=7,
                    padding=3,
                    bias=False,
                ),
                nn.Tanh() if final_tanh else nn.Identity(),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class YuE2VAEConfig(PretrainedConfig):
    model_type = "yue2_vae"
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
            "encoder_config",
            "decoder_config",
            "sample_rate",
            "latent_dim",
            "downsampling_ratio",
            "audio_channels",
            "release_variant",
            "decode_core_frames",
            "decode_halo_frames",
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
        encoder_config: Optional[Dict] = None,
        decoder_config: Optional[Dict] = None,
        sample_rate: int = 48000,
        latent_dim: int = 64,
        downsampling_ratio: int = 1920,
        audio_channels: int = 2,
        release_variant: str = "standard",
        decode_core_frames: int = 1024,
        decode_halo_frames: int = 16,
        **kwargs,
    ):
        clean_kwargs = {
            key: value for key, value in kwargs.items() if key in self._hf_fields
        }
        clean_kwargs.setdefault("architectures", ["YuE2VAE"])
        super().__init__(**clean_kwargs)

        self.encoder_config = encoder_config or dict(
            in_channels=2,
            channels=64,
            c_mults=[1, 2, 4, 8, 16, 32],
            strides=[2, 2, 4, 4, 5, 6],
            latent_dim=128,
            use_snake=True,
        )
        self.decoder_config = decoder_config or dict(
            out_channels=2,
            channels=64,
            c_mults=[1, 2, 4, 8, 16, 32],
            strides=[2, 2, 4, 4, 5, 6],
            latent_dim=64,
            use_snake=True,
            snake_type="vanilla",
            use_filter=False,
            final_tanh=False,
        )
        self.sample_rate = int(sample_rate)
        self.latent_dim = int(latent_dim)
        self.downsampling_ratio = int(downsampling_ratio)
        self.audio_channels = int(audio_channels)
        self.release_variant = release_variant
        self.decode_core_frames = int(decode_core_frames)
        self.decode_halo_frames = int(decode_halo_frames)

        if self.decode_core_frames < 1 or self.decode_halo_frames < 0:
            raise ValueError(
                "Core and halo framing specifications must be positive non-negative values."
            )
        if math.prod(self.decoder_config["strides"]) != self.downsampling_ratio:
            raise ValueError(
                "Cumulative stride geometry fails to reconstruct specified downsampling ratio."
            )
        if self.decoder_config["latent_dim"] != self.latent_dim:
            raise ValueError(
                "Decoder input dimensionality deviates from latent_dim invariant."
            )


def _dependency_interval(module: nn.Module, low: int, high: int) -> Tuple[int, int]:
    if isinstance(
        module,
        (nn.Sequential, OobleckEncoder, OobleckDecoder, EncoderBlock, DecoderBlock),
    ):
        layers = module if isinstance(module, nn.Sequential) else module.layers
        for child in reversed(list(layers)):
            low, high = _dependency_interval(child, low, high)
        return low, high
    if isinstance(module, ResidualUnit):
        a, b = _dependency_interval(module.layers, low, high)
        return min(a, low), max(b, high)
    if isinstance(module, nn.ConvTranspose1d):
        s, p, d, k = (
            module.stride[0],
            module.padding[0],
            module.dilation[0],
            module.kernel_size[0],
        )
        return -(-(low + p - d * (k - 1)) // s), (high + p) // s
    if isinstance(module, nn.Conv1d):
        s, p, d, k = (
            module.stride[0],
            module.padding[0],
            module.dilation[0],
            module.kernel_size[0],
        )
        return low * s - p, high * s - p + d * (k - 1)
    if isinstance(module, (SnakeBeta, nn.ELU, nn.Identity, nn.Tanh)):
        return low, high
    raise TypeError(
        f"Unmapped topological receptive layer type: {type(module).__name__}"
    )


def _output_length(module: nn.Module, length: int) -> int:
    if isinstance(
        module,
        (nn.Sequential, OobleckEncoder, OobleckDecoder, EncoderBlock, DecoderBlock),
    ):
        layers = module if isinstance(module, nn.Sequential) else module.layers
        for child in layers:
            length = _output_length(child, length)
        return length
    if isinstance(module, nn.ConvTranspose1d):
        return (
            (length - 1) * module.stride[0]
            - 2 * module.padding[0]
            + module.dilation[0] * (module.kernel_size[0] - 1)
            + module.output_padding[0]
            + 1
        )
    if isinstance(module, nn.Conv1d):
        return (
            length
            + 2 * module.padding[0]
            - module.dilation[0] * (module.kernel_size[0] - 1)
            - 1
        ) // module.stride[0] + 1
    if isinstance(module, (ResidualUnit, SnakeBeta, nn.ELU, nn.Identity, nn.Tanh)):
        return length
    raise TypeError(
        f"Unmapped sequence dimension transformation for: {type(module).__name__}"
    )


def vae_bottleneck_sample(
    raw_moments: torch.Tensor,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    precomputed_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    moments_f32 = raw_moments.float()
    mean, scale = moments_f32.chunk(2, dim=1)
    if deterministic:
        return mean
    stdev = F.softplus(scale) + 1e-4
    if precomputed_noise is not None:
        noise = precomputed_noise.to(device=mean.device, dtype=torch.float32)
    else:
        noise = torch.randn(
            mean.shape,
            dtype=torch.float32,
            device=mean.device,
            generator=generator,
        )
    return mean + stdev * noise


class YuE2VAE(PreTrainedModel):
    config_class = YuE2VAEConfig
    base_model_prefix = ""
    main_input_name = "audio"

    def __init__(self, config: YuE2VAEConfig, decoder_only: bool = False):
        with torch.inference_mode(False):
            super().__init__(config)
            self.decoder_only = bool(decoder_only)
            if not self.decoder_only:
                self.encoder = OobleckEncoder(**config.encoder_config)
            else:
                self.encoder = None
            self.decoder = OobleckDecoder(**config.decoder_config)
            self.to(dtype=torch.float32).eval().requires_grad_(False)

    def _apply(self, fn, recurse: bool = True):
        with torch.inference_mode(False):
            return super()._apply(fn, recurse=recurse)

    def _materialize_snake_caches(self) -> None:
        for mod in self.modules():
            if isinstance(mod, SnakeBeta):
                mod.materialize_eval_cache()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        *model_args,
        config: Optional[YuE2VAEConfig] = None,
        decoder_only: bool = False,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ) -> YuE2VAE:
        from safetensors.torch import load_file as load_safetensors

        with torch.inference_mode(False):
            path = Path(pretrained_model_name_or_path).expanduser()
            if not path.is_dir():
                from huggingface_hub import snapshot_download

                path = Path(
                    snapshot_download(
                        str(pretrained_model_name_or_path),
                        allow_patterns=["config.json", "*.safetensors"],
                    )
                )
            if config is None:
                config_file = path / "config.json"
                config_dict = json.loads(config_file.read_text(encoding="utf-8"))
                config = YuE2VAEConfig(**config_dict)

            sf_files = sorted(list(path.glob("*.safetensors")))
            if not sf_files:
                raise FileNotFoundError(
                    f"SafeTensors artifact missing in directory: {path}"
                )

            raw_state: Dict[str, torch.Tensor] = {}
            for sf in sf_files:
                raw_state.update(load_safetensors(str(sf), device="cpu"))

            has_encoder_weights = any(k.startswith("encoder.") for k in raw_state)
            effective_decoder_only = bool(decoder_only or not has_encoder_weights)
            model = cls(config, decoder_only=effective_decoder_only)

            filtered_state = {
                k: v
                for k, v in raw_state.items()
                if not effective_decoder_only or k.startswith("decoder.")
            }
            fused_state = fold_weight_norm_dict(filtered_state)
            model.load_state_dict(fused_state, strict=True)
            model.to(device=device, dtype=torch.float32).eval()
            model._materialize_snake_caches()
            return model

    @property
    def has_encoder(self) -> bool:
        return self.encoder is not None

    @property
    def decoder_device(self) -> torch.device:
        return next(self.decoder.parameters()).device

    @property
    def encoder_device(self) -> torch.device:
        if self.encoder is None:
            raise RuntimeError(
                "YuE2VAE was initialized with decoder_only=True; encoder unavailable."
            )
        return next(self.encoder.parameters()).device

    def _validate_audio(self, audio: torch.Tensor) -> torch.Tensor:
        audio_t = torch.as_tensor(audio)
        if audio_t.ndim == 2:
            audio_t = audio_t.unsqueeze(0)
        if audio_t.ndim != 3 or audio_t.shape[0] < 1 or audio_t.shape[-1] < 1:
            raise ValueError(
                f"Expected stereo audio tensor [B, 2, samples], received {audio_t.shape}"
            )
        if audio_t.shape[1] == 1 and self.config.audio_channels == 2:
            audio_t = audio_t.expand(-1, 2, -1)
        elif audio_t.shape[1] != self.config.audio_channels:
            raise ValueError(
                f"Expected {self.config.audio_channels} audio channels, received {audio_t.shape[1]}"
            )
        if not torch.isfinite(audio_t).all():
            raise FloatingPointError("Input audio tensor contains NaN or Inf values.")
        return audio_t

    def _validate_latents(self, latent: torch.Tensor) -> torch.Tensor:
        latent_t = torch.as_tensor(latent)
        if (
            latent_t.ndim != 3
            or latent_t.shape[1] != self.config.latent_dim
            or latent_t.shape[0] < 1
            or latent_t.shape[-1] < 1
        ):
            raise ValueError(
                f"Expected latents with shape [B, {self.config.latent_dim}, T], received {latent_t.shape}"
            )
        if not torch.isfinite(latent_t).all():
            raise FloatingPointError("Latent tensor contains NaN or Inf components.")
        return latent_t

    def required_encode_halo_frames(self, core_frames: Optional[int] = None) -> int:
        if self.encoder is None:
            raise RuntimeError("Encoder unavailable in decoder_only mode.")
        core = (
            self.config.decode_core_frames
            if core_frames is None
            else int(core_frames)
        )
        ratio = self.config.downsampling_ratio
        low, high = _dependency_interval(self.encoder, 0, core - 1)
        left_samples = max(0, -low)
        right_samples = max(0, high - (core * ratio - 1))
        return math.ceil(max(left_samples, right_samples) / ratio)

    def encode(
        self,
        audio: torch.Tensor,
        deterministic: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError(
                "YuE2VAE was initialized with decoder_only=True; cannot encode audio."
            )
        caller_in_inf = torch.is_inference_mode_enabled()
        with torch.inference_mode():
            audio_t = self._validate_audio(audio)
            ratio = self.config.downsampling_ratio
            rem = audio_t.shape[-1] % ratio
            if rem != 0:
                audio_t = F.pad(audio_t, (0, ratio - rem))
            target_frames = audio_t.shape[-1] // ratio
            dev = self.encoder_device
            with IEEE754FP32Guard(), torch.autocast(
                device_type=dev.type, enabled=False
            ):
                moments = self.encoder(
                    audio_t.to(device=dev, dtype=torch.float32)
                )[..., :target_frames]
                latents = vae_bottleneck_sample(
                    moments, deterministic=deterministic, generator=generator
                )
        if not caller_in_inf and latents.is_inference():
            with torch.inference_mode(False):
                return latents.clone()
        return latents

    def encode_tiled(
        self,
        audio: torch.Tensor,
        core_frames: Optional[int] = None,
        halo_frames: Optional[int] = None,
        deterministic: bool = True,
        generator: Optional[torch.Generator] = None,
        output_device: Union[str, torch.device] = "cpu",
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError(
                "YuE2VAE was initialized with decoder_only=True; cannot encode audio."
            )
        caller_in_inf = torch.is_inference_mode_enabled()
        with torch.inference_mode():
            audio_t = self._validate_audio(audio)
            ratio = self.config.downsampling_ratio
            rem = audio_t.shape[-1] % ratio
            if rem != 0:
                audio_t = F.pad(audio_t, (0, ratio - rem))
            total_samples = audio_t.shape[-1]
            total_frames = total_samples // ratio

            core = (
                self.config.decode_core_frames
                if core_frames is None
                else int(core_frames)
            )
            req_halo = self.required_encode_halo_frames(core)
            halo = max(
                req_halo,
                self.config.decode_halo_frames
                if halo_frames is None
                else int(halo_frames),
            )

            if total_frames <= core + 2 * halo:
                out = (
                    self.encode(
                        audio_t, deterministic=deterministic, generator=generator
                    )
                    .to(device=output_device, dtype=torch.float32)
                    .contiguous()
                )
                if on_progress is not None:
                    on_progress(1, 1)
                if not caller_in_inf and out.is_inference():
                    with torch.inference_mode(False):
                        return out.clone()
                return out

            dev = self.encoder_device
            shared_noise: Optional[torch.Tensor] = None
            if not deterministic:
                noise_dev = generator.device if generator is not None else dev
                shared_noise = torch.randn(
                    (audio_t.shape[0], self.config.latent_dim, total_frames),
                    dtype=torch.float32,
                    device=noise_dev,
                    generator=generator,
                )

            with torch.inference_mode(False):
                latents_out = torch.empty(
                    (audio_t.shape[0], self.config.latent_dim, total_frames),
                    dtype=torch.float32,
                    device=output_device,
                )
            tiles = (total_frames + core - 1) // core
            halo_samples = halo * ratio

            for tile_idx, f_start in enumerate(range(0, total_frames, core)):
                f_end = min(total_frames, f_start + core)
                s_start = f_start * ratio
                s_end = f_end * ratio
                s_left = max(0, s_start - halo_samples)
                s_right = min(total_samples, s_end + halo_samples)

                slice_audio = audio_t[..., s_left:s_right]
                tile_frames = (s_right - s_left) // ratio
                crop_f_start = (s_start - s_left) // ratio
                crop_f_len = f_end - f_start

                with IEEE754FP32Guard(), torch.autocast(
                    device_type=dev.type, enabled=False
                ):
                    moments = self.encoder(
                        slice_audio.to(device=dev, dtype=torch.float32)
                    )[..., :tile_frames]
                    moments_core = moments[
                        ..., crop_f_start : crop_f_start + crop_f_len
                    ]
                    noise_slice = (
                        shared_noise[..., f_start:f_end]
                        if shared_noise is not None
                        else None
                    )
                    crop = vae_bottleneck_sample(
                        moments_core,
                        deterministic=deterministic,
                        precomputed_noise=noise_slice,
                    )

                if crop.shape[-1] != crop_f_len:
                    raise RuntimeError(
                        "Encoded latent slice length diverged from receptive core bounds."
                    )
                latents_out[..., f_start:f_end].copy_(
                    crop.to(device=output_device, dtype=torch.float32)
                )
                del slice_audio, moments, moments_core, crop
                if on_progress is not None:
                    on_progress(tile_idx + 1, tiles)

            return latents_out

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        caller_in_inf = torch.is_inference_mode_enabled()
        with torch.inference_mode():
            latent_t = self._validate_latents(latent)
            dev = self.decoder_device
            with IEEE754FP32Guard(), torch.autocast(
                device_type=dev.type, enabled=False
            ):
                out = self.decoder(latent_t.to(device=dev, dtype=torch.float32))
        if not caller_in_inf and out.is_inference():
            with torch.inference_mode(False):
                return out.clone()
        return out

    def natural_output_length(self, frames: int) -> int:
        if int(frames) < 1:
            raise ValueError("Frame length evaluation must be strictly positive.")
        return _output_length(self.decoder, int(frames))

    def required_halo(self, core_frames: Optional[int] = None) -> int:
        core = (
            self.config.decode_core_frames
            if core_frames is None
            else int(core_frames)
        )
        ratio = self.config.downsampling_ratio
        low, high = _dependency_interval(self.decoder, 0, core * ratio - 1)
        return max(0, -low, high - core + 1)

    def decode_tiled(
        self,
        latent: torch.Tensor,
        core_frames: Optional[int] = None,
        halo_frames: Optional[int] = None,
        output_device: Union[str, torch.device] = "cpu",
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        caller_in_inf = torch.is_inference_mode_enabled()
        with torch.inference_mode():
            latent_t = self._validate_latents(latent)
            core = (
                self.config.decode_core_frames
                if core_frames is None
                else int(core_frames)
            )
            halo = (
                self.config.decode_halo_frames
                if halo_frames is None
                else int(halo_frames)
            )
            required = self.required_halo(core)
            if halo < required:
                raise ValueError(
                    f"Configured halo frames ({halo}) cannot fulfill structural receptive bound ({required})."
                )

            frames = latent_t.shape[-1]
            ratio = self.config.downsampling_ratio
            total_samples = self.natural_output_length(frames)
            if frames <= core:
                out = (
                    self.decode(latent_t)[..., :total_samples]
                    .to(device=output_device, dtype=torch.float32)
                    .contiguous()
                )
                if on_progress is not None:
                    on_progress(1, 1)
                if not caller_in_inf and out.is_inference():
                    with torch.inference_mode(False):
                        return out.clone()
                return out

            with torch.inference_mode(False):
                audio = torch.empty(
                    (latent_t.shape[0], self.config.audio_channels, total_samples),
                    dtype=torch.float32,
                    device=output_device,
                )
            tiles = (frames + core - 1) // core

            for tile_idx, start in enumerate(range(0, frames, core)):
                end = min(frames, start + core)
                left = max(0, start - halo)
                right = min(frames, end + halo)
                tile = self.decode(latent_t[..., left:right])

                out_start = start * ratio
                out_end = min(end * ratio, total_samples)
                crop_start = (start - left) * ratio
                crop_len = out_end - out_start
                crop = tile[..., crop_start : crop_start + crop_len]
                if crop.shape[-1] != crop_len:
                    raise RuntimeError(
                        "Reconstructed slice length diverged from computed receptive core bounds."
                    )

                audio[..., out_start:out_end].copy_(
                    crop.to(device=output_device, dtype=torch.float32)
                )
                del tile, crop
                if on_progress is not None:
                    on_progress(tile_idx + 1, tiles)

            return audio


__all__ = [
    "HAS_NEW_FP32_PRECISION_API",
    "IEEE754FP32Guard",
    "enforce_full_precision_reductions",
    "SovereignConv1d",
    "SovereignConvTranspose1d",
    "WNConv1d",
    "WNConvTranspose1d",
    "fold_weight_norm_dict",
    "snake_beta_fused",
    "SnakeBeta",
    "get_activation",
    "ResidualUnit",
    "EncoderBlock",
    "DecoderBlock",
    "OobleckEncoder",
    "OobleckDecoder",
    "YuE2VAEConfig",
    "vae_bottleneck_sample",
    "YuE2VAE",
]
