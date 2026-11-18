from __future__ import annotations

from collections import OrderedDict, deque
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple, Union, cast
import warnings

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CACHE_DIR = Path(os.environ.get("HF_HOME") or (ROOT_DIR / ".hf_cache")).resolve()
TMP_DIR = Path(os.environ.get("TMP") or (ROOT_DIR / "artifacts" / "tmp")).resolve()
for _d in (CACHE_DIR, CACHE_DIR / "hub", CACHE_DIR / "torch", TMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE_DIR / "hub"))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
os.environ.setdefault("TMP", str(TMP_DIR))
os.environ.setdefault("TEMP", str(TMP_DIR))
os.environ.pop("TRANSFORMERS_CACHE", None)
if (
    not os.environ.get("PYTORCH_ALLOC_CONF")
    and not os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    and not os.environ.get("PYTORCH_HIP_ALLOC_CONF")
):
    os.environ["PYTORCH_ALLOC_CONF"] = "garbage_collection_threshold:0.8"

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors

from models.modeling_vae import (
    IEEE754FP32Guard,
    YuE2VAE,
    enforce_full_precision_reductions,
)
from models.modeling_yue2 import (
    AudioPositionEmbedding,
    RMSNorm,
    StaticKVCache,
    YuE2Config,
    YuE2ForCausalLM,
    clear_yue2_workspaces,
    sdpa,
    synthesize_audio_pe_fp32,
)
from models.tokenizer import YuE2Tokenizer
from refinery import PhysicsLab, clear_refinery_cache, contender_cast, refineTensor
from schema import (
    GenerationRequest,
    GenerationResponse,
    INSTRUCTIONS,
    get_active_engine_defaults,
)

enforce_full_precision_reductions()

EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
LATENT_START, LATENT_END, LATENT_PAD = 184621, 184622, 184623
VOCAB_SIZE, CONTEXT = 184704, 24576
TARGET_SAMPLE_RATE = 48000

ABC_VOCAB_SLICE: Tuple[int, int] = (0, ABC_END + 1)
SEMANTIC_VOCAB_SLICE: Tuple[int, int] = (MUSIC_END, CODEC_OFFSET + CODEC_SIZE)
_INV_LN2 = 1.0 / math.log(2.0)

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

_FP32_BRIDGE_PREFIXES = (
    "llm2vae.",
    "vae2llm.",
    "time_embedder.",
    "latent_pos_embed.",
)

_POLYPHASE_KERNEL_CACHE: OrderedDict[
    Tuple[str, Optional[int], int, int, int, float], Tuple[torch.Tensor, int, int]
] = OrderedDict()
_POLYPHASE_CACHE_CAP = 16


def _contender_project_to_dtype(
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    *,
    phase_offset: int = 0,
    unbiased: bool = False,
) -> torch.Tensor:
    if tensor.dtype == target_dtype:
        return tensor
    if PhysicsLab.get_mantissa_width(tensor.dtype) > PhysicsLab.get_mantissa_width(
        target_dtype
    ):
        return contender_cast(
            tensor, target_dtype, phase_offset=phase_offset, unbiased=unbiased
        )
    return tensor.to(dtype=target_dtype)


def apply_sub_millisecond_declick(
    audio_tensor: torch.Tensor, fade_samples: int = 512
) -> torch.Tensor:
    if audio_tensor.ndim < 1:
        raise ValueError(
            f"Expected audio tensor with shape (*lead_shape, T), got {tuple(audio_tensor.shape)}"
        )
    if (
        fade_samples <= 0
        or audio_tensor.shape[-1] <= fade_samples * 2
        or audio_tensor.numel() == 0
    ):
        return audio_tensor
    if audio_tensor.is_inference() and not torch.is_inference_mode_enabled():
        audio_tensor = audio_tensor.clone()

    comp_dev = (
        torch.device("cpu")
        if audio_tensor.device.type == "mps"
        else audio_tensor.device
    )
    phase = torch.linspace(
        0.0,
        math.pi,
        fade_samples,
        device=comp_dev,
        dtype=torch.float64,
    )
    fade_f64 = 0.5 * (1.0 - torch.cos(phase))
    fade_f64[0] = 0.0
    fade_f64[-1] = 1.0
    tail_fade_f64 = torch.flip(fade_f64, dims=[0])

    head_f64 = (
        audio_tensor[..., :fade_samples].to(device=comp_dev, dtype=torch.float64)
        * fade_f64
    )
    tail_f64 = (
        audio_tensor[..., -fade_samples:].to(device=comp_dev, dtype=torch.float64)
        * tail_fade_f64
    )
    head_proj = _contender_project_to_dtype(head_f64, audio_tensor.dtype)
    tail_proj = _contender_project_to_dtype(tail_f64, audio_tensor.dtype)

    if comp_dev != audio_tensor.device:
        head_proj = head_proj.to(device=audio_tensor.device)
        tail_proj = tail_proj.to(device=audio_tensor.device)

    audio_tensor[..., :fade_samples] = head_proj
    audio_tensor[..., -fade_samples:] = tail_proj
    return audio_tensor


def _reflect_indices(idx: torch.Tensor, length: int) -> torch.Tensor:
    if length <= 1:
        return torch.zeros_like(idx)
    period = 2 * (length - 1)
    folded = torch.remainder(idx, period)
    return torch.where(folded < length, folded, period - folded)


def _reflect_pad_1d(x: torch.Tensor, pad_left: int, pad_right: int) -> torch.Tensor:
    length = int(x.shape[-1])
    if pad_left <= 0 and pad_right <= 0:
        return x
    if length <= 1:
        total = length + max(0, pad_left) + max(0, pad_right)
        return x[..., :1].expand(*x.shape[:-1], total).contiguous()
    if pad_left < length and pad_right < length:
        parts: List[torch.Tensor] = []
        if pad_left > 0:
            parts.append(torch.flip(x[..., 1 : pad_left + 1], dims=[-1]))
        parts.append(x)
        if pad_right > 0:
            parts.append(torch.flip(x[..., -(pad_right + 1) : -1], dims=[-1]))
        return torch.cat(parts, dim=-1)
    idx = torch.arange(
        -pad_left, length + pad_right, dtype=torch.int64, device=x.device
    )
    return x.index_select(-1, _reflect_indices(idx, length))


def _get_rational_polyphase_kernel(
    device: torch.device,
    up: int,
    down: int,
    half_length: int = 16,
    beta: float = 9.5,
) -> Tuple[torch.Tensor, int, int]:
    dev_idx = (
        torch.cuda.current_device()
        if (device.type == "cuda" and device.index is None and torch.cuda.is_available())
        else device.index
    )
    key = (device.type, dev_idx, int(up), int(down), int(half_length), float(beta))
    cached = _POLYPHASE_KERNEL_CACHE.get(key)
    if cached is not None and not cached[0].is_inference():
        _POLYPHASE_KERNEL_CACHE.move_to_end(key)
        return cached

    with torch.inference_mode(False):
        max_rate = max(up, down)
        cutoff = 0.85 / float(max_rate)
        half_taps = half_length * max_rate
        num_taps = 2 * half_taps + 1

        k = torch.arange(num_taps, dtype=torch.float64, device=device)
        m_tap = k - float(half_taps)
        arg = math.pi * cutoff * m_tap
        sinc = torch.where(m_tap == 0.0, torch.ones_like(m_tap), torch.sin(arg) / arg)
        beta_t = torch.full((), beta, dtype=torch.float64, device=device)
        inv_i0_beta = 1.0 / torch.i0(beta_t)
        win_arg = beta * torch.sqrt(
            torch.clamp_min(1.0 - (m_tap / float(half_taps)) ** 2, 0.0)
        )
        proto = sinc * (torch.i0(win_arg) * inv_i0_beta)

        taps_per_phase = (num_taps + up - 1) // up
        r_vec = torch.arange(up, dtype=torch.int64, device=device)
        s_r = r_vec * down + half_taps
        phi_r = torch.remainder(s_r, up)
        b_r = torch.div(s_r, up, rounding_mode="floor")

        j_grid = torch.arange(taps_per_phase, dtype=torch.int64, device=device)[None, :]
        tap_indices = phi_r[:, None] + j_grid * up
        valid_mask = tap_indices < num_taps
        clamped_idx = tap_indices.clamp(max=num_taps - 1)
        poly_r = torch.where(
            valid_mask, proto[clamped_idx], torch.zeros_like(proto[:1])
        )
        row_sums = poly_r.sum(dim=-1, keepdim=True)
        poly_r = torch.where(
            row_sums.abs() > 1e-15, poly_r / row_sums.clamp_min(1e-15), poly_r
        )
        poly_rev = torch.flip(poly_r, dims=[-1])

        start_r = b_r - (taps_per_phase - 1)
        min_start = (half_taps // up) - (taps_per_phase - 1)
        max_end = ((up - 1) * down + half_taps) // up
        win_width = max_end - min_start + 1

        offset_r = start_r - min_start
        u_grid = torch.arange(taps_per_phase, dtype=torch.int64, device=device)[None, :]
        col_idx = offset_r[:, None] + u_grid

        kernel_w_r = torch.zeros((win_width, up), dtype=torch.float64, device=device)
        kernel_w_r.scatter_(
            0,
            col_idx.transpose(0, 1).contiguous(),
            poly_rev.transpose(0, 1).contiguous(),
        )
        kernel_w_r = kernel_w_r.contiguous()

    result = (kernel_w_r, min_start, win_width)
    _POLYPHASE_KERNEL_CACHE[key] = result
    while len(_POLYPHASE_KERNEL_CACHE) > _POLYPHASE_CACHE_CAP:
        _POLYPHASE_KERNEL_CACHE.popitem(last=False)
    return result


def resample_polyphase_kaiser(
    audio: torch.Tensor,
    orig_sr: int,
    target_sr: int = TARGET_SAMPLE_RATE,
    *,
    _block_chunk_override: Optional[int] = None,
) -> torch.Tensor:
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"Sample rates must be positive: {orig_sr} -> {target_sr}")
    if audio.ndim < 1:
        raise ValueError(
            f"Expected audio tensor with shape (*lead_shape, T), got {tuple(audio.shape)}"
        )
    if orig_sr == target_sr or audio.shape[-1] == 0:
        return _contender_project_to_dtype(audio, torch.float32)

    orig_device = audio.device
    comp_dev = torch.device("cpu") if orig_device.type == "mps" else orig_device
    lead_shape = tuple(audio.shape[:-1])
    in_len = int(audio.shape[-1])
    out_len = max(1, int(round(in_len * float(target_sr) / float(orig_sr))))
    lead_numel = int(math.prod(lead_shape)) if lead_shape else 1

    if lead_numel == 0:
        return torch.empty((*lead_shape, out_len), dtype=torch.float32, device=orig_device)

    x_2d = (
        audio.to(device=comp_dev, dtype=torch.float64)
        .reshape(lead_numel, in_len)
        .contiguous()
    )

    gcd_val = math.gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // gcd_val
    down = int(orig_sr) // gcd_val
    half_length = 16
    beta = 9.5

    with IEEE754FP32Guard():
        if up <= 1024:
            kernel_w_r, min_start, win_width = _get_rational_polyphase_kernel(
                device=comp_dev, up=up, down=down, half_length=half_length, beta=beta
            )
            num_blocks = (out_len + up - 1) // up
            last_needed = (num_blocks - 1) * down + min_start + win_width - 1
            pad_left = max(0, -min_start)
            pad_right = max(0, last_needed - (in_len - 1))

            x_pad = _reflect_pad_1d(x_2d, pad_left, pad_right)
            start_offset = min_start + pad_left
            span_len = (num_blocks - 1) * down + win_width
            x_span = x_pad[:, start_offset : start_offset + span_len]

            if _block_chunk_override is not None:
                block_chunk = max(64, ((int(_block_chunk_override) + 63) // 64) * 64)
            else:
                block_chunk = max(64, ((32768 // up) // 64) * 64)

            if num_blocks <= block_chunk:
                windows = x_span.unfold(-1, win_width, down)
                out_blocks = torch.matmul(windows, kernel_w_r)
                out_f64 = out_blocks.reshape(lead_numel, num_blocks * up)[
                    :, :out_len
                ].reshape(*lead_shape, out_len)
                out_f32 = contender_cast(out_f64, torch.float32)
            else:
                out_f32 = torch.empty(
                    (*lead_shape, out_len), dtype=torch.float32, device=comp_dev
                )
                for q_start in range(0, num_blocks, block_chunk):
                    q_end = min(q_start + block_chunk, num_blocks)
                    q_cnt = q_end - q_start
                    s0 = q_start * down
                    s_len = (q_cnt - 1) * down + win_width
                    win_chunk = x_span[:, s0 : s0 + s_len].unfold(
                        -1, win_width, down
                    )
                    acc_blocks = torch.matmul(win_chunk, kernel_w_r).reshape(
                        lead_numel, q_cnt * up
                    )
                    m0 = q_start * up
                    m1 = min(m0 + q_cnt * up, out_len)
                    chunk_len = m1 - m0
                    acc_nd = acc_blocks[:, :chunk_len].reshape(
                        *lead_shape, chunk_len
                    )
                    out_f32[..., m0:m1] = contender_cast(acc_nd, torch.float32)

            return (
                out_f32
                if out_f32.device == orig_device
                else out_f32.to(device=orig_device)
            )

        ratio_scale = min(1.0, float(target_sr) / float(orig_sr))
        cutoff_cont = 0.85 * ratio_scale
        support = float(half_length) / max(ratio_scale, 1e-9)
        half_win = int(math.ceil(support))
        win_size = 2 * half_win + 1
        offsets_f64 = torch.arange(
            -half_win, half_win + 1, dtype=torch.float64, device=comp_dev
        )[None, :]
        beta_t = torch.full((), beta, dtype=torch.float64, device=comp_dev)
        inv_i0_beta = 1.0 / torch.i0(beta_t)
        inv_up_f64 = 1.0 / float(up)

        max_center_int = ((out_len - 1) * down) // up
        pad_left = half_win
        pad_right = max(0, max_center_int + half_win - (in_len - 1))
        x_pad = _reflect_pad_1d(x_2d, pad_left, pad_right)

        chunk_out = 16384
        out_f32 = torch.empty(
            (*lead_shape, out_len), dtype=torch.float32, device=comp_dev
        )
        for start in range(0, out_len, chunk_out):
            end = min(start + chunk_out, out_len)
            m_cnt = end - start
            c_lo = (start * down) // up
            c_hi = ((end - 1) * down) // up

            m_i64 = torch.arange(start, end, dtype=torch.int64, device=comp_dev)
            prod_i64 = m_i64 * down
            center_int = torch.div(prod_i64, up, rounding_mode="floor")
            rem_i64 = prod_i64 - center_int * up
            frac = (rem_i64.to(dtype=torch.float64) * inv_up_f64)[:, None]

            delta = frac - offsets_f64
            norm_dist = delta / support
            inside = torch.abs(norm_dist) <= 1.0
            arg = math.pi * cutoff_cont * delta
            sinc = torch.where(
                delta == 0.0, torch.ones_like(delta), torch.sin(arg) / arg
            )
            win_arg = beta * torch.sqrt(
                torch.clamp_min(1.0 - norm_dist * norm_dist, 0.0)
            )
            weights = torch.where(
                inside,
                sinc * (torch.i0(win_arg) * inv_i0_beta),
                torch.zeros_like(delta),
            )
            w_sum = torch.clamp_min(torch.sum(weights, dim=-1, keepdim=True), 1e-15)
            weights = weights / w_sum

            x_sub = x_pad[:, c_lo : c_hi + win_size]
            sub_windows = x_sub.unfold(-1, win_size, 1).permute(1, 0, 2)
            local_center = center_int - c_lo
            gathered_win = sub_windows.index_select(0, local_center)
            acc = (
                torch.bmm(gathered_win, weights.unsqueeze(-1))
                .squeeze(-1)
                .transpose(0, 1)
            )
            acc_nd = acc.reshape(*lead_shape, m_cnt)
            out_f32[..., start:end] = contender_cast(acc_nd, torch.float32)

    return out_f32 if out_f32.device == orig_device else out_f32.to(device=orig_device)


def load_reference_stereo_audio(
    audio_path: Path, max_duration: Optional[float] = None
) -> torch.Tensor:
    data, sr = sf.read(str(audio_path), dtype="float64", always_2d=True)
    if max_duration is not None and max_duration > 0.0:
        max_frames = int(round(max_duration * sr))
        data = data[:max_frames]
    if data.shape[0] == 0:
        raise ValueError(f"Reference audio file contains zero samples: {audio_path}")
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    with torch.inference_mode(False):
        frames_first = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float64))
        tensor_bct = frames_first.transpose(-1, -2).unsqueeze(0).contiguous()
        if sr != TARGET_SAMPLE_RATE:
            return resample_polyphase_kaiser(
                tensor_bct, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE
            )
        return contender_cast(tensor_bct, torch.float32)


def resolve_model_path(repo_or_path: Union[str, Path]) -> Path:
    direct_path = Path(repo_or_path).resolve()
    if direct_path.exists() and (direct_path.is_dir() or direct_path.is_file()):
        return direct_path

    repo_str = str(repo_or_path).replace("\\", "/")
    hub_candidates: List[Path] = []
    for env_key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if env_key in os.environ and os.environ[env_key].strip():
            hub_candidates.append(Path(os.environ[env_key]).resolve())
    if "HF_HOME" in os.environ and os.environ["HF_HOME"].strip():
        hf_home = Path(os.environ["HF_HOME"]).resolve()
        hub_candidates.append(hf_home / "hub")
        hub_candidates.append(hf_home)
    hub_candidates.append(CACHE_DIR / "hub")
    hub_candidates.append(CACHE_DIR)
    hub_candidates.append(ROOT_DIR.parent / ".hf_cache" / "hub")
    hub_candidates.append(ROOT_DIR.parent / ".hf_cache")
    hub_candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    repo_folder_name = "models--" + repo_str.replace("/", "--")
    pure_name = repo_str.split("/")[-1]

    for hub_dir in hub_candidates:
        if not hub_dir.exists():
            continue
        cand = hub_dir / repo_folder_name / "snapshots"
        if cand.exists():
            snapshots = sorted(
                list(cand.iterdir()), key=lambda p: p.stat().st_mtime, reverse=True
            )
            if snapshots:
                return snapshots[0].resolve()
        direct_named = hub_dir / pure_name
        if direct_named.exists() and (direct_named / "config.json").exists():
            return direct_named.resolve()

    for hub_dir in hub_candidates:
        if not hub_dir.exists():
            continue
        try:
            matches = list(hub_dir.rglob(f"{pure_name}/config.json"))
            if matches:
                return matches[0].parent.resolve()
        except Exception:
            pass

    return direct_path


class _RollingPenaltyState:
    __slots__ = (
        "window",
        "vocab_offset",
        "support_len",
        "penalty",
        "device",
        "ring",
        "counts",
        "_stage_slots",
        "_slot_cursor",
        "_cpu_idx",
        "_cpu_cnt",
        "_cpu_idx_np",
        "_cpu_cnt_np",
        "_dev_idx",
        "_dev_cnt",
        "_dev_penalty",
        "_pinned",
    )

    def __init__(
        self,
        window: int,
        vocab_offset: int,
        support_len: int,
        penalty: float,
        device: torch.device,
    ) -> None:
        self.window = max(0, int(window))
        self.vocab_offset = int(vocab_offset)
        self.support_len = int(support_len)
        self.penalty = float(penalty)
        self.device = device
        self.ring: deque[int] = deque()
        self.counts: Dict[int, int] = {}
        cap = max(1, self.window)
        dev_f64_ok = device.type not in ("cpu", "mps")
        self._stage_slots = 4 if dev_f64_ok else 1
        self._slot_cursor = 0
        with torch.inference_mode(False):
            use_pin = False
            if device.type == "cuda" and self.penalty != 1.0 and self.window > 0:
                try:
                    self._cpu_idx = torch.empty(
                        (self._stage_slots, cap), dtype=torch.long, pin_memory=True
                    )
                    self._cpu_cnt = torch.empty(
                        (self._stage_slots, cap), dtype=torch.float64, pin_memory=True
                    )
                    use_pin = True
                except Exception:
                    use_pin = False
            if not use_pin:
                self._cpu_idx = torch.empty((self._stage_slots, cap), dtype=torch.long)
                self._cpu_cnt = torch.empty(
                    (self._stage_slots, cap), dtype=torch.float64
                )
            self._pinned = use_pin
            self._cpu_idx_np = self._cpu_idx.numpy()
            self._cpu_cnt_np = self._cpu_cnt.numpy()
            if not dev_f64_ok:
                self._dev_idx = self._cpu_idx[0]
                self._dev_cnt = self._cpu_cnt[0]
                self._dev_penalty = torch.full(
                    (), self.penalty, dtype=torch.float64, device="cpu"
                )
            else:
                self._dev_idx = torch.empty(cap, dtype=torch.long, device=device)
                self._dev_cnt = torch.empty(cap, dtype=torch.float64, device=device)
                self._dev_penalty = torch.full(
                    (), self.penalty, dtype=torch.float64, device=device
                )

    def push(self, token: int) -> None:
        if self.window <= 0 or self.penalty == 1.0:
            return
        if len(self.ring) >= self.window:
            evicted = self.ring.popleft()
            if evicted >= 0:
                rem = self.counts[evicted] - 1
                if rem <= 0:
                    del self.counts[evicted]
                else:
                    self.counts[evicted] = rem
        loc = int(token) - self.vocab_offset
        if 0 <= loc < self.support_len:
            self.ring.append(loc)
            self.counts[loc] = self.counts.get(loc, 0) + 1
        else:
            self.ring.append(-1)

    def apply_inplace_f64(self, scores_1d: torch.Tensor) -> None:
        if self.penalty == 1.0 or not self.counts:
            return
        u_len = len(self.counts)
        slot = self._slot_cursor
        self._slot_cursor = (slot + 1) % self._stage_slots
        self._cpu_idx_np[slot, :u_len] = np.fromiter(
            self.counts.keys(), dtype=np.int64, count=u_len
        )
        self._cpu_cnt_np[slot, :u_len] = np.fromiter(
            self.counts.values(), dtype=np.float64, count=u_len
        )
        target_dev = scores_1d.device
        if target_dev.type == "cpu":
            idx_u = self._cpu_idx[slot, :u_len]
            cnt_u = self._cpu_cnt[slot, :u_len]
            pen_scalar = (
                self._dev_penalty
                if self._dev_penalty.device.type == "cpu"
                else torch.full((), self.penalty, dtype=torch.float64)
            )
        elif target_dev == self.device and self._dev_cnt.device == target_dev:
            self._dev_idx[:u_len].copy_(
                self._cpu_idx[slot, :u_len], non_blocking=self._pinned
            )
            self._dev_cnt[:u_len].copy_(
                self._cpu_cnt[slot, :u_len], non_blocking=self._pinned
            )
            idx_u = self._dev_idx[:u_len]
            cnt_u = self._dev_cnt[:u_len]
            pen_scalar = self._dev_penalty
        else:
            idx_u = self._cpu_idx[slot, :u_len].to(device=target_dev)
            cnt_u = self._cpu_cnt[slot, :u_len].to(
                device=target_dev, dtype=scores_1d.dtype
            )
            pen_scalar = torch.full(
                (), self.penalty, dtype=scores_1d.dtype, device=target_dev
            )
        s_sub = scores_1d.index_select(0, idx_u)
        alpha_u = torch.pow(pen_scalar, cnt_u)
        pen_sub = torch.where(s_sub < 0.0, s_sub * alpha_u, s_sub / alpha_u)
        scores_1d.index_copy_(0, idx_u, pen_sub)


def apply_windowed_penalty(
    logits: torch.Tensor,
    history: List[int],
    penalty: float,
    window: int,
    vocab_offset: int = 0,
    _inplace: bool = False,
) -> torch.Tensor:
    if penalty == 1.0 or window <= 0 or not history:
        return logits
    recent = history[-window:]
    support_len = int(logits.shape[-1])
    counts: Dict[int, int] = {}
    if vocab_offset != 0 or support_len < VOCAB_SIZE:
        for tok in recent:
            loc = tok - vocab_offset
            if 0 <= loc < support_len:
                counts[loc] = counts.get(loc, 0) + 1
    else:
        for tok in recent:
            if 0 <= tok < support_len:
                counts[tok] = counts.get(tok, 0) + 1
    if not counts:
        return logits

    if logits.device.type == "mps":
        work_dt = torch.float32
    else:
        work_dt = (
            logits.dtype
            if logits.dtype in (torch.float32, torch.float64)
            else torch.float64
        )
    if logits.dtype == work_dt:
        scores = (
            logits
            if (
                _inplace
                and (torch.is_inference_mode_enabled() or not logits.is_inference())
            )
            else logits.clone()
        )
    else:
        scores = logits.to(dtype=work_dt)
    device = scores.device
    idx_u = torch.tensor(list(counts.keys()), dtype=torch.long, device=device)
    cnt_u = torch.tensor(list(counts.values()), dtype=work_dt, device=device)
    p_t = torch.full((), float(penalty), dtype=work_dt, device=device)
    alpha_u = torch.pow(p_t, cnt_u)
    s_sub = scores.index_select(-1, idx_u)
    pen_sub = torch.where(s_sub < 0.0, s_sub * alpha_u, s_sub / alpha_u)
    scores.index_copy_(-1, idx_u, pen_sub)
    return scores if logits.dtype == work_dt else scores.to(dtype=logits.dtype)


def sample_categorical_distribution_with_entropy(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    history: List[int],
    step: int,
    min_tokens: int,
    penalty: float,
    penalty_window: int,
    phase: str,
    generator: torch.Generator,
    vocab_offset: int = 0,
    penalty_state: Optional[_RollingPenaltyState] = None,
) -> Tuple[int, float]:
    on_mps = logits.device.type == "mps"
    if on_mps:
        scores = logits.to(device="cpu", dtype=torch.float64)
    else:
        scores = (
            logits.clone()
            if logits.dtype == torch.float64
            else logits.to(dtype=torch.float64)
        )
    if scores.ndim > 1:
        scores = scores.reshape(-1)

    support_len = int(scores.shape[-1])
    end_token = ABC_END if phase == "abc" else MUSIC_END
    local_end = end_token - vocab_offset

    if phase == "abc":
        valid_hi = min(support_len, max(0, EOD - vocab_offset))
        if valid_hi < support_len:
            mask_tail = torch.ones(
                support_len - valid_hi, dtype=torch.bool, device=scores.device
            )
            if 0 <= local_end - valid_hi < mask_tail.numel():
                mask_tail[local_end - valid_hi] = False
            scores[valid_hi:] = scores[valid_hi:].masked_fill(mask_tail, -float("inf"))
    else:
        c_lo = max(0, CODEC_OFFSET - vocab_offset)
        c_hi = min(support_len, CODEC_OFFSET + CODEC_SIZE - vocab_offset)
        all_covered = (c_lo == 0 and c_hi == support_len) or (
            local_end == 0 and c_lo == 1 and c_hi == support_len
        )
        if not all_covered and (c_lo > 0 or c_hi < support_len):
            invalid = torch.ones(support_len, dtype=torch.bool, device=scores.device)
            if c_hi > c_lo:
                invalid[c_lo:c_hi] = False
            if 0 <= local_end < support_len:
                invalid[local_end] = False
            scores.masked_fill_(invalid, -float("inf"))

    if step < min_tokens and 0 <= local_end < support_len:
        scores[local_end] = -float("inf")

    if penalty_state is not None:
        penalty_state.apply_inplace_f64(scores)
    else:
        scores = apply_windowed_penalty(
            scores,
            history,
            penalty,
            penalty_window,
            vocab_offset=vocab_offset,
            _inplace=True,
        )

    if temperature <= 0.0:
        chosen_local = int(scores.argmax(-1).item())
        return chosen_local + vocab_offset, 0.0

    inv_temp = 1.0 / float(temperature)
    if temperature != 1.0:
        scores.mul_(inv_temp)

    k_val = max(1, min(int(top_k), support_len))
    top_vals, top_idx = torch.topk(scores, k_val, dim=-1, largest=True, sorted=True)
    any_finite = torch.isfinite(top_vals).any()
    safe_top_vals = torch.where(any_finite, top_vals, torch.zeros_like(top_vals))

    log_probs = torch.log_softmax(safe_top_vals, dim=-1)
    probs = torch.exp(log_probs)

    if top_p < 1.0:
        cum_probs = torch.cumsum(probs, dim=-1)
        exceeded = (cum_probs - probs) > float(top_p)
        exceeded[0] = False
        safe_top_vals = safe_top_vals.masked_fill(exceeded, -float("inf"))
        log_probs = torch.log_softmax(safe_top_vals, dim=-1)
        probs = torch.exp(log_probs)

    prob_sum = probs.sum(dim=-1, keepdim=True).clamp_min(1e-300)
    probs = probs / prob_sum

    safe_log2_p = torch.where(
        probs > 0.0,
        (log_probs - torch.log(prob_sum)) * _INV_LN2,
        torch.zeros_like(probs),
    )
    raw_ent = (-torch.sum(probs * safe_log2_p)).clamp_min(0.0)
    ent_f64 = torch.where(any_finite, raw_ent, torch.zeros_like(raw_ent))

    if on_mps and generator.device.type == "mps":
        probs_mps = probs.to(device=logits.device, dtype=torch.float32)
        sampled_pos = torch.multinomial(probs_mps, 1, generator=generator).to(
            device="cpu"
        )
    else:
        sampled_pos = torch.multinomial(probs, 1, generator=generator)

    sampled_local = top_idx.gather(0, sampled_pos).reshape(())
    fallback_local = max(0, min(local_end, support_len - 1))
    chosen_local = torch.where(
        any_finite,
        sampled_local,
        torch.full((), fallback_local, dtype=top_idx.dtype, device=scores.device),
    )

    readout = torch.stack((chosen_local.to(dtype=torch.float64), ent_f64)).tolist()
    return int(readout[0]) + vocab_offset, float(readout[1])


def sample_categorical_distribution(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    history: List[int],
    step: int,
    min_tokens: int,
    penalty: float,
    penalty_window: int,
    phase: str,
    generator: torch.Generator,
    vocab_offset: int = 0,
    penalty_state: Optional[_RollingPenaltyState] = None,
) -> int:
    tok, _ = sample_categorical_distribution_with_entropy(
        logits=logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        history=history,
        step=step,
        min_tokens=min_tokens,
        penalty=penalty,
        penalty_window=penalty_window,
        phase=phase,
        generator=generator,
        vocab_offset=vocab_offset,
        penalty_state=penalty_state,
    )
    return tok


def partition_song_chunks(
    prefix: List[int],
    codec: List[int],
    seed: int,
    context_len: int = CONTEXT,
    nar_text_only: bool = False,
    ref_latents_td: Optional[torch.Tensor] = None,
) -> List[Dict[str, object]]:
    prefix_len = len(prefix)
    ref_len = int(ref_latents_td.shape[0]) if ref_latents_td is not None else 0
    max_total_nar = (context_len - prefix_len - 3) // 2
    if max_total_nar < 1 or not codec:
        raise ValueError(
            "Sequence length exceeds permissible acoustic context boundaries."
        )

    if ref_len >= max_total_nar and ref_latents_td is not None:
        keep_ref = max(1, max_total_nar // 2)
        ref_latents_td = ref_latents_td[-keep_ref:]
        ref_len = keep_ref

    first_cap = max(1, min(max_total_nar - ref_len, len(codec)))
    ranges: List[Tuple[int, int]] = [(0, first_cap)]
    cursor = first_cap
    subseq_cap = max(1, min(max_total_nar, context_len))
    while cursor < len(codec):
        nxt = min(cursor + subseq_cap, len(codec))
        ranges.append((cursor, nxt))
        cursor = nxt

    with torch.inference_mode(False):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        total_noise_frames = ref_len + len(codec)
        full_noise = torch.randn(
            (total_noise_frames, 64),
            dtype=torch.float32,
            device="cpu",
            generator=generator,
        )
        ref_noise = full_noise[:ref_len] if ref_len > 0 else None
        codec_noise = full_noise[ref_len:]

        cond_end = prefix_len if nar_text_only else 0
        chunks: List[Dict[str, object]] = []
        for idx, (a, b) in enumerate(ranges):
            ar_toks = prefix + [c + CODEC_OFFSET for c in codec[a:b]] + [MUSIC_END]
            if idx == 0 and ref_latents_td is not None and ref_noise is not None:
                chk_noise = torch.cat([ref_noise, codec_noise[a:b]], dim=0)
                rc_proj = _contender_project_to_dtype(
                    ref_latents_td.detach(), torch.float32, phase_offset=56
                )
                chk_ref = rc_proj.clone() if rc_proj.is_inference() else rc_proj
            else:
                chk_noise = codec_noise[a:b]
                chk_ref = None
            chunks.append(
                {
                    "tokens": ar_toks,
                    "noise": chk_noise,
                    "nar_cond_end": cond_end,
                    "ref_clean": chk_ref,
                }
            )
    return chunks


def _format_seconds(seconds: float) -> str:
    s = max(0, int(seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class EngineProgressReporter:
    def __init__(self) -> None:
        self.current_stage: Optional[str] = None
        self.stage_start: float = time.perf_counter()
        self.last_render: float = 0.0
        self.render_interval: float = 0.10
        self.last_count: int = 0

    def __call__(self, stage: str, current: int, total: int) -> None:
        now = time.perf_counter()
        if stage != self.current_stage:
            if self.current_stage is not None:
                self._finish_stage(now)
            self.current_stage = stage
            self.stage_start = now
            self.last_render = 0.0
            self.last_count = 0

        self.last_count = current
        is_final = current >= total
        if not is_final and (now - self.last_render < self.render_interval):
            return
        self.last_render = now

        elapsed = max(now - self.stage_start, 1e-4)
        pct = (current / max(total, 1)) * 100.0
        rate = (current / elapsed) if (elapsed >= 0.25 and current >= 3) else 0.0
        remaining_units = max(0, total - current)
        eta = (remaining_units / rate) if rate > 0.0 else 0.0
        time_str = f"[{_format_seconds(elapsed)}<{_format_seconds(eta)}"

        if stage == "abc":
            prefix = "[1/4 CoT Planner]"
            rate_str = f"{rate:5.1f} tok/s" if rate > 0 else "  --.- tok/s"
            suffix = (
                f" {pct:5.1f}% | {current:4d}/{total:4d} tok {time_str}, {rate_str}]"
            )
        elif stage == "semantic":
            prefix = "[2/4 Semantic MoT]"
            rate_str = f"{rate:4.1f} fps" if rate > 0 else " --.- fps"
            audio_sec = current / 25.0
            rtf_str = (
                f", RTF: {elapsed / max(audio_sec, 1e-3):4.2f}x"
                if audio_sec >= 1.0
                else ""
            )
            suffix = f" {pct:5.1f}% | {current:5d}/{total:5d} f {time_str}, {rate_str}{rtf_str}]"
        elif stage == "nar":
            prefix = "[3/4 Flow Matching]"
            rate_str = f"{rate:4.1f} it/s" if rate > 0 else " --.- it/s"
            suffix = (
                f" {pct:5.1f}% | {current:4d}/{total:4d} stp {time_str}, {rate_str}]"
            )
        elif stage == "vae":
            prefix = "[4/4 Receptive VAE]"
            rate_str = f"{rate:4.1f} tile/s" if rate > 0 else " --.- tile/s"
            suffix = (
                f" {pct:5.1f}% | {current:3d}/{total:3d} tile {time_str}, {rate_str}]"
            )
        else:
            prefix = f"[{stage.upper()}]"
            suffix = f" {pct:5.1f}% | {current}/{total} {time_str}]"

        cols = max(60, min(shutil.get_terminal_size((80, 20)).columns, 140))
        avail = cols - len(prefix) - len(suffix) - 3
        bar_len = max(8, min(avail, 30))
        filled = min(bar_len, int(bar_len * current / max(total, 1)))
        bar = (
            "=" * filled + ">"
            if (filled < bar_len and current > 0)
            else "=" * filled
        )
        bar = bar.ljust(bar_len, " ")
        line = f"\r{prefix} [{bar}]{suffix}"
        sys.stdout.write(line.ljust(cols - 1)[: cols - 1])
        sys.stdout.flush()

        if is_final:
            self._finish_stage(now)

    def _finish_stage(self, now: float) -> None:
        elapsed = max(now - self.stage_start, 1e-4)
        stage_names = {
            "abc": "Symbolic CoT Planner",
            "semantic": "Semantic Codec MoT",
            "nar": "Continuous Flow Matching",
            "vae": "Receptive Tiled VAE",
        }
        name = stage_names.get(
            self.current_stage,
            self.current_stage.upper() if self.current_stage else "Stage",
        )
        cols = max(60, min(shutil.get_terminal_size((80, 20)).columns, 140))
        summary = f"\r[+] {name:<26} finalized in {elapsed:6.2f}s ({self.last_count} units)."
        sys.stdout.write(summary.ljust(cols - 1)[: cols - 1] + "\n")
        sys.stdout.flush()
        self.current_stage = None

    def close(self) -> None:
        if self.current_stage is not None:
            self._finish_stage(time.perf_counter())


class CachedNARChunkSolver:
    def __init__(
        self,
        model: YuE2ForCausalLM,
        chunk_tokens: List[int],
        initial_noise: torch.Tensor,
        nar_cond_end: int = 0,
        ref_clean: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.chunk_tokens = chunk_tokens
        self.initial_noise = initial_noise
        self.device = model.model.embed_tokens.weight.device
        self.trunk_dtype = model.trunk_dtype
        self._acc_dtype = (
            torch.float64 if self.device.type != "mps" else torch.float32
        )
        self.ar_length = len(chunk_tokens)
        self.cond_length = (
            min(int(nar_cond_end), self.ar_length)
            if nar_cond_end > 0
            else self.ar_length
        )
        self.nar_length = len(initial_noise) + 2

        with torch.inference_mode(False):
            if ref_clean is not None and ref_clean.shape[0] > 0:
                self.ref_len = int(ref_clean.shape[0])
                rc = _contender_project_to_dtype(
                    ref_clean.detach().to(device=self.device),
                    torch.float32,
                    phase_offset=57,
                )
                rn = _contender_project_to_dtype(
                    initial_noise[: self.ref_len].detach().to(device=self.device),
                    torch.float32,
                    phase_offset=57,
                )
                self.ref_clean: Optional[torch.Tensor] = (
                    rc.clone() if rc.is_inference() else rc
                )
                self.ref_noise: Optional[torch.Tensor] = (
                    rn.clone() if rn.is_inference() else rn
                )
                self._ref_clean_acc: Optional[torch.Tensor] = self.ref_clean.to(
                    dtype=self._acc_dtype
                )
                self._ref_noise_acc: Optional[torch.Tensor] = self.ref_noise.to(
                    dtype=self._acc_dtype
                )
            else:
                self.ref_len = 0
                self.ref_clean = None
                self.ref_noise = None
                self._ref_clean_acc = None
                self._ref_noise_acc = None

            total_span = self.ar_length + self.nar_length
            pos = torch.arange(self.ar_length, total_span, device=self.device)[None]
            cos_t, sin_t = model.model.rotary_emb(pos, seq_len_hint=total_span)
            self.cos = cos_t.clone() if cos_t.is_inference() else cos_t
            self.sin = sin_t.clone() if sin_t.is_inference() else sin_t
            local_ids = torch.arange(self.nar_length, device=self.device).clamp(
                max=model.config.max_latent_frames - 1
            )
            pe_f32 = model.latent_pos_embed.forward_fp32(local_ids)[None]
            self.pos_emb_f32 = pe_f32.clone() if pe_f32.is_inference() else pe_f32
            self.x_nar_buf_f32 = torch.zeros(
                (1, self.nar_length, model.config.latent_dim),
                dtype=torch.float32,
                device=self.device,
            )
            self.kv_buffers: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._prefill_ar()

    def _prefill_ar(self) -> None:
        with torch.inference_mode():
            backbone = self.model.model
            cond_len = self.cond_length
            act_dt = self.trunk_dtype
            tokens_tensor = torch.tensor(
                [self.chunk_tokens[:cond_len]], dtype=torch.long, device=self.device
            )
            positions = torch.arange(cond_len, device=self.device)[None]
            cos, sin = backbone.rotary_emb(positions, seq_len_hint=cond_len)
            emb = backbone.embed_tokens(tokens_tensor)
            x_f32 = emb if emb.dtype == torch.float32 else emb.float()
            total_kv_len = cond_len + self.nar_length
            num_kv_heads = self.model.config.num_key_value_heads
            head_dim = self.model.config.head_dim

            for layer in backbone.layers:
                q, k, v = layer.self_attn.project_qkv(
                    layer.input_layernorm.forward_to_dtype(x_f32, act_dt), cos, sin
                )
                q_tx = q.transpose(1, 2)
                k_tx = k.transpose(1, 2)
                v_tx = v.transpose(1, 2)
                with torch.inference_mode(False):
                    k_buf = torch.empty(
                        (1, num_kv_heads, total_kv_len, head_dim),
                        dtype=act_dt,
                        device=self.device,
                    )
                    v_buf = torch.empty(
                        (1, num_kv_heads, total_kv_len, head_dim),
                        dtype=act_dt,
                        device=self.device,
                    )
                k_buf[:, :, :cond_len, :].copy_(k_tx)
                v_buf[:, :, :cond_len, :].copy_(v_tx)
                self.kv_buffers.append((k_buf, v_buf))
                h = sdpa(q_tx, k_tx, v_tx, is_causal=(cond_len > 1)).transpose(1, 2)
                o_f32 = layer.self_attn.o_proj.forward_fp32(
                    h.reshape(1, cond_len, -1)
                )
                x_f32.add_(o_f32)
                m_f32 = layer.mlp.forward_fp32(
                    layer.post_attention_layernorm.forward_to_dtype(x_f32, act_dt)
                )
                x_f32.add_(m_f32)

    def _project_reference_manifold(self, state: torch.Tensor, tau: float) -> None:
        if (
            self.ref_len > 0
            and self.ref_clean is not None
            and self._ref_clean_acc is not None
            and self._ref_noise_acc is not None
        ):
            t_val = float(tau)
            if t_val <= 0.0:
                if state.dtype == self._acc_dtype:
                    state[: self.ref_len].copy_(self._ref_clean_acc)
                else:
                    state[: self.ref_len].copy_(
                        _contender_project_to_dtype(
                            self.ref_clean, state.dtype, phase_offset=57
                        )
                    )
            else:
                w_clean = 1.0 - t_val
                blended = self._ref_clean_acc * w_clean + self._ref_noise_acc * t_val
                if state.dtype == blended.dtype:
                    state[: self.ref_len].copy_(blended)
                else:
                    state[: self.ref_len].copy_(
                        _contender_project_to_dtype(
                            blended, state.dtype, phase_offset=57
                        )
                    )

    def velocity(self, state: torch.Tensor, raw_t: float) -> torch.Tensor:
        caller_in_inf = torch.is_inference_mode_enabled()
        with torch.inference_mode():
            state_dev = state if state.device == self.device else state.to(self.device)
            state_f32 = _contender_project_to_dtype(
                state_dev, torch.float32, phase_offset=57
            )
            self.x_nar_buf_f32[0, 0].zero_()
            self.x_nar_buf_f32[0, -1].zero_()
            self.x_nar_buf_f32[0, 1:-1].copy_(state_f32)
            x_f32 = self.model.project_latent_to_trunk_fp32(
                self.x_nar_buf_f32, raw_t, self.pos_emb_f32
            )
            cond_len = self.cond_length
            act_dt = self.trunk_dtype

            for layer, (k_buf, v_buf) in zip(self.model.model.layers, self.kv_buffers):
                q, k, v = layer.nar_self_attn.project_qkv(
                    layer.nar_input_layernorm.forward_to_dtype(x_f32, act_dt),
                    self.cos,
                    self.sin,
                )
                q_tx = q.transpose(1, 2)
                k_buf[:, :, cond_len:, :].copy_(k.transpose(1, 2))
                v_buf[:, :, cond_len:, :].copy_(v.transpose(1, 2))
                h = sdpa(q_tx, k_buf, v_buf, is_causal=False).transpose(1, 2)
                o_f32 = layer.nar_self_attn.o_proj.forward_fp32(
                    h.reshape(1, self.nar_length, -1)
                )
                x_f32.add_(o_f32)
                m_f32 = layer.nar_mlp.forward_fp32(
                    layer.nar_pre_mlp_layernorm.forward_to_dtype(x_f32, act_dt)
                )
                x_f32.add_(m_f32)

            vel_f32 = self.model.project_trunk_to_velocity_fp32(x_f32[:, 1:-1, :])[0]
        if not caller_in_inf and vel_f32.is_inference():
            with torch.inference_mode(False):
                return vel_f32.clone()
        return vel_f32

    @staticmethod
    def _logit_clamped(prob: float) -> float:
        p = min(max(float(prob), 1e-9), 1.0 - 1e-9)
        val = math.log(p / (1.0 - p))
        return min(max(val, -20.0), 20.0)

    def solve(
        self,
        steps: int = 32,
        method: str = "midpoint",
        step_callback: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        caller_in_inf = torch.is_inference_mode_enabled()
        with torch.inference_mode():
            acc_dt = self._acc_dtype
            init_dev = self.initial_noise.to(device=self.device)
            if init_dev.dtype == acc_dt:
                state_acc = init_dev.clone()
            elif PhysicsLab.get_mantissa_width(
                init_dev.dtype
            ) > PhysicsLab.get_mantissa_width(acc_dt):
                state_acc = _contender_project_to_dtype(
                    init_dev, acc_dt, phase_offset=57
                ).clone()
            else:
                state_acc = init_dev.to(dtype=acc_dt)

            scratch_acc = (
                torch.empty_like(state_acc) if method != "euler" else state_acc
            )
            inv_steps = 1.0 / float(steps)
            dt = inv_steps
            half_dt = 0.5 * dt

            for step in range(steps):
                t = float(steps - step) * inv_steps
                self._project_reference_manifold(state_acc, t)
                raw_t = self._logit_clamped(t)

                if method == "euler":
                    v = self.velocity(state_acc, raw_t).to(dtype=acc_dt)
                    state_acc.sub_(v, alpha=dt)
                elif method == "heun":
                    v1 = self.velocity(state_acc, raw_t).to(dtype=acc_dt)
                    torch.add(state_acc, v1, alpha=-dt, out=scratch_acc)
                    t_next = max(0.0, float(steps - (step + 1)) * inv_steps)
                    self._project_reference_manifold(scratch_acc, t_next)
                    raw_next = self._logit_clamped(t_next)
                    v2 = self.velocity(scratch_acc, raw_next).to(dtype=acc_dt)
                    v1.add_(v2)
                    state_acc.sub_(v1, alpha=half_dt)
                else:
                    v1 = self.velocity(state_acc, raw_t).to(dtype=acc_dt)
                    torch.add(state_acc, v1, alpha=-half_dt, out=scratch_acc)
                    t_mid = max(0.0, (float(steps - step) - 0.5) * inv_steps)
                    self._project_reference_manifold(scratch_acc, t_mid)
                    raw_mid = self._logit_clamped(t_mid)
                    v2 = self.velocity(scratch_acc, raw_mid).to(dtype=acc_dt)
                    state_acc.sub_(v2, alpha=dt)

                if step_callback is not None:
                    step_callback(step + 1, steps)

            self._project_reference_manifold(state_acc, 0.0)
            out_f32 = _contender_project_to_dtype(
                state_acc, torch.float32, phase_offset=57
            )
            if self.ref_len > 0 and self.ref_clean is not None:
                if out_f32.data_ptr() == self.ref_clean.data_ptr():
                    out_f32 = out_f32.clone()
                out_f32[: self.ref_len].copy_(self.ref_clean)

            result = out_f32.cpu()
            if not torch.isfinite(result).all():
                raise FloatingPointError(
                    "NAR flow-matching integration yielded non-finite values."
                )
        if not caller_in_inf and result.is_inference():
            with torch.inference_mode(False):
                return result.clone()
        return result

    def close(self) -> None:
        self.kv_buffers.clear()
        self.cos = self.sin = self.pos_emb_f32 = self.x_nar_buf_f32 = None
        self.ref_clean = self.ref_noise = None
        self._ref_clean_acc = self._ref_noise_acc = None


def _apply_non_degradative_placement(
    model: YuE2ForCausalLM,
    state_dict: Dict[str, torch.Tensor],
    target_device: torch.device,
    trunk_dtype: torch.dtype,
) -> None:
    with torch.inference_mode(False), torch.no_grad():
        prepared_state: Dict[str, torch.Tensor] = {}
        used_refinery = False
        m_trunk = PhysicsLab.get_mantissa_width(trunk_dtype)

        for idx, (key, raw_tensor) in enumerate(state_dict.items()):
            if key == "latent_pos_embed.pe" or key.endswith("rotary_emb.inv_freq"):
                continue
            tensor = (
                raw_tensor.detach().clone()
                if raw_tensor.is_inference()
                else raw_tensor.detach()
            )
            if not tensor.is_floating_point():
                prepared_state[key] = tensor
                continue

            key_lower = key.lower()
            if _FP8_DTYPES and tensor.dtype in _FP8_DTYPES:
                prepared_state[key] = tensor
            elif (
                key.startswith(_FP32_BRIDGE_PREFIXES)
                or "norm" in key_lower
                or "bias" in key_lower
            ):
                prepared_state[key] = _contender_project_to_dtype(
                    tensor, torch.float32, phase_offset=idx & 63
                )
            elif tensor.dtype == trunk_dtype:
                prepared_state[key] = tensor
            elif PhysicsLab.get_mantissa_width(tensor.dtype) > m_trunk:
                prepared_state[key] = refineTensor(
                    origin_tensor=tensor,
                    target_dtype=trunk_dtype,
                    original_on_disk_dtype=tensor.dtype,
                    target_device=target_device,
                    phase_offset=idx & 63,
                )
                used_refinery = True
            else:
                prepared_state[key] = tensor.to(dtype=trunk_dtype)

        if used_refinery:
            clear_refinery_cache()

        if (
            getattr(model.config, "tie_word_embeddings", False)
            and "lm_head.weight" not in prepared_state
            and "model.embed_tokens.weight" in prepared_state
        ):
            prepared_state["lm_head.weight"] = prepared_state[
                "model.embed_tokens.weight"
            ]

        model.load_state_dict(prepared_state, strict=False, assign=True)
        model.latent_pos_embed.pe = synthesize_audio_pe_fp32(
            model.config.max_latent_frames,
            model.config.hidden_size,
            device="cpu",
        )

        for mod in model.modules():
            if isinstance(mod, RMSNorm) and mod.weight.dtype != torch.float32:
                mod.weight = nn.Parameter(
                    _contender_project_to_dtype(
                        mod.weight.detach(),
                        torch.float32,
                        phase_offset=mod.phase_offset,
                    ),
                    requires_grad=False,
                )

        for name, param in model.named_parameters():
            if param.is_meta:
                raise RuntimeError(
                    f"Incomplete checkpoint load: parameter '{name}' remained on meta device."
                )
        for name, buf in model.named_buffers():
            if buf.is_meta:
                raise RuntimeError(
                    f"Incomplete checkpoint load: buffer '{name}' remained on meta device."
                )

        model.to(device=target_device).eval().requires_grad_(False)
        model.bind_fused_projections()


class MusicEngine:
    def __init__(
        self,
        repo_id: Optional[str] = None,
        vae_repo_id: Optional[str] = None,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        enforce_full_precision_reductions()
        self.repo_id = repo_id or "m-a-p/YuE2-3B"
        self.vae_repo_id = vae_repo_id or "m-a-p/YuE2-Vae"
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.model_path = resolve_model_path(self.repo_id)
        self.vae_path = resolve_model_path(self.vae_repo_id)

        ranks_path = self.model_path / "qwen.ranks.bin"
        if not ranks_path.exists():
            ranks_path = self.model_path / "qwen.tiktoken"
        self.tokenizer = YuE2Tokenizer(ranks_path if ranks_path.exists() else None)

        self.model: Optional[YuE2ForCausalLM] = None
        self.vae: Optional[YuE2VAE] = None
        self._current_offload_state: Optional[bool] = None
        self._vae_has_encoder: bool = False
        self._abc_prefix_cache: Optional[StaticKVCache] = None
        self._abc_cached_tokens: List[int] = []

    def _reset_prefix_cache(self) -> None:
        self._abc_prefix_cache = None
        self._abc_cached_tokens = []

    def _init_components(
        self, cpu_offload: bool, require_vae_encoder: bool = False
    ) -> None:
        enforce_full_precision_reductions()
        need_vae_reload = require_vae_encoder and not self._vae_has_encoder
        if (
            self.model is not None
            and self.vae is not None
            and self._current_offload_state == cpu_offload
            and not need_vae_reload
        ):
            return

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA execution requested but no compatible GPU detected."
            )

        with torch.inference_mode(False):
            if self.model is not None and self._current_offload_state != cpu_offload:
                self._reset_prefix_cache()
                del self.model, self.vae
                self.model = None
                self.vae = None
                clear_yue2_workspaces(self.device)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if self.model is None:
                cfg_file = self.model_path / "config.json"
                if not cfg_file.exists():
                    resolved = resolve_model_path(self.model_path)
                    cfg_file = resolved / "config.json"
                    self.model_path = resolved
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg_dict = json.load(f)
                cfg = YuE2Config(**cfg_dict)
                model_device = torch.device("cpu") if cpu_offload else self.device
                with torch.device("meta"):
                    meta_model = YuE2ForCausalLM(cfg)
                meta_model.latent_pos_embed = AudioPositionEmbedding(
                    cfg.max_latent_frames, cfg.hidden_size
                )
                sf_shards = sorted(list(self.model_path.glob("*.safetensors")))
                if not sf_shards:
                    raise FileNotFoundError(
                        f"No SafeTensors shards found in {self.model_path}"
                    )
                state: Dict[str, torch.Tensor] = {}
                for shard in sf_shards:
                    state.update(load_safetensors(str(shard), device="cpu"))
                _apply_non_degradative_placement(
                    meta_model,
                    state,
                    target_device=model_device,
                    trunk_dtype=self.dtype,
                )
                del state
                self.model = meta_model

            if self.vae is None or need_vae_reload:
                if self.vae is not None:
                    del self.vae
                    self.vae = None
                    gc.collect()
                vae_device = torch.device("cpu") if cpu_offload else self.device
                self.vae = YuE2VAE.from_pretrained(
                    self.vae_path,
                    decoder_only=not require_vae_encoder,
                    device=vae_device,
                )
                self._vae_has_encoder = self.vae.has_encoder

            self._current_offload_state = cpu_offload

    def _prefill_with_prefix_splicing(
        self,
        prefix_tokens: List[int],
        max_new_tokens: int,
        cpu_offload: bool,
        vocab_slice: Optional[Tuple[int, int]] = None,
        min_cache_capacity: int = 8192,
    ) -> Tuple[StaticKVCache, torch.Tensor, int]:
        assert self.model is not None
        req_cap = len(prefix_tokens) + max_new_tokens + 16
        trunk_dt = self.model.trunk_dtype
        effective_slice = (
            vocab_slice
            if vocab_slice is not None
            else (0, int(self.model.config.vocab_size))
        )

        if cpu_offload:
            self._reset_prefix_cache()
            cache = StaticKVCache(
                num_layers=self.model.config.num_hidden_layers,
                batch_size=1,
                num_kv_heads=self.model.config.num_key_value_heads,
                max_seq_len=req_cap,
                head_dim=self.model.config.head_dim,
                dtype=trunk_dt,
                device=self.device,
            )
            pre_in = torch.tensor([prefix_tokens], dtype=torch.long, device=self.device)
            logits = self.model.forward_vocab_slice_fp32(
                input_ids=pre_in,
                vocab_slice=effective_slice,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )[:, -1, :]
            return cache, logits, 0

        if self._abc_prefix_cache is not None and (
            self._abc_prefix_cache.device != self.device
            or self._abc_prefix_cache.dtype != trunk_dt
        ):
            self._reset_prefix_cache()

        max_ctx = int(self.model.config.max_position_embeddings)
        desired_cap = max(req_cap, int(min_cache_capacity))
        aligned_cap = ((desired_cap + 255) // 256) * 256
        if req_cap <= max_ctx:
            aligned_cap = min(aligned_cap, max_ctx)

        if self._abc_prefix_cache is None:
            self._abc_prefix_cache = StaticKVCache(
                num_layers=self.model.config.num_hidden_layers,
                batch_size=1,
                num_kv_heads=self.model.config.num_key_value_heads,
                max_seq_len=max(req_cap, aligned_cap),
                head_dim=self.model.config.head_dim,
                dtype=trunk_dt,
                device=self.device,
            )
            self._abc_cached_tokens = []

        cache = self._abc_prefix_cache
        shared = self.tokenizer.find_common_prefix(
            self._abc_cached_tokens, prefix_tokens
        )
        reused = min(shared, max(0, len(prefix_tokens) - 1), cache.get_seq_length())
        if cache.max_seq_len < req_cap:
            cache.ensure_capacity(max(req_cap, aligned_cap), keep_tokens=reused)
        else:
            cache.truncate(reused)

        delta_tokens = prefix_tokens[reused:]
        delta_in = torch.tensor([delta_tokens], dtype=torch.long, device=self.device)
        logits = self.model.forward_vocab_slice_fp32(
            input_ids=delta_in,
            vocab_slice=effective_slice,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )[:, -1, :]
        self._abc_cached_tokens = list(prefix_tokens)
        return cache, logits, reused

    @torch.inference_mode()
    def synthesize(
        self,
        request: GenerationRequest,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> GenerationResponse:
        request.validate()
        has_ref_audio = bool(
            request.reference_audio_path and request.reference_audio_path.strip()
        )
        self._init_components(
            bool(request.cpu_offload), require_vae_encoder=has_ref_audio
        )
        assert self.model is not None and self.vae is not None

        defaults = get_active_engine_defaults()
        reporter: Optional[EngineProgressReporter] = None
        if progress_callback is None:
            reporter = EngineProgressReporter()
            effective_callback = reporter
        else:
            effective_callback = progress_callback

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.perf_counter()
        explicit_seed = (
            request.seed
            if (request.seed is not None and request.seed >= 0)
            else random.randint(100000, 99999999)
        )
        random.seed(explicit_seed)
        np.random.seed(explicit_seed % (2**32))
        torch.manual_seed(explicit_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(explicit_seed)
        generator = torch.Generator(device=self.device).manual_seed(explicit_seed)

        ref_latents_td: Optional[torch.Tensor] = None
        ref_frames_count: int = 0
        if has_ref_audio and request.reference_audio_path is not None:
            ref_path = Path(request.reference_audio_path.strip())
            if not ref_path.is_absolute():
                ref_path = ROOT_DIR / ref_path
            ref_wave = load_reference_stereo_audio(
                ref_path, max_duration=request.reference_duration
            )
            if request.cpu_offload:
                self.vae.to(device=self.device, dtype=torch.float32)
            ref_bdt = self.vae.encode_tiled(
                ref_wave,
                core_frames=request.vae_core_frames or defaults["vae_core_frames"],
                halo_frames=request.vae_halo_frames or defaults["vae_halo_frames"],
                deterministic=True,
                output_device="cpu",
            )
            ref_latents_td = (
                ref_bdt.transpose(-1, -2)[0].to(dtype=torch.float32).contiguous()
            )
            ref_frames_count = int(ref_latents_td.shape[0])
            if request.cpu_offload:
                self.vae.to("cpu")
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()

        cot_mode = request.cot or defaults["cot"]
        instruction = INSTRUCTIONS.get(cot_mode, INSTRUCTIONS["full"])
        full_text = request.compile_full_text()
        style_text = request.compile_style()
        lyrics_text = request.sanitize_lyrics()
        base_tokens = [EOD] + self.tokenizer.encode_ordinary(full_text)
        max_ctx = int(self.model.config.max_position_embeddings)

        resolved_cfg = float(
            request.cfg_scale
            if request.cfg_scale is not None
            else defaults["cfg_scale"]
        )
        apply_cfg = abs(resolved_cfg - 1.0) > 1e-4

        remaining_duration = max(
            1.0, float(request.audio_duration) - (ref_frames_count / 25.0)
        )
        planned_semantic_tokens = min(int(round(remaining_duration * 25.0)), 15000)

        if request.cpu_offload:
            self.model.to(device=self.device)

        abc_output_text: Optional[str] = None
        abc_ids: List[int] = []
        abc_entropies: List[float] = []
        prefix_reused_tokens: int = 0

        if cot_mode != "off":
            if request.abc is not None and request.abc.strip():
                abc_output_text = request.abc.strip()
                abc_ids = self.tokenizer.encode_ordinary(abc_output_text)
            else:
                abc_prefix = base_tokens + [ABC_START]
                avail_abc = max_ctx - len(abc_prefix) - 32
                if avail_abc < 32:
                    raise ValueError(
                        "Prompt exceeds context window capacity for symbolic CoT generation."
                    )
                max_abc_tokens = min(4096, avail_abc)
                abc_horizon_hint = (
                    8192
                    if apply_cfg
                    else max(
                        8192,
                        min(
                            max_ctx,
                            len(abc_prefix) + 1024 + planned_semantic_tokens + 32,
                        ),
                    )
                )
                cache_abc, logits, prefix_reused_tokens = (
                    self._prefill_with_prefix_splicing(
                        abc_prefix,
                        max_new_tokens=max_abc_tokens,
                        cpu_offload=bool(request.cpu_offload),
                        vocab_slice=ABC_VOCAB_SLICE,
                        min_cache_capacity=abc_horizon_hint,
                    )
                )
                history: List[int] = []
                abc_temp = (
                    request.abc_temperature
                    if request.abc_temperature is not None
                    else defaults["abc_temperature"]
                )
                abc_top_p = (
                    request.abc_top_p
                    if request.abc_top_p is not None
                    else defaults["abc_top_p"]
                )
                abc_top_k = (
                    request.abc_top_k
                    if request.abc_top_k is not None
                    else defaults["abc_top_k"]
                )
                abc_pen = (
                    request.abc_repetition_penalty
                    if request.abc_repetition_penalty is not None
                    else defaults["abc_repetition_penalty"]
                )
                abc_win = (
                    request.abc_penalty_window
                    if request.abc_penalty_window is not None
                    else defaults["abc_penalty_window"]
                )
                abc_pen_state = _RollingPenaltyState(
                    window=abc_win,
                    vocab_offset=ABC_VOCAB_SLICE[0],
                    support_len=ABC_VOCAB_SLICE[1] - ABC_VOCAB_SLICE[0],
                    penalty=abc_pen,
                    device=self.device,
                )
                step_in_abc = torch.empty(
                    (1, 1), dtype=torch.long, device=self.device
                )

                for step in range(max_abc_tokens):
                    token, step_ent = sample_categorical_distribution_with_entropy(
                        logits=logits,
                        temperature=abc_temp,
                        top_p=abc_top_p,
                        top_k=abc_top_k,
                        history=history,
                        step=step,
                        min_tokens=32,
                        penalty=abc_pen,
                        penalty_window=abc_win,
                        phase="abc",
                        generator=generator,
                        vocab_offset=ABC_VOCAB_SLICE[0],
                        penalty_state=abc_pen_state,
                    )
                    abc_entropies.append(step_ent)
                    if token == ABC_END:
                        break
                    history.append(token)
                    abc_pen_state.push(token)
                    effective_callback("abc", step + 1, max_abc_tokens)
                    step_in_abc.fill_(token)
                    logits = self.model.forward_vocab_slice_fp32(
                        input_ids=step_in_abc,
                        vocab_slice=ABC_VOCAB_SLICE,
                        past_key_values=cache_abc,
                        use_cache=True,
                        logits_to_keep=1,
                    )[:, -1, :]

                abc_ids = history
                abc_output_text = self.tokenizer.decode(abc_ids)
                if not request.cpu_offload:
                    self._abc_cached_tokens = abc_prefix + abc_ids
                else:
                    del cache_abc

        if cot_mode == "off":
            song_prefix = base_tokens + [ABC_START, ABC_END, MUSIC_START]
        else:
            song_prefix = base_tokens + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]

        semantic_tokens: List[int] = []
        semantic_entropies: List[float] = []
        trunk_dt = self.model.trunk_dtype

        sem_temp = (
            request.temperature
            if request.temperature is not None
            else defaults["temperature"]
        )
        sem_top_p = (
            request.top_p if request.top_p is not None else defaults["top_p"]
        )
        sem_top_k = (
            request.top_k if request.top_k is not None else defaults["top_k"]
        )
        sem_pen = (
            request.repetition_penalty
            if request.repetition_penalty is not None
            else defaults["repetition_penalty"]
        )
        sem_win = (
            request.penalty_window
            if request.penalty_window is not None
            else defaults["penalty_window"]
        )
        sem_pen_state = _RollingPenaltyState(
            window=sem_win,
            vocab_offset=SEMANTIC_VOCAB_SLICE[0],
            support_len=SEMANTIC_VOCAB_SLICE[1] - SEMANTIC_VOCAB_SLICE[0],
            penalty=sem_pen,
            device=self.device,
        )

        if apply_cfg:
            neg_base = [EOD] + self.tokenizer.encode_ordinary(instruction)
            if cot_mode == "off":
                neg_prefix = neg_base + [ABC_START, ABC_END, MUSIC_START]
            else:
                neg_prefix = neg_base + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]

            max_pref_len = max(len(song_prefix), len(neg_prefix))
            avail_sem = max_ctx - max_pref_len - 16
            if avail_sem < 1:
                raise ValueError(
                    "Prompt and symbolic score exceed model context window."
                )
            max_semantic_tokens = min(planned_semantic_tokens, avail_sem)

            pad_pos = max_pref_len - len(song_prefix)
            pad_neg = max_pref_len - len(neg_prefix)
            batched_prefix = torch.tensor(
                [
                    [EOD] * pad_pos + song_prefix,
                    [EOD] * pad_neg + neg_prefix,
                ],
                dtype=torch.long,
                device=self.device,
            )
            attn_mask_full = torch.ones(
                (2, max_pref_len + max_semantic_tokens + 16),
                dtype=torch.bool,
                device=self.device,
            )
            if pad_pos > 0:
                attn_mask_full[0, :pad_pos] = False
            if pad_neg > 0:
                attn_mask_full[1, :pad_neg] = False

            cache_cfg = StaticKVCache(
                num_layers=self.model.config.num_hidden_layers,
                batch_size=2,
                num_kv_heads=self.model.config.num_key_value_heads,
                max_seq_len=max_pref_len + max_semantic_tokens + 16,
                head_dim=self.model.config.head_dim,
                dtype=trunk_dt,
                device=self.device,
            )
            cur_len = max_pref_len
            logits_pair = self.model.forward_vocab_slice_fp32(
                input_ids=batched_prefix,
                vocab_slice=SEMANTIC_VOCAB_SLICE,
                attention_mask=attn_mask_full[:, :cur_len],
                past_key_values=cache_cfg,
                use_cache=True,
                logits_to_keep=1,
            )[:, -1, :]

            pos_rope_offset = len(song_prefix)
            neg_rope_offset = len(neg_prefix)
            step_in_cfg = torch.empty((2, 1), dtype=torch.long, device=self.device)
            step_pos_cfg = torch.tensor(
                [[pos_rope_offset - 1], [neg_rope_offset - 1]],
                dtype=torch.long,
                device=self.device,
            )

            for step in range(max_semantic_tokens):
                if logits_pair.device.type == "mps":
                    pair_f64 = logits_pair.to(device="cpu", dtype=torch.float64)
                else:
                    pair_f64 = logits_pair.to(dtype=torch.float64)
                logits_pos_f64 = pair_f64[0]
                logits_neg_f64 = pair_f64[1]
                effective_logits_f64 = logits_neg_f64 + resolved_cfg * (
                    logits_pos_f64 - logits_neg_f64
                )
                token, step_ent = sample_categorical_distribution_with_entropy(
                    logits=effective_logits_f64,
                    temperature=sem_temp,
                    top_p=sem_top_p,
                    top_k=sem_top_k,
                    history=semantic_tokens,
                    step=step,
                    min_tokens=min(200, max_semantic_tokens),
                    penalty=sem_pen,
                    penalty_window=sem_win,
                    phase="semantic",
                    generator=generator,
                    vocab_offset=SEMANTIC_VOCAB_SLICE[0],
                    penalty_state=sem_pen_state,
                )
                semantic_entropies.append(step_ent)
                if token == MUSIC_END:
                    break
                semantic_tokens.append(token)
                sem_pen_state.push(token)
                effective_callback("semantic", step + 1, max_semantic_tokens)
                step_in_cfg.fill_(token)
                step_pos_cfg.add_(1)
                cur_len += 1
                logits_pair = self.model.forward_vocab_slice_fp32(
                    input_ids=step_in_cfg,
                    vocab_slice=SEMANTIC_VOCAB_SLICE,
                    attention_mask=attn_mask_full[:, :cur_len],
                    position_ids=step_pos_cfg,
                    past_key_values=cache_cfg,
                    use_cache=True,
                    logits_to_keep=1,
                    seq_len_hint=cur_len,
                )[:, -1, :]
            del cache_cfg
        else:
            avail_sem = max_ctx - len(song_prefix) - 16
            if avail_sem < 1:
                raise ValueError(
                    "Prompt and symbolic score exceed model context window."
                )
            max_semantic_tokens = min(planned_semantic_tokens, avail_sem)
            cache_pos, logits_pos, reused_sem = self._prefill_with_prefix_splicing(
                song_prefix,
                max_new_tokens=max_semantic_tokens,
                cpu_offload=bool(request.cpu_offload),
                vocab_slice=SEMANTIC_VOCAB_SLICE,
            )
            prefix_reused_tokens = max(prefix_reused_tokens, reused_sem)
            step_in_sem = torch.empty((1, 1), dtype=torch.long, device=self.device)
            for step in range(max_semantic_tokens):
                token, step_ent = sample_categorical_distribution_with_entropy(
                    logits=logits_pos,
                    temperature=sem_temp,
                    top_p=sem_top_p,
                    top_k=sem_top_k,
                    history=semantic_tokens,
                    step=step,
                    min_tokens=min(200, max_semantic_tokens),
                    penalty=sem_pen,
                    penalty_window=sem_win,
                    phase="semantic",
                    generator=generator,
                    vocab_offset=SEMANTIC_VOCAB_SLICE[0],
                    penalty_state=sem_pen_state,
                )
                semantic_entropies.append(step_ent)
                if token == MUSIC_END:
                    break
                semantic_tokens.append(token)
                sem_pen_state.push(token)
                effective_callback("semantic", step + 1, max_semantic_tokens)
                step_in_sem.fill_(token)
                logits_pos = self.model.forward_vocab_slice_fp32(
                    input_ids=step_in_sem,
                    vocab_slice=SEMANTIC_VOCAB_SLICE,
                    past_key_values=cache_pos,
                    use_cache=True,
                    logits_to_keep=1,
                )[:, -1, :]

            if not request.cpu_offload:
                cache_pos.truncate(len(song_prefix))
                self._abc_cached_tokens = list(song_prefix)
            else:
                del cache_pos

        raw_codec = [t - CODEC_OFFSET for t in semantic_tokens]
        if not raw_codec:
            raise RuntimeError(
                "Autoregressive generation terminated with zero acoustic codec frames."
            )

        nar_text_only = (
            bool(request.nar_text_only)
            if request.nar_text_only is not None
            else bool(defaults["nar_text_only"])
        )
        chunks = partition_song_chunks(
            song_prefix,
            raw_codec,
            explicit_seed,
            context_len=max_ctx,
            nar_text_only=nar_text_only,
            ref_latents_td=ref_latents_td,
        )
        first_ref = cast(Optional[torch.Tensor], chunks[0]["ref_clean"])
        if first_ref is not None:
            ref_frames_count = int(first_ref.shape[0])

        total_chunks = len(chunks)
        ode_steps = (
            request.num_inference_steps
            if request.num_inference_steps is not None
            else defaults["num_inference_steps"]
        )
        ode_method = (
            request.ode_method
            if request.ode_method is not None
            else defaults["ode_method"]
        )

        latent_pieces: List[torch.Tensor] = []
        for c_idx, chk in enumerate(chunks):
            chk_tokens = cast(List[int], chk["tokens"])
            chk_noise = cast(torch.Tensor, chk["noise"])
            chk_cond_end = int(cast(int, chk["nar_cond_end"]))
            chk_ref_clean = cast(Optional[torch.Tensor], chk["ref_clean"])
            solver = CachedNARChunkSolver(
                self.model,
                chk_tokens,
                chk_noise,
                nar_cond_end=chk_cond_end,
                ref_clean=chk_ref_clean,
            )

            def cb_step(cur: int, tot: int) -> None:
                effective_callback("nar", c_idx * tot + cur, total_chunks * tot)

            chunk_latent = solver.solve(
                steps=ode_steps, method=ode_method, step_callback=cb_step
            )
            solver.close()
            del solver
            latent_pieces.append(chunk_latent)

        latents_fp32 = (
            torch.cat(latent_pieces, dim=0)
            .unsqueeze(0)
            .transpose(-1, -2)
            .to(dtype=torch.float32)
            .contiguous()
        )

        if request.cpu_offload:
            self.model.to("cpu")
            clear_yue2_workspaces(self.device)
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
            self.vae.to(device=self.device, dtype=torch.float32)

        def cb_vae(cur: int, tot: int) -> None:
            effective_callback("vae", cur, tot)

        audio_tensor = self.vae.decode_tiled(
            latents_fp32,
            core_frames=request.vae_core_frames or defaults["vae_core_frames"],
            halo_frames=request.vae_halo_frames or defaults["vae_halo_frames"],
            output_device="cpu",
            on_progress=cb_vae,
        )

        if request.cpu_offload:
            self.vae.to("cpu")
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

        if reporter is not None:
            reporter.close()

        declick_enabled = (
            request.apply_declick
            if request.apply_declick is not None
            else defaults["apply_declick"]
        )
        if declick_enabled:
            audio_tensor = apply_sub_millisecond_declick(
                audio_tensor, fade_samples=512
            )

        elapsed_time = time.perf_counter() - start_time
        peak_vram_gb = 0.0
        if self.device.type == "cuda" and torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)

        audio_f64 = (
            audio_tensor.to(device="cpu", dtype=torch.float64)
            if audio_tensor.device.type == "mps"
            else audio_tensor.to(dtype=torch.float64)
        )
        peak_val = float(torch.max(torch.abs(audio_f64)).item())
        rms_val = float(torch.sqrt(torch.mean(audio_f64.pow(2))).item())
        peak_dbfs = 20.0 * math.log10(max(peak_val, 1e-12))
        rms_dbfs = 20.0 * math.log10(max(rms_val, 1e-12))
        crest_factor_db = peak_dbfs - rms_dbfs
        del audio_f64

        abc_mean_ent = (
            float(sum(abc_entropies) / len(abc_entropies)) if abc_entropies else 0.0
        )
        abc_ppl = float(2.0 ** min(abc_mean_ent, 30.0)) if abc_entropies else 1.0
        sem_mean_ent = (
            float(sum(semantic_entropies) / len(semantic_entropies))
            if semantic_entropies
            else 0.0
        )
        sem_ppl = (
            float(2.0 ** min(sem_mean_ent, 30.0)) if semantic_entropies else 1.0
        )

        audio_tc = audio_tensor.detach().transpose(-1, -2)[0].contiguous()
        audio_data = np.ascontiguousarray(audio_tc.cpu().numpy(), dtype=np.float32)

        out_path = Path(request.output_path)
        if not out_path.is_absolute():
            out_path = ROOT_DIR / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio_data, TARGET_SAMPLE_RATE, subtype="FLOAT")

        total_samples = int(audio_data.shape[0])
        actual_duration = total_samples / float(TARGET_SAMPLE_RATE)
        rtf = elapsed_time / max(actual_duration, 1e-6)

        return GenerationResponse(
            output_path=str(out_path),
            sample_rate=TARGET_SAMPLE_RATE,
            duration_seconds=actual_duration,
            total_samples=total_samples,
            generation_time_seconds=elapsed_time,
            real_time_factor=rtf,
            peak_vram_gb=peak_vram_gb,
            cpu_offload_active=bool(self._current_offload_state),
            cot_mode_used=cot_mode,
            abc_token_count=len(abc_ids),
            semantic_token_count=len(semantic_tokens),
            ode_steps_used=ode_steps,
            ode_method_used=ode_method,
            nar_text_only_used=nar_text_only,
            cfg_scale_used=resolved_cfg,
            effective_prompt=full_text,
            effective_style=style_text,
            effective_lyrics=lyrics_text,
            abc_notation=abc_output_text,
            declick_applied=bool(declick_enabled),
            peak_linear=peak_val,
            peak_dbfs=peak_dbfs,
            rms_dbfs=rms_dbfs,
            crest_factor_db=crest_factor_db,
            is_instrumental_used=request.is_instrumental,
            instrumental_branch_used=request.instrumental_branch,
            prefix_tokens_reused=prefix_reused_tokens,
            reference_frames_encoded=ref_frames_count,
            swiss5_fast_tables_active=self.tokenizer.has_fast_tables,
            abc_mean_entropy_bits=abc_mean_ent,
            abc_perplexity=abc_ppl,
            semantic_mean_entropy_bits=sem_mean_ent,
            semantic_perplexity=sem_ppl,
        )

    def generate(
        self,
        request: GenerationRequest,
        duration: Optional[float] = None,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Tuple[torch.Tensor, int]:
        if duration is not None:
            request.audio_duration = duration
        if seed is not None:
            request.seed = seed

        def cb(stage: str, cur: int, tot: int):
            if progress_callback is not None:
                pct = int((cur / max(tot, 1)) * 100)
                progress_callback(pct, f"Doodle Synthesizing {stage} ({cur}/{tot})...")

        resp = self.synthesize(
            request, progress_callback=cb if progress_callback else None
        )
        data, sr = sf.read(resp.output_path, dtype="float32", always_2d=True)
        with torch.inference_mode(False):
            tensor = (
                torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
                .transpose(-1, -2)
                .contiguous()
            )
        return tensor, sr

    @torch.inference_mode()
    def warmup(self) -> None:
        self._init_components(cpu_offload=False, require_vae_encoder=False)
        assert self.model is not None and self.vae is not None
        t0 = time.perf_counter()

        sys.stdout.write("  [1/3] Warmup: Autoregressive Backbone Prefill Pass...\n")
        sys.stdout.flush()
        dummy_ar = torch.tensor(
            [[EOD, ABC_START, 100, 200, ABC_END, MUSIC_START]],
            dtype=torch.long,
            device=self.device,
        )
        _ = self.model.forward_vocab_slice_fp32(
            input_ids=dummy_ar,
            vocab_slice=SEMANTIC_VOCAB_SLICE,
            use_cache=False,
            logits_to_keep=1,
        )

        sys.stdout.write(
            "  [2/3] Warmup: Non-Autoregressive MoT Flow-Matching Velocity...\n"
        )
        sys.stdout.flush()
        t_lat = 4
        ar_mask = torch.tensor(
            [[True] * 6 + [False] * 6], device=self.device, dtype=torch.bool
        )
        nar_mask = torch.tensor(
            [[False] * 6 + [True] * 6], device=self.device, dtype=torch.bool
        )
        nar_content_mask = torch.tensor(
            [[False] * 7 + [True] * t_lat + [False] * 1],
            device=self.device,
            dtype=torch.bool,
        )
        tokens = torch.tensor(
            [
                [
                    EOD,
                    ABC_START,
                    100,
                    200,
                    ABC_END,
                    MUSIC_START,
                    LATENT_START,
                    0,
                    0,
                    0,
                    0,
                    LATENT_END,
                ]
            ],
            dtype=torch.long,
            device=self.device,
        )
        x_t = torch.randn((t_lat, 64), device=self.device, dtype=torch.float32)
        _ = self.model.nar_velocity(
            tokens, ar_mask, nar_mask, nar_content_mask, x_t, 0.5
        )

        sys.stdout.write(
            "  [3/3] Warmup: Oobleck Convolutional VAE Tiled Decoder...\n"
        )
        sys.stdout.flush()
        dummy_latents = torch.randn(
            (1, 64, 25), device=self.device, dtype=torch.float32
        )
        _ = self.vae.decode_tiled(
            dummy_latents, core_frames=16, halo_frames=16, output_device="cpu"
        )

        sys.stdout.write(
            f"  [+] Engine Warmup Completed in {time.perf_counter() - t0:.2f}s.\n"
        )
        sys.stdout.flush()