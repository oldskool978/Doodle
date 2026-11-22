#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple, Union
import warnings

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

sys.modules["torchvision"] = None
sys.modules["torchvision.transforms"] = None
sys.modules["torchvision.io"] = None

import numpy as np
from safetensors.torch import (
    load as load_safetensors_bytes,
    save as save_safetensors_bytes,
)
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.utils._python_dispatch import TorchDispatchMode

from engine import (
    ABC_END,
    ABC_START,
    ABC_VOCAB_SLICE,
    CODEC_OFFSET,
    EOD,
    LATENT_END,
    LATENT_START,
    MUSIC_END,
    MUSIC_START,
    SEMANTIC_VOCAB_SLICE,
    TMP_DIR,
    CachedNARChunkSolver,
    _RollingPenaltyState,
    _apply_non_degradative_placement,
    apply_sub_millisecond_declick,
    apply_windowed_penalty,
    load_reference_stereo_audio,
    resample_polyphase_kaiser,
    resolve_model_path,
    sample_categorical_distribution_with_entropy,
)
import models.modeling_vae as modeling_vae_mod
from models.modeling_vae import (
    IEEE754FP32Guard,
    SnakeBeta,
    SovereignConv1d,
    SovereignConvTranspose1d,
    WNConv1d,
    WNConvTranspose1d,
    YuE2VAE,
    YuE2VAEConfig,
    enforce_full_precision_reductions,
    fold_weight_norm_dict,
    snake_beta_fused,
)
from models.modeling_yue2 import (
    Attention,
    AudioPositionEmbedding,
    MLP,
    QuantSafeLinear,
    RMSNorm,
    StaticKVCache,
    YuE2Config,
    YuE2ForCausalLM,
    _project_narrowing,
    clear_yue2_workspaces,
    sdpa,
    synthesize_audio_pe_fp32,
)
from models.tokenizer import (
    YuE2Tokenizer,
    find_common_prefix,
    is_swiss5_fast_container,
)
from refinery import (
    FP8_DTYPES,
    MIN_CONDUCTOR_ABI_VERSION,
    TheContender,
    TheRefinery,
    _exact_lattice_brackets_fp32,
    _get_fp8_sorted_lut,
    _resolve_dtype_limits,
    clear_refinery_cache,
    contender_cast,
    get_conductor_context,
    get_conductor_info,
    get_conductor_module,
    get_doubly_balanced_bayer_8x8,
    refineTensor,
    sample_bayer_lattice,
    warmup_conductor_gpu,
)
from schema import GenerationResponse


def resolve_test_device(preferred: str = "auto") -> torch.device:
    if preferred == "cuda" or (preferred == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if preferred == "mps" or (
        preferred == "auto"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def _f64_device(device: torch.device) -> torch.device:
    return torch.device("cpu") if device.type == "mps" else device


def _to_f64(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type == "mps":
        return tensor.to(device="cpu", dtype=torch.float64)
    return tensor.to(dtype=torch.float64)


class _ZeroConvDispatchAudit(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.violations: List[str] = []
        self.op_count: int = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.op_count += 1
        op_name = str(func).lower()
        if "convolution" in op_name or "miopen" in op_name:
            self.violations.append(str(func))
        return func(*args, **(kwargs or {}))


def audit_gate_1_tokenizer_and_simd_prefix() -> None:
    print("\n--- Gate 1: Sovereign TikTalkin SWISS5 & AVX2 Prefix-Splicing Audit ---")
    for length in range(1, 65):
        seq_a = list(range(500, 500 + length))
        assert find_common_prefix(seq_a, list(seq_a)) == length
        for diff_at in range(length):
            seq_b = list(seq_a)
            seq_b[diff_at] = -1
            got = find_common_prefix(seq_a, seq_b)
            assert got == diff_at, (
                f"Prefix lane mismatch at len={length}, idx={diff_at}: got {got}"
            )
    assert find_common_prefix([], [1, 2, 3]) == 0
    print("  [+] AVX2 Prefix Splicing       : Lengths 1..64 lane torture bit-exact.")

    backbone_cand = resolve_model_path("m-a-p/YuE2-3B")
    vocab_hint = backbone_cand if backbone_cand.exists() else None
    try:
        tok = YuE2Tokenizer(vocab_hint)
    except FileNotFoundError:
        print(
            "  [i] Vocabulary Container       : Not yet hydrated locally; run download_model.py to stage qwen.ranks.bin."
        )
        return

    assert is_swiss5_fast_container(tok.ranks_path), (
        f"Expected SWISS5 zero-init container at {tok.ranks_path}"
    )
    assert tok.has_fast_tables, "SWISS5 fast tables inactive on YuE2Tokenizer"

    sample = (
        "X:1\nT:Convergence\nM:4/4\nL:1/8\nK:Fm\n"
        "|: [CEG]2 ^^C>D [D_FA]4 | \u591c\u7a7a\u4e2d\u6700\u4eae\u7684\u661f \U0001d11e :|"
    )
    ids = tok.encode_ordinary(sample)
    assert tok.decode(ids) == sample, "UTF-8 / ABC roundtrip mismatch"

    structured = tok.encode_yue2_sequence(
        prompt_text="Generate music.\n[Tags]\nNeo-Soul, 96 bpm\n[Lyrics]\n[verse]\nTest\n",
        abc_text="X:1\nK:C\nCDEF|",
        codec_tokens=[CODEC_OFFSET + 10, CODEC_OFFSET + 20],
        latent_frames=128,
    )
    assert structured[0] == EOD and structured.count(184623) == 128
    tok.close()
    print(
        f"  [+] SWISS5 Zero-Init Container : {tok.ranks_path.name} (Fast Tables ACTIVE, CJK/SMP & RLE Verified)"
    )


def _eval_resampler_snr_db(cand: torch.Tensor, ref_f64: torch.Tensor, guard: int) -> float:
    cand_inner = _to_f64(cand[..., guard:-guard])
    ref_inner = ref_f64[..., guard:-guard]
    err = cand_inner - ref_inner
    sig_pow = float(torch.mean(ref_inner.pow(2)).item())
    err_pow = max(float(torch.mean(err.pow(2)).item()), 1e-20)
    return 10.0 * math.log10(sig_pow / err_pow)


def audit_gate_2_polyphase_kaiser_resampler(device: torch.device) -> None:
    print(
        "\n--- Gate 2: Multidimensional (*lead, T) Float64 Polyphase & Continuous-Phase Kaiser Audit ---"
    )
    f64_dev = _f64_device(device)
    if device.type == "cuda":
        warm_a = torch.ones((2, 4, 4), dtype=torch.float64, device=device)
        warm_b = torch.ones((4, 4), dtype=torch.float64, device=device)
        _ = torch.matmul(warm_a, warm_b)
        del warm_a, warm_b
        torch.cuda.synchronize(device)

    dur = 0.25
    guard = 256
    for label, orig_sr, target_sr in [
        ("Rational Polyphase (44.1k -> 48k, L=160, M=147)", 44100, 48000),
        ("Rational Decimate  (96.0k -> 44.1k, L=147, M=320)", 96000, 44100),
        ("Continuous-Phase   (44099 -> 48k, Prime Ratio)  ", 44099, 48000),
    ]:
        n_in = int(round(dur * orig_sr))
        t_in = torch.arange(n_in, dtype=torch.float64, device=f64_dev) / float(orig_sr)
        c0_in = 0.20 + 0.45 * torch.sin(2.0 * math.pi * 997.0 * t_in)
        c1_in = -0.15 + 0.45 * torch.cos(2.0 * math.pi * 6131.0 * t_in)
        c2_in = 0.10 + 0.40 * torch.sin(2.0 * math.pi * 2311.0 * t_in + 0.3)
        c3_in = -0.05 + 0.40 * torch.cos(2.0 * math.pi * 4177.0 * t_in - 0.5)

        in_1d = contender_cast(c0_in, torch.float32).to(device=device)
        in_2d = contender_cast(torch.stack([c0_in, c1_in], dim=0), torch.float32).to(
            device=device
        )
        in_3d = contender_cast(
            torch.stack(
                [torch.stack([c0_in, c1_in], dim=0), torch.stack([c2_in, c3_in], dim=0)],
                dim=0,
            ),
            torch.float32,
        ).to(device=device)

        out_1d = resample_polyphase_kaiser(in_1d, orig_sr=orig_sr, target_sr=target_sr)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out_2d = resample_polyphase_kaiser(in_2d, orig_sr=orig_sr, target_sr=target_sr)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ms = (time.perf_counter() - t0) * 1000.0
        out_3d = resample_polyphase_kaiser(in_3d, orig_sr=orig_sr, target_sr=target_sr)

        n_out = int(out_2d.shape[-1])
        assert out_1d.shape == (n_out,) and out_1d.dtype == torch.float32
        assert out_2d.shape == (2, n_out) and out_2d.dtype == torch.float32
        assert out_3d.shape == (2, 2, n_out) and out_3d.dtype == torch.float32

        t_out = torch.arange(n_out, dtype=torch.float64, device=f64_dev) / float(
            target_sr
        )
        r0 = 0.20 + 0.45 * torch.sin(2.0 * math.pi * 997.0 * t_out)
        r1 = -0.15 + 0.45 * torch.cos(2.0 * math.pi * 6131.0 * t_out)
        r2 = 0.10 + 0.40 * torch.sin(2.0 * math.pi * 2311.0 * t_out + 0.3)
        r3 = -0.05 + 0.40 * torch.cos(2.0 * math.pi * 4177.0 * t_out - 0.5)
        ref_2d = torch.stack([r0, r1], dim=0)
        ref_3d = torch.stack([ref_2d, torch.stack([r2, r3], dim=0)], dim=0)

        snr_1d = _eval_resampler_snr_db(out_1d, r0, guard)
        snr_2d = _eval_resampler_snr_db(out_2d, ref_2d, guard)
        snr_3d = _eval_resampler_snr_db(out_3d, ref_3d, guard)
        min_snr = min(snr_1d, snr_2d, snr_3d)
        assert min_snr > 80.0, (
            f"{label} passband SNR degraded: 1D={snr_1d:.2f}, 2D={snr_2d:.2f}, 3D={snr_3d:.2f} dB"
        )

        if orig_sr == 44100 and target_sr == 48000:
            out_3d_chunked = resample_polyphase_kaiser(
                in_3d,
                orig_sr=orig_sr,
                target_sr=target_sr,
                _block_chunk_override=64,
            )
            seam_err = float((out_3d - out_3d_chunked).abs().max().item())
            assert seam_err <= 1e-7, (
                f"64-aligned polyphase chunk seam discontinuity: {seam_err:.2e}"
            )

        print(
            f"  [+] {label} : [T]/[C,T]/[B,C,T] min SNR = {min_snr:6.2f} dB (>80 dB) | {ms:6.2f} ms"
        )

    n_48 = 24000
    t_48 = torch.arange(n_48, dtype=torch.float64, device=f64_dev) / 48000.0
    stop_tone = (
        torch.stack(
            [
                torch.sin(2.0 * math.pi * 16500.0 * t_48),
                torch.cos(2.0 * math.pi * 19200.0 * t_48),
            ],
            dim=0,
        )
        .to(dtype=torch.float32, device=device)
        .unsqueeze(0)
    )
    decimated = resample_polyphase_kaiser(stop_tone, orig_sr=48000, target_sr=24000)
    assert decimated.shape == (1, 2, 12000)
    inner_dec = _to_f64(decimated[:, :, 128:-128])
    atten_db = 10.0 * math.log10(
        max(float(torch.mean(inner_dec.pow(2)).item()), 1e-24) / 0.5
    )
    assert atten_db < -85.0, f"Stopband alias rejection insufficient: {atten_db:.2f} dB"

    for pulse_shape in ((2048,), (2, 2048), (2, 2, 2048)):
        pulse = torch.ones(pulse_shape, dtype=torch.float32, device=device)
        declicked = apply_sub_millisecond_declick(pulse.clone(), fade_samples=512)
        assert declicked.shape == pulse_shape
        assert float(declicked[..., 0].abs().max().item()) == 0.0
        assert float(declicked[..., -1].abs().max().item()) == 0.0
        assert float(declicked[..., 511].min().item()) == 1.0
        assert float(declicked[..., -512].min().item()) == 1.0

    with tempfile.TemporaryDirectory(dir=str(TMP_DIR)) as tmp_d:
        wav_p = Path(tmp_d) / "ref_probe.wav"
        t_wav = np.arange(4410, dtype=np.float64) / 44100.0
        mono_np = (0.3 * np.sin(2.0 * math.pi * 440.0 * t_wav))[:, None]
        sf.write(str(wav_p), mono_np, 44100, subtype="DOUBLE")
        ref_bct = load_reference_stereo_audio(wav_p)
        assert ref_bct.shape == (1, 2, 4800) and ref_bct.dtype == torch.float32

    print(
        f"  [+] Ultrasonic Alias & [B,C,T] : {atten_db:6.2f} dB (<-85 dB) | 1D/2D/3D Hann Zero-Endpoint & [1,2,T] Ingest Verified"
    )


def _build_synthetic_yue2(device: torch.device) -> YuE2ForCausalLM:
    cfg = YuE2Config(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=256,
        vocab_size=184704,
        max_position_embeddings=512,
        latent_dim=64,
        max_latent_frames=512,
        timestep_shift=1.0,
    )
    ref_cpu = YuE2ForCausalLM(cfg).eval()
    raw_sd: Dict[str, torch.Tensor] = {}
    for k, v in ref_cpu.state_dict().items():
        raw_sd[k] = v.to(dtype=torch.bfloat16) if v.is_floating_point() else v
    raw_sd["latent_pos_embed.pe"] = torch.zeros_like(
        ref_cpu.latent_pos_embed.pe, dtype=torch.bfloat16
    )
    raw_sd["model.rotary_emb.inv_freq"] = torch.ones(16, dtype=torch.bfloat16)

    with torch.device("meta"):
        meta_model = YuE2ForCausalLM(cfg)
    meta_model.latent_pos_embed = AudioPositionEmbedding(
        cfg.max_latent_frames, cfg.hidden_size
    )
    _apply_non_degradative_placement(
        meta_model, raw_sd, target_device=device, trunk_dtype=torch.bfloat16
    )
    return meta_model


@torch.inference_mode()
def audit_gate_3_placement_and_mixed_precision(
    device: torch.device,
) -> YuE2ForCausalLM:
    print(
        "\n--- Gate 3: Non-Degradative Placement, Split-K FP32 Lock, ABI v3 Fused Kernels & FP64 Markov Audit ---"
    )
    enforce_full_precision_reductions()
    if torch.cuda.is_available() and hasattr(torch.backends, "cuda"):
        mm_mod = getattr(torch.backends.cuda, "matmul", None)
        if mm_mod is not None:
            for attr in (
                "allow_bf16_reduced_precision_reduction",
                "allow_fp16_reduced_precision_reduction",
                "allow_fp16_accumulation",
            ):
                if hasattr(mm_mod, attr):
                    assert getattr(mm_mod, attr) is False, (
                        f"Split-K reduced-precision reduction remained unlocked: {attr}"
                    )

    model = _build_synthetic_yue2(device)

    for name, p in model.named_parameters():
        assert not p.is_meta, f"Meta parameter leaked: {name}"
        name_l = name.lower()
        if "norm" in name_l or "bias" in name_l:
            assert p.dtype == torch.float32, (
                f"Expected FP32 promotion on '{name}', got {p.dtype}"
            )
    for name, b in model.named_buffers():
        assert not b.is_meta, f"Meta buffer leaked: {name}"

    assert model.llm2vae.weight.dtype == torch.float32
    assert model.vae2llm.weight.dtype == torch.float32
    assert model.time_embedder.mlp[0].weight.dtype == torch.float32
    assert model.latent_pos_embed.pe.dtype == torch.float32
    for mod in model.modules():
        if isinstance(mod, RMSNorm):
            assert mod.weight.dtype == torch.float32
    assert model.model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16

    expected_pe = synthesize_audio_pe_fp32(
        model.config.max_latent_frames, model.config.hidden_size, device=device
    )
    pe_err = float((model.latent_pos_embed.pe - expected_pe).abs().max().item())
    assert pe_err == 0.0, f"Analytical PE corrupted by disk state_dict: max_err={pe_err}"

    layer0 = model.model.layers[0]
    assert layer0.self_attn._can_use_fused_qkv(), "Fused QKV inactive after placement"
    assert layer0.mlp._can_use_fused_gate_up(), "Fused Gate/Up inactive after placement"
    assert layer0.self_attn._phase_q != layer0.self_attn._phase_k

    x_fuse_f32 = torch.randn(
        (2, 8, model.config.hidden_size), dtype=torch.float32, device=device
    )
    delta_f32 = torch.randn(
        (2, 8, model.config.hidden_size), dtype=torch.float32, device=device
    )
    rms_out_buf = torch.empty(
        (2, 8, model.config.hidden_size), dtype=torch.bfloat16, device=device
    )
    rms_ptr = rms_out_buf.data_ptr()
    rms_eager = layer0.input_layernorm.forward_to_dtype(x_fuse_f32, torch.bfloat16)
    rms_zero = layer0.input_layernorm.forward_to_dtype(
        x_fuse_f32, torch.bfloat16, out=rms_out_buf
    )
    assert rms_zero.data_ptr() == rms_ptr
    assert torch.equal(rms_zero.view(torch.int16), rms_eager.view(torch.int16))

    res_ref = x_fuse_f32 + delta_f32
    nrm_ref = layer0.post_attention_layernorm.forward_to_dtype(
        res_ref, torch.bfloat16
    )
    res_oop, nrm_oop = layer0.post_attention_layernorm.forward_residual_to_dtype(
        x_fuse_f32, delta_f32, torch.bfloat16
    )
    assert res_oop.dtype == torch.float32 and nrm_oop.dtype == torch.bfloat16
    assert torch.equal(res_oop, res_ref), "forward_residual_to_dtype OOP sum mismatch"
    assert torch.equal(nrm_oop.view(torch.int16), nrm_ref.view(torch.int16)), (
        "forward_residual_to_dtype OOP norm mismatch"
    )

    res_inp = x_fuse_f32.clone()
    inp_ptr = res_inp.data_ptr()
    res_inp_out, nrm_inp = layer0.post_attention_layernorm.forward_residual_to_dtype(
        res_inp,
        delta_f32,
        torch.bfloat16,
        res_out=res_inp,
        norm_out=rms_out_buf,
    )
    assert res_inp_out.data_ptr() == inp_ptr and nrm_inp.data_ptr() == rms_ptr
    assert torch.equal(res_inp_out, res_ref), "forward_residual_to_dtype in-place sum mismatch"
    assert torch.equal(nrm_inp.view(torch.int16), nrm_ref.view(torch.int16)), (
        "forward_residual_to_dtype in-place norm mismatch"
    )

    with torch.inference_mode(False):
        rn_ste = RMSNorm(32, eps=1e-6, phase_offset=5).to(
            device=device, dtype=torch.float32
        )
        with torch.no_grad():
            rn_ste.weight.uniform_(0.7, 1.3)
        r_ste = torch.randn(
            (2, 4, 32), device=device, dtype=torch.float32, requires_grad=True
        )
        d_ste = torch.randn(
            (2, 4, 32), device=device, dtype=torch.float32, requires_grad=True
        )
        r_out_ste, n_out_ste = rn_ste.forward_residual_to_dtype(
            r_ste, d_ste, torch.bfloat16
        )
        g_r = torch.randn_like(r_out_ste)
        g_n = torch.randn((2, 4, 32), device=device, dtype=torch.bfloat16)
        torch.autograd.backward((r_out_ste, n_out_ste), (g_r, g_n))

        r_ref64 = _to_f64(r_ste.detach()).requires_grad_(True)
        d_ref64 = _to_f64(d_ste.detach()).requires_grad_(True)
        w_ref64 = _to_f64(rn_ste.weight.detach()).requires_grad_(True)
        s_ref64 = r_ref64 + d_ref64
        n_ref64 = (
            s_ref64
            * torch.rsqrt(s_ref64.pow(2).mean(-1, keepdim=True) + 1e-6)
            * w_ref64
        )
        torch.autograd.backward((s_ref64, n_ref64), (_to_f64(g_r), _to_f64(g_n)))
        assert r_ste.grad is not None and d_ste.grad is not None and rn_ste.weight.grad is not None
        assert float((_to_f64(r_ste.grad) - r_ref64.grad).abs().max().item()) < 2e-5
        assert float((_to_f64(d_ste.grad) - d_ref64.grad).abs().max().item()) < 2e-5
        assert float((_to_f64(rn_ste.weight.grad) - w_ref64.grad).abs().max().item()) < 2e-5

    pos_fuse = torch.arange(8, device=device).unsqueeze(0).expand(2, 8)
    cos_f, sin_f = model.model.rotary_emb(pos_fuse, seq_len_hint=8)
    q_bshd, k_bshd, v_bshd = layer0.self_attn.project_qkv(
        rms_eager, cos_f, sin_f, layout_bhsd=False
    )
    q_buf = torch.empty((2, 4, 8, 32), dtype=torch.bfloat16, device=device)
    k_buf = torch.empty((2, 2, 8, 32), dtype=torch.bfloat16, device=device)
    v_buf = torch.empty((2, 2, 8, 32), dtype=torch.bfloat16, device=device)
    qp, kp, vp = q_buf.data_ptr(), k_buf.data_ptr(), v_buf.data_ptr()
    q_bhsd, k_bhsd, v_bhsd = layer0.self_attn.project_qkv(
        rms_eager,
        cos_f,
        sin_f,
        layout_bhsd=True,
        q_out=q_buf,
        k_out=k_buf,
        v_out=v_buf,
    )
    assert q_bhsd.data_ptr() == qp and k_bhsd.data_ptr() == kp and v_bhsd.data_ptr() == vp
    assert torch.equal(
        q_bhsd.view(torch.int16), q_bshd.transpose(1, 2).contiguous().view(torch.int16)
    )
    assert torch.equal(
        k_bhsd.view(torch.int16), k_bshd.transpose(1, 2).contiguous().view(torch.int16)
    )
    assert torch.equal(
        v_bhsd.view(torch.int16), v_bshd.transpose(1, 2).contiguous().view(torch.int16)
    )

    k_sent = torch.full((2, 2, 20, 32), 7.5, dtype=torch.bfloat16, device=device)
    v_sent = torch.full((2, 2, 20, 32), -3.25, dtype=torch.bfloat16, device=device)
    q_str = layer0.self_attn.project_qkv_into_kv_cache(
        rms_eager, cos_f, sin_f, k_sent, v_sent, kv_seq_offset=6, q_out=q_buf
    )
    assert q_str.data_ptr() == qp
    assert torch.equal(q_str.view(torch.int16), q_bhsd.view(torch.int16))
    assert torch.equal(k_sent[:, :, 6:14, :].view(torch.int16), k_bhsd.view(torch.int16))
    assert torch.equal(v_sent[:, :, 6:14, :].view(torch.int16), v_bhsd.view(torch.int16))
    assert bool((k_sent[:, :, :6, :] == 7.5).all().item())
    assert bool((k_sent[:, :, 14:, :] == 7.5).all().item())
    assert bool((v_sent[:, :, :6, :] == -3.25).all().item())
    assert bool((v_sent[:, :, 14:, :] == -3.25).all().item())

    kv_gap = StaticKVCache(2, 2, 2, 24, 32, torch.bfloat16, device)
    kv_gap.key_cache[0].fill_(9.0)
    kv_gap.value_cache[0].fill_(-9.0)
    kv_gap.key_cache[1].fill_(9.0)
    kv_gap.value_cache[1].fill_(-9.0)
    gap_pos = torch.arange(5, 13, dtype=torch.long, device=device)
    for l_idx in range(2):
        prep = kv_gap.prepare_contiguous_write(l_idx, 8, 2, gap_pos)
        assert prep is not None
        kb, vb, kv_off, eff_end = prep
        assert kv_off == 5 and eff_end == 13
        assert bool((kb[:, :, :5, :] == 0.0).all().item())
        assert bool((vb[:, :, :5, :] == 0.0).all().item())
        _ = layer0.self_attn.project_qkv_into_kv_cache(
            rms_eager, cos_f, sin_f, kb, vb, kv_off
        )
        k_vis, v_vis = kv_gap.commit_contiguous_write(l_idx, eff_end)
        assert k_vis.shape == (2, 2, 13, 32) and v_vis.shape == (2, 2, 13, 32)
        assert torch.equal(k_vis[:, :, 5:13, :].view(torch.int16), k_bhsd.view(torch.int16))
        assert torch.equal(v_vis[:, :, 5:13, :].view(torch.int16), v_bhsd.view(torch.int16))
        assert bool((kb[:, :, 13:, :] == 9.0).all().item())
    assert kv_gap.get_seq_length() == 13

    mlp_act_buf = torch.empty(
        (2, 8, model.config.intermediate_size), dtype=torch.bfloat16, device=device
    )
    mlp_eager = layer0.mlp.forward_fp32(rms_eager)
    mlp_zero = layer0.mlp.forward_fp32(rms_eager, act_out=mlp_act_buf)
    assert torch.equal(mlp_eager, mlp_zero)

    for t_test in (1, 20):
        emb_in_f32 = torch.randn(
            (1, t_test, model.config.hidden_size), dtype=torch.float32, device=device
        )
        emb_orig = emb_in_f32.clone()
        pos_test = torch.arange(t_test, dtype=torch.long, device=device).unsqueeze(0)
        kv_fused = StaticKVCache(2, 1, 2, 32, 32, model.trunk_dtype, device)
        pre_fused, _ = model.model.forward_prenorm(
            inputs_embeds=emb_in_f32,
            position_ids=pos_test,
            past_key_values=kv_fused,
            use_cache=True,
            seq_len_hint=t_test,
        )
        assert torch.equal(emb_in_f32, emb_orig), (
            "Backbone.forward_prenorm mutated caller FP32 inputs_embeds in-place"
        )

        kv_ref = StaticKVCache(2, 1, 2, 32, 32, model.trunk_dtype, device)
        cos_t, sin_t = model.model.rotary_emb(pos_test, seq_len_hint=t_test)
        x_ref_l = emb_orig.clone()
        for l_idx, lyr in enumerate(model.model.layers):
            x_ref_l = lyr(
                x_ref_l,
                cos_t,
                sin_t,
                kv_ref,
                layer_idx=l_idx,
                compute_dtype=model.trunk_dtype,
                _inplace_residual=False,
            )
        fused_backbone_err = float((pre_fused - x_ref_l).abs().max().item())
        assert fused_backbone_err <= 1e-6, (
            f"Cross-layer fused Backbone diverged from reference (T={t_test}): {fused_backbone_err:.2e}"
        )
        for l_idx in range(2):
            assert torch.equal(
                kv_fused.key_cache[l_idx][:, :, :t_test, :].view(torch.int16),
                kv_ref.key_cache[l_idx][:, :, :t_test, :].view(torch.int16),
            ), f"Strided KV cache K mismatch at layer {l_idx}, T={t_test}"
            assert torch.equal(
                kv_fused.value_cache[l_idx][:, :, :t_test, :].view(torch.int16),
                kv_ref.value_cache[l_idx][:, :, :t_test, :].view(torch.int16),
            ), f"Strided KV cache V mismatch at layer {l_idx}, T={t_test}"

    mid_rot = torch.full(
        (1, 64, 2, 32), 1.0 + 0.42 * (2.0**-7), dtype=torch.float32, device=device
    )
    q_narrow = _project_narrowing(
        mid_rot, torch.bfloat16, phase_offset=layer0.self_attn._phase_q
    )
    k_narrow = _project_narrowing(
        mid_rot, torch.bfloat16, phase_offset=layer0.self_attn._phase_k
    )
    assert not torch.equal(q_narrow, k_narrow), (
        "Q and K narrowing shared identical dither realization"
    )

    exported_sd = model.state_dict()
    serialized_bytes = save_safetensors_bytes(exported_sd)
    reloaded_sd = load_safetensors_bytes(serialized_bytes)
    assert layer0.self_attn._can_use_fused_qkv(), (
        "state_dict() export de-fused live Attention projection"
    )
    assert layer0.mlp._can_use_fused_gate_up(), (
        "state_dict() export de-fused live MLP projection"
    )
    assert torch.equal(
        reloaded_sd["model.layers.0.self_attn.q_proj.weight"].to(device=device),
        layer0.self_attn.q_proj.weight,
    ), "SafeTensors roundtrip drift on fused q_proj.weight"

    for sub_mod in model.modules():
        sub_mod._apply(lambda t: t, recurse=False)
    for mod in model.model.layers.modules():
        if isinstance(mod, Attention):
            assert mod._pending_rebind is True
        elif isinstance(mod, MLP):
            assert mod._pending_rebind is True

    print(
        "  [+] Manifold Bridge & ABI v3   : FP32 bridges, Residual+RMSNorm STE, Strided KV scatter & Cross-layer Backbone verified."
    )

    probe_ids = torch.tensor(
        [[EOD, 42, ABC_START, MUSIC_START]], dtype=torch.long, device=device
    )
    full_out = model(input_ids=probe_ids, use_cache=False, logits_to_keep=1).logits
    assert full_out.dtype == torch.float32, (
        f"Mid-stream logit downcast detected: expected float32, got {full_out.dtype}"
    )
    assert (
        layer0.self_attn._can_use_fused_qkv() and not layer0.self_attn._pending_rebind
    ), "Lazy QKV re-bind failed on post-_apply forward"
    assert (
        layer0.mlp._can_use_fused_gate_up() and not layer0.mlp._pending_rebind
    ), "Lazy Gate/Up re-bind failed on post-_apply forward"

    max_slice_err = 0.0
    for sl_name, v_slice in (
        ("ABC", ABC_VOCAB_SLICE),
        ("SEMANTIC", SEMANTIC_VOCAB_SLICE),
    ):
        sliced_out = model.forward_vocab_slice_fp32(
            input_ids=probe_ids,
            vocab_slice=v_slice,
            use_cache=False,
            logits_to_keep=1,
        )
        assert sliced_out.dtype == torch.float32
        slice_err = float(
            (sliced_out - full_out[..., v_slice[0] : v_slice[1]]).abs().max().item()
        )
        max_slice_err = max(max_slice_err, slice_err)
        assert slice_err < 1e-6, f"{sl_name} active-vocab slice drift: {slice_err}"

    with torch.inference_mode(False):
        with torch.no_grad():
            nograd_slice = model.forward_vocab_slice_fp32(
                input_ids=probe_ids.clone(),
                vocab_slice=SEMANTIC_VOCAB_SLICE,
                use_cache=False,
                logits_to_keep=1,
            )
            assert float((nograd_slice - sliced_out).abs().max().item()) == 0.0

        tiny_cfg = YuE2Config(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            intermediate_size=128,
            vocab_size=256,
            max_position_embeddings=64,
            latent_dim=16,
            max_latent_frames=64,
        )
        tiny_train = YuE2ForCausalLM(tiny_cfg).to(
            device=device, dtype=torch.bfloat16
        )
        tiny_train.eval().requires_grad_(False)
        with torch.inference_mode():
            _ = tiny_train(
                input_ids=torch.tensor([[1, 2, 3, 4]], device=device),
                use_cache=False,
            )
        tiny_train.train(True).requires_grad_(True)
        train_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
        loss = tiny_train(
            input_ids=train_ids, labels=train_ids, use_cache=False
        ).loss
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        tl0 = tiny_train.model.layers[0]
        for grad_param in (
            tl0.input_layernorm.weight,
            tl0.self_attn.q_proj.weight,
            tl0.self_attn.k_proj.weight,
            tl0.self_attn.v_proj.weight,
            tl0.self_attn.q_norm.weight,
            tl0.self_attn.k_norm.weight,
            tl0.self_attn.o_proj.weight,
            tl0.post_attention_layernorm.weight,
            tl0.mlp.gate_proj.weight,
            tl0.mlp.up_proj.weight,
            tl0.mlp.down_proj.weight,
            tiny_train.model.norm.weight,
            tiny_train.lm_head.weight,
        ):
            assert grad_param.grad is not None and torch.isfinite(grad_param.grad).all()
        del tiny_train, loss

    clear_yue2_workspaces(device)

    tf32_drift = 0.0
    if device.type == "cuda" and hasattr(torch.backends, "cuda"):
        gen_tf32 = torch.Generator(device="cpu").manual_seed(31415)
        prenorm_probe = torch.randn(
            (2, 4, model.config.hidden_size), dtype=torch.float32, generator=gen_tf32
        ).to(device=device)
        lat_probe = torch.randn(
            (1, 6, model.config.latent_dim), dtype=torch.float32, generator=gen_tf32
        ).to(device=device)
        pe_probe = model.latent_pos_embed.forward_fp32(
            torch.arange(6, device=device)
        ).unsqueeze(0)
        mm_backend = torch.backends.cuda.matmul
        orig_mm_mode = str(mm_backend.fp32_precision)
        try:
            mm_backend.fp32_precision = "ieee"
            ref_lm = model.forward_lm_head_fp32(
                prenorm_probe, vocab_slice=SEMANTIC_VOCAB_SLICE
            )
            ref_in = model.project_latent_to_trunk_fp32(lat_probe, 0.35, pe_probe)
            ref_vel = model.project_trunk_to_velocity_fp32(prenorm_probe)

            mm_backend.fp32_precision = "tf32"
            cand_lm = model.forward_lm_head_fp32(
                prenorm_probe, vocab_slice=SEMANTIC_VOCAB_SLICE
            )
            cand_in = model.project_latent_to_trunk_fp32(lat_probe, 0.35, pe_probe)
            cand_vel = model.project_trunk_to_velocity_fp32(prenorm_probe)
        finally:
            mm_backend.fp32_precision = orig_mm_mode

        tf32_drift = max(
            float((ref_lm - cand_lm).abs().max().item()),
            float((ref_in - cand_in).abs().max().item()),
            float((ref_vel - cand_vel).abs().max().item()),
        )
        assert tf32_drift == 0.0, (
            f"TF32 truncation leaked into FP32 bridge/lm_head contractions: {tf32_drift}"
        )

    sem_len = SEMANTIC_VOCAB_SLICE[1] - SEMANTIC_VOCAB_SLICE[0]
    f64_dev = _f64_device(device)
    pen_base = torch.linspace(-2.5, 2.5, sem_len, dtype=torch.float64, device=f64_dev)
    pen_state = _RollingPenaltyState(
        window=6,
        vocab_offset=SEMANTIC_VOCAB_SLICE[0],
        support_len=sem_len,
        penalty=1.25,
        device=device,
    )
    pen_hist: List[int] = []
    for tok_cand in (
        SEMANTIC_VOCAB_SLICE[0],
        CODEC_OFFSET + 3,
        CODEC_OFFSET + 3,
        42,
        CODEC_OFFSET + 19,
        CODEC_OFFSET + 32767,
        CODEC_OFFSET + 7,
        CODEC_OFFSET + 3,
        CODEC_OFFSET + 11,
        SEMANTIC_VOCAB_SLICE[0],
    ):
        s_fast = pen_base.clone()
        pen_state.apply_inplace_f64(s_fast)
        s_ref = apply_windowed_penalty(
            pen_base,
            pen_hist,
            1.25,
            6,
            vocab_offset=SEMANTIC_VOCAB_SLICE[0],
        )
        assert torch.equal(s_fast, s_ref), "RollingPenaltyState diverged from reference"
        pen_hist.append(tok_cand)
        pen_state.push(tok_cand)

    gen_m = torch.Generator(device=device).manual_seed(9001)
    uniform_8 = torch.zeros(sem_len, dtype=torch.float32, device=device)
    tok_id, ent_bits = sample_categorical_distribution_with_entropy(
        logits=uniform_8,
        temperature=1.0,
        top_p=1.0,
        top_k=8,
        history=[],
        step=250,
        min_tokens=200,
        penalty=1.0,
        penalty_window=50,
        phase="semantic",
        generator=gen_m,
        vocab_offset=SEMANTIC_VOCAB_SLICE[0],
    )
    assert CODEC_OFFSET <= tok_id < CODEC_OFFSET + 32768 or tok_id == MUSIC_END
    assert abs(ent_bits - 3.0) < 1e-9, (
        f"Expected 3.000 bits on uniform K=8 support, got {ent_bits}"
    )
    print(
        f"  [+] Zero-Downcast Markov Head  : Slice L_inf={max_slice_err:.1e} | TF32 drift={tf32_drift:.1e} | H(U_8)={ent_bits:.6f}b"
    )

    if FP8_DTYPES:
        fp8_dt = FP8_DTYPES[0]
        q_lin = QuantSafeLinear(64, 32, bias=True).to(device=device)
        with torch.no_grad():
            w_ref = torch.randn((32, 64), device=device, dtype=torch.float32) * 0.1
            q_lin.weight = nn.Parameter(
                refineTensor(w_ref, fp8_dt, target_device=device), requires_grad=False
            )
            q_lin.bias = nn.Parameter(
                torch.randn(32, device=device, dtype=torch.float32) * 0.05,
                requires_grad=False,
            )
        for act_dt in (torch.bfloat16, torch.float32):
            x_in = torch.randn((2, 8, 64), device=device, dtype=act_dt)
            y_out = q_lin(x_in)
            assert y_out.dtype == act_dt and torch.isfinite(y_out).all()
        x_fp8 = torch.randn((2, 8, 64), device=device, dtype=torch.bfloat16).to(
            dtype=fp8_dt
        )
        y_fp8 = q_lin(x_fp8)
        assert y_fp8.dtype == torch.bfloat16 and torch.isfinite(y_fp8).all()
        print(
            f"  [+] QuantSafeLinear FP8        : {fp8_dt} x {{fp8, bf16, fp32}} dispatch finite & verified."
        )

    q_f32 = torch.randn((1, 4, 16, 32), device=device, dtype=torch.float32)
    k_f32 = torch.randn((1, 2, 16, 32), device=device, dtype=torch.float32)
    v_f32 = torch.randn((1, 2, 16, 32), device=device, dtype=torch.float32)
    out_sdpa = sdpa(q_f32, k_f32, v_f32, is_causal=True)
    assert out_sdpa.shape == (1, 4, 16, 32) and torch.isfinite(out_sdpa).all()
    print(
        "  [+] Backend-Aware GQA SDPA     : FP32 IEEE-guarded group-expansion & finite attention verified."
    )
    return model


@torch.inference_mode()
def audit_gate_4_batched_cfg_and_kv_splicing(model: YuE2ForCausalLM) -> None:
    print(
        "\n--- Gate 4: Batched CFG (B=2) Padding-Aware RoPE & StaticKVCache Truncation Audit ---"
    )
    device = model.model.embed_tokens.weight.device
    song_prefix = [EOD, 101, 202, 303, 404, ABC_START, 505, 606, ABC_END, MUSIC_START]
    neg_prefix = [EOD, 101, 202, ABC_START, 505, 606, ABC_END, MUSIC_START]

    c_pos = StaticKVCache(2, 1, 2, 12, 32, model.trunk_dtype, device)
    c_neg = StaticKVCache(2, 1, 2, 64, 32, model.trunk_dtype, device)
    lp_single = model.forward_vocab_slice_fp32(
        input_ids=torch.tensor([song_prefix], dtype=torch.long, device=device),
        vocab_slice=SEMANTIC_VOCAB_SLICE,
        past_key_values=c_pos,
        use_cache=True,
        logits_to_keep=1,
    )[:, -1, :]
    ln_single = model.forward_vocab_slice_fp32(
        input_ids=torch.tensor([neg_prefix], dtype=torch.long, device=device),
        vocab_slice=SEMANTIC_VOCAB_SLICE,
        past_key_values=c_neg,
        use_cache=True,
        logits_to_keep=1,
    )[:, -1, :]

    max_len = max(len(song_prefix), len(neg_prefix))
    pad_neg = max_len - len(neg_prefix)
    batched_in = torch.tensor(
        [song_prefix, [EOD] * pad_neg + neg_prefix], dtype=torch.long, device=device
    )
    attn_mask = torch.ones((2, 64), dtype=torch.bool, device=device)
    attn_mask[1, :pad_neg] = False
    c_cfg = StaticKVCache(2, 2, 2, 64, 32, model.trunk_dtype, device)
    pair_prefill = model.forward_vocab_slice_fp32(
        input_ids=batched_in,
        vocab_slice=SEMANTIC_VOCAB_SLICE,
        attention_mask=attn_mask[:, :max_len],
        past_key_values=c_cfg,
        use_cache=True,
        logits_to_keep=1,
    )[:, -1, :]

    diff_prefill = max(
        float((pair_prefill[0:1] - lp_single).abs().max().item()),
        float((pair_prefill[1:2] - ln_single).abs().max().item()),
    )
    assert diff_prefill < 2e-2, f"Batched CFG prefill RoPE drift: {diff_prefill}"

    next_tok = CODEC_OFFSET + 42
    lp_step = model.forward_vocab_slice_fp32(
        input_ids=torch.tensor([[next_tok]], dtype=torch.long, device=device),
        vocab_slice=SEMANTIC_VOCAB_SLICE,
        past_key_values=c_pos,
        use_cache=True,
        logits_to_keep=1,
    )[:, -1, :]
    ln_step = model.forward_vocab_slice_fp32(
        input_ids=torch.tensor([[next_tok]], dtype=torch.long, device=device),
        vocab_slice=SEMANTIC_VOCAB_SLICE,
        past_key_values=c_neg,
        use_cache=True,
        logits_to_keep=1,
    )[:, -1, :]

    step_pos = torch.tensor(
        [[len(song_prefix)], [len(neg_prefix)]], dtype=torch.long, device=device
    )
    pair_step = model.forward_vocab_slice_fp32(
        input_ids=torch.tensor(
            [[next_tok], [next_tok]], dtype=torch.long, device=device
        ),
        vocab_slice=SEMANTIC_VOCAB_SLICE,
        attention_mask=attn_mask[:, : max_len + 1],
        position_ids=step_pos,
        past_key_values=c_cfg,
        use_cache=True,
        logits_to_keep=1,
        seq_len_hint=max_len + 1,
    )[:, -1, :]

    diff_step = max(
        float((pair_step[0:1] - lp_step).abs().max().item()),
        float((pair_step[1:2] - ln_step).abs().max().item()),
    )
    assert diff_step < 2e-2, f"Batched CFG decode RoPE drift: {diff_step}"

    edited_prefix = song_prefix[:-3] + [707, 808, 909, 1010, ABC_END, MUSIC_START]
    shared = find_common_prefix(song_prefix, edited_prefix)
    c_pos.ensure_capacity(64, keep_tokens=shared)
    assert c_pos.max_seq_len == 64 and c_pos.get_seq_length() == shared
    spliced_logits = model.forward_vocab_slice_fp32(
        input_ids=torch.tensor(
            [edited_prefix[shared:]], dtype=torch.long, device=device
        ),
        vocab_slice=SEMANTIC_VOCAB_SLICE,
        past_key_values=c_pos,
        use_cache=True,
        logits_to_keep=1,
    )[:, -1, :]

    c_fresh = StaticKVCache(2, 1, 2, 64, 32, model.trunk_dtype, device)
    fresh_logits = model.forward_vocab_slice_fp32(
        input_ids=torch.tensor([edited_prefix], dtype=torch.long, device=device),
        vocab_slice=SEMANTIC_VOCAB_SLICE,
        past_key_values=c_fresh,
        use_cache=True,
        logits_to_keep=1,
    )[:, -1, :]
    splice_diff = float((spliced_logits - fresh_logits).abs().max().item())
    assert splice_diff < 2e-2, f"KV-cache prefix splice divergence: {splice_diff}"
    print(
        f"  [+] B=2 CFG & KV Growth Splice : Prefill delta={diff_prefill:.2e}, Step delta={diff_step:.2e}, Growth splice={splice_diff:.2e}"
    )


@torch.inference_mode()
def audit_gate_5_nar_fp32_ode_and_manifold_inpainting(model: YuE2ForCausalLM) -> None:
    print(
        "\n--- Gate 5: FP64-Accumulated Flow-Matching ODE, Graph Replay Parity & Manifold Inpainting Audit ---"
    )
    device = model.model.embed_tokens.weight.device
    has_cuda_graph = device.type == "cuda" and hasattr(torch.cuda, "CUDAGraph")
    chunk_tokens = [
        EOD,
        10,
        20,
        ABC_START,
        30,
        ABC_END,
        MUSIC_START,
        CODEC_OFFSET + 1,
        CODEC_OFFSET + 2,
        CODEC_OFFSET + 3,
        CODEC_OFFSET + 4,
        MUSIC_END,
    ]
    t_lat = 10
    gen = torch.Generator(device="cpu").manual_seed(1234)
    noise = torch.randn((t_lat, 64), dtype=torch.float32, generator=gen)
    ref_clean = torch.randn((4, 64), dtype=torch.float32, generator=gen)

    ar_len = len(chunk_tokens)
    nar_len = t_lat + 2
    full_seq = torch.tensor(
        [chunk_tokens + [LATENT_START] + [0] * t_lat + [LATENT_END]],
        dtype=torch.long,
        device=device,
    )
    ar_mask = torch.tensor(
        [[True] * ar_len + [False] * nar_len], dtype=torch.bool, device=device
    )
    nar_mask = ~ar_mask
    nar_content_mask = torch.tensor(
        [[False] * (ar_len + 1) + [True] * t_lat + [False]],
        dtype=torch.bool,
        device=device,
    )

    max_vel_diff = 0.0
    for nar_cond_end in (0, 7):
        solver_eager = CachedNARChunkSolver(
            model=model,
            chunk_tokens=chunk_tokens,
            initial_noise=noise,
            nar_cond_end=nar_cond_end,
            ref_clean=None,
            use_cuda_graph=False,
        )
        solver_graph = CachedNARChunkSolver(
            model=model,
            chunk_tokens=chunk_tokens,
            initial_noise=noise,
            nar_cond_end=nar_cond_end,
            ref_clean=None,
            use_cuda_graph=True,
        )
        if has_cuda_graph:
            assert solver_graph.graph_active, (
                f"Expected HIPGraph/CUDAGraph capture active on {device} (cond_end={nar_cond_end})"
            )

        x_probe = noise.to(device=device, dtype=torch.float32)
        v_eager_025 = solver_eager.velocity(x_probe, 0.25)
        v_graph_025 = solver_graph.velocity(x_probe, 0.25)
        assert torch.equal(v_graph_025, v_eager_025), (
            f"NAR velocity graph replay diverged from eager (cond_end={nar_cond_end}, t=0.25)"
        )

        v_direct = model.nar_velocity(
            tokens=full_seq,
            ar_mask=ar_mask,
            nar_mask=nar_mask,
            nar_content_mask=nar_content_mask,
            x_t=x_probe,
            t_value=0.25,
            nar_cond_end=nar_cond_end,
        )
        v_diff = float((v_graph_025 - v_direct).abs().max().item())
        max_vel_diff = max(max_vel_diff, v_diff)
        assert v_diff < 5e-3, (
            f"CachedNARChunkSolver diverged from nar_velocity (cond_end={nar_cond_end}): {v_diff}"
        )

        v_graph_neg = solver_graph.velocity(x_probe, -1.5)
        v_graph_neg_snap = v_graph_neg.clone()
        v_eager_neg = solver_eager.velocity(x_probe, -1.5)
        v_graph_pos = solver_graph.velocity(x_probe * 0.5, 2.0)
        v_eager_pos = solver_eager.velocity(x_probe * 0.5, 2.0)

        assert torch.equal(v_graph_neg, v_graph_neg_snap), (
            "NAR graph velocity(copy_out=True) aliased across consecutive calls"
        )
        assert torch.equal(v_graph_neg, v_eager_neg), (
            f"NAR velocity graph replay diverged at t=-1.5 (cond_end={nar_cond_end})"
        )
        assert torch.equal(v_graph_pos, v_eager_pos), (
            f"NAR velocity graph replay diverged at t=2.0 (cond_end={nar_cond_end})"
        )
        assert not torch.equal(v_graph_neg, v_graph_pos), (
            "NAR graph replay froze dynamic timestep or state buffer"
        )

        solver_eager.close()
        solver_graph.close()

    for method in ("euler", "heun", "midpoint"):
        for nar_cond_end in (0, 7):
            solver_eager = CachedNARChunkSolver(
                model=model,
                chunk_tokens=chunk_tokens,
                initial_noise=noise,
                nar_cond_end=nar_cond_end,
                ref_clean=ref_clean,
                use_cuda_graph=False,
            )
            solver_graph = CachedNARChunkSolver(
                model=model,
                chunk_tokens=chunk_tokens,
                initial_noise=noise,
                nar_cond_end=nar_cond_end,
                ref_clean=ref_clean,
                use_cuda_graph=True,
            )
            out_eager = solver_eager.solve(steps=8, method=method)
            out_graph = solver_graph.solve(steps=8, method=method)
            solver_eager.close()
            solver_graph.close()

            assert torch.equal(out_graph, out_eager), (
                f"NAR graph-replayed ODE ({method}, cond_end={nar_cond_end}) diverged from eager"
            )
            assert out_graph.dtype == torch.float32, (
                f"Expected float32 trajectory, got {out_graph.dtype}"
            )
            assert out_graph.shape == (t_lat, 64) and torch.isfinite(out_graph).all()
            anchor_err = float((out_graph[:4] - ref_clean).abs().max().item())
            assert anchor_err == 0.0, (
                f"Reference manifold drifted at tau=0 ({method}): {anchor_err}"
            )

    graph_tag = "Graph vs Eager 0-ULP bit-exact" if has_cuda_graph else "Eager verified"
    print(
        f"  [+] Euler / Heun / Midpoint    : {graph_tag} | Direct vel delta={max_vel_diff:.2e} | anchor L_inf = 0.0."
    )


def _audit_sovereign_conv1d_case(
    device: torch.device,
    gen: torch.Generator,
    *,
    batch: Optional[int],
    c_in: int,
    c_out: int,
    k_size: int,
    stride: int = 1,
    padding: Union[int, str] = 0,
    dilation: int = 1,
    groups: int = 1,
    t_in: int = 128,
    use_bias: bool = True,
) -> float:
    with torch.inference_mode(False):
        layer = SovereignConv1d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=k_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
        ).to(device=device, dtype=torch.float32)

        fan_in = max(1, (c_in // groups) * k_size)
        w_init = torch.randn(
            layer.weight.shape, dtype=torch.float32, generator=gen
        ) / math.sqrt(fan_in)
        b_init = (
            torch.randn((c_out,), dtype=torch.float32, generator=gen) * 0.1
            if use_bias
            else None
        )
        with torch.no_grad():
            layer.weight.copy_(w_init.to(device=device))
            if use_bias and layer.bias is not None and b_init is not None:
                layer.bias.copy_(b_init.to(device=device))

        x_shape = (c_in, t_in) if batch is None else (batch, c_in, t_in)
        x_cpu = torch.randn(x_shape, dtype=torch.float32, generator=gen) * 0.5

        x64 = x_cpu.to(dtype=torch.float64).requires_grad_(True)
        w64 = w_init.to(dtype=torch.float64).requires_grad_(True)
        b64 = (
            b_init.to(dtype=torch.float64).requires_grad_(True)
            if b_init is not None
            else None
        )
        ref_out = F.conv1d(
            x64, w64, b64, stride=stride, padding=padding, dilation=dilation, groups=groups
        )
        eff_n = max(1, int(ref_out.shape[-1]) * (1 if batch is None else batch))
        upstream_cpu = torch.randn(
            ref_out.shape, dtype=torch.float32, generator=gen
        ) * (0.25 / math.sqrt(eff_n))
        (ref_out * upstream_cpu.to(dtype=torch.float64)).sum().backward()

        with torch.inference_mode():
            inf_out = layer(x_cpu.to(device=device))
        assert inf_out.shape == ref_out.shape

        x_dev = x_cpu.to(device=device).requires_grad_(True)
        layer.zero_grad(set_to_none=True)
        grad_out = layer(x_dev)
        (grad_out * upstream_cpu.to(device=device)).sum().backward()

        assert x_dev.grad is not None and layer.weight.grad is not None
        errs = [
            float((inf_out.cpu().to(torch.float64) - ref_out.detach()).abs().max().item()),
            float((grad_out.detach().cpu().to(torch.float64) - ref_out.detach()).abs().max().item()),
            float((x_dev.grad.detach().cpu().to(torch.float64) - x64.grad).abs().max().item()),
            float((layer.weight.grad.detach().cpu().to(torch.float64) - w64.grad).abs().max().item()),
        ]
        if use_bias and layer.bias is not None and b64 is not None:
            assert layer.bias.grad is not None and b64.grad is not None
            errs.append(
                float((layer.bias.grad.detach().cpu().to(torch.float64) - b64.grad).abs().max().item())
            )
        max_err = max(errs)
        assert max_err < 1e-5, (
            f"SovereignConv1d parity failure ({x_shape}, K={k_size}, s={stride}, p={padding}, d={dilation}, G={groups}): {max_err:.2e}"
        )
        return max_err


def _audit_sovereign_conv_transpose1d_case(
    device: torch.device,
    gen: torch.Generator,
    *,
    batch: Optional[int],
    c_in: int,
    c_out: int,
    k_size: int,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    t_in: int = 48,
    use_bias: bool = True,
) -> Tuple[float, Tuple[int, ...]]:
    with torch.inference_mode(False):
        layer = SovereignConvTranspose1d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=k_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=use_bias,
            dilation=dilation,
        ).to(device=device, dtype=torch.float32)

        fan_in = max(1, c_in * k_size)
        w_init = torch.randn(
            layer.weight.shape, dtype=torch.float32, generator=gen
        ) / math.sqrt(fan_in)
        b_init = (
            torch.randn((c_out,), dtype=torch.float32, generator=gen) * 0.1
            if use_bias
            else None
        )
        with torch.no_grad():
            layer.weight.copy_(w_init.to(device=device))
            if use_bias and layer.bias is not None and b_init is not None:
                layer.bias.copy_(b_init.to(device=device))

        x_shape = (c_in, t_in) if batch is None else (batch, c_in, t_in)
        x_cpu = torch.randn(x_shape, dtype=torch.float32, generator=gen) * 0.5

        x64 = x_cpu.to(dtype=torch.float64).requires_grad_(True)
        w64 = w_init.to(dtype=torch.float64).requires_grad_(True)
        b64 = (
            b_init.to(dtype=torch.float64).requires_grad_(True)
            if b_init is not None
            else None
        )
        ref_out = F.conv_transpose1d(
            x64,
            w64,
            b64,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        )
        eff_n = max(1, int(ref_out.shape[-1]) * (1 if batch is None else batch))
        upstream_cpu = torch.randn(
            ref_out.shape, dtype=torch.float32, generator=gen
        ) * (0.25 / math.sqrt(eff_n))
        (ref_out * upstream_cpu.to(dtype=torch.float64)).sum().backward()

        with torch.inference_mode():
            inf_out = layer(x_cpu.to(device=device))
        assert inf_out.shape == ref_out.shape

        x_dev = x_cpu.to(device=device).requires_grad_(True)
        layer.zero_grad(set_to_none=True)
        grad_out = layer(x_dev)
        (grad_out * upstream_cpu.to(device=device)).sum().backward()

        assert x_dev.grad is not None and layer.weight.grad is not None
        errs = [
            float((inf_out.cpu().to(torch.float64) - ref_out.detach()).abs().max().item()),
            float((grad_out.detach().cpu().to(torch.float64) - ref_out.detach()).abs().max().item()),
            float((x_dev.grad.detach().cpu().to(torch.float64) - x64.grad).abs().max().item()),
            float((layer.weight.grad.detach().cpu().to(torch.float64) - w64.grad).abs().max().item()),
        ]
        if use_bias and layer.bias is not None and b64 is not None:
            assert layer.bias.grad is not None and b64.grad is not None
            errs.append(
                float((layer.bias.grad.detach().cpu().to(torch.float64) - b64.grad).abs().max().item())
            )
        max_err = max(errs)
        assert max_err < 1e-5, (
            f"SovereignConvTranspose1d parity failure ({x_shape}, K={k_size}, s={stride}, p={padding}, op={output_padding}, d={dilation}, G={groups}): {max_err:.2e}"
        )
        return max_err, tuple(inf_out.shape)


@torch.inference_mode()
def audit_gate_6_vae_fp32_guard_and_tiling(device: torch.device) -> None:
    print(
        "\n--- Gate 6: SovereignConv1d/Transpose1d Autograd, Conductor SnakeBeta, Zero-MIOpen & Odd-Stride VAE Audit ---"
    )
    if torch.cuda.is_available():
        mm_backend = torch.backends.cuda.matmul
        conv_backend = (
            torch.backends.cudnn.conv
            if hasattr(torch.backends, "cudnn")
            and hasattr(torch.backends.cudnn, "conv")
            and hasattr(torch.backends.cudnn.conv, "fp32_precision")
            else None
        )
        orig_mm_mode = str(mm_backend.fp32_precision)
        orig_conv_mode = (
            str(conv_backend.fp32_precision) if conv_backend is not None else None
        )
        orig_generic_mode = (
            str(torch.backends.fp32_precision)
            if hasattr(torch.backends, "fp32_precision")
            else None
        )
        orig_cudnn_mode = (
            str(torch.backends.cudnn.fp32_precision)
            if hasattr(torch.backends, "cudnn")
            and hasattr(torch.backends.cudnn, "fp32_precision")
            else None
        )
        try:
            if orig_generic_mode is not None:
                torch.backends.fp32_precision = "none"
            if orig_cudnn_mode is not None:
                torch.backends.cudnn.fp32_precision = "none"
            for initial_mode in ("tf32", "ieee"):
                mm_backend.fp32_precision = initial_mode
                if conv_backend is not None:
                    conv_backend.fp32_precision = initial_mode
                with IEEE754FP32Guard():
                    assert str(mm_backend.fp32_precision) == "ieee"
                    if conv_backend is not None:
                        assert str(conv_backend.fp32_precision) == "ieee"
                    with IEEE754FP32Guard():
                        assert str(mm_backend.fp32_precision) == "ieee"
                        if conv_backend is not None:
                            assert str(conv_backend.fp32_precision) == "ieee"
                    assert str(mm_backend.fp32_precision) == "ieee"
                    if conv_backend is not None:
                        assert str(conv_backend.fp32_precision) == "ieee"
                assert str(mm_backend.fp32_precision) == initial_mode, (
                    f"IEEE754FP32Guard clobbered cuda.matmul.fp32_precision (expected {initial_mode})"
                )
                if conv_backend is not None:
                    assert str(conv_backend.fp32_precision) == initial_mode, (
                        f"IEEE754FP32Guard clobbered cudnn.conv.fp32_precision (expected {initial_mode})"
                    )
        finally:
            if orig_generic_mode is not None:
                torch.backends.fp32_precision = orig_generic_mode
            if orig_cudnn_mode is not None:
                torch.backends.cudnn.fp32_precision = orig_cudnn_mode
            mm_backend.fp32_precision = orig_mm_mode
            if conv_backend is not None and orig_conv_mode is not None:
                conv_backend.fp32_precision = orig_conv_mode
        print(
            "  [+] IEEE754FP32Guard           : Strict LIFO state capture & TF32 suppression verified."
        )

    c1_inst = WNConv1d(16, 32, kernel_size=3, padding=1)
    ct1_inst = WNConvTranspose1d(32, 16, kernel_size=10, stride=5, padding=3)
    assert isinstance(c1_inst, SovereignConv1d) and isinstance(c1_inst, nn.Conv1d)
    assert isinstance(ct1_inst, SovereignConvTranspose1d) and isinstance(
        ct1_inst, nn.ConvTranspose1d
    )

    gen_conv = torch.Generator(device="cpu").manual_seed(4242)
    max_conv_err = 0.0
    for conv_kw in (
        dict(batch=1, c_in=32, c_out=48, k_size=1, stride=1, padding=0, dilation=1, groups=1, t_in=257, use_bias=True),
        dict(batch=2, c_in=24, c_out=32, k_size=1, stride=1, padding=0, dilation=1, groups=1, t_in=129, use_bias=False),
        dict(batch=None, c_in=16, c_out=24, k_size=1, stride=1, padding=0, dilation=1, groups=1, t_in=97, use_bias=True),
        dict(batch=2, c_in=24, c_out=24, k_size=7, stride=1, padding=3, dilation=1, groups=1, t_in=131, use_bias=True),
        dict(batch=1, c_in=24, c_out=24, k_size=7, stride=1, padding=9, dilation=3, groups=1, t_in=131, use_bias=True),
        dict(batch=1, c_in=16, c_out=16, k_size=7, stride=1, padding=27, dilation=9, groups=1, t_in=64, use_bias=True),
        dict(batch=1, c_in=16, c_out=16, k_size=7, stride=1, padding=27, dilation=9, groups=1, t_in=20, use_bias=False),
        dict(batch=1, c_in=16, c_out=32, k_size=4, stride=2, padding=1, dilation=1, groups=1, t_in=129, use_bias=True),
        dict(batch=1, c_in=16, c_out=32, k_size=8, stride=4, padding=2, dilation=1, groups=1, t_in=131, use_bias=True),
        dict(batch=1, c_in=16, c_out=32, k_size=10, stride=5, padding=3, dilation=1, groups=1, t_in=133, use_bias=True),
        dict(batch=1, c_in=16, c_out=32, k_size=12, stride=6, padding=3, dilation=1, groups=1, t_in=137, use_bias=True),
        dict(batch=2, c_in=24, c_out=36, k_size=5, stride=2, padding=2, dilation=2, groups=4, t_in=115, use_bias=True),
        dict(batch=1, c_in=16, c_out=16, k_size=5, stride=1, padding="same", dilation=2, groups=2, t_in=73, use_bias=True),
    ):
        max_conv_err = max(
            max_conv_err, _audit_sovereign_conv1d_case(device, gen_conv, **conv_kw)
        )

    for deconv_kw in (
        dict(batch=1, c_in=32, c_out=16, k_size=4, stride=2, padding=1, output_padding=0, dilation=1, groups=1, t_in=67, use_bias=True),
        dict(batch=2, c_in=24, c_out=16, k_size=8, stride=4, padding=2, output_padding=0, dilation=1, groups=1, t_in=53, use_bias=True),
        dict(batch=1, c_in=32, c_out=16, k_size=10, stride=5, padding=3, output_padding=0, dilation=1, groups=1, t_in=47, use_bias=True),
        dict(batch=1, c_in=24, c_out=12, k_size=10, stride=5, padding=3, output_padding=2, dilation=1, groups=1, t_in=31, use_bias=False),
        dict(batch=1, c_in=32, c_out=16, k_size=12, stride=6, padding=3, output_padding=0, dilation=1, groups=1, t_in=43, use_bias=True),
        dict(batch=1, c_in=16, c_out=12, k_size=6, stride=3, padding=0, output_padding=2, dilation=1, groups=1, t_in=29, use_bias=True),
        dict(batch=None, c_in=16, c_out=8, k_size=4, stride=2, padding=1, output_padding=0, dilation=1, groups=1, t_in=35, use_bias=True),
        dict(batch=2, c_in=24, c_out=16, k_size=7, stride=3, padding=2, output_padding=1, dilation=1, groups=1, t_in=41, use_bias=True),
        dict(batch=2, c_in=16, c_out=12, k_size=5, stride=2, padding=3, output_padding=1, dilation=2, groups=1, t_in=37, use_bias=True),
        dict(batch=2, c_in=24, c_out=16, k_size=6, stride=3, padding=2, output_padding=0, dilation=2, groups=4, t_in=33, use_bias=True),
    ):
        err_val, out_sh = _audit_sovereign_conv_transpose1d_case(
            device, gen_conv, **deconv_kw
        )
        max_conv_err = max(max_conv_err, err_val)
        if deconv_kw["stride"] == 5 and deconv_kw["output_padding"] == 0:
            assert out_sh[-1] == 5 * deconv_kw["t_in"] - 1

    orig_ws_bytes = modeling_vae_mod._MAX_CONV_WORKSPACE_BYTES
    try:
        modeling_vae_mod._MAX_CONV_WORKSPACE_BYTES = 4096
        max_conv_err = max(
            max_conv_err,
            _audit_sovereign_conv1d_case(
                device,
                gen_conv,
                batch=2,
                c_in=16,
                c_out=24,
                k_size=7,
                stride=1,
                padding=9,
                dilation=3,
                groups=1,
                t_in=211,
                use_bias=True,
            ),
            _audit_sovereign_conv1d_case(
                device,
                gen_conv,
                batch=2,
                c_in=16,
                c_out=24,
                k_size=5,
                stride=2,
                padding=2,
                dilation=2,
                groups=4,
                t_in=199,
                use_bias=True,
            ),
            _audit_sovereign_conv_transpose1d_case(
                device,
                gen_conv,
                batch=2,
                c_in=24,
                c_out=16,
                k_size=10,
                stride=5,
                padding=3,
                output_padding=0,
                dilation=1,
                groups=1,
                t_in=151,
                use_bias=True,
            )[0],
            _audit_sovereign_conv_transpose1d_case(
                device,
                gen_conv,
                batch=2,
                c_in=16,
                c_out=12,
                k_size=7,
                stride=3,
                padding=2,
                output_padding=1,
                dilation=2,
                groups=2,
                t_in=139,
                use_bias=True,
            )[0],
        )
    finally:
        modeling_vae_mod._MAX_CONV_WORKSPACE_BYTES = orig_ws_bytes

    print(
        f"  [+] Sovereign 1D Contractions  : isinstance, L2 multi-tile & Fwd/Bwd Autograd max L_inf={max_conv_err:.2e} (<1e-5)"
    )

    v_raw = torch.randn((16, 8, 7), dtype=torch.bfloat16)
    g_raw = torch.rand((16, 1, 1), dtype=torch.bfloat16) + 0.5
    folded = fold_weight_norm_dict({"conv.weight_v": v_raw, "conv.weight_g": g_raw})[
        "conv.weight"
    ]
    assert folded.dtype == torch.float32
    v64 = v_raw.to(dtype=torch.float64)
    g64 = g_raw.to(dtype=torch.float64)
    ref_folded = (
        v64
        * (
            g64
            / torch.clamp_min(
                torch.linalg.vector_norm(v64, ord=2, dim=(1, 2), keepdim=True), 1e-15
            )
        )
    ).to(dtype=torch.float32)
    wn_err = float((folded - ref_folded).abs().max().item())
    assert wn_err == 0.0, f"Weight-norm FP64->FP32 folding mismatch: {wn_err}"

    try:
        fold_weight_norm_dict({"conv.weight_v": v_raw})
        raise AssertionError("fold_weight_norm_dict failed to reject orphan weight_v")
    except KeyError:
        pass

    with torch.inference_mode(False):
        sb = SnakeBeta(8).to(device=device, dtype=torch.float32).eval()
        sb._apply(lambda t: t, recurse=False)
        sb_x = torch.randn((2, 8, 37), device=device, dtype=torch.float32)
        sb_x_orig = sb_x.clone()
        _ = sb(sb_x)
        new_alpha = torch.full((8,), 0.4, dtype=torch.float32, device=device)
        new_beta = torch.full((8,), -0.2, dtype=torch.float32, device=device)
        sb.load_state_dict({"alpha": new_alpha, "beta": new_beta})
        sb_out = sb(sb_x)
        assert torch.equal(sb_x, sb_x_orig), "SnakeBeta mutated input x in-place"
        exp_a = torch.exp(_to_f64(new_alpha)).to(device=device, dtype=torch.float32).view(1, 8, 1)
        exp_ib = (
            (1.0 / (torch.exp(_to_f64(new_beta)) + 1e-9))
            .to(device=device, dtype=torch.float32)
            .view(1, 8, 1)
        )
        sb_ref = torch.addcmul(sb_x, torch.sin(sb_x * exp_a).pow(2), exp_ib)
        snake_err = float((sb_out - sb_ref).abs().max().item())
        assert snake_err < 1e-5, f"SnakeBeta Conductor/ATen drift: {snake_err:.2e}"

        sb_out_buf = torch.empty_like(sb_x)
        sb_ptr = sb_out_buf.data_ptr()
        sb_pre = snake_beta_fused(sb_x, exp_a, exp_ib, out=sb_out_buf)
        assert sb_pre.data_ptr() == sb_ptr and torch.equal(sb_pre, sb_out)

        f64_dev = _f64_device(device)
        sb_x64 = torch.randn((1, 8, 17), device=f64_dev, dtype=torch.float64)
        sb_y64 = snake_beta_fused(
            sb_x64,
            exp_a.to(device=f64_dev, dtype=torch.float64),
            exp_ib.to(device=f64_dev, dtype=torch.float64),
        )
        assert sb_y64.dtype == torch.float64, "snake_beta_fused downcast FP64 input"

        sx_g = torch.randn(
            (2, 8, 33), device=device, dtype=torch.float32, requires_grad=True
        )
        sa_g = exp_a.detach().clone().requires_grad_(True)
        sib_g = exp_ib.detach().clone().requires_grad_(True)
        sy_g = snake_beta_fused(sx_g, sa_g, sib_g)
        g_up = torch.randn_like(sy_g)
        (sy_g * g_up).sum().backward()

        sx_64 = _to_f64(sx_g.detach()).requires_grad_(True)
        sa_64 = _to_f64(sa_g.detach()).requires_grad_(True)
        sib_64 = _to_f64(sib_g.detach()).requires_grad_(True)
        sy_64 = sx_64 + torch.sin(sx_64 * sa_64).pow(2) * sib_64
        (sy_64 * _to_f64(g_up)).sum().backward()
        assert sx_g.grad is not None and sa_g.grad is not None and sib_g.grad is not None
        assert float((_to_f64(sx_g.grad) - sx_64.grad).abs().max().item()) < 1e-4
        assert float((_to_f64(sa_g.grad) - sa_64.grad).abs().max().item()) < 1e-4
        assert float((_to_f64(sib_g.grad) - sib_64.grad).abs().max().item()) < 1e-4

    print(
        f"  [+] SnakeBeta Fused & Autograd : Non-aliasing, FP64-preserving, out= & Bwd max L_inf={snake_err:.2e}"
    )

    vae_cfg = YuE2VAEConfig(
        encoder_config=dict(
            in_channels=2,
            channels=12,
            c_mults=[1, 2, 4],
            strides=[2, 5, 6],
            latent_dim=16,
            use_snake=True,
        ),
        decoder_config=dict(
            out_channels=2,
            channels=12,
            c_mults=[1, 2, 4],
            strides=[2, 5, 6],
            latent_dim=8,
            use_snake=True,
            snake_type="vanilla",
            use_filter=False,
            final_tanh=False,
        ),
        sample_rate=48000,
        latent_dim=8,
        downsampling_ratio=60,
        audio_channels=2,
        decode_core_frames=24,
        decode_halo_frames=20,
    )
    vae = (
        YuE2VAE(vae_cfg, decoder_only=False)
        .to(device=device, dtype=torch.float32)
        .eval()
    )
    vae._materialize_snake_caches()

    core_f = 24
    req_dec_halo = vae.required_halo(core_f)
    req_enc_halo = vae.required_encode_halo_frames(core_f)
    halo_f = max(20, req_dec_halo, req_enc_halo)
    total_f = core_f * 4 + 2 * halo_f + 7

    gen = torch.Generator(device="cpu").manual_seed(77)
    short_lat = torch.randn((1, 8, 12), dtype=torch.float32, generator=gen)
    short_mono = vae.decode(short_lat).cpu()
    short_tiled = vae.decode_tiled(
        short_lat, core_frames=core_f, halo_frames=halo_f, output_device="cpu"
    )
    assert torch.equal(short_mono, short_tiled), "Single-core decode_tiled mismatch"

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with _ZeroConvDispatchAudit() as dispatch_audit:
            _ = vae.encode(
                torch.randn((1, 2, 240), dtype=torch.float32, generator=gen),
                deterministic=True,
            )
            _ = vae.decode(short_lat[:, :, :4])
            with torch.inference_mode(False):
                wn_live = weight_norm(
                    WNConv1d(8, 12, kernel_size=3, padding=1)
                ).to(device=device, dtype=torch.float32)
                dt_live = WNConvTranspose1d(
                    12, 8, kernel_size=10, stride=5, padding=3
                ).to(device=device, dtype=torch.float32)
                g_live = WNConvTranspose1d(
                    8, 4, kernel_size=7, stride=3, padding=2, dilation=2
                ).to(device=device, dtype=torch.float32)
                x_live = torch.randn(
                    (1, 8, 25), device=device, dtype=torch.float32, requires_grad=True
                )
                y_live = g_live(dt_live(wn_live(x_live)))
                y_live.mean().backward()
                assert x_live.grad is not None and torch.isfinite(x_live.grad).all()

    assert dispatch_audit.op_count > 0
    assert not dispatch_audit.violations, (
        f"Forbidden convolution/MIOpen dispatch detected: {dispatch_audit.violations}"
    )
    is_rocm = bool(getattr(torch.version, "hip", None))
    if is_rocm or device.type == "cuda":
        assert len(caught_warnings) == 0, (
            f"Unexpected warnings emitted during VAE dispatch audit: {[str(w.message) for w in caught_warnings]}"
        )
    backend_tag = "ROCm" if is_rocm else device.type.upper()
    print(
        f"  [+] Zero-Conv Dispatch Audit   : {dispatch_audit.op_count} ATen ops audited on {backend_tag} (0 conv/MIOpen, 0 warnings)"
    )

    latents = torch.randn((1, 8, total_f), dtype=torch.float32, generator=gen)
    mono_out = vae.decode(latents).cpu()
    tiled_out = vae.decode_tiled(
        latents, core_frames=core_f, halo_frames=halo_f, output_device="cpu"
    )
    assert mono_out.shape == tiled_out.shape and tiled_out.dtype == torch.float32
    dec_tile_err = float((mono_out - tiled_out).abs().max().item())
    assert dec_tile_err < 1e-5, (
        f"Tiled decode boundary discontinuity: {dec_tile_err:.2e}"
    )

    enc_audio = torch.randn((1, 2, total_f * 60), dtype=torch.float32, generator=gen)
    mono_enc = vae.encode(enc_audio, deterministic=True).cpu()
    tiled_enc = vae.encode_tiled(
        enc_audio,
        core_frames=core_f,
        halo_frames=halo_f,
        deterministic=True,
        output_device="cpu",
    )
    assert mono_enc.shape == (1, 8, total_f) and tiled_enc.shape == mono_enc.shape
    enc_tile_err = float((mono_enc - tiled_enc).abs().max().item())
    assert enc_tile_err < 1e-5, (
        f"Tiled encode boundary discontinuity: {enc_tile_err:.2e}"
    )

    gen_m_stoch = torch.Generator(device=device).manual_seed(888)
    gen_t_stoch = torch.Generator(device=device).manual_seed(888)
    mono_stoch = vae.encode(
        enc_audio, deterministic=False, generator=gen_m_stoch
    ).cpu()
    tiled_stoch = vae.encode_tiled(
        enc_audio,
        core_frames=core_f,
        halo_frames=halo_f,
        deterministic=False,
        generator=gen_t_stoch,
        output_device="cpu",
    )
    stoch_err = float((mono_stoch - tiled_stoch).abs().max().item())
    assert stoch_err < 1e-5, (
        f"Stochastic tiled encode boundary discontinuity: {stoch_err:.2e}"
    )
    print(
        f"  [+] Odd-Stride Tiled VAE Parity: Dec={dec_tile_err:.2e}, Enc(det)={enc_tile_err:.2e}, Enc(stoch)={stoch_err:.2e}"
    )


def _assert_bf16_straddle_containment(x_in: torch.Tensor) -> None:
    bf16_max = float(torch.finfo(torch.bfloat16).max)
    if x_in.dtype == torch.float64:
        x_cmp = torch.clamp(x_in, -bf16_max, bf16_max)
        q_rtn = x_cmp.to(dtype=torch.bfloat16)
        z_near = q_rtn.to(dtype=torch.float64)
        neg_inf = torch.tensor(
            -float("inf"), device=x_in.device, dtype=torch.bfloat16
        )
        pos_inf = torch.tensor(
            float("inf"), device=x_in.device, dtype=torch.bfloat16
        )
        s_low = torch.where(
            x_cmp >= z_near,
            z_near,
            torch.nextafter(q_rtn, neg_inf).to(dtype=torch.float64),
        )
        s_high = torch.where(
            x_cmp > z_near,
            torch.nextafter(q_rtn, pos_inf).to(dtype=torch.float64),
            z_near,
        )
        q_cont = contender_cast(x_in, torch.bfloat16).to(dtype=torch.float64)
    else:
        x_cmp = torch.clamp(x_in, -bf16_max, bf16_max)
        q_rtn = x_cmp.to(dtype=torch.bfloat16)
        z_near = q_rtn.to(dtype=torch.float32)
        neg_inf = torch.tensor(
            -float("inf"), device=x_in.device, dtype=torch.bfloat16
        )
        pos_inf = torch.tensor(
            float("inf"), device=x_in.device, dtype=torch.bfloat16
        )
        s_low = torch.where(
            x_cmp >= z_near,
            z_near,
            torch.nextafter(q_rtn, neg_inf).to(dtype=torch.float32),
        )
        s_high = torch.where(
            x_cmp > z_near,
            torch.nextafter(q_rtn, pos_inf).to(dtype=torch.float32),
            z_near,
        )
        q_cont = contender_cast(x_in, torch.bfloat16).to(dtype=torch.float32)
    assert bool(((q_cont == s_low) | (q_cont == s_high)).all().item()), (
        "Bitwise contender_cast escaped straddling BF16 lattice bounds!"
    )


def _assert_fp32_straddle_containment(x_f64: torch.Tensor) -> None:
    fp32_max = float(torch.finfo(torch.float32).max)
    x_clamped = torch.clamp(x_f64, -fp32_max, fp32_max)
    q_rtn = x_clamped.to(dtype=torch.float32)
    z_near = q_rtn.to(dtype=torch.float64)
    neg_inf = torch.tensor(-float("inf"), device=x_f64.device, dtype=torch.float32)
    pos_inf = torch.tensor(float("inf"), device=x_f64.device, dtype=torch.float32)
    s_low = torch.where(
        x_clamped >= z_near,
        z_near,
        torch.nextafter(q_rtn, neg_inf).to(dtype=torch.float64),
    )
    s_high = torch.where(
        x_clamped > z_near,
        torch.nextafter(q_rtn, pos_inf).to(dtype=torch.float64),
        z_near,
    )
    q_cont = contender_cast(x_f64, torch.float32).to(dtype=torch.float64)
    assert bool(((q_cont == s_low) | (q_cont == s_high)).all().item()), (
        "Bitwise contender_cast escaped straddling FP32 lattice bounds!"
    )


@torch.no_grad()
def audit_gate_7_contender_and_refinery_lattice(device: torch.device) -> None:
    print(
        "\n--- Gate 7: Doubly-Balanced 3D Bayer Lattice, Zero-Sync / Graph ABI v3 & Refinery Straddle Audit ---"
    )
    f64_dev = _f64_device(device)
    if device.type == "cuda":
        warmup_conductor_gpu(device)

    base_8x8 = get_doubly_balanced_bayer_8x8(torch.device("cpu"), torch.float64)
    assert base_8x8.shape == (8, 8)
    expected_quantiles = (
        torch.arange(64, dtype=torch.float64, device="cpu") * 2.0 - 63.0
    ) / 128.0
    assert torch.equal(torch.sort(base_8x8.reshape(-1)).values, expected_quantiles)
    assert float(base_8x8.sum(dim=1).abs().max().item()) == 0.0
    assert float(base_8x8.sum(dim=0).abs().max().item()) == 0.0
    assert float(base_8x8.view(4, 2, 4, 2).sum(dim=(1, 3)).abs().max().item()) == 0.0

    r_idx = torch.arange(8, device="cpu")[:, None]
    c_idx = torch.arange(8, device="cpu")[None, :]
    expected_pos = ((r_idx + c_idx) & 1) == 1
    assert torch.equal(base_8x8 > 0.0, expected_pos)

    cube_64 = sample_bayer_lattice(
        (64, 64, 64), device=torch.device("cpu"), dtype=torch.float64
    )
    assert float(cube_64.sum(dim=0).abs().max().item()) == 0.0
    assert float(cube_64.sum(dim=1).abs().max().item()) == 0.0
    assert float(cube_64.sum(dim=2).abs().max().item()) == 0.0
    assert float(cube_64[:8].sum(dim=0).abs().max().item()) == 0.0

    z_g = torch.arange(64, device="cpu").view(64, 1, 1)
    r_g = torch.arange(64, device="cpu").view(1, 64, 1)
    c_g = torch.arange(64, device="cpu").view(1, 1, 64)
    assert torch.equal(cube_64 > 0.0, ((z_g + r_g + c_g) & 1) == 1), (
        "3D Nyquist checkerboard sign seam detected across 8x8x8 macro-blocks!"
    )

    ulp_bf16 = 2.0**-7
    alpha_test = 0.42
    target_val = float(
        torch.tensor(1.0 + alpha_test * ulp_bf16, dtype=torch.float32).item()
    )
    max_col_coherence_ratio = 0.0
    for shape_2d in ((64, 64), (64, 128), (64, 2048)):
        field_2d = torch.full(
            shape_2d, target_val, dtype=torch.float32, device=device
        )
        q_2d = contender_cast(field_2d, torch.bfloat16).to(dtype=torch.float32)
        rtn_2d = field_2d.to(dtype=torch.bfloat16).to(dtype=torch.float32)
        col_err_c = float((q_2d.mean(dim=0) - target_val).abs().max().item())
        row_err_c = float((q_2d.mean(dim=1) - target_val).abs().max().item())
        col_err_r = float((rtn_2d.mean(dim=0) - target_val).abs().max().item())
        ratio_2d = max(col_err_c, row_err_c) / max(col_err_r, 1e-12)
        max_col_coherence_ratio = max(max_col_coherence_ratio, ratio_2d)
        assert max(col_err_c, row_err_c) < col_err_r * 0.12, (
            f"2D {shape_2d} sub-ULP coherence cancellation failed: cont={col_err_c:.2e} vs RTN={col_err_r:.2e}"
        )

    field_3d = torch.full(
        (8, 16, 128), target_val, dtype=torch.float32, device=device
    )
    q_3d = contender_cast(field_3d, torch.bfloat16)
    rtn_3d = field_3d.to(dtype=torch.bfloat16)
    col_err_3d = float(
        (q_3d.to(torch.float32).mean(dim=(0, 1)) - target_val).abs().max().item()
    )
    col_rtn_3d = float(
        (rtn_3d.to(torch.float32).mean(dim=(0, 1)) - target_val).abs().max().item()
    )
    max_col_coherence_ratio = max(
        max_col_coherence_ratio, col_err_3d / max(col_rtn_3d, 1e-12)
    )
    assert col_err_3d < col_rtn_3d * 0.12, (
        f"3D [8, 16, 128] sub-ULP coherence cancellation failed: cont={col_err_3d:.2e} vs RTN={col_rtn_3d:.2e}"
    )
    for z1 in range(8):
        for z2 in range(z1 + 1, 8):
            assert not torch.equal(q_3d[z1], q_3d[z2]), (
                f"Cross-slice dither collision detected between slices {z1} and {z2}"
            )

    print(
        f"  [+] 2D/3D Doubly-Balanced Bayer: Row/Col/Slice sum=0.0 | Coherence={max_col_coherence_ratio:.3f}x RTN | 8-Slice Decorrelated"
    )

    all_bf16_cpu = torch.arange(-32768, 32768, dtype=torch.int16, device="cpu").view(
        torch.bfloat16
    )
    finite_bf16 = all_bf16_cpu[torch.isfinite(all_bf16_cpu)].to(device=device)
    assert finite_bf16.numel() == 65280
    roundtrip_bf16 = contender_cast(
        finite_bf16.to(dtype=torch.float32), torch.bfloat16
    )
    assert bool((roundtrip_bf16 == finite_bf16).all().item()), (
        "Bitwise contender_cast perturbed an exact finite BF16 lattice point!"
    )

    signed_zeros_f32 = torch.tensor([-0.0, 0.0], dtype=torch.float32, device=device)
    for dt_sz in (torch.bfloat16, torch.float16):
        sz_out = contender_cast(signed_zeros_f32, dt_sz)
        assert bool(
            (torch.signbit(sz_out) == torch.tensor([True, False], device=device))
            .all()
            .item()
        ), f"Signed-zero signbit lost on FP32->{dt_sz} contender_cast!"

    signed_zeros_f64 = torch.tensor([-0.0, 0.0], dtype=torch.float64, device=f64_dev)
    for dt_sz64 in (torch.float32, torch.bfloat16):
        sz_64_out = contender_cast(signed_zeros_f64, dt_sz64)
        assert bool(
            (torch.signbit(sz_64_out) == torch.tensor([True, False], device=f64_dev))
            .all()
            .item()
        ), f"Signed-zero signbit lost on FP64->{dt_sz64} contender_cast!"

    gen = torch.Generator(device="cpu").manual_seed(2026)
    for shape_nd in ((13,), (64,), (129,), (4099,), (17, 67), (5, 9, 71)):
        mix_f32 = torch.randn(shape_nd, dtype=torch.float32, generator=gen).to(
            device=device
        )
        flat_32 = mix_f32.view(-1)
        flat_32[: min(4, flat_32.numel())] = torch.tensor(
            [0.0, -0.0, 1.5e-40, -3.2e38], dtype=torch.float32, device=device
        )[: min(4, flat_32.numel())]
        _assert_bf16_straddle_containment(mix_f32)

        mix_f64 = torch.randn(shape_nd, dtype=torch.float64, generator=gen).to(
            device=f64_dev
        )
        flat_64 = mix_f64.view(-1)
        flat_64[: min(4, flat_64.numel())] = torch.tensor(
            [0.0, -0.0, 1.5e-41, -3.3e38], dtype=torch.float64, device=f64_dev
        )[: min(4, flat_64.numel())]
        _assert_fp32_straddle_containment(mix_f64)
        _assert_bf16_straddle_containment(mix_f64)

    async_verified = "CPU"
    if device.type == "cuda":
        cnd_ctx = get_conductor_context()
        async_2d = torch.randn((64, 128), dtype=torch.float32, device=device)
        async_3d = torch.randn((8, 16, 128), dtype=torch.float32, device=device)
        async_del = torch.randn((8, 16, 128), dtype=torch.float32, device=device)
        async_tail = torch.randn((3, 9, 67), dtype=torch.float32, device=device)
        async_f64 = (
            torch.tensor(
                [-0.0, 0.0, 1.2e-42, -7.5e-41, 0.123456789, -1234.56789],
                dtype=torch.float64,
                device=device,
            )
            .repeat(64)
            .reshape(3, 8, 16)
        )
        async_w = torch.ones((128,), dtype=torch.float32, device=device)
        async_gu = torch.randn((8, 16, 256), dtype=torch.float32, device=device)
        async_qkv = torch.randn((2, 8, (8 + 8) * 32), dtype=torch.float32, device=device)
        async_qw = torch.ones((32,), dtype=torch.float32, device=device)
        async_kw = torch.ones((32,), dtype=torch.float32, device=device)
        async_cs = torch.ones((2, 8, 16), dtype=torch.float32, device=device)
        async_sn = torch.zeros((2, 8, 16), dtype=torch.float32, device=device)
        async_sx = torch.randn((2, 16, 128), dtype=torch.float32, device=device)
        async_sa = torch.full((16,), 1.25, dtype=torch.float32, device=device)
        async_sib = torch.full((16,), 0.85, dtype=torch.float32, device=device)

        buf_rms = torch.empty((8, 16, 128), dtype=torch.bfloat16, device=device)
        buf_res = torch.empty((8, 16, 128), dtype=torch.float32, device=device)
        buf_res_nrm = torch.empty((8, 16, 128), dtype=torch.bfloat16, device=device)
        buf_res_inp = async_3d.clone()
        buf_swp = torch.empty((8, 16, 128), dtype=torch.bfloat16, device=device)
        buf_q = torch.empty((2, 8, 8, 32), dtype=torch.bfloat16, device=device)
        buf_k = torch.empty((2, 4, 8, 32), dtype=torch.bfloat16, device=device)
        buf_v = torch.empty((2, 4, 8, 32), dtype=torch.bfloat16, device=device)
        buf_kc = torch.zeros((2, 4, 24, 32), dtype=torch.bfloat16, device=device)
        buf_vc = torch.zeros((2, 4, 24, 32), dtype=torch.bfloat16, device=device)
        buf_sy = torch.empty_like(async_sx)

        _ = contender_cast(async_2d, torch.bfloat16)
        _ = contender_cast(async_3d, torch.float16)
        _ = contender_cast(async_tail, torch.bfloat16)
        _ = contender_cast(async_f64, torch.float32)
        _ = contender_cast(async_f64, torch.bfloat16)
        if cnd_ctx is not None:
            cnd_ctx.rms_norm_contender(
                async_3d, async_w, torch.bfloat16, phase_offset=3, out=buf_rms
            )
            cnd_ctx.residual_rms_norm_contender(
                async_3d,
                async_del,
                async_w,
                torch.bfloat16,
                phase_offset=4,
                res_out=buf_res,
                norm_out=buf_res_nrm,
            )
            cnd_ctx.swiglu_packed_contender(
                async_gu, torch.bfloat16, phase_offset=5, out=buf_swp
            )
            cnd_ctx.qk_norm_rope_packed_contender(
                async_qkv,
                async_qw,
                async_kw,
                async_cs,
                async_sn,
                8,
                4,
                32,
                torch.bfloat16,
                phase_q=1,
                phase_k=2,
                phase_v=3,
                layout_bhsd=True,
                q_out=buf_q,
                k_out=buf_k,
                v_out=buf_v,
            )
            cnd_ctx.qk_norm_rope_packed_strided_contender(
                async_qkv,
                async_qw,
                async_kw,
                async_cs,
                async_sn,
                8,
                4,
                32,
                buf_kc,
                buf_vc,
                kv_seq_offset=6,
                dst_dtype=torch.bfloat16,
                phase_q=1,
                phase_k=2,
                phase_v=3,
                q_out=buf_q,
            )
            cnd_ctx.snake_beta_f32(async_sx, async_sa, async_sib, out=buf_sy)
        torch.cuda.synchronize(device)

        if hasattr(torch.cuda, "set_sync_debug_mode"):
            prev_sync_mode = torch.cuda.get_sync_debug_mode()
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*Synchronization debug mode is a prototype feature.*",
                        category=UserWarning,
                    )
                    torch.cuda.set_sync_debug_mode("error")
                out_2d = contender_cast(async_2d, torch.bfloat16)
                out_3d = contender_cast(async_3d, torch.float16)
                out_tail = contender_cast(async_tail, torch.bfloat16)
                out_f32 = contender_cast(async_f64, torch.float32)
                out_bf16 = contender_cast(async_f64, torch.bfloat16)
                if cnd_ctx is not None:
                    cnd_ctx.rms_norm_contender(
                        async_3d, async_w, torch.bfloat16, phase_offset=3, out=buf_rms
                    )
                    cnd_ctx.residual_rms_norm_contender(
                        async_3d,
                        async_del,
                        async_w,
                        torch.bfloat16,
                        phase_offset=4,
                        res_out=buf_res,
                        norm_out=buf_res_nrm,
                    )
                    cnd_ctx.residual_rms_norm_contender(
                        buf_res_inp,
                        async_del,
                        async_w,
                        torch.bfloat16,
                        phase_offset=4,
                        res_out=buf_res_inp,
                        norm_out=buf_rms,
                    )
                    cnd_ctx.swiglu_packed_contender(
                        async_gu, torch.bfloat16, phase_offset=5, out=buf_swp
                    )
                    cnd_ctx.qk_norm_rope_packed_contender(
                        async_qkv,
                        async_qw,
                        async_kw,
                        async_cs,
                        async_sn,
                        8,
                        4,
                        32,
                        torch.bfloat16,
                        phase_q=1,
                        phase_k=2,
                        phase_v=3,
                        layout_bhsd=True,
                        q_out=buf_q,
                        k_out=buf_k,
                        v_out=buf_v,
                    )
                    cnd_ctx.qk_norm_rope_packed_strided_contender(
                        async_qkv,
                        async_qw,
                        async_kw,
                        async_cs,
                        async_sn,
                        8,
                        4,
                        32,
                        buf_kc,
                        buf_vc,
                        kv_seq_offset=6,
                        dst_dtype=torch.bfloat16,
                        phase_q=1,
                        phase_k=2,
                        phase_v=3,
                        q_out=buf_q,
                    )
                    cnd_ctx.snake_beta_f32(async_sx, async_sa, async_sib, out=buf_sy)
            finally:
                torch.cuda.set_sync_debug_mode(prev_sync_mode)
            torch.cuda.synchronize(device)
            assert (
                out_2d.dtype == torch.bfloat16
                and out_3d.dtype == torch.float16
                and out_tail.dtype == torch.bfloat16
                and out_f32.dtype == torch.float32
                and out_bf16.dtype == torch.bfloat16
            )
            assert torch.equal(buf_res_inp, buf_res)
            assert torch.equal(buf_rms.view(torch.int16), buf_res_nrm.view(torch.int16))
            assert torch.equal(buf_kc[:, :, 6:14, :].view(torch.int16), buf_k.view(torch.int16))
            assert torch.equal(buf_vc[:, :, 6:14, :].view(torch.int16), buf_v.view(torch.int16))
            async_verified = "ZERO-SYNC(v3)"

        if cnd_ctx is not None and hasattr(torch.cuda, "CUDAGraph"):
            eager_res = buf_res.clone()
            eager_rnrm = buf_res_nrm.clone()
            eager_swp = buf_swp.clone()
            eager_q = buf_q.clone()
            eager_kc = buf_kc.clone()
            eager_vc = buf_vc.clone()
            eager_sy = buf_sy.clone()

            g_res = torch.zeros_like(buf_res)
            g_rnrm = torch.zeros_like(buf_res_nrm)
            g_swp = torch.zeros_like(buf_swp)
            g_q = torch.zeros_like(buf_q)
            g_kc = torch.zeros_like(buf_kc)
            g_vc = torch.zeros_like(buf_vc)
            g_sy = torch.zeros_like(buf_sy)

            def _graph_body() -> None:
                cnd_ctx.residual_rms_norm_contender(
                    async_3d,
                    async_del,
                    async_w,
                    torch.bfloat16,
                    phase_offset=4,
                    res_out=g_res,
                    norm_out=g_rnrm,
                )
                cnd_ctx.swiglu_packed_contender(
                    async_gu, torch.bfloat16, phase_offset=5, out=g_swp
                )
                cnd_ctx.qk_norm_rope_packed_strided_contender(
                    async_qkv,
                    async_qw,
                    async_kw,
                    async_cs,
                    async_sn,
                    8,
                    4,
                    32,
                    g_kc,
                    g_vc,
                    kv_seq_offset=6,
                    dst_dtype=torch.bfloat16,
                    phase_q=1,
                    phase_k=2,
                    phase_v=3,
                    q_out=g_q,
                )
                cnd_ctx.snake_beta_f32(async_sx, async_sa, async_sib, out=g_sy)

            cap_stream = torch.cuda.Stream(device=device)
            cap_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(cap_stream):
                _graph_body()
            torch.cuda.current_stream(device).wait_stream(cap_stream)
            torch.cuda.synchronize(device)

            g_res.zero_()
            g_rnrm.zero_()
            g_swp.zero_()
            g_q.zero_()
            g_kc.zero_()
            g_vc.zero_()
            g_sy.zero_()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=cap_stream):
                _graph_body()
            graph.replay()
            torch.cuda.synchronize(device)

            assert torch.equal(g_res, eager_res), "CUDAGraph residual_rms_norm sum mismatch"
            assert torch.equal(g_rnrm.view(torch.int16), eager_rnrm.view(torch.int16)), (
                "CUDAGraph residual_rms_norm norm mismatch"
            )
            assert torch.equal(g_swp.view(torch.int16), eager_swp.view(torch.int16)), (
                "CUDAGraph swiglu_packed mismatch"
            )
            assert torch.equal(g_q.view(torch.int16), eager_q.view(torch.int16)), (
                "CUDAGraph strided QK-RoPE Q mismatch"
            )
            assert torch.equal(g_kc.view(torch.int16), eager_kc.view(torch.int16)), (
                "CUDAGraph strided QK-RoPE K cache mismatch"
            )
            assert torch.equal(g_vc.view(torch.int16), eager_vc.view(torch.int16)), (
                "CUDAGraph strided QK-RoPE V cache mismatch"
            )
            assert torch.equal(g_sy, eager_sy), "CUDAGraph snake_beta_f32 mismatch"
            async_verified = f"{async_verified}+GRAPH"

    print(
        f"  [+] Bitwise BF16/FP32 Kernel   : 65,280 exact points, signed zeros, N-D containment & {async_verified} verified."
    )

    refinery = TheRefinery(device)
    contender = TheContender(device)

    x_chunk = torch.randn((320, 64), dtype=torch.float32, generator=gen).to(
        device=device
    )
    q_cont_2d = contender.quantize(x_chunk, torch.bfloat16, phase_offset=7)
    q_ref_2d = refinery.quantize(
        x_chunk,
        torch.bfloat16,
        dither_enabled=True,
        max_chunk_channels=128,
        phase_offset=7,
    )
    assert torch.equal(q_cont_2d.view(torch.int16), q_ref_2d.view(torch.int16)), (
        "64-aligned chunked TheRefinery diverged from single-shot TheContender!"
    )

    x_1d_fp16 = torch.randn((512,), dtype=torch.float32, generator=gen).to(
        device=device
    )
    q_fp16 = refineTensor(x_1d_fp16, torch.float16, target_device=device, phase_offset=11)
    assert q_fp16.dtype == torch.float16 and torch.isfinite(q_fp16).all()
    _, s_lo16, s_hi16 = _exact_lattice_brackets_fp32(x_1d_fp16, torch.float16)
    q_fp16_f32 = q_fp16.to(dtype=torch.float32)
    assert bool(((q_fp16_f32 == s_lo16) | (q_fp16_f32 == s_hi16)).all().item()), (
        "FP16 refineTensor escaped straddling lattice bounds!"
    )

    audited_fp8: List[str] = []
    for fp8_dt in FP8_DTYPES:
        lut_fp8 = _get_fp8_sorted_lut(device, fp8_dt)
        min_v, max_v = _resolve_dtype_limits(fp8_dt)
        assert lut_fp8.numel() > 16
        assert float(lut_fp8[0].item()) == min_v and float(lut_fp8[-1].item()) == max_v
        x_fp8_src = (
            torch.randn((2, 16, 64), dtype=torch.float32, generator=gen).to(
                device=device
            )
            * 12.0
        )
        q_fp8 = refineTensor(
            x_fp8_src, fp8_dt, target_device=device, phase_offset=19
        )
        assert q_fp8.dtype == fp8_dt
        q_fp8_f32 = q_fp8.to(dtype=torch.float32)
        assert torch.isfinite(q_fp8_f32).all()
        _, s_lo8, s_hi8 = _exact_lattice_brackets_fp32(x_fp8_src, fp8_dt)
        assert bool(((q_fp8_f32 == s_lo8) | (q_fp8_f32 == s_hi8)).all().item()), (
            f"{fp8_dt} refineTensor escaped straddling FP8 lattice bounds!"
        )
        audited_fp8.append(str(fp8_dt).split(".")[-1])

    clear_refinery_cache()
    print(
        f"  [+] Refinery Straddle & FP8    : 64-aligned chunk parity exact; FP16 + FP8 ({', '.join(audited_fp8) or 'none'}) contained."
    )


def _synth_audit_groove(
    bpm: float, duration_s: float = 8.0, sr: int = 48000
) -> np.ndarray:
    n_samples = int(round(duration_s * sr))
    audio = np.zeros((n_samples, 2), dtype=np.float32)
    beat_period = 60.0 / float(bpm)
    click_len = int(round(0.030 * sr))
    t_click = np.arange(click_len, dtype=np.float64) / float(sr)
    kick = (
        0.85
        * np.sin(2.0 * math.pi * (55.0 + 110.0 * np.exp(-t_click / 0.008)) * t_click)
        * np.exp(-t_click / 0.012)
    ).astype(np.float32)
    t_pos = 0.0
    while t_pos < duration_s:
        idx = int(round(t_pos * sr))
        end = min(idx + click_len, n_samples)
        if end > idx:
            seg = kick[: end - idx]
            audio[idx:end, 0] += seg
            audio[idx:end, 1] += seg
        t_pos += beat_period
    return audio


def audit_gate_8_conductor_symbolic_midi_acoustic(device: torch.device) -> None:
    print(
        "\n--- Gate 8: Native Conductor ABI v3 Symbolic ABC, Multi-Tempo SMF1 MIDI & Acoustic BPM Audit ---"
    )
    info = get_conductor_info()
    ctx = get_conductor_context()
    cnd_mod = get_conductor_module()
    if not info["active"] or ctx is None or cnd_mod is None:
        print(
            "  [i] Native Conductor Runtime   : Not staged locally; pure-ATen fallback active."
        )
        return

    assert int(info["abi_version"]) >= MIN_CONDUCTOR_ABI_VERSION, (
        f"Conductor ABI v{info['abi_version']} < required v{MIN_CONDUCTOR_ABI_VERSION}"
    )

    resp_probe = GenerationResponse(
        output_path="audit.wav",
        sample_rate=48000,
        duration_seconds=1.0,
        total_samples=48000,
        generation_time_seconds=0.1,
        real_time_factor=0.1,
        peak_vram_gb=0.0,
        cpu_offload_active=False,
        cot_mode_used="full",
        abc_token_count=0,
        semantic_token_count=25,
        ode_steps_used=32,
        ode_method_used="midpoint",
        cfg_scale_used=1.0,
        effective_prompt="",
        effective_style="",
        effective_lyrics="",
        declick_applied=True,
        peak_linear=0.5,
        peak_dbfs=-6.0,
        rms_dbfs=-12.0,
        crest_factor_db=6.0,
        conductor_active=bool(info["active"]),
        conductor_abi_version=int(info["abi_version"]),
        conductor_gpu_backend=str(info["gpu_backend"]),
        conductor_fast_lut_active=bool(info["has_fast_lut"]),
    )
    assert resp_probe.conductor_abi_version == int(info["abi_version"])

    b86, m86, mpqn86 = ctx.canonicalize_bpm(86.0)
    assert m86 == 86000 and mpqn86 == 697674 and abs(b86 - 86.0) < 1e-9
    b_back, m_back = ctx.bpm_from_midi_mpqn(mpqn86)
    assert m_back == 86000 and abs(b_back - 86.0) < 1e-9

    tks_perf, num_p, den_p = ctx.quantize_beats(
        [0.0, 0.33, 0.5, 1.0],
        profile=cnd_mod.CndIntentProfile.PERFORMED_AUDIO,
        return_rationals=True,
    )
    assert tks_perf == (0, 6720, 10080, 20160)
    assert num_p == (0, 1, 1, 1) and den_p == (1, 3, 2, 1)

    frames, samples, secs = ctx.project_ticks_to_grid([0, 20160, 80640], bpm=120.0)
    assert frames == (0, 12, 50) or frames == (0, 13, 50)
    assert samples == (0, 24000, 96000)
    assert abs(secs[2] - 2.0) < 1e-12
    ticks_back = ctx.project_frames_to_ticks([0, 50], bpm=120.0)
    assert ticks_back == (0, 80640)

    abc_multi = (
        "X:1\nT:MultiTempo\nM:4/4\nL:1/4\nQ:1/4=120\nK:C\n"
        "C D E F | [Q:1/4=60] G A B c |\n"
        "Q:1/4=240\nc B A G |\n"
    )
    ana = ctx.analyze_abc(abc_multi)
    assert ana.bar_count == 3 and ana.milli_bpm == 120000
    assert ana.bars[0].codec_frame_span == (0, 50)
    assert ana.bars[1].codec_frame_span == (50, 150)
    assert ana.bars[2].codec_frame_span == (150, 175)
    assert ana.total_samples_48k == 336000

    abc_midbar = "X:1\nM:4/4\nL:1/4\nQ:1/4=120\nK:C\nC D [Q:1/4=60] E F |\n"
    ana_mb = ctx.analyze_abc(abc_midbar)
    assert ana_mb.bar_count == 1
    assert ana_mb.bars[0].codec_frame_span == (0, 75)
    assert ana_mb.total_samples_48k == 144000

    rewritten = ctx.rewrite_abc_tempo(abc_multi, 96.0)
    ana_rw = ctx.analyze_abc(rewritten, include_bars=False)
    assert ana_rw.milli_bpm == 96000

    ext_notes, ext_tempos, ext_meta = ctx.abc_extract_midi(abc_multi)
    assert len(ext_notes) == 12 and len(ext_tempos) == 3
    assert [te.milli_bpm for te in ext_tempos] == [120000, 60000, 240000]
    assert ext_meta["time_signature"] == (4, 4)

    smf_abc, smf_meta = ctx.abc_to_midi_smf1(abc_multi)
    assert smf_meta["note_count"] == 12 and smf_meta["tempo_count"] == 3
    dec_abc_notes, dec_abc_meta = ctx.decode_midi_smf1(smf_abc)
    assert len(dec_abc_notes) == 12
    assert [te.milli_bpm for te in dec_abc_meta["tempos"]] == [120000, 60000, 240000]
    assert [n.pitch for n in dec_abc_notes[:4]] == [60, 62, 64, 65]

    smf_mb, _ = ctx.abc_to_midi_smf1(abc_midbar)
    dec_mb_notes, dec_mb_meta = ctx.decode_midi_smf1(smf_mb)
    assert len(dec_mb_notes) == 4 and len(dec_mb_meta["tempos"]) == 2
    assert dec_mb_meta["tempos"][0].tick == 0 and dec_mb_meta["tempos"][0].milli_bpm == 120000
    assert dec_mb_meta["tempos"][1].tick == 40320 and dec_mb_meta["tempos"][1].milli_bpm == 60000

    explicit_notes = [
        cnd_mod.MidiNoteEvent(start_tick=0, duration_ticks=20160, pitch=60, velocity=100),
        cnd_mod.MidiNoteEvent(start_tick=20160, duration_ticks=20160, pitch=64, velocity=96),
        cnd_mod.MidiNoteEvent(start_tick=40320, duration_ticks=40320, pitch=67, velocity=92),
    ]
    explicit_tempos = [
        cnd_mod.TempoEvent(tick=0, milli_bpm=96000),
        cnd_mod.TempoEvent(tick=40320, milli_bpm=128000),
    ]
    smf_ex = ctx.encode_midi_smf1(
        explicit_notes,
        milli_bpm=96000,
        tempos=explicit_tempos,
        time_signature=(4, 4),
        key_sf=-4,
        is_minor=True,
    )
    dec_ex_notes, dec_ex_meta = ctx.decode_midi_smf1(smf_ex)
    assert len(dec_ex_notes) == 3
    assert [n.pitch for n in dec_ex_notes] == [60, 64, 67]
    assert len(dec_ex_meta["tempos"]) == 2
    assert dec_ex_meta["tempos"][0].milli_bpm == 96000
    assert dec_ex_meta["tempos"][1].milli_bpm == 128000
    assert dec_ex_meta["key_sf"] == -4 and dec_ex_meta["is_minor"] is True

    groove_96 = _synth_audit_groove(96.0, duration_s=8.0, sr=48000)
    est_96_inter = ctx.estimate_acoustic_bpm(groove_96, 48000, planar=False)
    est_96_planar = ctx.estimate_acoustic_bpm(
        np.ascontiguousarray(groove_96.T), 48000, planar=True
    )
    assert abs(est_96_inter - 96.0) < 1.5, f"96 BPM interleaved drift: {est_96_inter}"
    assert abs(est_96_inter - est_96_planar) < 1e-6, (
        f"Planar vs interleaved BPM drift: {est_96_inter} vs {est_96_planar}"
    )

    groove_128 = _synth_audit_groove(128.0, duration_s=8.0, sr=48000)
    est_128 = ctx.estimate_acoustic_bpm(
        torch.from_numpy(groove_128.T).to(device=device), 48000, planar=True
    )
    assert abs(est_128 - 128.0) < 1.5, f"128 BPM estimation drift: {est_128}"

    print(
        f"  [+] Conductor ABI v{info['abi_version']}          : "
        f"GPU={info['gpu_backend']} | LUT001={'MMAP' if info['has_fast_lut'] else 'MEM'} | "
        f"Mid-Bar [Q:] & Multi-Tempo SMF1 Exact | BPM={est_96_inter:.2f}/{est_128:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Doodle Architectural Sanity-Check & Mathematical Convergence Auditor."
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Target compute device for tensor verification.",
    )
    args = parser.parse_args()
    device = resolve_test_device(args.device)

    print("=" * 88)
    print(
        f"     DOODLE ARCHITECTURAL CONVERGENCE & ZERO-REGRESSION AUDITOR ({device})"
    )
    print("=" * 88)

    t0 = time.perf_counter()
    audit_gate_1_tokenizer_and_simd_prefix()
    audit_gate_2_polyphase_kaiser_resampler(device)
    model = audit_gate_3_placement_and_mixed_precision(device)
    audit_gate_4_batched_cfg_and_kv_splicing(model)
    audit_gate_5_nar_fp32_ode_and_manifold_inpainting(model)
    audit_gate_6_vae_fp32_guard_and_tiling(device)
    audit_gate_7_contender_and_refinery_lattice(device)
    audit_gate_8_conductor_symbolic_midi_acoustic(device)
    clear_yue2_workspaces(device)

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 88)
    print(
        f"ALL 8 ARCHITECTURAL GATES PASSED IN {elapsed:.2f}s (ZERO DOWNCASTING, ZERO MIOpen/CONV, ABI v3 CONVERGED)."
    )
    print("=" * 88)


if __name__ == "__main__":
    main()