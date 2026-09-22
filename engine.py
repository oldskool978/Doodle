from __future__ import annotations

import gc
import json
import math
import os
import random
import sys
import time
import unicodedata
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CACHE_DIR = ROOT_DIR / ".hf_cache"
TMP_DIR = ROOT_DIR / "artifacts" / "tmp"

for d in [CACHE_DIR, TMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "transformers")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "hub")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["TMP"] = str(TMP_DIR)
os.environ["TEMP"] = str(TMP_DIR)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors

from models.modeling_vae import YuE2VAE, YuE2VAEConfig
from models.modeling_yue2 import StaticKVCache, YuE2Config, YuE2ForCausalLM
from models.tokenizer import YuE2Tokenizer
from schema import (
    GenerationRequest,
    GenerationResponse,
    INSTRUCTIONS,
    get_active_engine_defaults,
)

EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
LATENT_START, LATENT_END, LATENT_PAD = 184621, 184622, 184623
VOCAB_SIZE, CONTEXT = 184704, 24576


def apply_sub_millisecond_declick(audio_tensor: torch.Tensor, fade_samples: int = 512) -> torch.Tensor:
    if audio_tensor.shape[-1] <= fade_samples * 2:
        return audio_tensor
    fade = 0.5 * (
        1.0
        - torch.cos(
            torch.linspace(
                0.0,
                math.pi,
                fade_samples,
                device=audio_tensor.device,
                dtype=audio_tensor.dtype,
            )
        )
    )
    audio_tensor[..., :fade_samples] *= fade
    audio_tensor[..., -fade_samples:] *= torch.flip(fade, dims=[0])
    return audio_tensor


def resolve_model_path(repo_or_path: str) -> Path:
    direct_path = Path(repo_or_path)
    if direct_path.exists():
        return direct_path
    hub_path = CACHE_DIR / "hub"
    repo_folder_name = "models--" + repo_or_path.replace("/", "--")
    candidate = hub_path / repo_folder_name / "snapshots"
    if candidate.exists():
        snapshots = list(candidate.iterdir())
        if snapshots:
            return snapshots[0]
    return direct_path


def apply_windowed_penalty(
    logits: torch.Tensor,
    history: List[int],
    penalty: float,
    window: int,
) -> torch.Tensor:
    if penalty == 1.0 or not history:
        return logits
    recent = history[-window:]
    recent_tensor = torch.as_tensor(recent, dtype=torch.long, device=logits.device).reshape(1, -1)
    freq = torch.zeros_like(logits)
    freq.scatter_add_(-1, recent_tensor, torch.ones_like(recent_tensor, dtype=logits.dtype))
    alpha = penalty**freq
    return torch.where(logits < 0, logits * alpha, logits / alpha)


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
) -> int:
    scores = logits.float().clone()
    end_token = ABC_END if phase == "abc" else MUSIC_END
    allowed = torch.full_like(scores, -float("inf"))

    if phase == "abc":
        allowed[..., :EOD] = 0.0
    else:
        allowed[..., CODEC_OFFSET : CODEC_OFFSET + CODEC_SIZE] = 0.0
    allowed[..., end_token] = 0.0
    scores = scores + allowed

    if step < min_tokens:
        scores[..., end_token] = -float("inf")

    scores = apply_windowed_penalty(scores, history, penalty, penalty_window)

    if temperature <= 0.0:
        return int(scores.argmax(-1).item())

    if temperature != 1.0:
        scores = scores / temperature

    k_val = min(top_k, scores.shape[-1])
    threshold = scores.topk(k_val, dim=-1).values[..., -1, None]
    scores = scores.masked_fill(scores < threshold, -float("inf"))

    if top_p < 1.0:
        sorted_vals, sorted_indices = scores.sort(descending=True, dim=-1)
        probs = sorted_vals.softmax(dim=-1)
        cum_probs = probs.cumsum(dim=-1)
        exceeded = (cum_probs - probs) > top_p
        exceeded[..., 0] = False
        sorted_vals = sorted_vals.masked_fill(exceeded, -float("inf"))
        scores = scores.scatter(-1, sorted_indices, sorted_vals)

    probabilities = scores.softmax(dim=-1)
    next_id = torch.multinomial(probabilities, 1, generator=generator)
    return int(next_id.item())


def partition_song_chunks(prefix: List[int], codec: List[int], seed: int, context_len: int = CONTEXT):
    prefix_len = len(prefix)
    chunk_capacity = min((context_len - prefix_len - 3) // 2, context_len)
    if chunk_capacity < 1 or not codec:
        raise ValueError("Sequence length exceeds permissible acoustic context boundaries.")

    ranges = [(i, min(i + chunk_capacity, len(codec))) for i in range(0, len(codec), chunk_capacity)]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((len(codec), 64), dtype=torch.float32, device="cpu", generator=generator)

    chunks = []
    for a, b in ranges:
        ar_toks = prefix + [c + CODEC_OFFSET for c in codec[a:b]] + [MUSIC_END]
        chunks.append({"tokens": ar_toks, "noise": noise[a:b]})
    return chunks


class CachedNARChunkSolver:
    def __init__(self, model: YuE2ForCausalLM, chunk_tokens: List[int], initial_noise: torch.Tensor):
        self.model = model
        self.chunk_tokens = chunk_tokens
        self.initial_noise = initial_noise
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        self.ar_length = len(chunk_tokens)
        self.nar_length = len(initial_noise) + 2

        pos = torch.arange(self.ar_length, self.ar_length + self.nar_length, device=self.device)[None]
        self.cos, self.sin = model.model.rotary_emb(pos)
        local_ids = torch.arange(self.nar_length, device=self.device).clamp(max=model.config.max_latent_frames - 1)
        self.pos_emb = model.latent_pos_embed(local_ids)[None]

        self.kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._prefill_ar()

    @torch.inference_mode()
    def _prefill_ar(self) -> None:
        backbone = self.model.model
        tokens_tensor = torch.tensor([self.chunk_tokens], dtype=torch.long, device=self.device)
        positions = torch.arange(self.ar_length, device=self.device)[None]
        cos, sin = backbone.rotary_emb(positions)
        x = backbone.embed_tokens(tokens_tensor)

        for layer in backbone.layers:
            q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos, sin)
            self.kv_cache.append((k[0].clone(), v[0].clone()))
            q_tx = q.transpose(1, 2)
            k_tx = k.transpose(1, 2)
            v_tx = v.transpose(1, 2)
            h = F.scaled_dot_product_attention(
                q_tx,
                k_tx,
                v_tx,
                is_causal=True,
                enable_gqa=layer.self_attn.num_heads != layer.self_attn.num_kv_heads,
            ).transpose(1, 2)
            x = x + layer.self_attn.o_proj(h.reshape(1, self.ar_length, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))

    @torch.inference_mode()
    def velocity(self, state: torch.Tensor, raw_t: float) -> torch.Tensor:
        x_nar = F.pad(state, (0, 0, 1, 1))
        t_shifted = self.model._shift_t_value(raw_t, self.device, self.dtype)
        x = self.model.vae2llm(x_nar[None])
        x = x + self.model.time_embedder(t_shifted.expand(self.nar_length))[None]
        x = x + self.pos_emb

        for layer, (ar_k, ar_v) in zip(self.model.model.layers, self.kv_cache):
            q, k, v = layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(x), self.cos, self.sin)
            k_comb = torch.cat((ar_k, k[0]), dim=0)
            v_comb = torch.cat((ar_v, v[0]), dim=0)

            q_tx = q.transpose(1, 2)
            k_tx = k_comb.unsqueeze(0).transpose(1, 2)
            v_tx = v_comb.unsqueeze(0).transpose(1, 2)

            h = F.scaled_dot_product_attention(
                q_tx,
                k_tx,
                v_tx,
                is_causal=False,
                enable_gqa=layer.nar_self_attn.num_heads != layer.nar_self_attn.num_kv_heads,
            ).transpose(1, 2)

            x = x + layer.nar_self_attn.o_proj(h.reshape(1, self.nar_length, -1))
            x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))

        return self.model.llm2vae(self.model.model.norm(x))[0, 1:-1]

    @torch.inference_mode()
    def solve(
        self,
        steps: int = 32,
        method: str = "midpoint",
        step_callback: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        state = self.initial_noise.to(device=self.device, dtype=self.dtype)
        dt = 1.0 / steps

        for step in range(steps):
            t = 1.0 - step * dt
            raw_t = torch.logit(torch.tensor(t, dtype=torch.float64, device="cpu")).clamp(-20.0, 20.0).item()

            if method == "euler":
                v = self.velocity(state, raw_t)
                state = state - v * dt
            elif method == "heun":
                v1 = self.velocity(state, raw_t)
                state_pred = state - v1 * dt
                t_next = max(0.0, t - dt)
                raw_next = torch.logit(torch.tensor(t_next, dtype=torch.float64, device="cpu")).clamp(-20.0, 20.0).item()
                v2 = self.velocity(state_pred, raw_next)
                state = state - 0.5 * (v1 + v2) * dt
            else:
                v1 = self.velocity(state, raw_t)
                state_mid = state - v1 * (dt / 2.0)
                raw_mid = torch.logit(torch.tensor(t - dt / 2.0, dtype=torch.float64, device="cpu")).clamp(-20.0, 20.0).item()
                v2 = self.velocity(state_mid, raw_mid)
                state = state - v2 * dt

            if step_callback is not None:
                step_callback(step + 1, steps)

        result = state.float().cpu()
        if not torch.isfinite(result).all():
            raise FloatingPointError("NAR flow-matching integration yielded non-finite values.")
        return result

    def close(self) -> None:
        self.kv_cache.clear()
        self.cos = self.sin = self.pos_emb = None


class MusicEngine:
    def __init__(
        self,
        repo_id: Optional[str] = None,
        vae_repo_id: Optional[str] = None,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.repo_id = repo_id or "m-a-p/YuE2-3B"
        self.vae_repo_id = vae_repo_id or "m-a-p/YuE2-Vae"
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype

        self.model_path = resolve_model_path(self.repo_id)
        self.vae_path = resolve_model_path(self.vae_repo_id)

        ranks_path = self.model_path / "qwen.ranks.bin"
        if not ranks_path.exists():
            ranks_path = self.model_path / "qwen.tiktoken"
        self.tokenizer = YuE2Tokenizer(ranks_path)

        self.model: Optional[YuE2ForCausalLM] = None
        self.vae: Optional[YuE2VAE] = None
        self._current_offload_state: Optional[bool] = None

    def _init_components(self, cpu_offload: bool) -> None:
        if self.model is not None and self.vae is not None and self._current_offload_state == cpu_offload:
            return

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA execution requested but no compatible GPU detected.")

        if self.model is not None:
            del self.model, self.vae
            self.model = None
            self.vae = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cfg_file = self.model_path / "config.json"
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg = YuE2Config(**cfg_dict)

        model_device = torch.device("cpu") if cpu_offload else self.device
        self.model = YuE2ForCausalLM(cfg).to(device=model_device, dtype=self.dtype).eval()

        weights_file = self.model_path / "model.safetensors"
        state = load_safetensors(str(weights_file), device="cpu")
        self.model.load_state_dict(state, strict=False)
        del state
        self.model.to(device=model_device, dtype=self.dtype).eval()

        vae_device = torch.device("cpu") if cpu_offload else self.device
        self.vae = YuE2VAE.from_pretrained(self.vae_path, decoder_only=True, device=vae_device)

        self._current_offload_state = cpu_offload

    @torch.inference_mode()
    def synthesize(
        self,
        request: GenerationRequest,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> GenerationResponse:
        request.validate()
        self._init_components(request.cpu_offload)
        defaults = get_active_engine_defaults()

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.perf_counter()
        explicit_seed = request.seed if (request.seed is not None and request.seed >= 0) else random.randint(100000, 99999999)
        random.seed(explicit_seed)
        np.random.seed(explicit_seed % (2**32))
        torch.manual_seed(explicit_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(explicit_seed)

        generator = torch.Generator(device=self.device).manual_seed(explicit_seed)

        cot_mode = request.cot or defaults["cot"]
        instruction = INSTRUCTIONS.get(cot_mode, INSTRUCTIONS["full"])
        full_text = request.compile_full_text()
        style_text = request.compile_style()
        lyrics_text = request.sanitize_lyrics()

        base_tokens = [EOD] + self.tokenizer.encode(full_text)

        if request.cpu_offload:
            self.model.to(self.device)

        abc_output_text: Optional[str] = None
        abc_ids: List[int] = []

        if cot_mode != "off":
            if request.abc is not None and request.abc.strip():
                abc_output_text = request.abc.strip()
                abc_ids = self.tokenizer.encode(abc_output_text)
            else:
                abc_prefix = base_tokens + [ABC_START]
                cache = StaticKVCache(
                    num_layers=self.model.config.num_hidden_layers,
                    batch_size=1,
                    num_kv_heads=self.model.config.num_key_value_heads,
                    max_seq_len=len(abc_prefix) + 4096,
                    head_dim=self.model.config.head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
                pre_in = torch.tensor([abc_prefix], dtype=torch.long, device=self.device)
                logits = self.model(input_ids=pre_in, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1, :]

                history: List[int] = []
                for step in range(4096):
                    token = sample_categorical_distribution(
                        logits=logits,
                        temperature=request.abc_temperature if request.abc_temperature is not None else defaults["abc_temperature"],
                        top_p=request.abc_top_p if request.abc_top_p is not None else defaults["abc_top_p"],
                        top_k=request.abc_top_k if request.abc_top_k is not None else defaults["abc_top_k"],
                        history=history,
                        step=step,
                        min_tokens=32,
                        penalty=request.abc_repetition_penalty if request.abc_repetition_penalty is not None else defaults["abc_repetition_penalty"],
                        penalty_window=request.abc_penalty_window if request.abc_penalty_window is not None else defaults["abc_penalty_window"],
                        phase="abc",
                        generator=generator,
                    )
                    if token == ABC_END:
                        break
                    history.append(token)
                    if progress_callback is not None:
                        progress_callback("abc", step + 1, 4096)
                    step_in = torch.tensor([[token]], dtype=torch.long, device=self.device)
                    logits = self.model(input_ids=step_in, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1, :]

                abc_ids = history
                abc_output_text = self.tokenizer.decode(abc_ids)
                del cache

        if cot_mode == "off":
            song_prefix = base_tokens + [ABC_START, ABC_END, MUSIC_START]
        else:
            song_prefix = base_tokens + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]

        resolved_cfg = request.cfg_scale if request.cfg_scale is not None else defaults["cfg_scale"]
        apply_cfg = abs(resolved_cfg - 1.0) > 1e-4

        max_semantic_tokens = min(int(request.audio_duration * 25.0), 9000)
        cache_pos = StaticKVCache(
            num_layers=self.model.config.num_hidden_layers,
            batch_size=1,
            num_kv_heads=self.model.config.num_key_value_heads,
            max_seq_len=len(song_prefix) + max_semantic_tokens + 16,
            head_dim=self.model.config.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        pre_in_pos = torch.tensor([song_prefix], dtype=torch.long, device=self.device)
        logits_pos = self.model(input_ids=pre_in_pos, past_key_values=cache_pos, use_cache=True, logits_to_keep=1).logits[:, -1, :]

        cache_neg: Optional[StaticKVCache] = None
        logits_neg: Optional[torch.Tensor] = None
        if apply_cfg:
            neg_base = [EOD] + self.tokenizer.encode(instruction)
            if cot_mode == "off":
                neg_prefix = neg_base + [MUSIC_START]
            else:
                neg_prefix = neg_base + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]

            cache_neg = StaticKVCache(
                num_layers=self.model.config.num_hidden_layers,
                batch_size=1,
                num_kv_heads=self.model.config.num_key_value_heads,
                max_seq_len=len(neg_prefix) + max_semantic_tokens + 16,
                head_dim=self.model.config.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            pre_in_neg = torch.tensor([neg_prefix], dtype=torch.long, device=self.device)
            logits_neg = self.model(input_ids=pre_in_neg, past_key_values=cache_neg, use_cache=True, logits_to_keep=1).logits[:, -1, :]

        semantic_tokens: List[int] = []
        for step in range(max_semantic_tokens):
            if apply_cfg and logits_neg is not None:
                effective_logits = logits_neg + resolved_cfg * (logits_pos - logits_neg)
            else:
                effective_logits = logits_pos

            token = sample_categorical_distribution(
                logits=effective_logits,
                temperature=request.temperature if request.temperature is not None else defaults["temperature"],
                top_p=request.top_p if request.top_p is not None else defaults["top_p"],
                top_k=request.top_k if request.top_k is not None else defaults["top_k"],
                history=semantic_tokens,
                step=step,
                min_tokens=200,
                penalty=request.repetition_penalty if request.repetition_penalty is not None else defaults["repetition_penalty"],
                penalty_window=request.penalty_window if request.penalty_window is not None else defaults["penalty_window"],
                phase="semantic",
                generator=generator,
            )
            if token == MUSIC_END:
                break
            semantic_tokens.append(token)
            if progress_callback is not None:
                progress_callback("semantic", step + 1, max_semantic_tokens)

            step_in = torch.tensor([[token]], dtype=torch.long, device=self.device)
            logits_pos = self.model(input_ids=step_in, past_key_values=cache_pos, use_cache=True, logits_to_keep=1).logits[:, -1, :]
            if apply_cfg and cache_neg is not None:
                logits_neg = self.model(input_ids=step_in, past_key_values=cache_neg, use_cache=True, logits_to_keep=1).logits[:, -1, :]

        del cache_pos, cache_neg

        raw_codec = [t - CODEC_OFFSET for t in semantic_tokens]
        if not raw_codec:
            raise RuntimeError("Autoregressive generation terminated with zero acoustic codec frames.")

        chunks = partition_song_chunks(song_prefix, raw_codec, explicit_seed, context_len=CONTEXT)
        total_chunks = len(chunks)
        ode_steps = request.num_inference_steps if request.num_inference_steps is not None else defaults["num_inference_steps"]
        ode_method = request.ode_method if request.ode_method is not None else defaults["ode_method"]

        latent_pieces = []
        for c_idx, chk in enumerate(chunks):
            solver = CachedNARChunkSolver(self.model, chk["tokens"], chk["noise"])
            def cb_step(cur: int, tot: int):
                if progress_callback is not None:
                    progress_callback("nar", c_idx * tot + cur, total_chunks * tot)
            chunk_latent = solver.solve(steps=ode_steps, method=ode_method, step_callback=cb_step)
            solver.close()
            del solver
            latent_pieces.append(chunk_latent)

        latents_fp32 = torch.cat(latent_pieces, dim=0).T.unsqueeze(0)

        if request.cpu_offload:
            self.model.to("cpu")
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
            self.vae.to(self.device)

        def cb_vae(cur: int, tot: int):
            if progress_callback is not None:
                progress_callback("vae", cur, tot)

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

        if audio_tensor.ndim == 3:
            audio_tensor = audio_tensor.squeeze(0)

        if request.apply_declick if request.apply_declick is not None else defaults["apply_declick"]:
            audio_tensor = apply_sub_millisecond_declick(audio_tensor, fade_samples=512)

        elapsed_time = time.perf_counter() - start_time
        peak_vram_gb = 0.0
        if self.device.type == "cuda" and torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)

        peak_val = torch.max(torch.abs(audio_tensor)).item()
        rms_val = torch.sqrt(torch.mean(audio_tensor**2)).item()
        peak_dbfs = 20.0 * math.log10(max(peak_val, 1e-12))
        rms_dbfs = 20.0 * math.log10(max(rms_val, 1e-12))
        crest_factor_db = peak_dbfs - rms_dbfs

        audio_data = audio_tensor.detach().cpu().numpy()
        if audio_data.shape[0] < audio_data.shape[1]:
            audio_data = audio_data.T
        audio_data = np.ascontiguousarray(audio_data, dtype=np.float32)

        out_path = Path(request.output_path)
        if not out_path.is_absolute():
            out_path = ROOT_DIR / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(str(out_path), audio_data, 48000, subtype="FLOAT")
        total_samples = audio_data.shape[0]
        actual_duration = total_samples / 48000.0
        rtf = elapsed_time / max(actual_duration, 1e-6)

        return GenerationResponse(
            output_path=str(out_path),
            sample_rate=48000,
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
            cfg_scale_used=resolved_cfg,
            effective_prompt=full_text,
            effective_style=style_text,
            effective_lyrics=lyrics_text,
            abc_notation=abc_output_text,
            declick_applied=request.apply_declick if request.apply_declick is not None else defaults["apply_declick"],
            peak_linear=peak_val,
            peak_dbfs=peak_dbfs,
            rms_dbfs=rms_dbfs,
            crest_factor_db=crest_factor_db,
            is_instrumental_used=request.is_instrumental,
            instrumental_branch_used=request.instrumental_branch,
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

        resp = self.synthesize(request, progress_callback=cb if progress_callback else None)
        data, sr = sf.read(resp.output_path, dtype="float32")
        tensor = torch.from_numpy(data.T if data.ndim > 1 else data[None, :])
        return tensor, sr

    @torch.inference_mode()
    def warmup(self) -> None:
        self._init_components(cpu_offload=False)
        dummy_ar = torch.tensor([[EOD, ABC_START, 100, 200, ABC_END, MUSIC_START]], dtype=torch.long, device=self.device)
        _ = self.model(input_ids=dummy_ar, use_cache=False)

        t_lat = 4
        ar_mask = torch.tensor([[True] * 6 + [False] * 6], device=self.device, dtype=torch.bool)
        nar_mask = torch.tensor([[False] * 6 + [True] * 6], device=self.device, dtype=torch.bool)
        nar_content_mask = torch.tensor([[False] * 7 + [True] * t_lat + [False] * 1], device=self.device, dtype=torch.bool)
        tokens = torch.tensor([[EOD, ABC_START, 100, 200, ABC_END, MUSIC_START, LATENT_START, 0, 0, 0, 0, LATENT_END]], dtype=torch.long, device=self.device)
        x_t = torch.randn((t_lat, 64), device=self.device, dtype=self.dtype)
        _ = self.model.nar_velocity(tokens, ar_mask, nar_mask, nar_content_mask, x_t, 0.5)

        dummy_latents = torch.randn((1, 64, 25), device=self.device, dtype=torch.float32)
        _ = self.vae.decode_tiled(dummy_latents, core_frames=16, halo_frames=16, output_device="cpu")