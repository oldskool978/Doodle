from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast


def sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = False) -> torch.Tensor:
    grouped = query.shape[1] != key.shape[1]
    if grouped and query.device.type == "mps":
        groups = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        grouped = False
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        enable_gqa=grouped,
    )


def _causal_mask(
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    key_length: int,
    batch_size: int,
) -> torch.Tensor:
    device = cache_position.device
    visible = torch.arange(key_length, device=device)[None, :] <= cache_position[:, None]
    visible = visible[None, None].expand(batch_size, 1, -1, -1)
    if attention_mask is None:
        return visible
    mask = attention_mask.to(device=device)
    if mask.ndim == 2:
        if mask.shape[0] != batch_size or mask.shape[1] > key_length:
            raise ValueError("2D attention mask must span current batch and populated cache sequences.")
        if mask.shape[1] < key_length:
            mask = F.pad(mask, (0, key_length - mask.shape[1]), value=0)
        return visible & mask[:, None, None, :].bool()
    if mask.ndim != 4 or mask.shape[-2:] != visible.shape[-2:]:
        raise ValueError("Dimensional layout of attention mask fails 4D causal verification.")
    if mask.dtype == torch.bool:
        return visible & mask
    return mask.masked_fill(~visible, float("-inf"))


class YuE2Config(PretrainedConfig):
    model_type = "yue2"
    _hf_fields = frozenset({
        "model_type", "architectures", "auto_map", "transformers_version",
        "dtype", "torch_dtype", "return_dict", "output_hidden_states",
        "output_attentions", "use_cache", "tie_word_embeddings", "torchscript",
        "is_decoder", "is_encoder_decoder", "add_cross_attention",
        "bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id",
        "attn_implementation",
    })
    _inference_fields = frozenset({
        "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "intermediate_size", "vocab_size", "rms_norm_eps", "rope_theta",
        "max_position_embeddings", "tie_word_embeddings", "latent_type", "latent_dim",
        "max_latent_frames", "timestep_shift",
    })

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
            raise ValueError("YuE2 specification requires continuous latent_type='vae'.")
        clean_kwargs = {key: value for key, value in kwargs.items() if key in self._hf_fields}
        super().__init__(tie_word_embeddings=tie_word_embeddings, return_dict=return_dict, **clean_kwargs)
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
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 1000000.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self._inv_freq: Optional[torch.Tensor] = None

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._inv_freq is None or self._inv_freq.device != position_ids.device:
            self._inv_freq = 1.0 / (
                self.base ** (
                    torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=position_ids.device) / self.head_dim
                )
            )
        pos = position_ids.float().unsqueeze(-1)
        angles = pos * self._inv_freq
        return angles.cos(), angles.sin()


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, config: YuE2Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def project_qkv(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        rc, rs = cos.unsqueeze(2), sin.unsqueeze(2)
        q = _apply_rotary(q, rc, rs)
        k = _apply_rotary(k, rc, rs)
        return q, k, v

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
        b, t, _ = x.shape
        q, k, v = self.project_qkv(x, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, layer_idx, {"cache_position": cache_position})
        if attention_mask is not None:
            out = sdpa(q, k, v, attn_mask=attention_mask[..., : k.shape[2]])
        else:
            out = sdpa(q, k, v, is_causal=(t > 1 and k.shape[2] == t))
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


class MLP(nn.Module):
    def __init__(self, config: YuE2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: YuE2Config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.nar_input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.nar_self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)
        self.nar_pre_mlp_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.nar_mlp = MLP(config)

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
    ) -> torch.Tensor:
        if ar_mask is not None:
            mask_3d = ar_mask.unsqueeze(-1)
            mask_4d = ar_mask.unsqueeze(-1).unsqueeze(-1)

            ln_ar = self.input_layernorm(x)
            ln_nar = self.nar_input_layernorm(x)

            q_ar, k_ar, v_ar = self.self_attn.project_qkv(ln_ar, cos, sin)
            q_nar, k_nar, v_nar = self.nar_self_attn.project_qkv(ln_nar, cos, sin)

            query = torch.where(mask_4d, q_ar, q_nar)
            key = torch.where(mask_4d, k_ar, k_nar)
            value = torch.where(mask_4d, v_ar, v_nar)

            b, s = x.shape[:2]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            if attention_mask is not None and attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.to(query.dtype)

            core_out = sdpa(query, key, value, attn_mask=attention_mask)
            core_out = core_out.transpose(1, 2).reshape(b, s, -1)

            o_ar = self.self_attn.o_proj(core_out)
            o_nar = self.nar_self_attn.o_proj(core_out)
            h = torch.where(mask_3d, o_ar, o_nar)
            x = x + h

            ar_out = self.mlp(self.post_attention_layernorm(x))
            nar_out = self.nar_mlp(self.nar_pre_mlp_layernorm(x))
            mlp_out = torch.where(mask_3d, ar_out, nar_out)
        else:
            h = self.self_attn(
                self.input_layernorm(x),
                cos,
                sin,
                past_key_value,
                layer_idx,
                attention_mask,
                cache_position,
            )
            x = x + h
            mlp_out = self.mlp(self.post_attention_layernorm(x))

        x = x + mlp_out
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(next(self.parameters()).dtype))


class AudioPositionEmbedding(nn.Module):
    def __init__(self, max_frames: int, hidden_size: int):
        super().__init__()
        pe = torch.zeros(max_frames, hidden_size)
        position = torch.arange(0, max_frames, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

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
        self.max_seq_len = max_seq_len
        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = [
            torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.value_cache: List[torch.Tensor] = [
            torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen_tokens

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None):
        t = key_states.shape[2]
        pos = self._seen_tokens
        end = pos + t
        if end > self.max_seq_len:
            raise ValueError(f"StaticKVCache capacity limit {self.max_seq_len} exceeded by request ({end}).")
        self.key_cache[layer_idx][:, :, pos:end] = key_states
        self.value_cache[layer_idx][:, :, pos:end] = value_states
        if layer_idx == self.num_layers - 1:
            self._seen_tokens = end
        return self.key_cache[layer_idx][:, :, :end], self.value_cache[layer_idx][:, :, :end]

    def reset(self):
        self._seen_tokens = 0


class Backbone(nn.Module):
    def __init__(self, config: YuE2Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

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
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        cos, sin = self.rotary_emb(position_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                cos,
                sin,
                past_key_values if use_cache else None,
                layer_idx=i,
                attention_mask=attention_mask,
                ar_mask=ar_mask,
                cache_position=cache_position,
            )
        return self.norm(x), past_key_values


class YuE2ForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = YuE2Config
    base_model_prefix = "model"

    def __init__(self, config: YuE2Config):
        super().__init__(config)
        self.model = Backbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.llm2vae = nn.Linear(config.hidden_size, config.latent_dim)
        self.vae2llm = nn.Linear(config.latent_dim, config.hidden_size)
        self.time_embedder = TimestepEmbedder(config.hidden_size)
        self.latent_pos_embed = AudioPositionEmbedding(config.max_latent_frames, config.hidden_size)
        self.post_init()

    def _shift_t_value(self, t_value: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t_sig = torch.sigmoid(torch.tensor(t_value, dtype=dtype, device=device))
        shift = self.config.timestep_shift
        return shift * t_sig / (1 + (shift - 1) * t_sig)

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
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Supply either input_ids or inputs_embeds strictly.")
        use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", True)
        return_dict = return_dict if return_dict is not None else getattr(self.config, "return_dict", True)
        tensor = input_ids if input_ids is not None else inputs_embeds
        batch_size, seq_len = tensor.shape[:2]
        device = tensor.device

        past_len = past_key_values.get_seq_length() if past_key_values is not None and use_cache else 0
        if cache_position is None:
            cache_position = torch.arange(past_len, past_len + seq_len, device=device)
        if position_ids is None:
            position_ids = cache_position[None]

        key_length = past_len + seq_len
        if use_cache and past_key_values is not None and hasattr(past_key_values, "get_max_cache_shape"):
            cap = past_key_values.get_max_cache_shape()
            if cap is not None and cap > 0:
                key_length = cap

        needs_mask = attention_mask is not None or key_length != past_len + seq_len or (past_len > 0 and seq_len > 1)
        causal_mask = _causal_mask(attention_mask, cache_position, key_length, batch_size) if needs_mask else None

        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=causal_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
        )

        selected = hidden_states[:, -logits_to_keep:, :] if isinstance(logits_to_keep, int) and logits_to_keep else hidden_states
        logits = self.lm_head(selected)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits, past_key_values) if use_cache else (logits,)
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values if use_cache else None)

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
        dtype = next(self.parameters()).dtype
        b, s = tokens.shape
        token_emb = self.model.embed_tokens(tokens)

        t_shifted = self._shift_t_value(t_value, device, dtype)
        t_lat = x_t.shape[0]

        nar_indices = nar_mask[0].nonzero(as_tuple=True)[0]
        content_indices = nar_content_mask[0].nonzero(as_tuple=True)[0]
        n_nar = nar_indices.shape[0]

        x_nar = torch.zeros(n_nar, x_t.shape[1], device=device, dtype=dtype)
        x_nar[1 : 1 + t_lat] = x_t.to(dtype)

        latent_hidden_nar = self.vae2llm(x_nar.unsqueeze(0))
        time_emb = self.time_embedder(t_shifted.expand(n_nar)).unsqueeze(0)
        latent_hidden_nar = latent_hidden_nar + time_emb

        pos_ids = torch.arange(n_nar, device=device).clamp(max=self.config.max_latent_frames - 1)
        pos_emb = self.latent_pos_embed(pos_ids).unsqueeze(0)
        latent_hidden_nar = latent_hidden_nar + pos_emb

        token_emb[0, nar_indices] = latent_hidden_nar[0]

        ar_q = ar_mask.unsqueeze(2).float()
        ar_k = ar_mask.unsqueeze(1).float()
        nar_q = nar_mask.unsqueeze(2).float()
        nar_k = nar_mask.unsqueeze(1).float()
        causal = torch.tril(torch.ones(s, s, device=device))

        if nar_cond_end > 0:
            text_k = torch.zeros(1, 1, s, device=device)
            text_k[0, 0, :nar_cond_end] = 1.0
            mask = (ar_q * ar_k * causal) + (nar_q * text_k) + (nar_q * nar_k)
        else:
            mask = (ar_q * ar_k * causal) + (nar_q * ar_k) + (nar_q * nar_k)

        attn_mask = mask.unsqueeze(1).masked_fill(mask.unsqueeze(1) == 0, float("-inf")).masked_fill(mask.unsqueeze(1) > 0, 0.0)
        position_ids = torch.arange(s, device=device).unsqueeze(0)

        hidden_states, _ = self.model(
            inputs_embeds=token_emb,
            position_ids=position_ids,
            use_cache=False,
            attention_mask=attn_mask,
            ar_mask=ar_mask,
        )

        nar_pred = self.llm2vae(hidden_states)
        return nar_pred[0, content_indices]