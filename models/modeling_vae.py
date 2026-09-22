from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel


def checkpoint(function, *args, **kwargs):
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    kwargs.setdefault("use_reentrant", False)
    return torch_checkpoint(function, *args, **kwargs)


def WNConv1d(*args, **kwargs) -> nn.Conv1d:
    return nn.Conv1d(*args, **kwargs)


def WNConvTranspose1d(*args, **kwargs) -> nn.ConvTranspose1d:
    return nn.ConvTranspose1d(*args, **kwargs)


def fold_weight_norm_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    fused_dict = {}
    consumed_keys = set()
    for key in list(state_dict.keys()):
        if key in consumed_keys:
            continue
        if key.endswith(".weight_g"):
            base_key = key[:-9]
            v_key = f"{base_key}.weight_v"
            if v_key not in state_dict:
                raise KeyError(f"Corrupt state dict: located {key} but missing counterpart {v_key}")
            g = state_dict[key]
            v = state_dict[v_key]
            reduce_dims = tuple(range(1, v.ndim))
            norm = torch.linalg.vector_norm(v, ord=2, dim=reduce_dims, keepdim=True)
            w_fused = v * (g / torch.clamp_min(norm, 1e-12))
            fused_dict[f"{base_key}.weight"] = w_fused.contiguous()
            consumed_keys.update([key, v_key])
        elif not key.endswith(".weight_v"):
            fused_dict[key] = state_dict[key]
            consumed_keys.add(key)
    return fused_dict


def snake_beta(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return x + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


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
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return snake_beta(x, alpha, beta)


def get_activation(activation: Literal["elu", "snake", "none"], channels: Optional[int] = None) -> nn.Module:
    if activation == "elu":
        return nn.ELU()
    if activation == "snake":
        return SnakeBeta(channels)
    if activation == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation specifier: {activation}")


class ResidualUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, act_type: str):
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
            x = checkpoint(self.layers, x)
        else:
            x = self.layers(x)
        return x + residual


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
        channels: int = 128,
        latent_dim: int = 32,
        c_mults: Tuple[int, ...] = (1, 2, 4, 8),
        strides: Tuple[int, ...] = (2, 4, 8, 8),
        use_snake: bool = False,
        antialias_activation: bool = False,
    ):
        super().__init__()
        if antialias_activation:
            raise ValueError("Upstream Oobleck encoder specification rejects antialias_activation.")
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
        channels: int = 128,
        latent_dim: int = 32,
        c_mults: Tuple[int, ...] = (1, 2, 4, 8),
        strides: Tuple[int, ...] = (2, 4, 8, 8),
        use_snake: bool = False,
        snake_type: str = "vanilla",
        antialias_activation: bool = False,
        use_nearest_upsample: bool = False,
        use_filter: bool = False,
        final_tanh: bool = True,
    ):
        super().__init__()
        if antialias_activation or use_nearest_upsample or use_filter:
            raise ValueError("Unsupported operational modes enabled for Oobleck decoder.")
        if use_snake and snake_type != "vanilla":
            raise ValueError("Oobleck specification requires vanilla SnakeBeta activation.")
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
    _hf_fields = frozenset({
        "model_type", "architectures", "auto_map", "transformers_version",
        "dtype", "torch_dtype", "return_dict", "output_hidden_states",
        "output_attentions", "use_cache", "tie_word_embeddings", "torchscript",
        "is_decoder", "is_encoder_decoder", "add_cross_attention",
        "bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id",
        "attn_implementation",
    })
    _inference_fields = frozenset({
        "encoder_config", "decoder_config", "sample_rate", "latent_dim",
        "downsampling_ratio", "audio_channels", "release_variant",
        "decode_core_frames", "decode_halo_frames",
    })

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
        clean_kwargs = {key: value for key, value in kwargs.items() if key in self._hf_fields}
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
            raise ValueError("Core and halo framing specifications must be positive non-negative values.")
        if math.prod(self.decoder_config["strides"]) != self.downsampling_ratio:
            raise ValueError("Cumulative stride geometry fails to reconstruct specified downsampling ratio.")
        if self.decoder_config["latent_dim"] != self.latent_dim:
            raise ValueError("Decoder input dimensionality deviates from latent_dim invariant.")


def _dependency_interval(module: nn.Module, low: int, high: int) -> Tuple[int, int]:
    if isinstance(module, (nn.Sequential, OobleckDecoder, DecoderBlock)):
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
    raise TypeError(f"Unmapped topological receptive layer type: {type(module).__name__}")


def _output_length(module: nn.Module, length: int) -> int:
    if isinstance(module, (nn.Sequential, OobleckDecoder, DecoderBlock)):
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
    raise TypeError(f"Unmapped sequence dimension transformation for: {type(module).__name__}")


class YuE2VAE(PreTrainedModel):
    config_class = YuE2VAEConfig
    base_model_prefix = ""
    main_input_name = "audio"

    def __init__(self, config: YuE2VAEConfig, decoder_only: bool = False):
        super().__init__(config)
        self.decoder_only = bool(decoder_only)
        if not self.decoder_only:
            self.encoder = OobleckEncoder(**config.encoder_config)
        self.decoder = OobleckDecoder(**config.decoder_config)
        self.eval().requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        *model_args,
        config: Optional[YuE2VAEConfig] = None,
        decoder_only: bool = False,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        from safetensors.torch import load_file as load_safetensors

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

        model = cls(config, decoder_only=decoder_only)
        sf_files = sorted(list(path.glob("*.safetensors")))
        if not sf_files:
            raise FileNotFoundError(f"SafeTensors artifact missing in directory: {path}")

        raw_state = {}
        for sf in sf_files:
            raw_state.update(load_safetensors(str(sf), device="cpu"))

        filtered_state = {
            k: v
            for k, v in raw_state.items()
            if not decoder_only or k.startswith("decoder.")
        }
        fused_state = fold_weight_norm_dict(filtered_state)

        model.load_state_dict(fused_state, strict=True)
        model.to(device=device, dtype=torch.float32).eval()
        return model

    @property
    def decoder_device(self) -> torch.device:
        return next(self.decoder.parameters()).device

    def _validate_latents(self, latent: torch.Tensor) -> torch.Tensor:
        latent_t = torch.as_tensor(latent)
        if (
            latent_t.ndim != 3
            or latent_t.shape[1] != self.config.latent_dim
            or latent_t.shape[0] < 1
            or latent_t.shape[-1] < 1
        ):
            raise ValueError(f"Expected latents with shape [B, {self.config.latent_dim}, T], received {latent_t.shape}")
        if not torch.isfinite(latent_t).all():
            raise FloatingPointError("Latent tensor contains NaN or Inf components.")
        return latent_t

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        latent_t = self._validate_latents(latent)
        with torch.autocast(device_type=self.decoder_device.type, enabled=False):
            return self.decoder(latent_t.to(device=self.decoder_device, dtype=torch.float32))

    def natural_output_length(self, frames: int) -> int:
        if int(frames) < 1:
            raise ValueError("Frame length evaluation must be strictly positive.")
        return _output_length(self.decoder, int(frames))

    def required_halo(self, core_frames: Optional[int] = None) -> int:
        core = self.config.decode_core_frames if core_frames is None else core_frames
        ratio = self.config.downsampling_ratio
        low, high = _dependency_interval(self.decoder, 0, core * ratio - 1)
        return max(0, -low, high - core + 1)

    @torch.inference_mode()
    def decode_tiled(
        self,
        latent: torch.Tensor,
        core_frames: Optional[int] = None,
        halo_frames: Optional[int] = None,
        output_device: Union[str, torch.device] = "cpu",
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        latent_t = self._validate_latents(latent)
        core = self.config.decode_core_frames if core_frames is None else core_frames
        halo = self.config.decode_halo_frames if halo_frames is None else halo_frames
        required = self.required_halo(core)
        if halo < required:
            raise ValueError(f"Configured halo frames ({halo}) cannot fulfill structural receptive bound ({required}).")

        frames = latent_t.shape[-1]
        ratio = self.config.downsampling_ratio
        total_samples = self.natural_output_length(frames)

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
                raise RuntimeError("Reconstructed slice length diverged from computed receptive core bounds.")

            audio[..., out_start:out_end].copy_(crop.to(output_device))
            del tile, crop

            if on_progress is not None:
                on_progress(tile_idx + 1, tiles)

        return audio