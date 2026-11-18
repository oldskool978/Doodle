from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple
import warnings

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

sys.modules["torchvision"] = None
sys.modules["torchvision.transforms"] = None
sys.modules["torchvision.io"] = None

CACHE_DIR = Path(os.environ.get("HF_HOME") or (ROOT_DIR / ".hf_cache")).resolve()
for _d in (CACHE_DIR, CACHE_DIR / "hub", CACHE_DIR / "torch"):
    _d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE_DIR / "hub"))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
os.environ.pop("TRANSFORMERS_CACHE", None)
if (
    not os.environ.get("PYTORCH_ALLOC_CONF")
    and not os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    and not os.environ.get("PYTORCH_HIP_ALLOC_CONF")
):
    os.environ["PYTORCH_ALLOC_CONF"] = "garbage_collection_threshold:0.8"

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import (
    load_file as load_safetensors,
    save_file as save_safetensors,
)

from engine import MusicEngine
from models.modeling_vae import enforce_full_precision_reductions
from models.modeling_yue2 import synthesize_audio_pe_fp32
from models.tokenizer import compile_qwen_ranks_binary, is_swiss5_fast_container

enforce_full_precision_reductions()

try:
    from refinery import (
        DTYPE_LOOKUP,
        DTYPE_MAP,
        FP8_DTYPES,
        PhysicsLab,
        clear_refinery_cache,
        contender_cast,
        refineTensor,
    )

    HAS_REFINERY = True
except ImportError:
    HAS_REFINERY = False
    PhysicsLab = None  # type: ignore[assignment]
    DTYPE_MAP = {
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
    DTYPE_LOOKUP = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "e4m3fn": getattr(torch, "float8_e4m3fn", None),
        "e4m3fnuz": getattr(torch, "float8_e4m3fnuz", None),
        "e5m2": getattr(torch, "float8_e5m2", None),
        "e5m2fnuz": getattr(torch, "float8_e5m2fnuz", None),
        "e8m0fnu": getattr(torch, "float8_e8m0fnu", None),
    }
    FP8_DTYPES = [
        v
        for k, v in DTYPE_LOOKUP.items()
        if k in ("e4m3fn", "e4m3fnuz", "e5m2", "e5m2fnuz", "e8m0fnu") and v is not None
    ]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

DEFAULT_REPO_ID = "m-a-p/YuE2-3B"
DEFAULT_VAE_REPO_ID = "m-a-p/YuE2-Vae"
CONDITIONING_MARKER = ".doodle_conditioning.json"

PRECISION_CHOICES = [
    "native",
    "bf16",
    "fp16",
    "e4m3fn",
    "e4m3fnuz",
    "e5m2",
    "e5m2fnuz",
]
QUANT_ENGINES = ["dto_ca", "clean_rtn"]
DEFAULT_QUANT_ENGINE = "dto_ca" if HAS_REFINERY else "clean_rtn"

IGNORE_DOWNLOAD_PATTERNS = [
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.msgpack",
    "*.h5",
    "assets/*",
    "examples/*",
]

FP32_MANIFOLD_BRIDGES = (
    "llm2vae.",
    "vae2llm.",
    "time_embedder.",
    "latent_pos_embed.",
)

PROTECTED_NAMESPACES = (
    "embed_tokens",
    "lm_head",
    "llm2vae",
    "vae2llm",
    "time_embedder",
    "latent_pos_embed",
    "vae",
    "vocoder",
    "decoder",
    "encoder",
)

PROTECTED_KEY_PATTERNS = (
    "norm",
    "bias",
    "scale",
    "inv_freq",
    "pos_embed",
    "alpha",
    "beta",
)

DEAD_KEY_PATTERNS = (
    "rotary_emb.inv_freq",
    "loss",
    "training",
    "optimizer",
    "total_ops",
    "total_params",
)


def _resolve_compute_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _snapshot_download_compat(
    repo_id: str, max_workers: int, ignore_patterns: List[str]
) -> Path:
    kwargs: Dict[str, object] = {
        "repo_id": repo_id,
        "repo_type": "model",
        "max_workers": max_workers,
        "ignore_patterns": ignore_patterns,
    }
    try:
        params = inspect.signature(snapshot_download).parameters
        if "resume_download" in params:
            kwargs["resume_download"] = True
        if "local_dir_use_symlinks" in params:
            kwargs["local_dir_use_symlinks"] = False
    except (TypeError, ValueError):
        pass
    return Path(snapshot_download(**kwargs)).resolve()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def is_dead_key(key: str) -> bool:
    key_lower = key.lower()
    return any(pattern in key_lower for pattern in DEAD_KEY_PATTERNS)


def is_fp32_bridge_or_norm(key: str) -> bool:
    key_lower = key.lower()
    if key.startswith(FP32_MANIFOLD_BRIDGES):
        return True
    if "norm" in key_lower or "bias" in key_lower:
        return True
    return False


def is_quantizable(key: str, file_path: Path, tensor: torch.Tensor) -> bool:
    if not tensor.is_floating_point() or tensor.numel() < 128 or tensor.ndim < 2:
        return False
    path_lower = str(file_path).lower()
    if "vae" in path_lower:
        return False
    key_lower = key.lower()
    if any(p in key_lower for p in PROTECTED_NAMESPACES):
        return False
    if any(p in key_lower for p in PROTECTED_KEY_PATTERNS):
        return False
    return True


def quantize_clean_rtn(
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    compute_device: torch.device,
    *,
    phase_offset: int = 0,
) -> torch.Tensor:
    if HAS_REFINERY:
        eff_dev = (
            torch.device("cpu")
            if (compute_device.type == "mps" and tensor.dtype == torch.float64)
            else compute_device
        )
        src_t = (
            tensor.to(device=eff_dev, dtype=torch.float32)
            if (tensor.dtype == torch.bfloat16 and target_dtype != torch.bfloat16)
            else tensor.to(device=eff_dev)
        )
        return (
            contender_cast(src_t, target_dtype, phase_offset=phase_offset)
            .cpu()
            .contiguous()
        )

    eff_dev = (
        torch.device("cpu") if compute_device.type == "mps" else compute_device
    )
    t_f64 = tensor.to(device=eff_dev, dtype=torch.float64)
    try:
        finfo = torch.finfo(target_dtype)
        lo, hi = float(finfo.min), float(finfo.max)
    except Exception:
        name = str(target_dtype).split(".")[-1]
        if name == "float8_e4m3fnuz":
            lo, hi = (-240.0, 240.0)
        elif name in ("float8_e5m2", "float8_e5m2fnuz"):
            lo, hi = (-57344.0, 57344.0)
        else:
            lo, hi = (-448.0, 448.0)
    clamped = torch.clamp(t_f64, lo, hi)
    if target_dtype in (torch.bfloat16, torch.float16, torch.float32):
        out = clamped.to(dtype=target_dtype)
    else:
        out = clamped.to(dtype=torch.float32).to(dtype=target_dtype)
    return out.cpu().contiguous()


def read_conditioning_marker(repo_path: Path) -> Optional[dict]:
    marker = repo_path / CONDITIONING_MARKER
    if not marker.is_file():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return None


def verify_upstream_manifest(repo_path: Path) -> None:
    existing_marker = read_conditioning_marker(repo_path)
    if existing_marker is not None and existing_marker.get("modified", False):
        logging.info(
            "Snapshot previously conditioned (%s / %s); skipping pristine upstream SHA check.",
            existing_marker.get("precision", "unknown"),
            existing_marker.get("quant_engine", "unknown"),
        )
        return

    manifest_path = repo_path / "weights_manifest.json"
    if not manifest_path.exists():
        logging.warning(
            "No weights_manifest.json found at %s. Skipping SHA validation.",
            manifest_path,
        )
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    files = manifest.get("files", {})
    for filename, meta in files.items():
        target = repo_path / filename
        if not target.exists():
            raise FileNotFoundError(f"Manifest specifies missing file: {target}")
        expected_bytes = meta.get("bytes")
        if expected_bytes and target.stat().st_size != expected_bytes:
            raise ValueError(
                f"Byte size mismatch for {filename}: expected {expected_bytes}, got {target.stat().st_size}"
            )
        expected_sha = meta.get("sha256")
        if expected_sha:
            logging.info("Verifying upstream SHA-256 for %s...", filename)
            computed_sha = sha256_file(target)
            if computed_sha != expected_sha:
                raise ValueError(
                    f"Integrity failure on {filename}: expected {expected_sha}, computed {computed_sha}"
                )
            logging.info("SHA-256 integrity verified for %s.", filename)


def verify_vae_snapshot(vae_path: Path) -> None:
    cfg = vae_path / "config.json"
    sfs = sorted(list(vae_path.glob("*.safetensors")))
    if not cfg.is_file() or not sfs:
        raise FileNotFoundError(
            f"Incomplete YuE2-Vae snapshot in {vae_path} (missing config.json or *.safetensors)."
        )

    state_keys = set()
    for sf_file in sfs:
        if sf_file.stat().st_size == 0:
            raise ValueError(f"Corrupt zero-byte VAE shard: {sf_file}")
        loaded = load_safetensors(str(sf_file), device="cpu")
        for k, v in loaded.items():
            if v.is_floating_point() and not torch.isfinite(v.float()).all():
                raise FloatingPointError(
                    f"Non-finite parameter '{k}' detected in VAE shard {sf_file.name}"
                )
        state_keys.update(loaded.keys())
        del loaded

    has_enc = any(k.startswith("encoder.") for k in state_keys)
    has_dec = any(k.startswith("decoder.") for k in state_keys)
    if not has_dec:
        raise ValueError(
            f"YuE2-Vae snapshot {vae_path} lacks mandatory decoder weights."
        )
    logging.info(
        "Verified YuE2-Vae snapshot (%d tensors | Encoder: %s | Decoder: %s).",
        len(state_keys),
        "PRESENT" if has_enc else "ABSENT",
        "PRESENT",
    )


def condition_model_shards(
    model_dir: Path,
    precision: str,
    target_dtype: Optional[torch.dtype],
    quant_engine: str,
    compute_device: torch.device,
) -> None:
    enforce_full_precision_reductions()
    existing = read_conditioning_marker(model_dir)
    if (
        existing is not None
        and existing.get("precision") == precision
        and existing.get("quant_engine") == quant_engine
    ):
        logging.info(
            "Model shards already converged at precision=%s (%s). Skipping redundant pass.",
            precision.upper(),
            quant_engine.upper(),
        )
        return

    if (
        existing is not None
        and existing.get("modified", False)
        and existing.get("precision") not in ("native", precision)
    ):
        raise RuntimeError(
            f"Snapshot at {model_dir} was previously quantized to {existing.get('precision')}. "
            "Delete snapshot to re-download pristine weights before applying a different target precision."
        )

    safetensor_files = sorted(list(model_dir.rglob("*.safetensors")))
    total_files = len(safetensor_files)
    if total_files == 0:
        logging.warning("No SafeTensors weights located in %s", model_dir)
        return

    if target_dtype is None:
        logging.info(
            "Auditing pristine native SafeTensors finiteness (zero disk mutation): %s",
            model_dir,
        )
        audited_tensors = 0
        for idx, sf_path in enumerate(safetensor_files, start=1):
            logging.info(
                "[%d/%d] Verifying Native Container: %s", idx, total_files, sf_path.name
            )
            state_dict = load_safetensors(str(sf_path), device="cpu")
            for key, tensor in state_dict.items():
                if tensor.is_floating_point() and not torch.isfinite(tensor.float()).all():
                    raise FloatingPointError(
                        f"Non-finite parameter '{key}' detected in {sf_path.name}"
                    )
                audited_tensors += 1
            del state_dict
            gc.collect()

        marker_payload = {
            "precision": "native",
            "quant_engine": quant_engine,
            "modified": False,
            "timestamp": time.time(),
        }
        tmp_marker = (model_dir / CONDITIONING_MARKER).with_suffix(".tmp")
        tmp_marker.write_text(
            json.dumps(marker_payload, indent=2) + "\n", encoding="utf-8"
        )
        tmp_marker.replace(model_dir / CONDITIONING_MARKER)
        logging.info(
            "Native Snapshot Verified: %d finite tensors confirmed, 0 bytes modified.",
            audited_tensors,
        )
        return

    logging.info("Conditioning snapshot shards: %s", model_dir)
    logging.info(
        "Target Precision: %s | Quant Engine: %s | Device: %s",
        precision.upper(),
        quant_engine.upper(),
        str(compute_device),
    )

    quantized_tensors_count = 0
    promoted_fp32_count = 0
    synthesized_pe_count = 0
    preserved_tensors_count = 0
    stripped_keys_count = 0
    global_tensor_cursor = 0
    any_shard_rewritten = False

    with torch.inference_mode(False), torch.no_grad():
        for idx, sf_path in enumerate(safetensor_files, start=1):
            logging.info(
                "[%d/%d] Conditioning Container: %s", idx, total_files, sf_path.name
            )
            state_dict = load_safetensors(str(sf_path), device="cpu")
            processed_dict: Dict[str, torch.Tensor] = {}
            shard_modified = False

            for key, tensor in state_dict.items():
                if is_dead_key(key):
                    stripped_keys_count += 1
                    shard_modified = True
                    continue

                if tensor.is_floating_point() and not torch.isfinite(tensor.float()).all():
                    raise FloatingPointError(
                        f"Non-finite source parameter '{key}' detected in {sf_path.name}"
                    )

                phase_off = global_tensor_cursor & 63
                global_tensor_cursor += 1

                if key == "latent_pos_embed.pe" and tensor.ndim == 2:
                    out_tensor = synthesize_audio_pe_fp32(
                        int(tensor.shape[0]), int(tensor.shape[1]), device="cpu"
                    )
                    synthesized_pe_count += 1
                    shard_modified = True
                elif (
                    tensor.is_floating_point()
                    and is_fp32_bridge_or_norm(key)
                    and tensor.dtype != torch.float32
                ):
                    if (
                        HAS_REFINERY
                        and PhysicsLab is not None
                        and PhysicsLab.get_mantissa_width(tensor.dtype)
                        > PhysicsLab.get_mantissa_width(torch.float32)
                    ):
                        eff_dev = (
                            torch.device("cpu")
                            if (
                                compute_device.type == "mps"
                                and tensor.dtype == torch.float64
                            )
                            else compute_device
                        )
                        out_tensor = (
                            contender_cast(
                                tensor.to(device=eff_dev),
                                torch.float32,
                                phase_offset=phase_off,
                            )
                            .cpu()
                            .contiguous()
                        )
                    else:
                        out_tensor = tensor.to(dtype=torch.float32).contiguous()
                    promoted_fp32_count += 1
                    shard_modified = True
                elif tensor.dtype != target_dtype and is_quantizable(
                    key, sf_path, tensor
                ):
                    src_for_quant = (
                        tensor.to(dtype=torch.float32)
                        if (
                            tensor.dtype == torch.bfloat16
                            and target_dtype != torch.bfloat16
                        )
                        else tensor
                    )
                    eff_dev = (
                        torch.device("cpu")
                        if (
                            compute_device.type == "mps"
                            and src_for_quant.dtype == torch.float64
                        )
                        else compute_device
                    )
                    if quant_engine == "dto_ca" and HAS_REFINERY:
                        refined = refineTensor(
                            origin_tensor=src_for_quant,
                            target_dtype=target_dtype,
                            fp8_dtypes=FP8_DTYPES,
                            dtype_map=DTYPE_LOOKUP,
                            original_on_disk_dtype=tensor.dtype,
                            target_device=eff_dev,
                            phase_offset=phase_off,
                        )
                        out_tensor = refined.cpu().contiguous()
                    else:
                        out_tensor = quantize_clean_rtn(
                            src_for_quant,
                            target_dtype,
                            eff_dev,
                            phase_offset=phase_off,
                        )
                    quantized_tensors_count += 1
                    shard_modified = True
                else:
                    out_tensor = tensor
                    preserved_tensors_count += 1

                if (
                    out_tensor.is_floating_point()
                    and not torch.isfinite(out_tensor.float()).all()
                ):
                    raise FloatingPointError(
                        f"Non-finite conditioned parameter '{key}' produced in {sf_path.name}"
                    )
                processed_dict[key] = out_tensor

            if shard_modified:
                tmp_sf = sf_path.with_suffix(".safetensors.tmp")
                save_safetensors(processed_dict, str(tmp_sf))
                tmp_sf.replace(sf_path)
                any_shard_rewritten = True

            del state_dict, processed_dict
            gc.collect()

    if compute_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    if HAS_REFINERY:
        clear_refinery_cache()

    marker_payload = {
        "precision": precision,
        "quant_engine": quant_engine,
        "modified": any_shard_rewritten,
        "timestamp": time.time(),
    }
    tmp_marker = (model_dir / CONDITIONING_MARKER).with_suffix(".tmp")
    tmp_marker.write_text(
        json.dumps(marker_payload, indent=2) + "\n", encoding="utf-8"
    )
    tmp_marker.replace(model_dir / CONDITIONING_MARKER)

    logging.info(
        "Conditioning Complete: %d quantized (%s), %d promoted to FP32, %d analytical PE synthesized, %d preserved, %d dead keys stripped.",
        quantized_tensors_count,
        quant_engine.upper(),
        promoted_fp32_count,
        synthesized_pe_count,
        preserved_tensors_count,
        stripped_keys_count,
    )


def _resolve_fp8_arch_default(precision: str) -> str:
    if precision in ("e4m3fn", "e5m2") and torch.cuda.is_available():
        arch_name = str(
            getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
        ).lower()
        arch_base = arch_name.split(":", 1)[0]
        if arch_base == "gfx942":
            cand = f"{precision}uz"
            if DTYPE_MAP.get(cand) is not None:
                return cand
    return precision


def download_and_condition_weights(
    repo_id: str,
    vae_repo_id: str,
    precision: str = "native",
    quant_engine: str = DEFAULT_QUANT_ENGINE,
    device_str: str = "auto",
    max_workers: int = 8,
    skip_warmup: bool = False,
) -> Tuple[Path, Path]:
    enforce_full_precision_reductions()
    logging.info("Target Cache Anchor: %s", CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Initiating backbone acquisition for: %s", repo_id)
    backbone_path = _snapshot_download_compat(
        repo_id=repo_id,
        max_workers=max_workers,
        ignore_patterns=IGNORE_DOWNLOAD_PATTERNS,
    )

    logging.info("Initiating VAE acquisition for: %s", vae_repo_id)
    vae_path = _snapshot_download_compat(
        repo_id=vae_repo_id,
        max_workers=max_workers,
        ignore_patterns=IGNORE_DOWNLOAD_PATTERNS,
    )

    verify_upstream_manifest(backbone_path)
    verify_vae_snapshot(vae_path)

    logging.info("Auditing sovereign TikTalkin SWISS5 binary cache...")
    ranks_bin = compile_qwen_ranks_binary(backbone_path)
    swiss5_ok = is_swiss5_fast_container(ranks_bin)
    logging.info(
        "Binary ranks cache confirmed: %s (%d bytes | SWISS5 Zero-Init: %s)",
        ranks_bin.name,
        ranks_bin.stat().st_size,
        "ACTIVE" if swiss5_ok else "FALLBACK",
    )

    resolved_prec = _resolve_fp8_arch_default(precision.lower())
    target_dtype = DTYPE_MAP.get(resolved_prec, None)
    compute_device = _resolve_compute_device(device_str)

    condition_model_shards(
        backbone_path, resolved_prec, target_dtype, quant_engine, compute_device
    )

    if not skip_warmup:
        logging.info("=" * 80)
        logging.info("INITIALIZING MUSIC ENGINE WARMUP PASS")
        logging.info("=" * 80)
        t0 = time.perf_counter()
        trunk_warmup_dtype = (
            target_dtype
            if target_dtype in (torch.bfloat16, torch.float16)
            else torch.bfloat16
        )
        engine = MusicEngine(
            repo_id=str(backbone_path),
            vae_repo_id=str(vae_path),
            device=compute_device,
            dtype=trunk_warmup_dtype,
        )
        engine.warmup()
        t_warmup = time.perf_counter() - t0
        logging.info("MusicEngine warmup executed successfully in %.4fs.", t_warmup)
        del engine
        gc.collect()
        if compute_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    logging.info("Acquisition, conditioning, and verification complete.")
    logging.info("Backbone Path: %s", backbone_path)
    logging.info("VAE Path:      %s", vae_path)
    return backbone_path, vae_path


def prompt_precision_menu() -> Tuple[str, str, str]:
    print("\n" + "=" * 80)
    print("             DOODLE: YUE2-3B ACQUISITION & PRECISION SELECTOR")
    print("=" * 80)
    print(
        " [1] Native (None)   - Pristine untouched checkpoint (SHA-256 preserved + runtime FP32 Bridges)"
    )
    print(
        " [2] BF16 (Standard) - Doubly-Balanced 3D Bayer Contender BFloat16 (FP32 Bridges + Analytical PE)"
    )
    print(
        " [3] FP16            - Doubly-Balanced 3D Bayer Straddle Float16 (FP32 Bridges + Analytical PE)"
    )
    print(
        " [4] FP8 E4M3FN      - 3-bit Mantissa 3D Bayer Straddle FP8 (Embeddings, Heads, FP32 Bridges & PE preserved)"
    )
    print(
        " [5] FP8 E5M2        - 2-bit Mantissa 3D Bayer Straddle FP8 (Embeddings, Heads, FP32 Bridges & PE preserved)"
    )
    print("-" * 80)
    choice = input("Select precision [1-5] (Default: 1): ").strip()

    mapping = {
        "1": "native",
        "native": "native",
        "2": "bf16",
        "bf16": "bf16",
        "3": "fp16",
        "fp16": "fp16",
        "4": "e4m3fn",
        "e4m3fn": "e4m3fn",
        "5": "e5m2",
        "e5m2": "e5m2",
    }
    precision = mapping.get(choice.lower(), "native")
    if precision == "native":
        return "native", DEFAULT_QUANT_ENGINE, "auto"

    quant_engine = DEFAULT_QUANT_ENGINE
    if HAS_REFINERY:
        print("\n" + "-" * 80)
        print(" Select Quantization Engine:")
        print(
            " [1] Contender Refinery (DTO-CA) - 64-Aligned Chunked 3D Doubly-Balanced Bayer Straddle Projection (Default)"
        )
        print(
            " [2] Contender Direct (RTN+)     - Single-Shot 3D Doubly-Balanced Grace-Bayer Bracketed Projection"
        )
        print("-" * 80)
        eng_choice = input("Select engine [1-2] (Default: 1): ").strip()
        quant_engine = "clean_rtn" if eng_choice == "2" else "dto_ca"

    print("\n" + "-" * 80)
    print(" Select Compute Target for Conversion Pass:")
    print(" [1] GPU (CUDA/ROCm/MPS) - Hardware-accelerated execution")
    print(" [2] CPU (System RAM)    - Zero VRAM allocation execution")
    print("-" * 80)
    dev_choice = input("Select device [1-2] (Default: 1): ").strip()
    device_str = "cpu" if dev_choice == "2" else "auto"

    return precision, quant_engine, device_str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire, condition, and verify YuE2 model weights."
    )
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--vae_repo_id", type=str, default=DEFAULT_VAE_REPO_ID)
    parser.add_argument(
        "--precision",
        "--quant_target",
        dest="precision",
        type=str,
        choices=PRECISION_CHOICES,
        default=None,
    )
    parser.add_argument(
        "--engine",
        dest="quant_engine",
        type=str,
        choices=QUANT_ENGINES,
        default=DEFAULT_QUANT_ENGINE,
    )
    parser.add_argument(
        "--device", type=str, choices=["auto", "cuda", "mps", "cpu"], default=None
    )
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--skip_warmup", action="store_true")

    args = parser.parse_args()
    precision = args.precision
    quant_engine = args.quant_engine
    device_str = args.device

    if precision is None:
        if sys.stdin.isatty():
            precision, quant_engine, dev_input = prompt_precision_menu()
            if device_str is None:
                device_str = dev_input
        else:
            precision = "native"

    if device_str is None:
        device_str = "auto"

    resolved_prec = _resolve_fp8_arch_default(precision.lower())
    if resolved_prec != "native" and DTYPE_MAP.get(resolved_prec) is None:
        logging.error(
            "Selected precision '%s' is not supported in this PyTorch runtime.",
            resolved_prec,
        )
        sys.exit(1)

    try:
        download_and_condition_weights(
            repo_id=args.repo_id,
            vae_repo_id=args.vae_repo_id,
            precision=resolved_prec,
            quant_engine=quant_engine,
            device_str=device_str,
            max_workers=args.max_workers,
            skip_warmup=args.skip_warmup,
        )
    except Exception as e:
        logging.error(
            "Acquisition, conditioning, or warmup failure: %s", str(e), exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()