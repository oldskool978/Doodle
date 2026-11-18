from __future__ import annotations

import os
from pathlib import Path
import struct
from typing import AbstractSet, Any, Iterable, List, Literal, Optional, Sequence, Set, Union
import unicodedata

try:
    from tiktalkin import (
        Encoding,
        compile_qwen_ranks_binary as _native_compile,
        find_common_prefix as _native_find_common_prefix,
    )
except ImportError:
    try:
        from ..tiktalkin import (
            Encoding,
            compile_qwen_ranks_binary as _native_compile,
            find_common_prefix as _native_find_common_prefix,
        )
    except (ImportError, ValueError):
        from tiktalkin.tiktalkin import (
            Encoding,
            compile_qwen_ranks_binary as _native_compile,
            find_common_prefix as _native_find_common_prefix,
        )

DOODLE_ROOT = Path(__file__).resolve().parent.parent
_SWISS5_MAGIC = b"TIKTALKIN_SWISS5"
_SWISS_FAST_MAGIC = 0x54544B35
_SWISS_HDR_FMT = "<16sIIIIIIQQQQQIIQIIQQQ"
_SWISS_HDR_SIZE = struct.calcsize(_SWISS_HDR_FMT)


def is_swiss5_fast_container(bin_path: Path) -> bool:
    if not bin_path.is_file() or bin_path.stat().st_size < 1024 * 1024:
        return False
    try:
        with open(bin_path, "rb") as f:
            hdr = f.read(_SWISS_HDR_SIZE)
        if len(hdr) < _SWISS_HDR_SIZE:
            return False
        unpacked = struct.unpack(_SWISS_HDR_FMT, hdr)
        return unpacked[0] == _SWISS5_MAGIC and int(unpacked[16]) == _SWISS_FAST_MAGIC
    except Exception:
        return False


def _locate_bundled_ranks() -> Optional[Path]:
    candidates: List[Path] = []
    env_ranks = os.environ.get("TIKTALKIN_RANKS_PATH")
    if env_ranks:
        candidates.append(Path(env_ranks).resolve())
    candidates.extend(
        [
            DOODLE_ROOT / "tiktalkin" / "qwen.ranks.bin",
            DOODLE_ROOT / "tiktalkin" / "dist" / "bin" / "qwen.ranks.bin",
            DOODLE_ROOT / "dist" / "bin" / "qwen.ranks.bin",
            DOODLE_ROOT / "qwen.ranks.bin",
        ]
    )
    for cand in candidates:
        if is_swiss5_fast_container(cand):
            return cand.resolve()
    for cand in candidates:
        if cand.is_file() and cand.stat().st_size > 1024 * 1024:
            return cand.resolve()
    return None


def compile_qwen_ranks_binary(model_dir: Union[str, Path]) -> Path:
    target_dir = Path(model_dir).resolve()
    if target_dir.is_file():
        if target_dir.suffix == ".bin" and is_swiss5_fast_container(target_dir):
            return target_dir
        dest_bin = target_dir.with_suffix(".ranks.bin")
        return _native_compile(target_dir, dest_bin)

    bin_path = target_dir / "qwen.ranks.bin"
    if is_swiss5_fast_container(bin_path):
        return bin_path

    tiktoken_path = target_dir / "qwen.tiktoken"
    if not tiktoken_path.is_file():
        matches = sorted(target_dir.rglob("*.tiktoken"))
        if matches:
            tiktoken_path = matches[0]

    if tiktoken_path.is_file():
        return _native_compile(tiktoken_path, bin_path)

    if bin_path.is_file() and bin_path.stat().st_size > 1024 * 1024:
        return bin_path

    bundled = _locate_bundled_ranks()
    if bundled is not None:
        return bundled

    raise FileNotFoundError(f"Missing YuE2 vocabulary source in {target_dir}")


def find_common_prefix(
    tokens_a: Union[Sequence[int], Any],
    tokens_b: Union[Sequence[int], Any],
) -> int:
    return int(_native_find_common_prefix(tokens_a, tokens_b))


def load_ranks_binary(ranks_path: Union[str, Path, None] = None) -> Encoding:
    return YuE2Tokenizer(ranks_path)._enc


class YuE2Tokenizer:
    def __init__(self, vocab_source: Union[str, Path, None] = None):
        source_path = Path(vocab_source).resolve() if vocab_source else None
        resolved_bin: Optional[Path] = None

        if source_path is not None:
            if source_path.is_file() and source_path.suffix == ".bin":
                if is_swiss5_fast_container(source_path):
                    resolved_bin = source_path
                else:
                    sibling_tik = source_path.with_name("qwen.tiktoken")
                    if sibling_tik.is_file():
                        resolved_bin = _native_compile(sibling_tik, source_path)
                    elif source_path.stat().st_size > 1024 * 1024:
                        resolved_bin = source_path
            elif source_path.is_file() and source_path.suffix == ".tiktoken":
                cand = source_path.with_suffix(".ranks.bin")
                if not is_swiss5_fast_container(cand):
                    _native_compile(source_path, cand)
                resolved_bin = cand
            elif source_path.is_dir():
                cand = source_path / "qwen.ranks.bin"
                tik_cand = source_path / "qwen.tiktoken"
                if is_swiss5_fast_container(cand):
                    resolved_bin = cand
                elif tik_cand.is_file():
                    resolved_bin = _native_compile(tik_cand, cand)
                elif cand.is_file() and cand.stat().st_size > 1024 * 1024:
                    resolved_bin = cand

        if resolved_bin is None or not resolved_bin.is_file():
            resolved_bin = _locate_bundled_ranks()

        if resolved_bin is None or not resolved_bin.is_file():
            for tik in (
                DOODLE_ROOT / "tiktalkin" / "qwen.tiktoken",
                DOODLE_ROOT / "qwen.tiktoken",
            ):
                if tik.is_file() and tik.stat().st_size > 1024 * 1024:
                    target_bin = tik.with_suffix(".ranks.bin")
                    resolved_bin = _native_compile(tik, target_bin)
                    break

        if resolved_bin is None or not resolved_bin.is_file():
            raise FileNotFoundError(
                f"Unable to resolve valid YuE2 qwen.ranks.bin container inside {DOODLE_ROOT}."
            )

        self._enc = Encoding("YuE2", ranks_path=resolved_bin)
        self.n_vocab: int = self._enc.n_vocab
        self.base_vocab_size: int = self._enc.base_vocab_size
        self.eot_token: int = self._enc.eot_token
        self.ranks_path: Path = self._enc.ranks_path
        self.has_fast_tables: bool = self._enc.has_fast_tables

    @property
    def special_tokens_set(self) -> Set[str]:
        return self._enc.special_tokens_set

    @property
    def special_token_to_id(self) -> dict[str, int]:
        return self._enc.special_token_to_id

    def encode_ordinary(self, text: str) -> List[int]:
        if not text:
            return []
        normalized = unicodedata.normalize("NFC", text)
        return self._enc.encode_ordinary(normalized)

    def encode(
        self,
        text: str,
        allowed_special: Union[Literal["all"], AbstractSet[str], Sequence[str]] = (),
    ) -> List[int]:
        if not text:
            return []
        normalized = unicodedata.normalize("NFC", text)
        if not allowed_special:
            return self._enc.encode_ordinary(normalized)
        allowed_arg: Union[Literal["all"], AbstractSet[str]] = (
            "all" if allowed_special == "all" else set(allowed_special)
        )
        return self._enc.encode(
            normalized, allowed_special=allowed_arg, disallowed_special=()
        )

    def encode_yue2_sequence(
        self,
        prompt_text: str,
        abc_text: Optional[str] = None,
        codec_tokens: Optional[Sequence[int]] = None,
        latent_frames: int = 0,
        max_sequence_length: int = 24576,
    ) -> List[int]:
        return self._enc.encode_yue2_sequence(
            prompt_text=prompt_text,
            abc_text=abc_text,
            codec_tokens=codec_tokens,
            latent_frames=latent_frames,
            max_sequence_length=max_sequence_length,
        )

    def encode_batch(
        self,
        texts: Iterable[str],
        *,
        num_threads: int = 8,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
    ) -> List[List[int]]:
        norm_texts = [unicodedata.normalize("NFC", t) for t in texts]
        if not allowed_special:
            return self._enc.encode_ordinary_batch(norm_texts, num_threads=num_threads)
        return self._enc.encode_batch(
            norm_texts,
            num_threads=num_threads,
            allowed_special=allowed_special,
            disallowed_special=(),
        )

    def decode(self, ids: Union[Sequence[int], Any], errors: str = "replace") -> str:
        return self._enc.decode(ids, errors=errors)

    def decode_batch(
        self,
        batch: Iterable[Union[Sequence[int], Any]],
        *,
        num_threads: int = 8,
        errors: str = "replace",
    ) -> List[str]:
        return self._enc.decode_batch(batch, num_threads=num_threads, errors=errors)

    def find_common_prefix(
        self,
        tokens_a: Union[Sequence[int], Any],
        tokens_b: Union[Sequence[int], Any],
    ) -> int:
        return int(_native_find_common_prefix(tokens_a, tokens_b))

    def close(self) -> None:
        self._enc.close()