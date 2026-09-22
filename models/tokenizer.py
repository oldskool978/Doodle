from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Set, Union
import unicodedata

try:
    from tiktalkin import Encoding, compile_qwen_ranks_binary as _native_compile
except ImportError:
    try:
        from ..tiktalkin import Encoding, compile_qwen_ranks_binary as _native_compile
    except (ImportError, ValueError):
        from tiktalkin.tiktalkin import Encoding, compile_qwen_ranks_binary as _native_compile


def compile_qwen_ranks_binary(model_dir: Path) -> Path:
    bin_path = model_dir / "qwen.ranks.bin"
    if bin_path.exists() and bin_path.stat().st_size > 1024 * 1024:
        return bin_path
    tiktoken_path = model_dir / "qwen.tiktoken"
    if not tiktoken_path.exists():
        raise FileNotFoundError(f"Missing upstream vocabulary file: {tiktoken_path}")
    return _native_compile(tiktoken_path, bin_path)


def load_ranks_binary(ranks_path: Union[str, Path, None] = None) -> Encoding:
    return YuE2Tokenizer(ranks_path)._enc


class YuE2Tokenizer:
    def __init__(self, vocab_source: Union[str, Path, None] = None):
        source_path = Path(vocab_source) if vocab_source else None
        resolved_bin: Optional[Path] = None

        if source_path is not None:
            if source_path.is_file() and source_path.suffix == ".bin":
                resolved_bin = source_path
            elif source_path.is_file() and source_path.suffix == ".tiktoken":
                cand = source_path.with_suffix(".ranks.bin")
                if not cand.exists():
                    _native_compile(source_path, cand)
                resolved_bin = cand
            elif source_path.is_dir():
                cand = source_path / "qwen.ranks.bin"
                if cand.exists():
                    resolved_bin = cand
                else:
                    tik_cand = source_path / "qwen.tiktoken"
                    if tik_cand.exists():
                        resolved_bin = _native_compile(tik_cand, cand)

        if resolved_bin is None or not resolved_bin.exists():
            default_local = Path(__file__).resolve().parent.parent / "tiktalkin" / "qwen.ranks.bin"
            if default_local.exists():
                resolved_bin = default_local

        self._enc = Encoding("YuE2", ranks_path=resolved_bin)
        self.n_vocab = self._enc.n_vocab

    def encode_ordinary(self, text: str) -> List[int]:
        normalized = unicodedata.normalize("NFC", text)
        return self._enc.encode_ordinary(normalized)

    def encode(self, text: str, allowed_special: Union[str, Set[str]] = ()) -> List[int]:
        normalized = unicodedata.normalize("NFC", text)
        if not allowed_special:
            return self._enc.encode_ordinary(normalized)
        return self._enc.encode(normalized, allowed_special=allowed_special, disallowed_special=())

    def decode(self, ids: Sequence[int], errors: str = "replace") -> str:
        return self._enc.decode(ids, errors=errors)