from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
import os
from pathlib import Path
import re
import sys
from typing import (
    AbstractSet,
    Any,
    Collection,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

ROOT_DIR = Path(__file__).resolve().parent
IS_WIN = sys.platform == "win32"
_DLL_NAME = "tiktalkin.dll" if IS_WIN else "libtiktalkin.so"


class SpecialDisk(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("id", ctypes.c_int32),
        ("length", ctypes.c_uint16),
        ("literal", ctypes.c_char * 58),
    ]


class Telemetry(ctypes.Structure):
    _fields_ = [
        ("total_tokens", ctypes.c_uint32),
        ("table_capacity", ctypes.c_uint32),
        ("sso_tokens", ctypes.c_uint32),
        ("max_psl", ctypes.c_uint32),
        ("string_blob_bytes", ctypes.c_uint64),
        ("load_factor", ctypes.c_double),
        ("sso_ratio", ctypes.c_double),
        ("mean_psl", ctypes.c_double),
        ("psl_variance", ctypes.c_double),
        ("compile_seconds", ctypes.c_double),
    ]


class _TikTalkinBinding:
    def __init__(self, dll_path: Path):
        self.lib = ctypes.CDLL(str(dll_path))

        self.lib.tiktalkin_init.argtypes = [ctypes.c_char_p]
        self.lib.tiktalkin_init.restype = ctypes.c_void_p

        self.lib.tiktalkin_encode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_uint32,
        ]
        self.lib.tiktalkin_encode.restype = ctypes.c_int32

        self.lib.tiktalkin_encode_ordinary.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_uint32,
        ]
        self.lib.tiktalkin_encode_ordinary.restype = ctypes.c_int32

        self.lib.tiktalkin_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        self.lib.tiktalkin_decode.restype = ctypes.c_int32

        self.lib.tiktalkin_find_common_prefix.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_uint32,
        ]
        self.lib.tiktalkin_find_common_prefix.restype = ctypes.c_uint32

        self.lib.tiktalkin_destroy.argtypes = [ctypes.c_void_p]
        self.lib.tiktalkin_destroy.restype = None

        self.lib.tiktalkin_compile_vocab_with_telemetry.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(Telemetry),
        ]
        self.lib.tiktalkin_compile_vocab_with_telemetry.restype = ctypes.c_int32

        self.lib.tiktalkin_compile_vocab_v4.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(SpecialDisk),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.POINTER(Telemetry),
        ]
        self.lib.tiktalkin_compile_vocab_v4.restype = ctypes.c_int32


def _locate_native_library() -> Path:
    candidates = [
        ROOT_DIR / _DLL_NAME,
        ROOT_DIR / "dist" / "bin" / _DLL_NAME,
        ROOT_DIR / "build" / _DLL_NAME,
        ROOT_DIR.parent / "dist" / "bin" / _DLL_NAME,
        ROOT_DIR.parent / "tiktalkin" / _DLL_NAME,
        ROOT_DIR.parent / "tiktalkin" / "dist" / "bin" / _DLL_NAME,
        ROOT_DIR.parent / "doodle" / "tiktalkin" / _DLL_NAME,
        ROOT_DIR.parent / "doodle" / "tiktalkin" / "dist" / "bin" / _DLL_NAME,
    ]
    env_override = os.environ.get("TIKTALKIN_DLL_PATH")
    if env_override:
        candidates.insert(0, Path(env_override).resolve())

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Unable to locate {_DLL_NAME}. Placed native binary required in {ROOT_DIR}."
    )


_BINDING: Optional[_TikTalkinBinding] = None


def get_binding() -> _TikTalkinBinding:
    global _BINDING
    if _BINDING is None:
        _BINDING = _TikTalkinBinding(_locate_native_library())
    return _BINDING


def build_default_yue2_specials() -> List[Tuple[str, int]]:
    specials: List[Tuple[str, int]] = [
        ("<|endoftext|>", 151643),
        ("<|im_start|>", 151644),
        ("<|im_end|>", 151645),
        ("<R>", 151646),
        ("<S>", 151647),
        ("<X>", 151648),
        ("<mask >", 151649),
        ("<sep>", 151650),
    ]
    for i in range(196):
        specials.append((f"<extra_{i}>", 151651 + i))
    specials.append(("<abc>", 151847))
    specials.append(("</abc>", 151848))
    specials.append(("<extra_198>", 151849))
    specials.append(("<extra_199>", 151850))
    specials.append(("<music_start>", 151851))
    specials.append(("<music_end>", 151852))
    specials.append(("<latent_start>", 184621))
    specials.append(("<latent_end>", 184622))
    specials.append(("<latent_pad>", 184623))
    return specials


def compile_vocab(
    in_tiktoken_path: Union[str, Path],
    out_bin_path: Union[str, Path],
    out_telemetry_json: Optional[Union[str, Path]] = None,
    regex_pattern: Optional[str] = None,
    specials_list: Optional[Sequence[Tuple[str, int]]] = None,
    eot_token_id: int = 151643,
) -> Telemetry:
    binding = get_binding()
    in_path_str = str(Path(in_tiktoken_path).resolve()).encode("utf-8")
    out_path_str = str(Path(out_bin_path).resolve()).encode("utf-8")
    telem_path_str = (
        str(Path(out_telemetry_json).resolve()).encode("utf-8")
        if out_telemetry_json
        else None
    )
    pat_str = regex_pattern.encode("utf-8") if regex_pattern else None

    effective_specials = (
        specials_list if specials_list is not None else build_default_yue2_specials()
    )
    specials_count = len(effective_specials)
    specials_arr = (SpecialDisk * specials_count)()
    for idx, (literal, s_id) in enumerate(effective_specials):
        specials_arr[idx].id = int(s_id)
        lit_bytes = literal.encode("utf-8")[:57]
        specials_arr[idx].length = len(lit_bytes)
        specials_arr[idx].literal = lit_bytes

    telem = Telemetry()
    rc = binding.lib.tiktalkin_compile_vocab_v4(
        in_path_str,
        out_path_str,
        pat_str,
        specials_arr,
        specials_count,
        int(eot_token_id),
        telem_path_str,
        ctypes.byref(telem),
    )
    if rc != 0:
        raise RuntimeError(f"Vocabulary compilation failed with error code: {rc}")
    return telem


def compile_qwen_ranks_binary(
    source_dir_or_file: Union[str, Path],
    target_bin_path: Optional[Union[str, Path]] = None,
) -> Path:
    src = Path(source_dir_or_file).resolve()
    if src.is_dir():
        vocab_path = src / "qwen.tiktoken"
        if not vocab_path.exists():
            matches = list(src.rglob("*.tiktoken"))
            if not matches:
                raise FileNotFoundError(f"No .tiktoken file located in {src}")
            vocab_path = matches[0]
    else:
        vocab_path = src

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file missing: {vocab_path}")

    if target_bin_path is None:
        dest_bin = vocab_path.with_suffix(".ranks.bin")
    else:
        dest_bin = Path(target_bin_path).resolve()

    dest_bin.parent.mkdir(parents=True, exist_ok=True)
    compile_vocab(vocab_path, dest_bin)
    return dest_bin


def find_common_prefix(
    tokens_a: Union[Sequence[int], Any],
    tokens_b: Union[Sequence[int], Any],
) -> int:
    if hasattr(tokens_a, "detach"):
        tokens_a = tokens_a.detach().cpu().numpy().ravel()
    if hasattr(tokens_b, "detach"):
        tokens_b = tokens_b.detach().cpu().numpy().ravel()

    len_a = len(tokens_a)
    len_b = len(tokens_b)
    limit = len_a if len_a < len_b else len_b
    if limit == 0:
        return 0

    try:
        import numpy as np

        is_arr_a = isinstance(tokens_a, np.ndarray)
        is_arr_b = isinstance(tokens_b, np.ndarray)
        if is_arr_a or is_arr_b or limit > 32:
            arr_a = np.ascontiguousarray(tokens_a[:limit], dtype=np.int32)
            arr_b = np.ascontiguousarray(tokens_b[:limit], dtype=np.int32)
            binding = get_binding()
            if hasattr(binding.lib, "tiktalkin_find_common_prefix"):
                ptr_a = arr_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
                ptr_b = arr_b.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
                return int(
                    binding.lib.tiktalkin_find_common_prefix(
                        ptr_a, limit, ptr_b, limit
                    )
                )
            mismatches = np.flatnonzero(arr_a != arr_b)
            return int(mismatches[0]) if mismatches.size > 0 else limit
    except Exception:
        pass

    for i in range(limit):
        if tokens_a[i] != tokens_b[i]:
            return i
    return limit


class Encoding:
    def __init__(
        self,
        name: str = "YuE2",
        ranks_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        self._name = name
        self._binding = get_binding()

        resolved_ranks: Optional[Path] = None
        if ranks_path is not None:
            p = Path(ranks_path).resolve()
            if p.is_file() and p.suffix == ".bin":
                resolved_ranks = p
            elif p.is_file() and p.suffix == ".tiktoken":
                bin_candidate = p.with_suffix(".ranks.bin")
                if not bin_candidate.exists():
                    compile_vocab(p, bin_candidate)
                resolved_ranks = bin_candidate
            elif p.is_dir():
                bin_candidate = p / "qwen.ranks.bin"
                if bin_candidate.exists():
                    resolved_ranks = bin_candidate
                else:
                    tik_candidate = p / "qwen.tiktoken"
                    if tik_candidate.exists():
                        resolved_ranks = compile_qwen_ranks_binary(
                            tik_candidate, bin_candidate
                        )

        if resolved_ranks is None or not resolved_ranks.exists():
            search_candidates = [
                ROOT_DIR / "qwen.ranks.bin",
                ROOT_DIR / "dist" / "bin" / "qwen.ranks.bin",
                ROOT_DIR.parent / "qwen.ranks.bin",
                ROOT_DIR.parent / "dist" / "bin" / "qwen.ranks.bin",
                ROOT_DIR.parent / "tiktalkin" / "qwen.ranks.bin",
                ROOT_DIR.parent / "tiktalkin" / "dist" / "bin" / "qwen.ranks.bin",
                ROOT_DIR.parent / "doodle" / "tiktalkin" / "qwen.ranks.bin",
                ROOT_DIR.parent / "doodle" / "tiktalkin" / "dist" / "bin" / "qwen.ranks.bin",
            ]
            env_ranks = os.environ.get("TIKTALKIN_RANKS_PATH")
            if env_ranks:
                search_candidates.insert(0, Path(env_ranks).resolve())

            for cand in search_candidates:
                if cand.is_file():
                    resolved_ranks = cand
                    break

        if resolved_ranks is None or not resolved_ranks.exists():
            raise FileNotFoundError(
                f"Could not resolve ranks container. Placed binary required in {ROOT_DIR}."
            )

        self._ranks_path = resolved_ranks
        self._ctx = self._binding.lib.tiktalkin_init(
            str(self._ranks_path).encode("utf-8")
        )
        if not self._ctx:
            raise RuntimeError(
                f"Failed to initialize TikTalkin context from {self._ranks_path}"
            )

        self._n_vocab = 184704
        self._eot_token = 151643

        defaults = build_default_yue2_specials()
        self._special_token_to_id: Dict[str, int] = {
            literal: s_id for literal, s_id in defaults
        }
        self._special_tokens_set: Set[str] = set(self._special_token_to_id.keys())
        self._id_to_special_token: Dict[int, str] = {
            s_id: literal for literal, s_id in defaults
        }

    def close(self) -> None:
        if hasattr(self, "_ctx") and self._ctx:
            self._binding.lib.tiktalkin_destroy(self._ctx)
            self._ctx = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> Encoding:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_vocab(self) -> int:
        return self._n_vocab

    @property
    def eot_token(self) -> int:
        return self._eot_token

    @property
    def special_tokens_set(self) -> Set[str]:
        return self._special_tokens_set

    @property
    def special_token_to_id(self) -> Dict[str, int]:
        return self._special_token_to_id

    def encode_ordinary(self, text: str) -> List[int]:
        if not text:
            return []
        if not self._ctx:
            raise RuntimeError("Operation attempted on closed TikTalkin context.")
        raw = text.encode("utf-8")
        byte_len = len(raw)
        max_tokens = max(byte_len, 16)
        out_buf = (ctypes.c_int32 * max_tokens)()

        count = self._binding.lib.tiktalkin_encode_ordinary(
            self._ctx, raw, byte_len, out_buf, max_tokens
        )
        return list(out_buf[:count])

    def encode(
        self,
        text: str,
        *,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
    ) -> List[int]:
        if not text:
            return []
        if not self._ctx:
            raise RuntimeError("Operation attempted on closed TikTalkin context.")

        if disallowed_special == "all":
            allowed_reference = (
                self._special_tokens_set
                if allowed_special == "all"
                else set(allowed_special)
            )
            disallowed = self._special_tokens_set - allowed_reference
            for special in disallowed:
                if special in text:
                    raise ValueError(
                        f"Encountered text corresponding to disallowed special token {special!r}."
                    )
        elif disallowed_special:
            for special in disallowed_special:
                if special in text:
                    raise ValueError(
                        f"Encountered text corresponding to disallowed special token {special!r}."
                    )

        if allowed_special == "all":
            raw = text.encode("utf-8")
            byte_len = len(raw)
            max_tokens = max(byte_len, 16)
            out_buf = (ctypes.c_int32 * max_tokens)()
            count = self._binding.lib.tiktalkin_encode(
                self._ctx, raw, byte_len, out_buf, max_tokens
            )
            return list(out_buf[:count])

        if not allowed_special:
            return self.encode_ordinary(text)

        allowed_set = set(allowed_special)
        if allowed_set >= self._special_tokens_set:
            raw = text.encode("utf-8")
            byte_len = len(raw)
            max_tokens = max(byte_len, 16)
            out_buf = (ctypes.c_int32 * max_tokens)()
            count = self._binding.lib.tiktalkin_encode(
                self._ctx, raw, byte_len, out_buf, max_tokens
            )
            return list(out_buf[:count])

        valid_allowed = [s for s in allowed_set if s in self._special_tokens_set]
        if not valid_allowed:
            return self.encode_ordinary(text)

        valid_allowed_set = set(valid_allowed)
        valid_allowed.sort(key=len, reverse=True)
        pattern = f"({'|'.join(re.escape(s) for s in valid_allowed)})"
        parts = re.split(pattern, text)

        result: List[int] = []
        for part in parts:
            if not part:
                continue
            if part in valid_allowed_set:
                result.append(self._special_token_to_id[part])
            else:
                result.extend(self.encode_ordinary(part))
        return result

    def encode_batch(
        self,
        texts: Iterable[str],
        *,
        num_threads: int = 8,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
    ) -> List[List[int]]:
        text_list = list(texts)
        if len(text_list) <= 1 or num_threads <= 1:
            return [
                self.encode(
                    t,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
                for t in text_list
            ]
        with ThreadPoolExecutor(
            max_workers=min(num_threads, len(text_list))
        ) as executor:
            return list(
                executor.map(
                    lambda t: self.encode(
                        t,
                        allowed_special=allowed_special,
                        disallowed_special=disallowed_special,
                    ),
                    text_list,
                )
            )

    def encode_ordinary_batch(
        self,
        texts: Iterable[str],
        *,
        num_threads: int = 8,
    ) -> List[List[int]]:
        text_list = list(texts)
        if len(text_list) <= 1 or num_threads <= 1:
            return [self.encode_ordinary(t) for t in text_list]
        with ThreadPoolExecutor(
            max_workers=min(num_threads, len(text_list))
        ) as executor:
            return list(executor.map(self.encode_ordinary, text_list))

    def decode(self, tokens: Union[Sequence[int], Any], errors: str = "replace") -> str:
        if hasattr(tokens, "detach"):
            tokens = tokens.detach().cpu().numpy().ravel()
        if hasattr(tokens, "__len__") and len(tokens) == 0:
            return ""
        if not self._ctx:
            raise RuntimeError("Operation attempted on closed TikTalkin context.")

        try:
            import numpy as np

            contiguous_arr = np.ascontiguousarray(tokens, dtype=np.int32).ravel()
            count = int(contiguous_arr.size)
            if count == 0:
                return ""
            c_tokens = contiguous_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        except Exception:
            count = len(tokens)
            if count == 0:
                return ""
            arr_type = ctypes.c_int32 * count
            c_tokens = arr_type(*tokens)

        alloc_size = max(count * 8 + 64, 1024)
        buf = ctypes.create_string_buffer(alloc_size)

        needed = self._binding.lib.tiktalkin_decode(
            self._ctx, c_tokens, count, buf, alloc_size
        )
        if needed >= alloc_size:
            buf = ctypes.create_string_buffer(needed + 1)
            needed = self._binding.lib.tiktalkin_decode(
                self._ctx, c_tokens, count, buf, needed + 1
            )

        return buf.raw[:needed].decode("utf-8", errors=errors)

    def decode_batch(
        self,
        batch: Iterable[Union[Sequence[int], Any]],
        *,
        num_threads: int = 8,
        errors: str = "replace",
    ) -> List[str]:
        batch_list = list(batch)
        if len(batch_list) <= 1 or num_threads <= 1:
            return [self.decode(toks, errors=errors) for toks in batch_list]
        with ThreadPoolExecutor(
            max_workers=min(num_threads, len(batch_list))
        ) as executor:
            return list(
                executor.map(lambda toks: self.decode(toks, errors=errors), batch_list)
            )


def get_encoding(
    encoding_name: str = "YuE2", ranks_path: Optional[Union[str, Path]] = None
) -> Encoding:
    return Encoding(name=encoding_name, ranks_path=ranks_path)


def encoding_for_model(model_name: str = "YuE2") -> Encoding:
    return Encoding(name=model_name)