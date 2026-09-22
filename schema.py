from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

DOODLE_ROOT = Path(__file__).resolve().parent
DEFAULT_PRESET_FILENAME = "default.json"

SUPPORTED_COT = ["full", "melody", "off"]
SUPPORTED_ODE_METHODS = ["midpoint", "euler", "heun"]
SUPPORTED_INSTRUMENTAL_BRANCHES = ["cues", "tags_only"]

INSTRUCTIONS: Dict[str, str] = {
    "off": "Generate music with codec tokens from the given conditions.",
    "melody": "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.",
    "full": "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.",
}

BASELINE_ENGINE_DEFAULTS: Dict[str, Any] = {
    "cot": "full",
    "temperature": 1.0000,
    "top_p": 0.9500,
    "top_k": 100,
    "repetition_penalty": 1.2000,
    "penalty_window": 50,
    "abc_temperature": 0.7000,
    "abc_top_p": 0.9000,
    "abc_top_k": 30,
    "abc_repetition_penalty": 1.0050,
    "abc_penalty_window": 100,
    "cfg_scale": 1.0000,
    "num_inference_steps": 32,
    "ode_method": "midpoint",
    "vae_core_frames": 1024,
    "vae_halo_frames": 16,
    "apply_declick": True,
    "cpu_offload": False,
    "is_instrumental": False,
    "instrumental_branch": "cues",
}

_PRESET_CACHE: Dict[str, Any] = {
    "mtime": -1.0,
    "path": None,
    "has_custom_default": False,
    "defaults": dict(BASELINE_ENGINE_DEFAULTS),
}

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")


def clean_caption(c: str) -> str:
    text = _SPECIAL_TAG_RE.sub("", c)
    text = re.sub(r"^[ \t]*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def normalize_lyrics(lyrics: Optional[str]) -> str:
    if not isinstance(lyrics, str) or not lyrics.strip():
        return "[intro]\n[verse]\n[chorus]\n[outro]"
    raw = lyrics.replace("\r\n", "\n").replace("\r", "\n")
    raw = _SPECIAL_TAG_RE.sub("", raw)
    lines = []
    for line in raw.splitlines():
        match = _LEADING_TAGS_RE.match(line)
        if match:
            for tag in re.findall(r"\[[^\]]+\]", match.group(1)):
                lines.append(tag.strip().lower())
            rem = line[match.end():].strip()
            if rem:
                lines.append(rem)
        else:
            clean = line.strip()
            if clean:
                lines.append(clean)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def harvest_engine_preset(data: Dict[str, Any]) -> Dict[str, Any]:
    if "engine_defaults" in data and isinstance(data["engine_defaults"], dict):
        data = data["engine_defaults"]
    harvested: Dict[str, Any] = {}

    if "cot" in data and isinstance(data["cot"], str):
        c = data["cot"].strip().lower()
        if c in SUPPORTED_COT:
            harvested["cot"] = c

    for key, (lo, hi) in [
        ("temperature", (0.0001, 5.0)),
        ("top_p", (0.0001, 1.0)),
        ("repetition_penalty", (0.01, 5.0)),
        ("abc_temperature", (0.0001, 5.0)),
        ("abc_top_p", (0.0001, 1.0)),
        ("abc_repetition_penalty", (0.01, 5.0)),
        ("cfg_scale", (0.0, 20.0)),
    ]:
        if key in data and data[key] is not None:
            try:
                harvested[key] = max(lo, min(hi, float(data[key])))
            except (ValueError, TypeError):
                pass

    for key, (lo, hi) in [
        ("top_k", (1, 500)),
        ("penalty_window", (1, 200)),
        ("abc_top_k", (1, 500)),
        ("abc_penalty_window", (1, 200)),
        ("num_inference_steps", (1, 200)),
        ("vae_core_frames", (128, 4096)),
        ("vae_halo_frames", (16, 128)),
    ]:
        if key in data and data[key] is not None:
            try:
                harvested[key] = max(lo, min(hi, int(data[key])))
            except (ValueError, TypeError):
                pass

    if "ode_method" in data and isinstance(data["ode_method"], str):
        method = data["ode_method"].strip().lower()
        if method in SUPPORTED_ODE_METHODS:
            harvested["ode_method"] = method

    for key in ["apply_declick", "cpu_offload", "is_instrumental"]:
        if key in data and data[key] is not None:
            harvested[key] = bool(data[key])

    if "instrumental_branch" in data and isinstance(data["instrumental_branch"], str):
        br = data["instrumental_branch"].strip().lower()
        if br in SUPPORTED_INSTRUMENTAL_BRANCHES:
            harvested["instrumental_branch"] = br

    return harvested


def locate_default_preset_file() -> Optional[Path]:
    candidates = [
        DOODLE_ROOT / DEFAULT_PRESET_FILENAME,
        DOODLE_ROOT / "presets" / DEFAULT_PRESET_FILENAME,
        DOODLE_ROOT.parent / "config" / DEFAULT_PRESET_FILENAME,
    ]
    env_preset = os.environ.get("DOODLE_DEFAULT_JSON")
    if env_preset:
        candidates.insert(0, Path(env_preset).resolve())
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def get_active_engine_defaults() -> Dict[str, Any]:
    global _PRESET_CACHE
    preset_file = locate_default_preset_file()
    if preset_file is None:
        if _PRESET_CACHE["has_custom_default"]:
            _PRESET_CACHE["has_custom_default"] = False
            _PRESET_CACHE["mtime"] = -1.0
            _PRESET_CACHE["path"] = None
            _PRESET_CACHE["defaults"] = dict(BASELINE_ENGINE_DEFAULTS)
        return dict(_PRESET_CACHE["defaults"])

    try:
        current_mtime = preset_file.stat().st_mtime
    except OSError:
        return dict(_PRESET_CACHE["defaults"])

    if _PRESET_CACHE["path"] == preset_file and _PRESET_CACHE["mtime"] == current_mtime:
        return dict(_PRESET_CACHE["defaults"])

    try:
        with open(preset_file, "r", encoding="utf-8") as f:
            raw_payload = json.load(f)
        if isinstance(raw_payload, dict):
            extracted = harvest_engine_preset(raw_payload)
            resolved = dict(BASELINE_ENGINE_DEFAULTS)
            resolved.update(extracted)
            _PRESET_CACHE["mtime"] = current_mtime
            _PRESET_CACHE["path"] = preset_file
            _PRESET_CACHE["has_custom_default"] = True
            _PRESET_CACHE["defaults"] = resolved
            return dict(resolved)
    except Exception:
        pass
    return dict(_PRESET_CACHE["defaults"])


def has_custom_default_preset() -> bool:
    get_active_engine_defaults()
    return bool(_PRESET_CACHE["has_custom_default"])


class GenerationRequest(BaseModel):
    genre: str = Field(default="", max_length=100)
    subgenre: str = Field(default="", max_length=100)
    bpm: int = Field(default=0, ge=0, le=300)
    key: str = Field(default="", max_length=40)
    mood: str = Field(default="", max_length=500)
    vocals: str = Field(default="", max_length=800)
    vocal_lead: Optional[str] = Field(default="", max_length=800)
    instrumental_lead: Optional[str] = Field(default="", max_length=800)
    arrangement: str = Field(default="", max_length=2000)
    lyrics: str = Field(default="", max_length=6000)
    instrumental_lyrics: str = Field(default="", max_length=6000)
    is_instrumental: bool = Field(default=False)
    instrumental_branch: str = Field(default="cues")
    raw_prompt: Optional[str] = Field(default=None, max_length=5000)
    prompt: Optional[str] = Field(default=None, max_length=5000)
    cot: Optional[str] = Field(default=None)
    abc: Optional[str] = Field(default=None, max_length=8000)
    temperature: Optional[float] = Field(default=None, ge=0.0001, le=5.0)
    top_p: Optional[float] = Field(default=None, ge=0.0001, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=500)
    repetition_penalty: Optional[float] = Field(default=None, ge=0.01, le=5.0)
    penalty_window: Optional[int] = Field(default=None, ge=1, le=200)
    abc_temperature: Optional[float] = Field(default=None, ge=0.0001, le=5.0)
    abc_top_p: Optional[float] = Field(default=None, ge=0.0001, le=1.0)
    abc_top_k: Optional[int] = Field(default=None, ge=1, le=500)
    abc_repetition_penalty: Optional[float] = Field(default=None, ge=0.01, le=5.0)
    abc_penalty_window: Optional[int] = Field(default=None, ge=1, le=200)
    cfg_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=200)
    ode_method: Optional[str] = Field(default=None)
    vae_core_frames: Optional[int] = Field(default=None, ge=128, le=4096)
    vae_halo_frames: Optional[int] = Field(default=None, ge=16, le=128)
    audio_duration: float = Field(default=240.0, ge=1.0, le=600.0)
    seed: Optional[int] = Field(default=None, ge=0)
    output_path: str = Field(default="output.wav")
    device: str = Field(default="cuda")
    apply_declick: Optional[bool] = Field(default=None)
    cpu_offload: Optional[bool] = Field(default=None)
    repo_id: Optional[str] = Field(default=None)
    vae_repo_id: Optional[str] = Field(default=None)
    blocks: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    instrumental_blocks: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

    @field_validator("bpm", mode="before")
    @classmethod
    def coerce_bpm(cls, v: Any) -> int:
        if v is None or (isinstance(v, str) and not v.strip()):
            return 0
        try:
            return max(0, min(300, int(v)))
        except (ValueError, TypeError):
            return 0

    @field_validator("seed", mode="before")
    @classmethod
    def coerce_seed(cls, v: Any) -> Optional[int]:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

    @field_validator(
        "temperature",
        "top_p",
        "repetition_penalty",
        "abc_temperature",
        "abc_top_p",
        "abc_repetition_penalty",
        "cfg_scale",
        "audio_duration",
        mode="before",
    )
    @classmethod
    def coerce_floats(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return v

    @field_validator(
        "top_k",
        "penalty_window",
        "abc_top_k",
        "abc_penalty_window",
        "num_inference_steps",
        "vae_core_frames",
        "vae_halo_frames",
        mode="before",
    )
    @classmethod
    def coerce_ints(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return v

    @model_validator(mode="after")
    def synchronize_active_engine_preset(self) -> GenerationRequest:
        active = get_active_engine_defaults()
        use_custom = has_custom_default_preset()

        if self.cot is None or (use_custom and self.cot == BASELINE_ENGINE_DEFAULTS["cot"]):
            self.cot = str(active["cot"])
        if self.temperature is None or (use_custom and self.temperature == BASELINE_ENGINE_DEFAULTS["temperature"]):
            self.temperature = float(active["temperature"])
        if self.top_p is None or (use_custom and self.top_p == BASELINE_ENGINE_DEFAULTS["top_p"]):
            self.top_p = float(active["top_p"])
        if self.top_k is None or (use_custom and self.top_k == BASELINE_ENGINE_DEFAULTS["top_k"]):
            self.top_k = int(active["top_k"])
        if self.repetition_penalty is None or (use_custom and self.repetition_penalty == BASELINE_ENGINE_DEFAULTS["repetition_penalty"]):
            self.repetition_penalty = float(active["repetition_penalty"])
        if self.penalty_window is None or (use_custom and self.penalty_window == BASELINE_ENGINE_DEFAULTS["penalty_window"]):
            self.penalty_window = int(active["penalty_window"])

        if self.abc_temperature is None or (use_custom and self.abc_temperature == BASELINE_ENGINE_DEFAULTS["abc_temperature"]):
            self.abc_temperature = float(active["abc_temperature"])
        if self.abc_top_p is None or (use_custom and self.abc_top_p == BASELINE_ENGINE_DEFAULTS["abc_top_p"]):
            self.abc_top_p = float(active["abc_top_p"])
        if self.abc_top_k is None or (use_custom and self.abc_top_k == BASELINE_ENGINE_DEFAULTS["abc_top_k"]):
            self.abc_top_k = int(active["abc_top_k"])
        if self.abc_repetition_penalty is None or (use_custom and self.abc_repetition_penalty == BASELINE_ENGINE_DEFAULTS["abc_repetition_penalty"]):
            self.abc_repetition_penalty = float(active["abc_repetition_penalty"])
        if self.abc_penalty_window is None or (use_custom and self.abc_penalty_window == BASELINE_ENGINE_DEFAULTS["abc_penalty_window"]):
            self.abc_penalty_window = int(active["abc_penalty_window"])

        if self.cfg_scale is None or (use_custom and self.cfg_scale == BASELINE_ENGINE_DEFAULTS["cfg_scale"]):
            self.cfg_scale = float(active["cfg_scale"])
        if self.num_inference_steps is None or (use_custom and self.num_inference_steps == BASELINE_ENGINE_DEFAULTS["num_inference_steps"]):
            self.num_inference_steps = int(active["num_inference_steps"])
        if self.ode_method is None or (use_custom and self.ode_method == BASELINE_ENGINE_DEFAULTS["ode_method"]):
            self.ode_method = str(active["ode_method"])
        if self.vae_core_frames is None or (use_custom and self.vae_core_frames == BASELINE_ENGINE_DEFAULTS["vae_core_frames"]):
            self.vae_core_frames = int(active["vae_core_frames"])
        if self.vae_halo_frames is None or (use_custom and self.vae_halo_frames == BASELINE_ENGINE_DEFAULTS["vae_halo_frames"]):
            self.vae_halo_frames = int(active["vae_halo_frames"])
        if self.apply_declick is None or (use_custom and self.apply_declick == BASELINE_ENGINE_DEFAULTS["apply_declick"]):
            self.apply_declick = bool(active["apply_declick"])
        if self.cpu_offload is None or (use_custom and self.cpu_offload == BASELINE_ENGINE_DEFAULTS["cpu_offload"]):
            self.cpu_offload = bool(active["cpu_offload"])

        if not self.is_instrumental:
            if not self.vocal_lead and self.vocals:
                if not self.instrumental_lead or self.vocals != self.instrumental_lead:
                    self.vocal_lead = self.vocals
                else:
                    self.vocals = ""
            elif not self.vocals and self.vocal_lead:
                self.vocals = self.vocal_lead
        else:
            if not self.instrumental_lead and self.vocals:
                if not self.vocal_lead or self.vocals != self.vocal_lead:
                    self.instrumental_lead = self.vocals
                else:
                    self.vocals = ""
            elif not self.vocals and self.instrumental_lead:
                self.vocals = self.instrumental_lead
        return self

    def compile_style(self) -> str:
        override = self.raw_prompt or self.prompt
        if override and override.strip():
            return clean_caption(override.strip())

        parts: List[str] = []
        genre_desc = " / ".join(filter(None, [self.genre.strip(), self.subgenre.strip()]))
        if genre_desc:
            parts.append(genre_desc)

        if self.bpm and self.bpm > 0:
            parts.append(f"{self.bpm} bpm")

        key_clean = self.key.strip() if self.key else ""
        if key_clean:
            key_match = re.match(r"^([A-G][b#]?)(?:\s*(major|minor|min|maj|m))?\s*$", key_clean, re.IGNORECASE)
            if key_match:
                raw_root = key_match.group(1)
                key_root = raw_root[0].upper() + raw_root[1:].lower() if len(raw_root) > 1 else raw_root.upper()
                mode_token = (key_match.group(2) or "").lower()
                scale_mode = "minor" if mode_token in ("minor", "min", "m") else "major"
                parts.append(f"{key_root} {scale_mode}")
            else:
                parts.append(key_clean)

        if self.mood and self.mood.strip():
            parts.append(self.mood.strip().rstrip("."))

        lead_desc = (self.instrumental_lead if self.is_instrumental else self.vocal_lead) or self.vocals
        if lead_desc and lead_desc.strip():
            parts.append(lead_desc.strip().rstrip("."))

        if self.arrangement and self.arrangement.strip():
            parts.append(self.arrangement.strip().rstrip("."))

        compiled = ", ".join(filter(None, parts))
        return clean_caption(compiled)

    def sanitize_lyrics(self) -> str:
        if self.is_instrumental:
            branch = self.instrumental_branch.lower()
            if branch == "cues":
                active_blocks = self.instrumental_blocks or self.blocks
                if active_blocks:
                    compiled = []
                    for b in active_blocks:
                        if not isinstance(b, dict):
                            continue
                        lbl = b.get("label") or b.get("type") or "verse"
                        txt = (b.get("text") or "").strip()
                        clean_lbl = re.sub(r"[\[\]]", "", str(lbl)).strip().lower() or "verse"
                        if txt:
                            clean_txt = txt if txt.startswith("(") and txt.endswith(")") else f"({txt})"
                            compiled.append(f"[{clean_lbl}]\n{clean_txt}")
                        else:
                            compiled.append(f"[{clean_lbl}]")
                    return normalize_lyrics("\n\n".join(compiled))

                active_raw = self.instrumental_lyrics if self.instrumental_lyrics.strip() else self.lyrics
                if active_raw and active_raw.strip():
                    lines = active_raw.replace("\r\n", "\n").splitlines()
                    compiled = []
                    for line in lines:
                        trimmed = line.strip()
                        if not trimmed:
                            continue
                        match = _LEADING_TAGS_RE.match(trimmed)
                        if match:
                            for t in re.findall(r"\[[^\]]+\]", match.group(1)):
                                compiled.append(t.strip().lower())
                            rem = trimmed[match.end():].strip()
                            if rem:
                                compiled.append(rem if rem.startswith("(") and rem.endswith(")") else f"({rem})")
                        else:
                            compiled.append(trimmed if trimmed.startswith("(") and trimmed.endswith(")") else f"({trimmed})")
                    return normalize_lyrics("\n".join(compiled))
                return normalize_lyrics("[intro]\n\n[theme a]\n\n[verse]\n\n[chorus]\n\n[solo]\n\n[breakdown]\n\n[outro]")

            elif branch == "tags_only":
                active_blocks = self.instrumental_blocks or self.blocks
                if active_blocks:
                    tags = [f"[{re.sub(r'[\[\]]', '', str(b.get('label') or b.get('type') or 'verse')).strip().lower()}]" for b in active_blocks if isinstance(b, dict)]
                    return normalize_lyrics("\n\n".join(tags))

                active_raw = self.instrumental_lyrics if self.instrumental_lyrics.strip() else self.lyrics
                if active_raw and active_raw.strip():
                    tags = [m.group(0).lower() for m in re.finditer(r"\[([^\]]+)\]", active_raw)]
                    if tags:
                        return normalize_lyrics("\n\n".join(tags))
                return normalize_lyrics("[intro]\n\n[theme a]\n\n[verse]\n\n[chorus]\n\n[solo]\n\n[breakdown]\n\n[outro]")

        if self.blocks:
            compiled_blocks = []
            for b in self.blocks:
                if not isinstance(b, dict):
                    continue
                lbl = re.sub(r"[\[\]]", "", str(b.get("label") or b.get("type") or "verse")).strip().lower() or "verse"
                txt = (b.get("text") or "").strip()
                compiled_blocks.append(f"[{lbl}]\n{txt}" if txt else f"[{lbl}]")
            return normalize_lyrics("\n\n".join(compiled_blocks))

        if self.lyrics and self.lyrics.strip():
            return normalize_lyrics(self.lyrics)
        return ""

    def compile_full_text(self) -> str:
        cot_mode = self.cot or "full"
        instruction = INSTRUCTIONS.get(cot_mode, INSTRUCTIONS["full"])
        style = self.compile_style()
        lyrics = self.sanitize_lyrics()
        return f"{instruction}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n"

    def validate(self) -> None:
        if self.audio_duration <= 0.0 or self.audio_duration > 600.0:
            raise ValueError(f"Audio duration {self.audio_duration}s exceeds supported span (0.0 < t <= 600.0).")
        if self.bpm is not None and self.bpm != 0 and (self.bpm < 30 or self.bpm > 300):
            raise ValueError(f"BPM {self.bpm} outside musical range (30-300 or 0 for unmetered).")
        if self.cot not in SUPPORTED_COT:
            raise ValueError(f"CoT mode '{self.cot}' invalid. Supported: {SUPPORTED_COT}")
        if self.ode_method not in SUPPORTED_ODE_METHODS:
            raise ValueError(f"ODE method '{self.ode_method}' invalid. Supported: {SUPPORTED_ODE_METHODS}")
        if self.instrumental_branch not in SUPPORTED_INSTRUMENTAL_BRANCHES:
            raise ValueError(f"Instrumental branch '{self.instrumental_branch}' invalid. Supported: {SUPPORTED_INSTRUMENTAL_BRANCHES}")
        if self.num_inference_steps < 1 or self.num_inference_steps > 200:
            raise ValueError(f"Inference steps {self.num_inference_steps} out of bounds (1-200).")
        if self.cfg_scale < 0.0 or self.cfg_scale > 20.0:
            raise ValueError(f"Guidance scale {self.cfg_scale} out of bounds (0.0-20.0).")
        if self.temperature <= 0.0 or self.temperature > 5.0:
            raise ValueError(f"Semantic temperature {self.temperature} out of bounds (0.0 < T <= 5.0).")
        if self.top_p <= 0.0 or self.top_p > 1.0:
            raise ValueError(f"Semantic Top-P {self.top_p} out of bounds (0.0 < p <= 1.0).")
        if self.top_k < 1 or self.top_k > 500:
            raise ValueError(f"Semantic Top-K {self.top_k} out of bounds (1-500).")
        if self.abc_temperature <= 0.0 or self.abc_temperature > 5.0:
            raise ValueError(f"ABC temperature {self.abc_temperature} out of bounds (0.0 < T <= 5.0).")
        if self.abc_top_p <= 0.0 or self.abc_top_p > 1.0:
            raise ValueError(f"ABC Top-P {self.abc_top_p} out of bounds (0.0 < p <= 1.0).")
        if self.abc_top_k < 1 or self.abc_top_k > 500:
            raise ValueError(f"ABC Top-K {self.abc_top_k} out of bounds (1-500).")
        if self.vae_halo_frames < 16:
            raise ValueError(f"Configured halo frames ({self.vae_halo_frames}) violates analytical receptive bound (>=16).")

    def save_preset(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)
        temp_path.replace(path)
        if path.name == DEFAULT_PRESET_FILENAME:
            global _PRESET_CACHE
            _PRESET_CACHE["mtime"] = -1.0
            get_active_engine_defaults()

    @classmethod
    def load_preset(cls, path: Path) -> GenerationRequest:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class GenerationResponse(BaseModel):
    output_path: str
    sample_rate: int
    duration_seconds: float
    total_samples: int
    generation_time_seconds: float
    real_time_factor: float
    peak_vram_gb: float
    cpu_offload_active: bool
    cot_mode_used: str
    abc_token_count: int
    semantic_token_count: int
    ode_steps_used: int
    ode_method_used: str
    cfg_scale_used: float
    effective_prompt: str
    effective_style: str
    effective_lyrics: str
    abc_notation: Optional[str] = None
    declick_applied: bool
    peak_linear: float
    peak_dbfs: float
    rms_dbfs: float
    crest_factor_db: float
    is_instrumental_used: bool = False
    instrumental_branch_used: str = "cues"