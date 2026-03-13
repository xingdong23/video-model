from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


MODULE_ROOT = Path(__file__).resolve().parent
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm")
AUDIO_SUFFIXES = (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus")


@dataclass(frozen=True)
class SubtitlePaths:
    root: Path = MODULE_ROOT
    output_dir: Path = MODULE_ROOT / "output"


@dataclass(frozen=True)
class WhisperRuntime:
    requested_device: str
    resolved_device: str
    model_name: str
    compute_type: str
    total_vram_gb: Optional[float]


@dataclass(frozen=True)
class SubtitleCorrectionSettings:
    api_key: str
    api_base: str
    model_name: str
    request_timeout: int


def get_paths() -> SubtitlePaths:
    return SubtitlePaths()


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_SUFFIXES


def _load_torch():
    try:
        import torch  # type: ignore
    except Exception:
        return None
    return torch


def resolve_whisper_runtime(
    device: str = "auto",
    model_name: Optional[str] = None,
    compute_type: str = "auto",
    buffer_gb: float = 1.0,
) -> WhisperRuntime:
    requested_device = (device or os.getenv("SUBTITLE_DEVICE", "auto")).strip().lower()
    if requested_device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be one of: auto, cpu, cuda")

    requested_model_name = (model_name or os.getenv("SUBTITLE_MODEL") or "").strip() or None
    requested_compute_type = (
        compute_type or os.getenv("SUBTITLE_COMPUTE_TYPE", "auto")
    ).strip().lower()

    torch = _load_torch()
    cuda_available = False
    total_vram_gb = None

    if torch is not None:
        try:
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                device_props = torch.cuda.get_device_properties(torch.device("cuda"))
                total_vram_gb = float(device_props.total_memory) / float(1024 ** 3)
        except Exception:
            cuda_available = False
            total_vram_gb = None

    if requested_device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA device requested, but CUDA is unavailable in the current environment.")

    if requested_device == "cpu":
        resolved_device = "cpu"
    elif requested_device == "cuda":
        resolved_device = "cuda"
    else:
        resolved_device = "cuda" if cuda_available else "cpu"

    if requested_model_name:
        resolved_model_name = requested_model_name
    elif resolved_device == "cuda":
        required_vram_gb = 8.0 + float(buffer_gb)
        if total_vram_gb is not None and total_vram_gb >= required_vram_gb:
            resolved_model_name = "large-v3"
        else:
            resolved_model_name = "medium"
    else:
        resolved_model_name = "medium"

    if requested_compute_type and requested_compute_type != "auto":
        resolved_compute_type = requested_compute_type
    else:
        resolved_compute_type = "float16" if resolved_device == "cuda" else "int8"

    return WhisperRuntime(
        requested_device=requested_device,
        resolved_device=resolved_device,
        model_name=resolved_model_name,
        compute_type=resolved_compute_type,
        total_vram_gb=total_vram_gb,
    )


def resolve_ffmpeg_bin(ffmpeg_bin: Optional[str] = None) -> str:
    env_override = os.getenv("SUBTITLE_FFMPEG_BIN")
    candidates = []

    if ffmpeg_bin:
        candidates.append(Path(ffmpeg_bin).expanduser().resolve())
    elif env_override:
        candidates.append(Path(env_override).expanduser().resolve())

    which_ffmpeg = shutil.which("ffmpeg")
    if which_ffmpeg:
        candidates.append(Path(which_ffmpeg))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        "ffmpeg executable not found. Install ffmpeg, pass --ffmpeg-bin, or set SUBTITLE_FFMPEG_BIN."
    )


def normalize_api_key(api_key: Optional[str]) -> Optional[str]:
    raw = (api_key or "").strip()
    if not raw:
        return None

    if raw.startswith("[") and raw.endswith("]"):
        try:
            import json

            parsed = json.loads(raw)
            if isinstance(parsed, list) and parsed:
                first = str(parsed[0]).strip()
                return first or None
        except Exception:
            raw = raw.strip("[]\"' ")

    return raw or None


def resolve_correction_settings(
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model_name: Optional[str] = None,
    request_timeout: Optional[int] = None,
) -> SubtitleCorrectionSettings:
    resolved_api_key = normalize_api_key(api_key or os.getenv("SUBTITLE_LLM_API_KEY"))
    resolved_api_base = (api_base or os.getenv("SUBTITLE_LLM_API_BASE") or "").strip()
    resolved_model_name = (model_name or os.getenv("SUBTITLE_LLM_MODEL") or "").strip()
    timeout_value = int(request_timeout or os.getenv("SUBTITLE_LLM_TIMEOUT", "120"))

    if not resolved_api_key:
        raise ValueError(
            "Subtitle correction requires an API key. Pass --api-key or set SUBTITLE_LLM_API_KEY."
        )
    if not resolved_api_base:
        raise ValueError(
            "Subtitle correction requires an API base URL. Pass --api-base or set SUBTITLE_LLM_API_BASE."
        )
    if not resolved_model_name:
        raise ValueError(
            "Subtitle correction requires a model name. Pass --llm-model or set SUBTITLE_LLM_MODEL."
        )

    return SubtitleCorrectionSettings(
        api_key=resolved_api_key,
        api_base=resolved_api_base.rstrip("/"),
        model_name=resolved_model_name,
        request_timeout=timeout_value,
    )
