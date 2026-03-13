from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


MODULE_ROOT = Path(__file__).resolve().parent
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v")
AUDIO_SUFFIXES = (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus")


@dataclass(frozen=True)
class BgmPaths:
    root: Path = MODULE_ROOT
    output_dir: Path = MODULE_ROOT / "output"
    library_dir: Path = MODULE_ROOT / "library"


def get_paths() -> BgmPaths:
    return BgmPaths()


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_SUFFIXES


def _resolve_binary(
    explicit_path: Optional[str],
    env_var: str,
    binary_name: str,
    fallback_bin: Optional[Path] = None,
) -> str:
    candidates: list[Path] = []

    raw_explicit = (explicit_path or "").strip()
    if raw_explicit:
        candidates.append(Path(raw_explicit).expanduser().resolve())

    raw_env = (os.getenv(env_var) or "").strip()
    if raw_env:
        candidates.append(Path(raw_env).expanduser().resolve())

    if fallback_bin is not None:
        sibling = fallback_bin.with_name(binary_name)
        candidates.append(sibling)

    which_path = shutil.which(binary_name)
    if which_path:
        candidates.append(Path(which_path).expanduser().resolve())

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        "%s executable not found. Install %s, pass --%s-bin, or set %s."
        % (binary_name, binary_name, binary_name, env_var)
    )


def resolve_ffmpeg_bin(ffmpeg_bin: Optional[str] = None) -> str:
    return _resolve_binary(
        explicit_path=ffmpeg_bin,
        env_var="BGM_FFMPEG_BIN",
        binary_name="ffmpeg",
    )


def resolve_ffprobe_bin(
    ffprobe_bin: Optional[str] = None,
    ffmpeg_bin: Optional[str] = None,
) -> str:
    fallback_ffmpeg = None
    raw_ffmpeg_bin = (ffmpeg_bin or "").strip()
    if raw_ffmpeg_bin:
        fallback_ffmpeg = Path(raw_ffmpeg_bin).expanduser().resolve()

    return _resolve_binary(
        explicit_path=ffprobe_bin,
        env_var="BGM_FFPROBE_BIN",
        binary_name="ffprobe",
        fallback_bin=fallback_ffmpeg,
    )
