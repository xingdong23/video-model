from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


_DIGIHUMAN_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = {"env_prefix": "DIGIHUMAN_"}

    host: str = "0.0.0.0"
    port: int = 8000

    # Auth — set DIGIHUMAN_API_KEY to enable; empty = no auth
    api_key: str = ""

    # Uvicorn
    timeout_keep_alive: int = 30
    timeout_graceful_shutdown: int = 30

    # Storage
    storage_backend: str = "local"
    storage_root: str = str(_DIGIHUMAN_ROOT / "storage")
    file_ttl_seconds: int = 3600 * 24  # 24 hours

    # CORS
    cors_origins: str = "*"

    # Voice engine
    voice_model_dir: Optional[str] = None
    voice_fp16: bool = True

    # Digital-human engine
    digital_human_tuilionnx_dir: Optional[str] = None
    digital_human_ffmpeg_bin: Optional[str] = None
    digital_human_runtime: str = "auto"
    digital_human_warmup: bool = False

    # Subtitle engine
    subtitle_model_name: Optional[str] = None
    subtitle_device: str = "auto"
    subtitle_compute_type: str = "auto"
    subtitle_ffmpeg_bin: Optional[str] = None

    # BGM engine
    bgm_library_dir: Optional[str] = None
    bgm_ffmpeg_bin: Optional[str] = None

    # Rewrite engine
    rewrite_api_key: Optional[str] = None
    rewrite_base_url: Optional[str] = None
    rewrite_model: Optional[str] = None

    @property
    def digihuman_root(self) -> Path:
        return _DIGIHUMAN_ROOT


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
