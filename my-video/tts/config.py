from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class VoicePaths:
    root: Path = ROOT_DIR
    vendor_dir: Path = ROOT_DIR / "vendor"
    cosyvoice_dir: Path = ROOT_DIR / "vendor" / "cosyvoice"
    third_party_dir: Path = ROOT_DIR / "vendor" / "third_party"
    speakers_dir: Path = ROOT_DIR / "speakers"
    models_dir: Path = ROOT_DIR / "models"
    output_dir: Path = ROOT_DIR / "output"

    @property
    def default_model_dir(self) -> Path:
        return self.models_dir / "CosyVoice2-0.5B"


def get_paths() -> VoicePaths:
    return VoicePaths()


def resolve_model_dir(model_dir: str | os.PathLike | None = None) -> Path:
    paths = get_paths()
    resolved = Path(model_dir).expanduser().resolve() if model_dir else paths.default_model_dir
    if not resolved.exists():
        raise FileNotFoundError(
            f"CosyVoice model directory not found: {resolved}. "
            "Place CosyVoice2-0.5B under digihuman/tts/models/ or pass --model-dir."
        )
    return resolved
