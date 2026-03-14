from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import onnxruntime as ort
import torch


MODULE_ROOT = Path(__file__).resolve().parent
MY_VIDEO_ROOT = MODULE_ROOT.parent
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")


@dataclass(frozen=True)
class DigitalHumanPaths:
    root: Path = MODULE_ROOT
    my_video_root: Path = MY_VIDEO_ROOT
    local_faces_dir: Path = MODULE_ROOT / "faces"
    output_dir: Path = MODULE_ROOT / "output"
    models_dir: Path = MODULE_ROOT / "models"
    default_tuilionnx_dir: Path = MODULE_ROOT / "models" / "tuilionnx"


def get_paths() -> DigitalHumanPaths:
    return DigitalHumanPaths()


@dataclass(frozen=True)
class RuntimeSelection:
    requested: str
    resolved: str
    torch_device: str
    onnx_providers: tuple[str, ...]
    description: str


def resolve_runtime(runtime: str = "auto") -> RuntimeSelection:
    normalized = runtime.lower().strip()
    if normalized not in {"auto", "cuda", "cpu"}:
        raise ValueError("runtime must be one of: auto, cuda, cpu")

    available_providers = set(ort.get_available_providers())
    cuda_ready = torch.cuda.is_available() and "CUDAExecutionProvider" in available_providers

    if normalized == "cuda":
        if not cuda_ready:
            raise RuntimeError(
                "CUDA runtime requested, but torch/onnxruntime CUDA support is unavailable on this machine."
            )
        return RuntimeSelection(
            requested=normalized,
            resolved="cuda",
            torch_device="cuda",
            onnx_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
            description="NVIDIA CUDA path",
        )

    if normalized == "cpu":
        return RuntimeSelection(
            requested=normalized,
            resolved="cpu",
            torch_device="cpu",
            onnx_providers=("CPUExecutionProvider",),
            description=f"CPU experimental path on {platform.system()} {platform.machine()}",
        )

    if cuda_ready:
        return RuntimeSelection(
            requested=normalized,
            resolved="cuda",
            torch_device="cuda",
            onnx_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
            description="Auto-selected NVIDIA CUDA path",
        )

    return RuntimeSelection(
        requested=normalized,
        resolved="cpu",
        torch_device="cpu",
        onnx_providers=("CPUExecutionProvider",),
        description=f"Auto-selected CPU experimental path on {platform.system()} {platform.machine()}",
    )


def resolve_tuilionnx_dir(tuilionnx_dir: str | os.PathLike | None = None) -> Path:
    paths = get_paths()
    env_override = os.getenv("DIGITAL_HUMAN_TUILIONNX_DIR")
    candidate = Path(tuilionnx_dir or env_override or paths.default_tuilionnx_dir).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"TuiliONNX directory not found: {candidate}. "
            "Put the local model bundle under my-video/digital_human/models/tuilionnx, "
            "pass --tuilionnx-dir, or set DIGITAL_HUMAN_TUILIONNX_DIR."
        )
    return candidate


def resolve_ffmpeg_bin(ffmpeg_bin: str | os.PathLike | None = None) -> str:
    env_override = os.getenv("DIGITAL_HUMAN_FFMPEG_BIN")
    candidates = []

    if ffmpeg_bin:
        candidates.append(Path(ffmpeg_bin).expanduser().resolve())
    elif env_override:
        candidates.append(Path(env_override).expanduser().resolve())

    which_ffmpeg = shutil.which("ffmpeg")
    if which_ffmpeg:
        candidates.append(Path(which_ffmpeg))

    # Fallback: look next to the current Python interpreter (e.g. conda env)
    conda_ffmpeg = Path(sys.executable).resolve().parent / "ffmpeg"
    if conda_ffmpeg.exists():
        candidates.append(conda_ffmpeg)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        "ffmpeg executable not found. Install ffmpeg, pass --ffmpeg-bin, "
        "or set DIGITAL_HUMAN_FFMPEG_BIN."
    )
