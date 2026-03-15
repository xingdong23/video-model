from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .config import (
    VIDEO_SUFFIXES,
    RuntimeSelection,
    get_paths,
    resolve_ffmpeg_bin,
    resolve_runtime,
    resolve_tuilionnx_dir,
)


@dataclass(frozen=True)
class GenerationResult:
    output_path: Path
    reference_video: Path
    audio_path: Path
    elapsed_seconds: float
    runtime: str
    runtime_description: str
    diagnostics: dict[str, float]


logger = logging.getLogger(__name__)


class DigitalHumanEngine:
    def __init__(
        self,
        tuilionnx_dir: str | os.PathLike | None = None,
        ffmpeg_bin: str | os.PathLike | None = None,
        runtime: str = "auto",
    ):
        self.paths = get_paths()
        self._tuilionnx_dir = tuilionnx_dir
        self._ffmpeg_bin = ffmpeg_bin
        self._runtime = runtime
        self._runner_cache = {}
        self._runner_lock = threading.Lock()
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.local_faces_dir.mkdir(parents=True, exist_ok=True)

    @property
    def tuilionnx_dir(self) -> Path:
        return resolve_tuilionnx_dir(self._tuilionnx_dir)

    @property
    def checkpoints_dir(self) -> Path:
        return self.tuilionnx_dir / "checkpoints"

    @property
    def ffmpeg_bin(self) -> str:
        return resolve_ffmpeg_bin(self._ffmpeg_bin)

    @property
    def runtime(self) -> RuntimeSelection:
        return resolve_runtime(self._runtime)

    def list_face_videos(self) -> List[str]:
        files: dict[str, tuple[float, str]] = {}
        if self.paths.local_faces_dir.exists():
            for item in self.paths.local_faces_dir.iterdir():
                if not item.is_file() or item.suffix.lower() not in VIDEO_SUFFIXES:
                    continue
                files[item.name] = (item.stat().st_mtime, str(item))
        return [name for name, _ in sorted(files.items(), key=lambda entry: entry[1][0], reverse=True)]

    def resolve_reference_video(
        self,
        face: str | None = None,
        video: str | os.PathLike | None = None,
    ) -> Path:
        if video:
            resolved = Path(video).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Reference video not found: {resolved}")
            return resolved

        if not face:
            raise ValueError("Either --video or --face is required.")

        candidate = self.paths.local_faces_dir / face
        if candidate.exists():
            return candidate.resolve()

        raise FileNotFoundError(
            f"Face preset not found: {face}. Checked {self.paths.local_faces_dir}."
        )

    def _get_runner(
        self,
        *,
        human_path: Path,
        hubert_path: Path,
        runtime_selection: RuntimeSelection,
    ):
        from .pipeline import LstmSync

        cache_key = (
            str(human_path),
            str(hubert_path),
            runtime_selection.requested,
            runtime_selection.resolved,
            runtime_selection.onnx_providers,
            self.ffmpeg_bin,
        )
        with self._runner_lock:
            runner = self._runner_cache.get(cache_key)
            if runner is None:
                runner = LstmSync(
                    human_path=human_path,
                    hubert_path=hubert_path,
                    checkpoints_dir=self.checkpoints_dir,
                    runtime=runtime_selection,
                    ffmpeg_bin=self.ffmpeg_bin,
                )
                self._runner_cache[cache_key] = runner
                logger.info("Created digital human runner cache entry for %s", human_path.name)
            return runner

    def prepare_runtime(
        self,
        *,
        beautify_teeth: bool = False,
        runtime: str | None = None,
    ) -> RuntimeSelection:
        runtime_selection = resolve_runtime(runtime or self._runtime)
        human_path = self.checkpoints_dir / ("256.onnx" if beautify_teeth else "256_m.onnx")
        hubert_path = self.checkpoints_dir / "chinese-hubert-large"
        if not human_path.exists():
            raise FileNotFoundError(f"Human model not found: {human_path}")
        if not hubert_path.exists():
            raise FileNotFoundError(f"HuBERT model directory not found: {hubert_path}")
        runner = self._get_runner(
            human_path=human_path,
            hubert_path=hubert_path,
            runtime_selection=runtime_selection,
        )
        runner.preload_models()
        logger.info("Digital human runtime prepared: %s", runtime_selection.description)
        return runtime_selection

    def generate(
        self,
        audio: str | os.PathLike,
        face: str | None = None,
        video: str | os.PathLike | None = None,
        output_path: str | os.PathLike | None = None,
        batch_size: int = 16,
        sync_offset: int = 0,
        scale_h: float = 1.6,
        scale_w: float = 3.6,
        compress_inference: bool = False,
        beautify_teeth: bool = False,
        runtime: str | None = None,
        progress_callback=None,
    ) -> GenerationResult:
        audio_path = Path(audio).expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        reference_video = self.resolve_reference_video(face=face, video=video)
        output = Path(output_path).expanduser().resolve() if output_path else self.paths.output_dir / "output.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)

        runtime_selection = resolve_runtime(runtime or self._runtime)
        human_path = self.checkpoints_dir / ("256.onnx" if beautify_teeth else "256_m.onnx")
        hubert_path = self.checkpoints_dir / "chinese-hubert-large"
        if not human_path.exists():
            raise FileNotFoundError(f"Human model not found: {human_path}")
        if not hubert_path.exists():
            raise FileNotFoundError(f"HuBERT model directory not found: {hubert_path}")

        temp_dir = self.paths.output_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        video_fps25_path = temp_dir / f"fps25_temp_{timestamp}.mp4"
        video_temp_path = temp_dir / f"temp_{timestamp}"
        audio_temp_path = temp_dir / f"temp_{timestamp}.wav"

        start = time.perf_counter()
        lstm_sync = self._get_runner(
            human_path=human_path,
            hubert_path=hubert_path,
            runtime_selection=runtime_selection,
        )
        lstm_sync.configure_request(
            batch_size=batch_size,
            sync_offset=sync_offset,
            scale_h=scale_h,
            scale_w=scale_w,
            compress_inference_check_box=compress_inference,
            progress_callback=progress_callback,
        )
        try:
            result = lstm_sync.run(
                video_path=reference_video,
                video_fps25_path=video_fps25_path,
                video_temp_path=video_temp_path,
                audio_path=audio_path,
                audio_temp_path=audio_temp_path,
                video_out_path=output,
                compress_inference_check_box=compress_inference,
            )
        finally:
            for f in [video_fps25_path, audio_temp_path]:
                Path(f).unlink(missing_ok=True)
            # video_temp_path gets .mp4 appended in pipeline
            for suffix in (".mp4", ".avi"):
                Path(f"{video_temp_path}{suffix}").unlink(missing_ok=True)
        elapsed = time.perf_counter() - start
        logger.info("Digital human generation completed in %.1fs", elapsed)
        return GenerationResult(
            output_path=result,
            reference_video=reference_video,
            audio_path=audio_path,
            elapsed_seconds=elapsed,
            runtime=runtime_selection.resolved,
            runtime_description=runtime_selection.description,
            diagnostics=dict(getattr(lstm_sync, "last_run_stats", {})),
        )
