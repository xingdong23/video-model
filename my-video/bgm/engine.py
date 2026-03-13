from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import get_paths, is_audio_file, is_video_file, resolve_ffmpeg_bin, resolve_ffprobe_bin

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BgmLibraryTrack:
    name: str
    path: Path
    relative_path: Path


@dataclass(frozen=True)
class VideoAudioInfo:
    duration_seconds: float
    has_audio: bool


@dataclass(frozen=True)
class BgmMixResult:
    video_path: Path
    output_path: Path
    track_path: Path
    track_name: str
    used_library_track: bool
    had_original_audio: bool
    volume: float
    original_volume: float
    fade_out_seconds: float
    loop_enabled: bool


@dataclass(frozen=True)
class _SelectedTrack:
    name: str
    path: Path
    used_library_track: bool


class BgmEngine:
    def __init__(
        self,
        library_dir: str | os.PathLike | None = None,
        ffmpeg_bin: Optional[str] = None,
        ffprobe_bin: Optional[str] = None,
    ):
        self.paths = get_paths()
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self._library_dir = (
            Path(library_dir).expanduser().resolve()
            if library_dir is not None
            else self.paths.library_dir
        )
        self._ffmpeg_bin_override = ffmpeg_bin
        self._ffprobe_bin_override = ffprobe_bin
        self._resolved_ffmpeg_bin: Optional[str] = None
        self._resolved_ffprobe_bin: Optional[str] = None

    @property
    def library_dir(self) -> Path:
        self._library_dir.mkdir(parents=True, exist_ok=True)
        return self._library_dir

    @property
    def ffmpeg_bin(self) -> str:
        if self._resolved_ffmpeg_bin is None:
            self._resolved_ffmpeg_bin = resolve_ffmpeg_bin(self._ffmpeg_bin_override)
        return self._resolved_ffmpeg_bin

    @property
    def ffprobe_bin(self) -> str:
        if self._resolved_ffprobe_bin is None:
            self._resolved_ffprobe_bin = resolve_ffprobe_bin(
                ffprobe_bin=self._ffprobe_bin_override,
                ffmpeg_bin=self.ffmpeg_bin,
            )
        return self._resolved_ffprobe_bin

    def list_library_tracks(self) -> list[BgmLibraryTrack]:
        tracks: list[BgmLibraryTrack] = []
        for path in sorted(self.library_dir.rglob("*")):
            if not path.is_file() or not is_audio_file(path):
                continue
            relative_path = path.relative_to(self.library_dir)
            track_name = relative_path.with_suffix("").as_posix()
            tracks.append(
                BgmLibraryTrack(
                    name=track_name,
                    path=path.resolve(),
                    relative_path=relative_path,
                )
            )
        return tracks

    def inspect_video(self, video_path: str | os.PathLike) -> VideoAudioInfo:
        source_path = Path(video_path).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError("Video file not found: %s" % source_path)
        if not is_video_file(source_path):
            raise ValueError("Unsupported video file type: %s" % source_path.suffix)

        command = [
            self.ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            str(source_path),
        ]
        result = self._run(command, failure_message="Failed to inspect video with ffprobe")
        payload = json.loads(result.stdout or "{}")

        duration_raw = (payload.get("format") or {}).get("duration")
        try:
            duration_seconds = float(duration_raw)
        except (TypeError, ValueError):
            raise RuntimeError("Unable to determine video duration for %s" % source_path)

        has_audio = any(
            stream.get("codec_type") == "audio"
            for stream in (payload.get("streams") or [])
        )
        return VideoAudioInfo(duration_seconds=duration_seconds, has_audio=has_audio)

    def mix(
        self,
        *,
        video_path: str | os.PathLike,
        bgm_path: str | os.PathLike | None = None,
        bgm_name: Optional[str] = None,
        output_path: str | os.PathLike | None = None,
        bgm_volume: float = 0.35,
        original_volume: float = 1.0,
        random_choice: bool = False,
        loop_bgm: bool = True,
        fade_out_seconds: float = 0.0,
    ) -> BgmMixResult:
        source_video = Path(video_path).expanduser().resolve()
        if not source_video.exists():
            raise FileNotFoundError("Video file not found: %s" % source_video)
        if not is_video_file(source_video):
            raise ValueError("Unsupported video file type: %s" % source_video.suffix)

        selected_track = self._resolve_track(
            bgm_path=bgm_path,
            bgm_name=bgm_name,
            random_choice=random_choice,
        )
        if bgm_volume < 0:
            raise ValueError("bgm_volume must be >= 0")
        if original_volume < 0:
            raise ValueError("original_volume must be >= 0")
        if fade_out_seconds < 0:
            raise ValueError("fade_out_seconds must be >= 0")

        info = self.inspect_video(source_video)
        destination = self._resolve_output_path(source_video=source_video, output_path=output_path)
        temp_output = self._make_temp_output(destination)
        fade_duration = min(float(fade_out_seconds), float(info.duration_seconds))
        command = self._build_ffmpeg_command(
            source_video=source_video,
            selected_track=selected_track,
            destination=temp_output,
            info=info,
            bgm_volume=float(bgm_volume),
            original_volume=float(original_volume),
            loop_bgm=loop_bgm,
            fade_out_seconds=fade_duration,
        )

        try:
            self._run(command, failure_message="Failed to mix background music into video")
            if temp_output.stat().st_size <= 0:
                raise RuntimeError("ffmpeg produced an empty output file: %s" % temp_output)
            shutil.move(str(temp_output), str(destination))
        finally:
            if temp_output.exists():
                temp_output.unlink()

        return BgmMixResult(
            video_path=source_video,
            output_path=destination,
            track_path=selected_track.path,
            track_name=selected_track.name,
            used_library_track=selected_track.used_library_track,
            had_original_audio=info.has_audio,
            volume=float(bgm_volume),
            original_volume=float(original_volume),
            fade_out_seconds=fade_duration,
            loop_enabled=loop_bgm,
        )

    def _resolve_track(
        self,
        *,
        bgm_path: str | os.PathLike | None,
        bgm_name: Optional[str],
        random_choice: bool,
    ) -> _SelectedTrack:
        provided_path = Path(bgm_path).expanduser().resolve() if bgm_path else None
        normalized_name = (bgm_name or "").strip()

        selection_count = int(provided_path is not None) + int(bool(normalized_name)) + int(random_choice)
        if selection_count != 1:
            raise ValueError(
                "Select exactly one BGM source: bgm_path, bgm_name, or random_choice."
            )

        if provided_path is not None:
            if not provided_path.exists():
                raise FileNotFoundError("BGM file not found: %s" % provided_path)
            if not is_audio_file(provided_path):
                raise ValueError("Unsupported BGM file type: %s" % provided_path.suffix)
            return _SelectedTrack(
                name=provided_path.stem,
                path=provided_path,
                used_library_track=False,
            )

        tracks = self.list_library_tracks()
        if not tracks:
            raise FileNotFoundError(
                "No BGM tracks found in %s. Add audio files or use bgm_path."
                % self.library_dir
            )

        if random_choice:
            chosen = random.choice(tracks)
            return _SelectedTrack(
                name=chosen.name,
                path=chosen.path,
                used_library_track=True,
            )

        for track in tracks:
            if track.name == normalized_name:
                return _SelectedTrack(
                    name=track.name,
                    path=track.path,
                    used_library_track=True,
                )

        available = ", ".join(track.name for track in tracks[:20])
        if len(tracks) > 20:
            available = "%s, ..." % available
        raise ValueError(
            "BGM track %r not found in %s. Available tracks: %s"
            % (normalized_name, self.library_dir, available or "<empty>")
        )

    def _resolve_output_path(
        self,
        *,
        source_video: Path,
        output_path: str | os.PathLike | None,
    ) -> Path:
        if output_path:
            destination = Path(output_path).expanduser().resolve()
        else:
            destination = (self.paths.output_dir / ("%s_bgm%s" % (source_video.stem, source_video.suffix))).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def _make_temp_output(self, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="%s." % destination.stem,
            suffix=destination.suffix,
            dir=str(destination.parent),
            delete=False,
        ) as handle:
            temp_path = Path(handle.name).resolve()
        temp_path.unlink(missing_ok=True)
        return temp_path

    def _build_ffmpeg_command(
        self,
        *,
        source_video: Path,
        selected_track: _SelectedTrack,
        destination: Path,
        info: VideoAudioInfo,
        bgm_volume: float,
        original_volume: float,
        loop_bgm: bool,
        fade_out_seconds: float,
    ) -> list[str]:
        bgm_chain = [
            "aformat=sample_fmts=fltp:channel_layouts=stereo",
            "volume=%s" % self._format_number(bgm_volume),
            "aresample=async=1:first_pts=0",
        ]
        if loop_bgm:
            bgm_chain.append("atrim=duration=%s" % self._format_number(info.duration_seconds))
        else:
            bgm_chain.append("apad=whole_dur=%s" % self._format_number(info.duration_seconds))
            bgm_chain.append("atrim=duration=%s" % self._format_number(info.duration_seconds))
        if fade_out_seconds > 0:
            fade_start = max(info.duration_seconds - fade_out_seconds, 0.0)
            bgm_chain.append(
                "afade=t=out:st=%s:d=%s"
                % (
                    self._format_number(fade_start),
                    self._format_number(fade_out_seconds),
                )
            )
        bgm_chain.append("asetpts=PTS-STARTPTS")

        if info.has_audio:
            original_chain = [
                "aformat=sample_fmts=fltp:channel_layouts=stereo",
                "volume=%s" % self._format_number(original_volume),
                "aresample=async=1:first_pts=0",
                "apad=whole_dur=%s" % self._format_number(info.duration_seconds),
                "atrim=duration=%s" % self._format_number(info.duration_seconds),
            ]
            filter_complex = (
                "[0:a:0]%s[orig_audio];"
                "[1:a:0]%s[bgm_audio];"
                "[orig_audio][bgm_audio]amix=inputs=2:duration=longest:normalize=0[final_audio]"
            ) % (",".join(original_chain), ",".join(bgm_chain))
        else:
            filter_complex = "[1:a:0]%s[final_audio]" % ",".join(bgm_chain)

        command = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            str(source_video),
        ]
        if loop_bgm:
            command.extend(["-stream_loop", "-1"])
        command.extend(["-i", str(selected_track.path)])
        command.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                "0:v:0",
                "-map",
                "[final_audio]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-shortest",
                "-movflags",
                "+faststart",
                str(destination),
            ]
        )
        return command

    @staticmethod
    def _format_number(value: float) -> str:
        rendered = ("%.6f" % float(value)).rstrip("0").rstrip(".")
        return rendered or "0"

    @staticmethod
    def _run(command: list[str], failure_message: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            if details:
                raise RuntimeError("%s: %s" % (failure_message, details))
            raise RuntimeError(failure_message)
        return result
