from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .config import get_paths

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from bgm.engine import BgmEngine, BgmMixResult
    from digital_human.engine import DigitalHumanEngine, GenerationResult
    from subtitle.engine import SubtitleEngine, SubtitleGenerationResult
    from voice.engine import VoiceEngine


@dataclass(frozen=True)
class WorkflowResult:
    audio_path: Path
    raw_video_path: Path
    final_video_path: Path
    subtitle_path: Optional[Path]
    bgm_track: Optional[Path]
    audio_generated: bool
    subtitle_generated: bool
    subtitle_burned: bool
    bgm_applied: bool
    video_runtime: str
    video_runtime_description: str
    video_elapsed_seconds: float


class WorkflowEngine:
    def __init__(
        self,
        voice_model_dir: str | os.PathLike | None = None,
        tuilionnx_dir: str | os.PathLike | None = None,
        ffmpeg_bin: str | os.PathLike | None = None,
        runtime: str = "auto",
        subtitle_model_name: Optional[str] = None,
        subtitle_device: str = "auto",
        subtitle_compute_type: str = "auto",
        bgm_library_dir: str | os.PathLike | None = None,
    ):
        self.paths = get_paths()
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self._voice_model_dir = voice_model_dir
        self._tuilionnx_dir = tuilionnx_dir
        self._ffmpeg_bin = ffmpeg_bin
        self._runtime = runtime
        self._subtitle_model_name = subtitle_model_name
        self._subtitle_device = subtitle_device
        self._subtitle_compute_type = subtitle_compute_type
        self._bgm_library_dir = bgm_library_dir
        self._voice_engine = None
        self._digital_human_engine = None
        self._subtitle_engine = None
        self._bgm_engine = None

    @property
    def voice_engine(self):
        if self._voice_engine is None:
            from voice.engine import VoiceEngine

            self._voice_engine = VoiceEngine(model_dir=self._voice_model_dir)
        return self._voice_engine

    @property
    def digital_human_engine(self):
        if self._digital_human_engine is None:
            from digital_human.engine import DigitalHumanEngine

            self._digital_human_engine = DigitalHumanEngine(
                tuilionnx_dir=self._tuilionnx_dir,
                ffmpeg_bin=self._ffmpeg_bin,
                runtime=self._runtime,
            )
        return self._digital_human_engine

    @property
    def subtitle_engine(self):
        if self._subtitle_engine is None:
            from subtitle.engine import SubtitleEngine

            self._subtitle_engine = SubtitleEngine(
                model_name=self._subtitle_model_name,
                device=self._subtitle_device,
                compute_type=self._subtitle_compute_type,
                ffmpeg_bin=self._ffmpeg_bin,
            )
        return self._subtitle_engine

    @property
    def bgm_engine(self):
        if self._bgm_engine is None:
            from bgm.engine import BgmEngine

            self._bgm_engine = BgmEngine(
                library_dir=self._bgm_library_dir,
                ffmpeg_bin=self._ffmpeg_bin,
            )
        return self._bgm_engine

    def run(
        self,
        *,
        audio: Optional[str] = None,
        text: Optional[str] = None,
        speaker: Optional[str] = None,
        prompt_text: Optional[str] = None,
        prompt_audio: Optional[str] = None,
        speed: float = 1.0,
        face: Optional[str] = None,
        video: Optional[str] = None,
        output: Optional[str] = None,
        raw_video_output: Optional[str] = None,
        audio_output: Optional[str] = None,
        subtitle_output: Optional[str] = None,
        with_subtitles: bool = False,
        subtitle_correct: bool = False,
        subtitle_language: Optional[str] = "zh",
        subtitle_max_chars: int = 20,
        subtitle_beam_size: int = 10,
        subtitle_best_of: int = 5,
        subtitle_vad_filter: bool = True,
        subtitle_vad_min_silence_ms: int = 1000,
        subtitle_speech_pad_ms: int = 300,
        subtitle_api_key: Optional[str] = None,
        subtitle_api_base: Optional[str] = None,
        subtitle_llm_model: Optional[str] = None,
        subtitle_request_timeout: Optional[int] = None,
        subtitle_font_path: Optional[str] = None,
        subtitle_font_name: Optional[str] = None,
        subtitle_font_index: int = 0,
        subtitle_font_size: int = 24,
        subtitle_font_color: str = "#FFFFFF",
        subtitle_outline_color: str = "#000000",
        subtitle_outline: int = 1,
        subtitle_wrap_style: int = 2,
        subtitle_bottom_margin: int = 30,
        bgm_path: Optional[str] = None,
        bgm_name: Optional[str] = None,
        bgm_random: bool = False,
        bgm_volume: float = 0.35,
        bgm_original_volume: float = 1.0,
        bgm_fade_out: float = 0.0,
        bgm_loop: bool = True,
        batch_size: int = 4,
        sync_offset: int = 0,
        scale_h: float = 1.6,
        scale_w: float = 3.6,
        compress_inference: bool = False,
        beautify_teeth: bool = False,
        runtime: Optional[str] = None,
    ) -> WorkflowResult:
        final_output = self._resolve_final_output(output=output, with_subtitles=with_subtitles)
        audio_path, audio_generated = self._resolve_audio(
            audio=audio,
            text=text,
            speaker=speaker,
            prompt_text=prompt_text,
            prompt_audio=prompt_audio,
            speed=speed,
            audio_output=audio_output,
            output_hint=final_output,
        )

        raw_video_path = self._resolve_raw_video_output(
            final_output=final_output,
            raw_video_output=raw_video_output,
            with_subtitles=with_subtitles,
        )
        video_result = self.digital_human_engine.generate(
            audio=audio_path,
            face=face,
            video=video,
            output_path=raw_video_path,
            batch_size=batch_size,
            sync_offset=sync_offset,
            scale_h=scale_h,
            scale_w=scale_w,
            compress_inference=compress_inference,
            beautify_teeth=beautify_teeth,
            runtime=runtime or self._runtime,
        )

        if not with_subtitles:
            bgm_result = self._maybe_apply_bgm(
                current_video_path=video_result.output_path,
                final_output=final_output,
                bgm_path=bgm_path,
                bgm_name=bgm_name,
                bgm_random=bgm_random,
                bgm_volume=bgm_volume,
                bgm_original_volume=bgm_original_volume,
                bgm_fade_out=bgm_fade_out,
                bgm_loop=bgm_loop,
            )
            return WorkflowResult(
                audio_path=audio_path,
                raw_video_path=video_result.output_path,
                final_video_path=bgm_result.output_path if bgm_result is not None else video_result.output_path,
                subtitle_path=None,
                bgm_track=bgm_result.track_path if bgm_result is not None else None,
                audio_generated=audio_generated,
                subtitle_generated=False,
                subtitle_burned=False,
                bgm_applied=bgm_result is not None,
                video_runtime=video_result.runtime,
                video_runtime_description=video_result.runtime_description,
                video_elapsed_seconds=video_result.elapsed_seconds,
            )

        subtitle_path = self._resolve_subtitle_output(
            subtitle_output=subtitle_output,
            final_output=final_output,
        )
        subtitle_result = self.subtitle_engine.generate_srt(
            input_path=str(audio_path),
            output_path=str(subtitle_path),
            language=subtitle_language,
            max_chars=subtitle_max_chars,
            beam_size=subtitle_beam_size,
            best_of=subtitle_best_of,
            vad_filter=subtitle_vad_filter,
            vad_min_silence_ms=subtitle_vad_min_silence_ms,
            speech_pad_ms=subtitle_speech_pad_ms,
            apply_correction=subtitle_correct,
            correction_api_key=subtitle_api_key,
            correction_api_base=subtitle_api_base,
            correction_model_name=subtitle_llm_model,
            correction_timeout=subtitle_request_timeout,
        )
        burn_result = self.subtitle_engine.burn_subtitles(
            video_path=str(video_result.output_path),
            subtitle_path=str(subtitle_result.srt_path),
            output_path=str(final_output),
            font_path=subtitle_font_path,
            font_name=subtitle_font_name,
            font_index=subtitle_font_index,
            font_size=subtitle_font_size,
            font_color=subtitle_font_color,
            outline_color=subtitle_outline_color,
            outline=subtitle_outline,
            wrap_style=subtitle_wrap_style,
            bottom_margin=subtitle_bottom_margin,
        )
        bgm_result = self._maybe_apply_bgm(
            current_video_path=burn_result.output_path,
            final_output=final_output,
            bgm_path=bgm_path,
            bgm_name=bgm_name,
            bgm_random=bgm_random,
            bgm_volume=bgm_volume,
            bgm_original_volume=bgm_original_volume,
            bgm_fade_out=bgm_fade_out,
            bgm_loop=bgm_loop,
        )
        return WorkflowResult(
            audio_path=audio_path,
            raw_video_path=video_result.output_path,
            final_video_path=bgm_result.output_path if bgm_result is not None else burn_result.output_path,
            subtitle_path=subtitle_result.srt_path,
            bgm_track=bgm_result.track_path if bgm_result is not None else None,
            audio_generated=audio_generated,
            subtitle_generated=True,
            subtitle_burned=True,
            bgm_applied=bgm_result is not None,
            video_runtime=video_result.runtime,
            video_runtime_description=video_result.runtime_description,
            video_elapsed_seconds=video_result.elapsed_seconds,
        )

    def _resolve_audio(
        self,
        *,
        audio: Optional[str],
        text: Optional[str],
        speaker: Optional[str],
        prompt_text: Optional[str],
        prompt_audio: Optional[str],
        speed: float,
        audio_output: Optional[str],
        output_hint: Path,
    ) -> tuple[Path, bool]:
        if audio:
            if text or speaker or prompt_text or prompt_audio:
                raise ValueError("Do not combine --audio with text-to-speech options.")
            audio_path = Path(audio).expanduser().resolve()
            if not audio_path.exists():
                raise FileNotFoundError("Audio file not found: %s" % audio_path)
            return audio_path, False

        normalized_text = (text or "").strip()
        if not normalized_text:
            raise ValueError("Provide either --audio or --text.")

        using_zero_shot = bool(prompt_text or prompt_audio)
        if using_zero_shot:
            if speaker:
                raise ValueError("Do not combine --speaker with zero-shot prompt options.")
            if not (prompt_text and prompt_audio):
                raise ValueError("Zero-shot mode requires both --prompt-text and --prompt-audio.")
            destination = self._resolve_audio_output(audio_output=audio_output, output_hint=output_hint)
            result = self.voice_engine.synthesize_zero_shot_to_file(
                text=normalized_text,
                prompt_text=prompt_text,
                prompt_audio_path=prompt_audio,
                output_path=destination,
                speed=speed,
            )
            return result, True

        normalized_speaker = (speaker or "").strip()
        if not normalized_speaker:
            raise ValueError("TTS mode requires --speaker when --audio is not provided.")
        destination = self._resolve_audio_output(audio_output=audio_output, output_hint=output_hint)
        result = self.voice_engine.synthesize_to_file(
            text=normalized_text,
            speaker=normalized_speaker,
            output_path=destination,
            speed=speed,
        )
        return result, True

    def _resolve_final_output(self, output: Optional[str], with_subtitles: bool) -> Path:
        if output:
            destination = Path(output).expanduser().resolve()
        else:
            suffix = "subtitled" if with_subtitles else "video"
            destination = (self.paths.output_dir / f"workflow_{uuid.uuid4().hex[:12]}_{suffix}.mp4").resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def _resolve_audio_output(self, audio_output: Optional[str], output_hint: Path) -> Path:
        if audio_output:
            destination = Path(audio_output).expanduser().resolve()
        else:
            destination = (self.paths.output_dir / ("%s.wav" % output_hint.stem)).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    @staticmethod
    def _resolve_raw_video_output(
        final_output: Path,
        raw_video_output: Optional[str],
        with_subtitles: bool,
    ) -> Path:
        if raw_video_output:
            destination = Path(raw_video_output).expanduser().resolve()
        elif with_subtitles:
            destination = final_output.with_name("%s_raw%s" % (final_output.stem, final_output.suffix))
        else:
            destination = final_output
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    @staticmethod
    def _resolve_subtitle_output(subtitle_output: Optional[str], final_output: Path) -> Path:
        if subtitle_output:
            destination = Path(subtitle_output).expanduser().resolve()
        else:
            destination = final_output.with_suffix(".srt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def _maybe_apply_bgm(
        self,
        *,
        current_video_path: Path,
        final_output: Path,
        bgm_path: Optional[str],
        bgm_name: Optional[str],
        bgm_random: bool,
        bgm_volume: float,
        bgm_original_volume: float,
        bgm_fade_out: float,
        bgm_loop: bool,
    ):
        if not any((bgm_path, bgm_name, bgm_random)):
            return None

        return self.bgm_engine.mix(
            video_path=str(current_video_path),
            bgm_path=bgm_path,
            bgm_name=bgm_name,
            output_path=str(final_output),
            bgm_volume=bgm_volume,
            original_volume=bgm_original_volume,
            random_choice=bgm_random,
            loop_bgm=bgm_loop,
            fade_out_seconds=bgm_fade_out,
        )
