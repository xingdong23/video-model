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
        audio_config=None,
        digital_human_config=None,
        subtitle_config=None,
        subtitle_style_config=None,
        bgm_config=None,
        # Legacy kwargs for CLI / direct usage
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
        # Resolve config objects vs legacy kwargs.
        # Explicit kwargs (audio, video, prompt_audio) take precedence — the API router
        # resolves file_id -> path and passes it as the explicit kwarg.
        if audio_config is not None:
            _audio = audio or getattr(audio_config, "file_id", None)
            _text = text or getattr(audio_config, "text", None)
            _speaker = speaker or getattr(audio_config, "speaker", None)
            _prompt_text = prompt_text or getattr(audio_config, "prompt_text", None)
            _prompt_audio = prompt_audio or getattr(audio_config, "prompt_audio_file_id", None)
            _speed = getattr(audio_config, "speed", 1.0)
        else:
            _audio, _text, _speaker = audio, text, speaker
            _prompt_text, _prompt_audio, _speed = prompt_text, prompt_audio, speed

        if digital_human_config is not None:
            dc = digital_human_config
            _face = face or dc.face
            _video = video or dc.video_file_id
            _batch_size = dc.batch_size
            _sync_offset = dc.sync_offset
            _scale_h = dc.scale_h
            _scale_w = dc.scale_w
            _compress_inference = dc.compress_inference
            _beautify_teeth = dc.beautify_teeth
            _runtime = dc.runtime
        else:
            _face, _video = face, video
            _batch_size, _sync_offset = batch_size, sync_offset
            _scale_h, _scale_w = scale_h, scale_w
            _compress_inference, _beautify_teeth = compress_inference, beautify_teeth
            _runtime = runtime

        _with_subtitles = subtitle_config is not None if audio_config is not None else with_subtitles
        if subtitle_config is not None:
            sc = subtitle_config
            _subtitle_correct = sc.correct
            _subtitle_language = sc.language
            _subtitle_max_chars = sc.max_chars
            _subtitle_beam_size = sc.beam_size
            _subtitle_best_of = sc.best_of
            _subtitle_vad_filter = sc.vad_filter
            _subtitle_vad_min_silence_ms = sc.vad_min_silence_ms
            _subtitle_speech_pad_ms = sc.speech_pad_ms
            _subtitle_api_key = sc.api_key
            _subtitle_api_base = sc.api_base
            _subtitle_llm_model = sc.llm_model
            _subtitle_request_timeout = sc.request_timeout
        else:
            _subtitle_correct = subtitle_correct
            _subtitle_language = subtitle_language
            _subtitle_max_chars = subtitle_max_chars
            _subtitle_beam_size = subtitle_beam_size
            _subtitle_best_of = subtitle_best_of
            _subtitle_vad_filter = subtitle_vad_filter
            _subtitle_vad_min_silence_ms = subtitle_vad_min_silence_ms
            _subtitle_speech_pad_ms = subtitle_speech_pad_ms
            _subtitle_api_key = subtitle_api_key
            _subtitle_api_base = subtitle_api_base
            _subtitle_llm_model = subtitle_llm_model
            _subtitle_request_timeout = subtitle_request_timeout

        if subtitle_style_config is not None:
            ss = subtitle_style_config
            _subtitle_font_path = ss.font_path
            _subtitle_font_name = ss.font_name
            _subtitle_font_index = ss.font_index
            _subtitle_font_size = ss.font_size
            _subtitle_font_color = ss.font_color
            _subtitle_outline_color = ss.outline_color
            _subtitle_outline = ss.outline
            _subtitle_wrap_style = ss.wrap_style
            _subtitle_bottom_margin = ss.bottom_margin
        else:
            _subtitle_font_path = subtitle_font_path
            _subtitle_font_name = subtitle_font_name
            _subtitle_font_index = subtitle_font_index
            _subtitle_font_size = subtitle_font_size
            _subtitle_font_color = subtitle_font_color
            _subtitle_outline_color = subtitle_outline_color
            _subtitle_outline = subtitle_outline
            _subtitle_wrap_style = subtitle_wrap_style
            _subtitle_bottom_margin = subtitle_bottom_margin

        if bgm_config is not None:
            bc = bgm_config
            _bgm_path = bc.path
            _bgm_name = bc.name
            _bgm_random = bc.random
            _bgm_volume = bc.volume
            _bgm_original_volume = bc.original_volume
            _bgm_fade_out = bc.fade_out
            _bgm_loop = bc.loop
        else:
            _bgm_path = bgm_path
            _bgm_name = bgm_name
            _bgm_random = bgm_random
            _bgm_volume = bgm_volume
            _bgm_original_volume = bgm_original_volume
            _bgm_fade_out = bgm_fade_out
            _bgm_loop = bgm_loop

        final_output = self._resolve_final_output(output=output, with_subtitles=_with_subtitles)
        audio_path, audio_generated = self._resolve_audio(
            audio=_audio,
            text=_text,
            speaker=_speaker,
            prompt_text=_prompt_text,
            prompt_audio=_prompt_audio,
            speed=_speed,
            audio_output=audio_output,
            output_hint=final_output,
        )

        raw_video_path = self._resolve_raw_video_output(
            final_output=final_output,
            raw_video_output=raw_video_output,
            with_subtitles=_with_subtitles,
        )
        video_result = self.digital_human_engine.generate(
            audio=audio_path,
            face=_face,
            video=_video,
            output_path=raw_video_path,
            batch_size=_batch_size,
            sync_offset=_sync_offset,
            scale_h=_scale_h,
            scale_w=_scale_w,
            compress_inference=_compress_inference,
            beautify_teeth=_beautify_teeth,
            runtime=_runtime or self._runtime,
        )

        if not _with_subtitles:
            bgm_result = self._maybe_apply_bgm(
                current_video_path=video_result.output_path,
                final_output=final_output,
                bgm_path=_bgm_path,
                bgm_name=_bgm_name,
                bgm_random=_bgm_random,
                bgm_volume=_bgm_volume,
                bgm_original_volume=_bgm_original_volume,
                bgm_fade_out=_bgm_fade_out,
                bgm_loop=_bgm_loop,
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
            language=_subtitle_language,
            max_chars=_subtitle_max_chars,
            beam_size=_subtitle_beam_size,
            best_of=_subtitle_best_of,
            vad_filter=_subtitle_vad_filter,
            vad_min_silence_ms=_subtitle_vad_min_silence_ms,
            speech_pad_ms=_subtitle_speech_pad_ms,
            apply_correction=_subtitle_correct,
            correction_api_key=_subtitle_api_key,
            correction_api_base=_subtitle_api_base,
            correction_model_name=_subtitle_llm_model,
            correction_timeout=_subtitle_request_timeout,
        )
        burn_result = self.subtitle_engine.burn_subtitles(
            video_path=str(video_result.output_path),
            subtitle_path=str(subtitle_result.srt_path),
            output_path=str(final_output),
            font_path=_subtitle_font_path,
            font_name=_subtitle_font_name,
            font_index=_subtitle_font_index,
            font_size=_subtitle_font_size,
            font_color=_subtitle_font_color,
            outline_color=_subtitle_outline_color,
            outline=_subtitle_outline,
            wrap_style=_subtitle_wrap_style,
            bottom_margin=_subtitle_bottom_margin,
        )
        bgm_result = self._maybe_apply_bgm(
            current_video_path=burn_result.output_path,
            final_output=final_output,
            bgm_path=_bgm_path,
            bgm_name=_bgm_name,
            bgm_random=_bgm_random,
            bgm_volume=_bgm_volume,
            bgm_original_volume=_bgm_original_volume,
            bgm_fade_out=_bgm_fade_out,
            bgm_loop=_bgm_loop,
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
