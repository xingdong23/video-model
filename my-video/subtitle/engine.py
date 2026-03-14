from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional

from fontTools.ttLib import TTFont

from .config import (
    get_paths,
    is_audio_file,
    is_video_file,
    resolve_correction_settings,
    resolve_ffmpeg_bin,
    resolve_whisper_runtime,
)

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


logger = logging.getLogger(__name__)

TRAILING_PUNCTUATION = "。，！？、；：''（）【】《》…—～·,.!?:;\"'()[]{}"
SPLIT_PUNCTUATION = "。，！？.!?,；;：:"
DEFAULT_CORRECTION_PROMPT = """请对以下语音识别生成的字幕文本执行精准纠错，严格遵循：
1. 仅修正错别字或多音字错误
2. 保持原始文本字数、行数和口语化风格
3. 结合上下文语义修正同音错字
4. 不要移除或添加标点符号
5. 禁止扩写、删改或风格调整
6. 每行内容单独修正，保持原有的行分隔

请直接输出修正后的完整文本，按原格式每行一条，不要任何解释。

需要修正的字幕文本（每行一条）：
"""


@dataclass(frozen=True)
class SubtitleLine:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class SubtitleGenerationResult:
    input_path: Path
    srt_path: Path
    entries_count: int
    model_name: str
    device: str
    compute_type: str
    detected_language: Optional[str]
    correction_applied: bool = False


@dataclass(frozen=True)
class SubtitleCorrectionResult:
    srt_path: Path
    backup_path: Path
    entries_count: int
    model_name: str


@dataclass(frozen=True)
class SubtitleBurnResult:
    video_path: Path
    subtitle_path: Path
    output_path: Path


class SubtitleEngine:
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
        compute_type: str = "auto",
        ffmpeg_bin: Optional[str] = None,
    ):
        self.paths = get_paths()
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self._requested_model_name = model_name
        self._requested_device = device
        self._requested_compute_type = compute_type
        self.runtime = None
        self._ffmpeg_bin = ffmpeg_bin
        self._model = None

    @property
    def model(self):
        if self.runtime is None:
            self.runtime = resolve_whisper_runtime(
                device=self._requested_device,
                model_name=self._requested_model_name,
                compute_type=self._requested_compute_type,
            )
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                model_size_or_path=self.runtime.model_name,
                device=self.runtime.resolved_device,
                compute_type=self.runtime.compute_type,
            )
        return self._model

    def generate_srt(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        language: Optional[str] = "zh",
        max_chars: int = 20,
        beam_size: int = 10,
        best_of: int = 5,
        vad_filter: bool = True,
        vad_min_silence_ms: int = 1000,
        speech_pad_ms: int = 300,
        apply_correction: bool = False,
        correction_api_key: Optional[str] = None,
        correction_api_base: Optional[str] = None,
        correction_model_name: Optional[str] = None,
        correction_timeout: Optional[int] = None,
    ) -> SubtitleGenerationResult:
        source_path = Path(input_path).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError("Input file not found: %s" % source_path)
        if not is_audio_file(source_path) and not is_video_file(source_path):
            raise ValueError("Unsupported input file type: %s" % source_path.suffix)

        destination = (
            Path(output_path).expanduser().resolve()
            if output_path
            else (self.paths.output_dir / ("%s.srt" % source_path.stem)).resolve()
        )
        destination.parent.mkdir(parents=True, exist_ok=True)

        transcribe_language = (language or "").strip().lower() or None
        if transcribe_language == "auto":
            transcribe_language = None

        if is_video_file(source_path):
            with tempfile.TemporaryDirectory(prefix="subtitle-audio-", dir=str(self.paths.output_dir)) as temp_dir:
                extracted_audio = self._extract_audio(source_path, Path(temp_dir))
                result = self._transcribe_to_srt(
                    input_path=source_path,
                    audio_path=extracted_audio,
                    destination=destination,
                    language=transcribe_language,
                    max_chars=max_chars,
                    beam_size=beam_size,
                    best_of=best_of,
                    vad_filter=vad_filter,
                    vad_min_silence_ms=vad_min_silence_ms,
                    speech_pad_ms=speech_pad_ms,
                )
                if apply_correction:
                    self.correct_subtitles(
                        subtitle_path=str(result.srt_path),
                        api_key=correction_api_key,
                        api_base=correction_api_base,
                        model_name=correction_model_name,
                        request_timeout=correction_timeout,
                    )
                    result = SubtitleGenerationResult(
                        input_path=result.input_path,
                        srt_path=result.srt_path,
                        entries_count=result.entries_count,
                        model_name=result.model_name,
                        device=result.device,
                        compute_type=result.compute_type,
                        detected_language=result.detected_language,
                        correction_applied=True,
                    )
                return result

        result = self._transcribe_to_srt(
            input_path=source_path,
            audio_path=source_path,
            destination=destination,
            language=transcribe_language,
            max_chars=max_chars,
            beam_size=beam_size,
            best_of=best_of,
            vad_filter=vad_filter,
            vad_min_silence_ms=vad_min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        if apply_correction:
            self.correct_subtitles(
                subtitle_path=str(result.srt_path),
                api_key=correction_api_key,
                api_base=correction_api_base,
                model_name=correction_model_name,
                request_timeout=correction_timeout,
            )
            result = SubtitleGenerationResult(
                input_path=result.input_path,
                srt_path=result.srt_path,
                entries_count=result.entries_count,
                model_name=result.model_name,
                device=result.device,
                compute_type=result.compute_type,
                detected_language=result.detected_language,
                correction_applied=True,
            )
        return result

    def correct_subtitles(
        self,
        subtitle_path: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_name: Optional[str] = None,
        request_timeout: Optional[int] = None,
    ) -> SubtitleCorrectionResult:
        srt_path = Path(subtitle_path).expanduser().resolve()
        if not srt_path.exists():
            raise FileNotFoundError("Subtitle file not found: %s" % srt_path)

        settings = resolve_correction_settings(
            api_key=api_key,
            api_base=api_base,
            model_name=model_name,
            request_timeout=request_timeout,
        )

        entries = self._parse_srt_entries(srt_path.read_text(encoding="utf-8"))
        if not entries:
            raise RuntimeError("No valid subtitle entries found in %s" % srt_path)

        source_lines = [entry["text"] for entry in entries]
        corrected_lines = self._request_corrected_lines(source_lines, settings)
        normalized_lines = self._normalize_corrected_lines(source_lines, corrected_lines)
        corrected_content = self._build_srt_content(entries, normalized_lines)

        backup_path = srt_path.with_suffix("%s.backup" % srt_path.suffix)
        shutil.copy2(srt_path, backup_path)
        srt_path.write_text(corrected_content, encoding="utf-8")

        return SubtitleCorrectionResult(
            srt_path=srt_path,
            backup_path=backup_path,
            entries_count=len(entries),
            model_name=settings.model_name,
        )

    def burn_subtitles(
        self,
        video_path: str,
        subtitle_path: str,
        output_path: Optional[str] = None,
        font_path: Optional[str] = None,
        font_name: Optional[str] = None,
        font_index: int = 0,
        font_size: int = 24,
        font_color: str = "#FFFFFF",
        outline_color: str = "#000000",
        outline: int = 1,
        wrap_style: int = 2,
        bottom_margin: int = 30,
    ) -> SubtitleBurnResult:
        input_video = Path(video_path).expanduser().resolve()
        input_subtitle = Path(subtitle_path).expanduser().resolve()
        if not input_video.exists():
            raise FileNotFoundError("Video file not found: %s" % input_video)
        if not input_subtitle.exists():
            raise FileNotFoundError("Subtitle file not found: %s" % input_subtitle)

        destination = (
            Path(output_path).expanduser().resolve()
            if output_path
            else input_video.with_name("%s_subtitled%s" % (input_video.stem, input_video.suffix))
        )
        destination.parent.mkdir(parents=True, exist_ok=True)

        fonts_dir = None
        resolved_font_name = font_name

        # Auto-detect font: use provided path, or fall back to bundled CJK font
        effective_font_path = font_path
        if not effective_font_path:
            bundled_font = self.paths.root / "fonts" / "NotoSansSC-Regular-Static.ttf"
            if bundled_font.exists():
                effective_font_path = str(bundled_font)
                logger.info("Using bundled CJK font: %s", bundled_font)

        if effective_font_path:
            font_file = Path(effective_font_path).expanduser().resolve()
            if not font_file.exists():
                raise FileNotFoundError("Font file not found: %s" % font_file)
            fonts_dir = font_file.parent
            if not resolved_font_name:
                resolved_font_name = self._read_font_name(font_file, font_index)

        style_parts = []
        if resolved_font_name:
            style_parts.append("FontName=%s" % resolved_font_name)
        style_parts.extend(
            [
                "FontSize=%s" % int(font_size),
                "PrimaryColour=%s" % self._convert_ass_color(font_color),
                "OutlineColour=%s" % self._convert_ass_color(outline_color),
                "Outline=%s" % int(outline),
                "WrapStyle=%s" % int(wrap_style),
                "MarginV=%s" % int(bottom_margin),
            ]
        )

        subtitle_filter = "subtitles=filename='%s'" % self._escape_filter_value(input_subtitle)
        if fonts_dir is not None:
            subtitle_filter += ":fontsdir='%s'" % self._escape_filter_value(fonts_dir)
        subtitle_filter += ":force_style='%s'" % ",".join(style_parts)

        ffmpeg_bin = resolve_ffmpeg_bin(self._ffmpeg_bin)
        if self._has_ffmpeg_filter(ffmpeg_bin, "subtitles"):
            return self._burn_subtitles_with_ffmpeg(
                ffmpeg_bin=ffmpeg_bin,
                input_video=input_video,
                destination=destination,
                input_subtitle=input_subtitle,
                subtitle_filter=subtitle_filter,
            )

        return self._burn_subtitles_with_moviepy(
            ffmpeg_bin=ffmpeg_bin,
            input_video=input_video,
            input_subtitle=input_subtitle,
            destination=destination,
            font_path=font_path,
            font_name=font_name,
            font_index=font_index,
            font_size=font_size,
            font_color=font_color,
            outline_color=outline_color,
            outline=outline,
            bottom_margin=bottom_margin,
        )

    @staticmethod
    def _fontconfig_env() -> dict:
        """Build env with FONTCONFIG_PATH so libass can find fonts."""
        env = os.environ.copy()
        # Ensure fontconfig can locate its config (needed in conda / container envs)
        if "FONTCONFIG_PATH" not in env:
            import sys
            conda_fc = Path(sys.executable).resolve().parent.parent / "etc" / "fonts"
            if conda_fc.is_dir():
                env["FONTCONFIG_PATH"] = str(conda_fc)
        return env

    def _burn_subtitles_with_ffmpeg(
        self,
        ffmpeg_bin: str,
        input_video: Path,
        destination: Path,
        input_subtitle: Path,
        subtitle_filter: str,
    ) -> SubtitleBurnResult:
        env = self._fontconfig_env()
        # Try NVENC first, fall back to libx264
        nvenc_command = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(input_video),
            "-vf",
            subtitle_filter,
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-cq",
            "20",
            "-c:a",
            "copy",
            str(destination),
        ]
        result = subprocess.run(
            nvenc_command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        if result.returncode != 0:
            logger.info("NVENC not available for subtitle burning, falling back to libx264")
            command = [
                ffmpeg_bin,
                "-y",
                "-i",
                str(input_video),
                "-vf",
                subtitle_filter,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-c:a",
                "copy",
                str(destination),
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            if result.returncode != 0:
                raise RuntimeError("ffmpeg failed while burning subtitles:\n%s" % result.stderr.strip())

        return SubtitleBurnResult(
            video_path=input_video,
            subtitle_path=input_subtitle,
            output_path=destination,
        )

    def _burn_subtitles_with_moviepy(
        self,
        ffmpeg_bin: str,
        input_video: Path,
        input_subtitle: Path,
        destination: Path,
        font_path: Optional[str],
        font_name: Optional[str],
        font_index: int,
        font_size: int,
        font_color: str,
        outline_color: str,
        outline: int,
        bottom_margin: int,
    ) -> SubtitleBurnResult:
        import numpy
        from PIL import Image, ImageColor, ImageDraw, ImageFont

        try:
            from moviepy.editor import VideoFileClip  # type: ignore
        except ImportError:
            from moviepy import VideoFileClip  # type: ignore

        entries = self._parse_srt_entries(input_subtitle.read_text(encoding="utf-8"))
        if not entries:
            raise RuntimeError("No valid subtitle entries found in %s" % input_subtitle)

        timeline = []
        for entry in entries:
            timeline.append(
                {
                    "start_seconds": self._parse_srt_timestamp(entry["start_time"]),
                    "end_seconds": self._parse_srt_timestamp(entry["end_time"]),
                    "text": entry["text"],
                }
            )

        font = self._load_pillow_font(font_path, font_name, font_size, font_index)
        fill_color = ImageColor.getrgb(font_color)
        stroke_color = ImageColor.getrgb(outline_color)

        def render_frame(get_frame, timestamp):
            frame = get_frame(timestamp)
            text = self._find_subtitle_text(timeline, float(timestamp))
            if not text:
                return frame

            image = Image.fromarray(frame).convert("RGBA")
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            wrapped_text = self._wrap_text_to_width(
                text=text,
                draw=draw,
                font=font,
                max_width=int(image.size[0] * 0.86),
            )
            bbox = draw.multiline_textbbox(
                (0, 0),
                wrapped_text,
                font=font,
                spacing=6,
                stroke_width=max(int(outline), 0),
            )
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = max(int((image.size[0] - text_width) / 2), 0)
            y = max(int(image.size[1] - bottom_margin - text_height), 0)
            draw.multiline_text(
                (x, y),
                wrapped_text,
                font=font,
                fill=fill_color + (255,),
                spacing=6,
                align="center",
                stroke_width=max(int(outline), 0),
                stroke_fill=stroke_color + (255,),
            )
            combined = Image.alpha_composite(image, overlay).convert("RGB")
            return numpy.array(combined)

        with tempfile.TemporaryDirectory(prefix="subtitle-burn-", dir=str(self.paths.output_dir)) as temp_dir:
            rendered_path = Path(temp_dir) / "rendered_video.mp4"
            clip = VideoFileClip(str(input_video))
            try:
                if hasattr(clip, "transform"):
                    rendered = clip.transform(lambda get_frame, t: render_frame(get_frame, t))
                else:
                    rendered = clip.fl(lambda get_frame, t: render_frame(get_frame, t))  # type: ignore[attr-defined]
                try:
                    rendered.write_videofile(
                        str(rendered_path),
                        codec="libx264",
                        audio=False,
                        fps=clip.fps,
                        logger=None,
                    )
                finally:
                    rendered.close()
            finally:
                clip.close()

            self._mux_original_audio(
                ffmpeg_bin=ffmpeg_bin,
                rendered_video=rendered_path,
                source_video=input_video,
                destination=destination,
            )

        return SubtitleBurnResult(
            video_path=input_video,
            subtitle_path=input_subtitle,
            output_path=destination,
        )

    def _transcribe_to_srt(
        self,
        input_path: Path,
        audio_path: Path,
        destination: Path,
        language: Optional[str],
        max_chars: int,
        beam_size: int,
        best_of: int,
        vad_filter: bool,
        vad_min_silence_ms: int,
        speech_pad_ms: int,
    ) -> SubtitleGenerationResult:
        segments, info = self.model.transcribe(
            str(audio_path),
            beam_size=int(beam_size),
            best_of=int(best_of),
            word_timestamps=True,
            vad_filter=bool(vad_filter),
            vad_parameters={
                "min_silence_duration_ms": int(vad_min_silence_ms),
                "speech_pad_ms": int(speech_pad_ms),
            },
            without_timestamps=False,
            language=language,
        )

        segments_list = list(segments)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        subtitle_lines = self._build_subtitle_lines(segments_list, max_chars=max_chars)
        if not subtitle_lines:
            raise RuntimeError("No subtitle lines were produced from %s" % input_path)

        self._write_srt(destination, subtitle_lines)

        detected_language = None
        if info is not None and hasattr(info, "language"):
            detected_language = getattr(info, "language")

        return SubtitleGenerationResult(
            input_path=input_path,
            srt_path=destination,
            entries_count=len(subtitle_lines),
            model_name=self.runtime.model_name,
            device=self.runtime.resolved_device,
            compute_type=self.runtime.compute_type,
            detected_language=detected_language,
        )

    def _extract_audio(self, video_path: Path, temp_dir: Path) -> Path:
        ffmpeg_bin = resolve_ffmpeg_bin(self._ffmpeg_bin)
        temp_audio = temp_dir / "input.wav"
        command = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(temp_audio),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError("ffmpeg failed while extracting audio:\n%s" % result.stderr.strip())
        return temp_audio

    def _build_subtitle_lines(self, segments: Iterable[object], max_chars: int) -> List[SubtitleLine]:
        lines = []
        for segment in segments:
            words = list(getattr(segment, "words", None) or [])
            if words:
                lines.extend(self._build_lines_from_words(segment, words, max_chars=max_chars))
                continue

            clean_text = self._clean_subtitle_text(getattr(segment, "text", ""))
            if not clean_text:
                continue

            start = float(getattr(segment, "start", 0.0) or 0.0)
            end = float(getattr(segment, "end", start) or start)
            lines.append(SubtitleLine(text=clean_text, start=start, end=max(end, start + 0.01)))
        return lines

    def _request_corrected_lines(self, source_lines: List[str], settings) -> List[str]:
        import requests

        endpoint = settings.api_base
        if not endpoint.endswith("/chat/completions"):
            endpoint = endpoint + "/chat/completions"

        payload = {
            "model": settings.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的字幕校对助手，精通中文语言和常见词汇。",
                },
                {
                    "role": "user",
                    "content": DEFAULT_CORRECTION_PROMPT + "\n".join(source_lines),
                },
            ],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": "Bearer %s" % settings.api_key,
            "Content-Type": "application/json",
        }

        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=settings.request_timeout,
        )
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Subtitle correction response did not include choices.")

        message = choices[0].get("message") or {}
        corrected_text = (message.get("content") or "").strip()
        if not corrected_text:
            raise RuntimeError("Subtitle correction response did not include corrected text.")

        return [line.strip() for line in corrected_text.splitlines()]

    @staticmethod
    def _normalize_corrected_lines(source_lines: List[str], corrected_lines: List[str]) -> List[str]:
        normalized = [line for line in corrected_lines if line != ""]
        if len(normalized) < len(source_lines):
            normalized.extend(source_lines[len(normalized) :])
        elif len(normalized) > len(source_lines):
            normalized = normalized[: len(source_lines)]
        return normalized

    @staticmethod
    def _parse_srt_entries(srt_content: str) -> List[dict]:
        entries = []
        chunks = [chunk.strip() for chunk in (srt_content or "").strip().split("\n\n") if chunk.strip()]
        for chunk in chunks:
            lines = chunk.splitlines()
            if len(lines) < 3:
                continue
            index = lines[0].strip()
            timestamp = lines[1].strip()
            if " --> " not in timestamp:
                continue
            start_time, end_time = timestamp.split(" --> ", 1)
            text = " ".join(part.strip() for part in lines[2:] if part.strip())
            entries.append(
                {
                    "index": index,
                    "start_time": start_time,
                    "end_time": end_time,
                    "text": text,
                }
            )
        return entries

    @staticmethod
    def _build_srt_content(entries: List[dict], corrected_lines: List[str]) -> str:
        blocks = []
        for index, entry in enumerate(entries):
            corrected_text = corrected_lines[index] if index < len(corrected_lines) else entry["text"]
            blocks.append(
                "%s\n%s --> %s\n%s"
                % (entry["index"], entry["start_time"], entry["end_time"], corrected_text)
            )
        return "\n\n".join(blocks) + "\n"

    def _build_lines_from_words(self, segment: object, words: List[object], max_chars: int) -> List[SubtitleLine]:
        lines = []
        current_text = ""
        segment_start = getattr(words[0], "start", None)
        if segment_start is None:
            segment_start = getattr(segment, "start", 0.0)
        current_start = float(segment_start or 0.0)
        current_end = current_start

        for word in words:
            word_text = getattr(word, "word", "") or ""
            if not word_text:
                continue

            word_end = getattr(word, "end", None)
            if word_end is None:
                word_end = getattr(segment, "end", current_end)
            current_text += word_text
            current_end = float(word_end or current_end)

            should_split = any(symbol in word_text for symbol in SPLIT_PUNCTUATION)
            if max_chars > 0 and self._visible_length(current_text) > int(max_chars):
                should_split = True

            if should_split:
                clean_text = self._clean_subtitle_text(current_text)
                if clean_text:
                    lines.append(
                        SubtitleLine(
                            text=clean_text,
                            start=current_start,
                            end=max(current_end, current_start + 0.01),
                        )
                    )
                current_text = ""
                current_start = current_end

        if current_text.strip():
            clean_text = self._clean_subtitle_text(current_text)
            if clean_text:
                segment_end = getattr(segment, "end", current_end)
                lines.append(
                    SubtitleLine(
                        text=clean_text,
                        start=current_start,
                        end=max(float(segment_end or current_end), current_start + 0.01),
                    )
                )

        return lines

    @staticmethod
    def _clean_subtitle_text(text: str) -> str:
        clean_text = (text or "").strip()
        while clean_text and clean_text[-1] in TRAILING_PUNCTUATION:
            clean_text = clean_text[:-1].strip()
        return clean_text

    @staticmethod
    def _visible_length(text: str) -> int:
        return len((text or "").replace(" ", "").replace("\n", "").strip())

    @staticmethod
    def _write_srt(destination: Path, lines: List[SubtitleLine]) -> None:
        with destination.open("w", encoding="utf-8") as handle:
            for index, line in enumerate(lines, 1):
                handle.write("%s\n" % index)
                handle.write(
                    "%s --> %s\n" % (
                        SubtitleEngine._format_timestamp(line.start),
                        SubtitleEngine._format_timestamp(line.end),
                    )
                )
                handle.write("%s\n\n" % line.text)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        total_milliseconds = max(int(round(float(seconds) * 1000.0)), 0)
        hours, remainder = divmod(total_milliseconds, 3600000)
        minutes, remainder = divmod(remainder, 60000)
        whole_seconds, milliseconds = divmod(remainder, 1000)
        return "%02d:%02d:%02d,%03d" % (hours, minutes, whole_seconds, milliseconds)

    @staticmethod
    def _convert_ass_color(hex_color: str) -> str:
        value = (hex_color or "#FFFFFF").strip()
        if not value.startswith("#") or len(value) != 7:
            raise ValueError("Color must use #RRGGBB format: %s" % value)
        red = int(value[1:3], 16)
        green = int(value[3:5], 16)
        blue = int(value[5:7], 16)
        return "&H%02X%02X%02X" % (blue, green, red)

    @staticmethod
    def _escape_filter_value(path: Path) -> str:
        value = str(path).replace("\\", "/")
        value = value.replace(":", "\\:")
        value = value.replace("'", "\\'")
        value = value.replace(",", "\\,")
        value = value.replace("[", "\\[")
        value = value.replace("]", "\\]")
        return value

    @staticmethod
    def _has_ffmpeg_filter(ffmpeg_bin: str, filter_name: str) -> bool:
        result = subprocess.run(
            [ffmpeg_bin, "-filters"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = "%s\n%s" % (result.stdout, result.stderr)
        if result.returncode != 0:
            return False

        for line in output.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == filter_name:
                return True
        return False

    @staticmethod
    def _mux_original_audio(
        ffmpeg_bin: str,
        rendered_video: Path,
        source_video: Path,
        destination: Path,
    ) -> None:
        command = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(rendered_video),
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-shortest",
            str(destination),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError("ffmpeg failed while muxing original audio:\n%s" % result.stderr.strip())

    @staticmethod
    def _parse_srt_timestamp(value: str) -> float:
        timestamp = value.strip()
        hours = int(timestamp[0:2])
        minutes = int(timestamp[3:5])
        seconds = int(timestamp[6:8])
        milliseconds = int(timestamp[9:12])
        return float(hours * 3600 + minutes * 60 + seconds) + float(milliseconds) / 1000.0

    @staticmethod
    def _find_subtitle_text(timeline: List[dict], timestamp: float) -> Optional[str]:
        for entry in timeline:
            if entry["start_seconds"] <= timestamp <= entry["end_seconds"]:
                return entry["text"]
        return None

    @staticmethod
    def _wrap_text_to_width(text, draw, font, max_width: int) -> str:
        lines = []
        current = ""
        for char in text:
            candidate = current + char
            bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=0)
            width = bbox[2] - bbox[0]
            if current and width > max_width:
                lines.append(current)
                current = char
            else:
                current = candidate
        if current:
            lines.append(current)
        return "\n".join(lines)

    @staticmethod
    def _load_pillow_font(
        font_path: Optional[str],
        font_name: Optional[str],
        font_size: int,
        font_index: int,
    ):
        from PIL import ImageFont

        candidates = []
        if font_path:
            candidates.append(Path(font_path).expanduser().resolve())
        if font_name:
            candidates.append(Path(font_name).expanduser())

        # Bundled CJK font (highest priority fallback)
        bundled_font = Path(__file__).resolve().parent / "fonts" / "NotoSansSC-Regular-Static.ttf"
        if bundled_font.exists():
            candidates.append(bundled_font)

        candidates.extend(
            [
                Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                Path("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"),
                Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
            ]
        )

        for candidate in candidates:
            try:
                return ImageFont.truetype(str(candidate), size=int(font_size), index=int(font_index))
            except Exception:
                logging.warning("Failed to load font: %s", candidate, exc_info=True)
                continue

        return ImageFont.load_default()

    @staticmethod
    def _read_font_name(font_path: Path, font_index: int) -> str:
        font = TTFont(str(font_path), fontNumber=int(font_index))
        try:
            name_table = font["name"]
            family_name = None  # nameID=1: Font Family
            full_name = None    # nameID=4: Full Name
            for record in name_table.names:
                if record.platformID != 3:
                    continue
                if record.nameID == 1 and family_name is None:
                    family_name = record.string.decode("utf-16-be")
                if record.nameID == 4 and full_name is None:
                    full_name = record.string.decode("utf-16-be")
            # Prefer family name (nameID=1) for libass/fontconfig compatibility
            return family_name or full_name or font_path.stem
        finally:
            font.close()
