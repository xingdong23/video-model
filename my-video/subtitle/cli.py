from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if TYPE_CHECKING:
    if __package__ in (None, ""):
        from subtitle.engine import SubtitleEngine  # type: ignore
    else:
        from .engine import SubtitleEngine


def _load_subtitle_engine():
    if __package__ in (None, ""):
        from subtitle.engine import SubtitleEngine  # type: ignore

        return SubtitleEngine
    from .engine import SubtitleEngine

    return SubtitleEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone subtitle generator and burn-in tool for DigiHuman"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate an SRT subtitle file from audio or video")
    generate.add_argument("--input", required=True, help="Input audio or video file path")
    generate.add_argument("--output", help="Output SRT path")
    generate.add_argument("--language", default="zh", help="Whisper language code, or 'auto'")
    generate.add_argument("--model", help="Whisper model name or local model directory")
    generate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    generate.add_argument("--compute-type", default="auto")
    generate.add_argument("--max-chars", type=int, default=20, help="Max characters before splitting a subtitle line")
    generate.add_argument("--beam-size", type=int, default=10)
    generate.add_argument("--best-of", type=int, default=5)
    generate.add_argument("--no-vad-filter", dest="vad_filter", action="store_false")
    generate.add_argument("--vad-min-silence-ms", type=int, default=1000)
    generate.add_argument("--speech-pad-ms", type=int, default=300)
    generate.add_argument("--correct", action="store_true", help="Apply optional LLM subtitle correction after SRT generation")
    generate.add_argument("--api-key", help="Correction API key, or use SUBTITLE_LLM_API_KEY")
    generate.add_argument("--api-base", help="Correction API base URL, or use SUBTITLE_LLM_API_BASE")
    generate.add_argument("--llm-model", help="Correction model name, or use SUBTITLE_LLM_MODEL")
    generate.add_argument("--request-timeout", type=int, help="Correction HTTP timeout in seconds")
    generate.add_argument("--ffmpeg-bin", help="ffmpeg path, only needed when the input is a video")
    generate.set_defaults(vad_filter=True)

    correct = subparsers.add_parser("correct", help="Correct an existing SRT subtitle file with an OpenAI-compatible API")
    correct.add_argument("--subtitle", required=True, help="Input subtitle .srt path")
    correct.add_argument("--api-key", help="Correction API key, or use SUBTITLE_LLM_API_KEY")
    correct.add_argument("--api-base", help="Correction API base URL, or use SUBTITLE_LLM_API_BASE")
    correct.add_argument("--llm-model", help="Correction model name, or use SUBTITLE_LLM_MODEL")
    correct.add_argument("--request-timeout", type=int, help="Correction HTTP timeout in seconds")

    burn = subparsers.add_parser("burn", help="Burn an existing subtitle file into a video")
    burn.add_argument("--video", required=True, help="Input video path")
    burn.add_argument("--subtitle", required=True, help="Input subtitle .srt path")
    burn.add_argument("--output", help="Output video path")
    burn.add_argument("--font-path", help="Optional font file path")
    burn.add_argument("--font-name", help="Optional explicit font family name")
    burn.add_argument("--font-index", type=int, default=0, help="TTC font index when --font-path points to a collection")
    burn.add_argument("--font-size", type=int, default=24)
    burn.add_argument("--font-color", default="#FFFFFF")
    burn.add_argument("--outline-color", default="#000000")
    burn.add_argument("--outline", type=int, default=1)
    burn.add_argument("--wrap-style", type=int, default=2)
    burn.add_argument("--bottom-margin", type=int, default=30)
    burn.add_argument("--ffmpeg-bin", help="ffmpeg path")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "generate":
            SubtitleEngine = _load_subtitle_engine()
            engine = SubtitleEngine(
                model_name=args.model,
                device=args.device,
                compute_type=args.compute_type,
                ffmpeg_bin=args.ffmpeg_bin,
            )
            result = engine.generate_srt(
                input_path=args.input,
                output_path=args.output,
                language=args.language,
                max_chars=args.max_chars,
                beam_size=args.beam_size,
                best_of=args.best_of,
                vad_filter=args.vad_filter,
                vad_min_silence_ms=args.vad_min_silence_ms,
                speech_pad_ms=args.speech_pad_ms,
                apply_correction=args.correct,
                correction_api_key=args.api_key,
                correction_api_base=args.api_base,
                correction_model_name=args.llm_model,
                correction_timeout=args.request_timeout,
            )
            print(result.srt_path)
            print("entries=%s" % result.entries_count)
            print("model=%s" % result.model_name)
            print("device=%s" % result.device)
            print("compute_type=%s" % result.compute_type)
            if result.detected_language:
                print("language=%s" % result.detected_language)
            print("correction_applied=%s" % ("true" if result.correction_applied else "false"))
            return

        if args.command == "correct":
            SubtitleEngine = _load_subtitle_engine()
            engine = SubtitleEngine()
            result = engine.correct_subtitles(
                subtitle_path=args.subtitle,
                api_key=args.api_key,
                api_base=args.api_base,
                model_name=args.llm_model,
                request_timeout=args.request_timeout,
            )
            print(result.srt_path)
            print("entries=%s" % result.entries_count)
            print("llm_model=%s" % result.model_name)
            print("backup=%s" % result.backup_path)
            return

        SubtitleEngine = _load_subtitle_engine()
        engine = SubtitleEngine(ffmpeg_bin=args.ffmpeg_bin)
        burn_result = engine.burn_subtitles(
            video_path=args.video,
            subtitle_path=args.subtitle,
            output_path=args.output,
            font_path=args.font_path,
            font_name=args.font_name,
            font_index=args.font_index,
            font_size=args.font_size,
            font_color=args.font_color,
            outline_color=args.outline_color,
            outline=args.outline,
            wrap_style=args.wrap_style,
            bottom_margin=args.bottom_margin,
        )
        print(burn_result.output_path)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, "%s\n" % exc)


if __name__ == "__main__":
    main()
