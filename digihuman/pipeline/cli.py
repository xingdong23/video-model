from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if TYPE_CHECKING:
    if __package__ in (None, ""):
        from pipeline.engine import WorkflowEngine  # type: ignore
    else:
        from .engine import WorkflowEngine


def _load_workflow_engine():
    if __package__ in (None, ""):
        from pipeline.engine import WorkflowEngine  # type: ignore

        return WorkflowEngine
    from .engine import WorkflowEngine

    return WorkflowEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Integrated DigiHuman pipeline for audio, digital human, and subtitles")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the integrated workflow")
    run.add_argument("--audio", help="Existing input audio path")
    run.add_argument("--text", help="Text for TTS when --audio is not provided")
    run.add_argument("--speaker", help="Speaker name for standard TTS mode")
    run.add_argument("--prompt-text", help="Prompt transcript for zero-shot TTS mode")
    run.add_argument("--prompt-audio", help="Prompt audio path for zero-shot TTS mode")
    run.add_argument("--speed", type=float, default=1.0)
    run.add_argument("--face", help="Preset face video filename")
    run.add_argument("--video", help="Explicit reference video path")
    run.add_argument("--output", help="Final output video path")
    run.add_argument("--raw-video-output", help="Optional intermediate digital human video path")
    run.add_argument("--audio-output", help="Optional generated audio output path")
    run.add_argument("--subtitle-output", help="Optional subtitle .srt output path")
    run.add_argument("--with-subtitles", action="store_true", help="Generate subtitles and burn them into the final video")
    run.add_argument("--subtitle-correct", action="store_true", help="Apply optional LLM correction after subtitle generation")
    run.add_argument("--subtitle-language", default="zh", help="Subtitle ASR language code, or 'auto'")
    run.add_argument("--subtitle-model", help="Subtitle Whisper model name or local model directory")
    run.add_argument("--subtitle-device", choices=["auto", "cpu", "cuda"], default="auto")
    run.add_argument("--subtitle-compute-type", default="auto")
    run.add_argument("--subtitle-max-chars", type=int, default=20)
    run.add_argument("--subtitle-beam-size", type=int, default=10)
    run.add_argument("--subtitle-best-of", type=int, default=5)
    run.add_argument("--no-subtitle-vad-filter", dest="subtitle_vad_filter", action="store_false")
    run.add_argument("--subtitle-vad-min-silence-ms", type=int, default=1000)
    run.add_argument("--subtitle-speech-pad-ms", type=int, default=300)
    run.add_argument("--subtitle-api-key", help="Correction API key")
    run.add_argument("--subtitle-api-base", help="Correction API base URL")
    run.add_argument("--subtitle-llm-model", help="Correction model name")
    run.add_argument("--subtitle-request-timeout", type=int, help="Correction HTTP timeout in seconds")
    run.add_argument("--subtitle-font-path", help="Optional subtitle font file path")
    run.add_argument("--subtitle-font-name", help="Optional explicit subtitle font family name")
    run.add_argument("--subtitle-font-index", type=int, default=0)
    run.add_argument("--subtitle-font-size", type=int, default=24)
    run.add_argument("--subtitle-font-color", default="#FFFFFF")
    run.add_argument("--subtitle-outline-color", default="#000000")
    run.add_argument("--subtitle-outline", type=int, default=1)
    run.add_argument("--subtitle-wrap-style", type=int, default=2)
    run.add_argument("--subtitle-bottom-margin", type=int, default=30)
    bgm_source = run.add_mutually_exclusive_group()
    bgm_source.add_argument("--bgm-path", help="Explicit BGM audio path")
    bgm_source.add_argument("--bgm-name", help="Preset BGM track name from audio_mixer/library")
    bgm_source.add_argument("--bgm-random", action="store_true", help="Randomly select a preset BGM track from audio_mixer/library")
    run.add_argument("--bgm-library-dir", help="Optional override for the BGM library directory")
    run.add_argument("--bgm-volume", type=float, default=0.35)
    run.add_argument("--bgm-original-volume", type=float, default=1.0)
    run.add_argument("--bgm-fade-out", type=float, default=0.0)
    run.add_argument("--no-bgm-loop", dest="bgm_loop", action="store_false")
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--sync-offset", type=int, default=0)
    run.add_argument("--scale-h", type=float, default=1.6)
    run.add_argument("--scale-w", type=float, default=3.6)
    run.add_argument("--compress-inference", action="store_true")
    run.add_argument("--beautify-teeth", action="store_true")
    run.add_argument("--voice-model-dir")
    run.add_argument("--tuilionnx-dir")
    run.add_argument("--ffmpeg-bin")
    run.add_argument("--runtime", choices=["auto", "tensorrt", "cuda", "cpu"], default="auto")
    run.set_defaults(subtitle_vad_filter=True, bgm_loop=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        WorkflowEngine = _load_workflow_engine()
        engine = WorkflowEngine(
            voice_model_dir=args.voice_model_dir,
            tuilionnx_dir=args.tuilionnx_dir,
            ffmpeg_bin=args.ffmpeg_bin,
            runtime=args.runtime,
            subtitle_model_name=args.subtitle_model,
            subtitle_device=args.subtitle_device,
            subtitle_compute_type=args.subtitle_compute_type,
            bgm_library_dir=args.bgm_library_dir,
        )
        result = engine.run(
            audio=args.audio,
            text=args.text,
            speaker=args.speaker,
            prompt_text=args.prompt_text,
            prompt_audio=args.prompt_audio,
            speed=args.speed,
            face=args.face,
            video=args.video,
            output=args.output,
            raw_video_output=args.raw_video_output,
            audio_output=args.audio_output,
            subtitle_output=args.subtitle_output,
            with_subtitles=args.with_subtitles,
            subtitle_correct=args.subtitle_correct,
            subtitle_language=args.subtitle_language,
            subtitle_max_chars=args.subtitle_max_chars,
            subtitle_beam_size=args.subtitle_beam_size,
            subtitle_best_of=args.subtitle_best_of,
            subtitle_vad_filter=args.subtitle_vad_filter,
            subtitle_vad_min_silence_ms=args.subtitle_vad_min_silence_ms,
            subtitle_speech_pad_ms=args.subtitle_speech_pad_ms,
            subtitle_api_key=args.subtitle_api_key,
            subtitle_api_base=args.subtitle_api_base,
            subtitle_llm_model=args.subtitle_llm_model,
            subtitle_request_timeout=args.subtitle_request_timeout,
            subtitle_font_path=args.subtitle_font_path,
            subtitle_font_name=args.subtitle_font_name,
            subtitle_font_index=args.subtitle_font_index,
            subtitle_font_size=args.subtitle_font_size,
            subtitle_font_color=args.subtitle_font_color,
            subtitle_outline_color=args.subtitle_outline_color,
            subtitle_outline=args.subtitle_outline,
            subtitle_wrap_style=args.subtitle_wrap_style,
            subtitle_bottom_margin=args.subtitle_bottom_margin,
            bgm_path=args.bgm_path,
            bgm_name=args.bgm_name,
            bgm_random=args.bgm_random,
            bgm_volume=args.bgm_volume,
            bgm_original_volume=args.bgm_original_volume,
            bgm_fade_out=args.bgm_fade_out,
            bgm_loop=args.bgm_loop,
            batch_size=args.batch_size,
            sync_offset=args.sync_offset,
            scale_h=args.scale_h,
            scale_w=args.scale_w,
            compress_inference=args.compress_inference,
            beautify_teeth=args.beautify_teeth,
            runtime=args.runtime,
        )
        print(result.final_video_path)
        print("raw_video=%s" % result.raw_video_path)
        print("audio=%s" % result.audio_path)
        print("audio_generated=%s" % ("true" if result.audio_generated else "false"))
        print("subtitle_generated=%s" % ("true" if result.subtitle_generated else "false"))
        print("subtitle_burned=%s" % ("true" if result.subtitle_burned else "false"))
        print("bgm_applied=%s" % ("true" if result.bgm_applied else "false"))
        if result.subtitle_path:
            print("subtitle=%s" % result.subtitle_path)
        if result.bgm_track:
            print("bgm_track=%s" % result.bgm_track)
        print("video_runtime=%s" % result.video_runtime)
        print("video_runtime_detail=%s" % result.video_runtime_description)
        print("video_elapsed=%.2fs" % result.video_elapsed_seconds)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, "%s\n" % exc)


if __name__ == "__main__":
    main()
