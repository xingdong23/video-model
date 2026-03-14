from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if TYPE_CHECKING:
    if __package__ in (None, ""):
        from tts.engine import VoiceEngine  # type: ignore
    else:
        from .engine import VoiceEngine


def _load_voice_engine():
    if __package__ in (None, ""):
        from tts.engine import VoiceEngine  # type: ignore

        return VoiceEngine
    from .engine import VoiceEngine

    return VoiceEngine


def _load_create_app():
    if __package__ in (None, ""):
        from tts.server import create_app  # type: ignore

        return create_app
    from .server import create_app

    return create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone CosyVoice wrapper for DigiHuman")
    subparsers = parser.add_subparsers(dest="command", required=True)

    speakers = subparsers.add_parser("speakers", help="List available speakers")
    speakers.add_argument("--model-dir")

    synthesize = subparsers.add_parser("synthesize", help="Generate WAV audio from text")
    synthesize.add_argument("--text", required=True)
    synthesize.add_argument("--speaker", required=True)
    synthesize.add_argument("--speed", type=float, default=1.0)
    synthesize.add_argument("--output")
    synthesize.add_argument("--model-dir")

    zero_shot = subparsers.add_parser("zero-shot", help="Generate WAV audio from reference audio and text")
    zero_shot.add_argument("--text", required=True)
    zero_shot.add_argument("--prompt-text", required=True)
    zero_shot.add_argument("--prompt-audio", required=True)
    zero_shot.add_argument("--speed", type=float, default=1.0)
    zero_shot.add_argument("--output")
    zero_shot.add_argument("--model-dir")

    export_voice = subparsers.add_parser("export-voice", help="Export a reusable custom voice .pt from reference audio")
    export_voice.add_argument("--voice-name", required=True)
    export_voice.add_argument("--prompt-text", required=True)
    export_voice.add_argument("--prompt-audio", required=True)
    export_voice.add_argument("--output")
    export_voice.add_argument("--model-dir")

    serve = subparsers.add_parser("serve", help="Run local HTTP service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=9880)
    serve.add_argument("--model-dir")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "serve":
        create_app = _load_create_app()
        app = create_app(model_dir=args.model_dir)
        app.run(host=args.host, port=args.port, debug=False)
        return

    VoiceEngine = _load_voice_engine()
    engine = VoiceEngine(model_dir=args.model_dir)

    if args.command == "speakers":
        for speaker in engine.list_speakers():
            print(speaker)
        return

    if args.command == "synthesize":
        output = Path(args.output).expanduser().resolve() if args.output else engine.paths.output_dir / "generated.wav"
        result = engine.synthesize_to_file(
            text=args.text,
            speaker=args.speaker,
            output_path=output,
            speed=args.speed,
        )
        print(result)
        return

    if args.command == "zero-shot":
        output = Path(args.output).expanduser().resolve() if args.output else engine.paths.output_dir / "zero_shot.wav"
        result = engine.synthesize_zero_shot_to_file(
            text=args.text,
            prompt_text=args.prompt_text,
            prompt_audio_path=args.prompt_audio,
            output_path=output,
            speed=args.speed,
        )
        print(result)
        return

    output = Path(args.output).expanduser().resolve() if args.output else engine.paths.speakers_dir / f"{args.voice_name}.pt"
    result = engine.export_custom_voice(
        voice_name=args.voice_name,
        prompt_text=args.prompt_text,
        prompt_audio_path=args.prompt_audio,
        output_path=output,
    )
    print(result)


if __name__ == "__main__":
    main()
