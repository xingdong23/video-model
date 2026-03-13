from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from digital_human.engine import DigitalHumanEngine  # type: ignore
else:
    from .engine import DigitalHumanEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone TuiliONNX digital human video generator for my-video")
    subparsers = parser.add_subparsers(dest="command", required=True)

    faces = subparsers.add_parser("faces", help="List available preset face videos")
    faces.add_argument("--tuilionnx-dir")
    faces.add_argument("--ffmpeg-bin")
    faces.add_argument("--runtime", choices=["auto", "cuda", "cpu"], default="auto")

    generate = subparsers.add_parser("generate", help="Generate a digital human video")
    generate.add_argument("--audio", required=True, help="Input audio file path")
    generate.add_argument("--face", help="Preset face video filename")
    generate.add_argument("--video", help="Explicit reference video path")
    generate.add_argument("--output", help="Output mp4 path")
    generate.add_argument("--batch-size", type=int, default=4)
    generate.add_argument("--sync-offset", type=int, default=0)
    generate.add_argument("--scale-h", type=float, default=1.6)
    generate.add_argument("--scale-w", type=float, default=3.6)
    generate.add_argument("--compress-inference", action="store_true")
    generate.add_argument("--beautify-teeth", action="store_true")
    generate.add_argument("--tuilionnx-dir")
    generate.add_argument("--ffmpeg-bin")
    generate.add_argument("--runtime", choices=["auto", "cuda", "cpu"], default="auto")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        engine = DigitalHumanEngine(
            tuilionnx_dir=args.tuilionnx_dir,
            ffmpeg_bin=args.ffmpeg_bin,
            runtime=args.runtime,
        )

        if args.command == "faces":
            for face in engine.list_face_videos():
                print(face)
            return

        result = engine.generate(
            audio=args.audio,
            face=args.face,
            video=args.video,
            output_path=args.output,
            batch_size=args.batch_size,
            sync_offset=args.sync_offset,
            scale_h=args.scale_h,
            scale_w=args.scale_w,
            compress_inference=args.compress_inference,
            beautify_teeth=args.beautify_teeth,
            runtime=args.runtime,
        )
        print(result.output_path)
        print(f"reference={result.reference_video}")
        print(f"runtime={result.runtime}")
        print(f"runtime_detail={result.runtime_description}")
        print(f"elapsed={result.elapsed_seconds:.2f}s")
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
