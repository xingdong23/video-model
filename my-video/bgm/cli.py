from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if TYPE_CHECKING:
    if __package__ in (None, ""):
        from bgm.engine import BgmEngine  # type: ignore
    else:
        from .engine import BgmEngine


def _load_bgm_engine():
    if __package__ in (None, ""):
        from bgm.engine import BgmEngine  # type: ignore

        return BgmEngine
    from .engine import BgmEngine

    return BgmEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone background music mixer for my-video"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list",
        help="List preset BGM tracks from the module library directory",
    )
    list_parser.add_argument("--library-dir", help="Optional override for the BGM library directory")

    mix = subparsers.add_parser("mix", help="Mix background music into a video")
    mix.add_argument("--video", required=True, help="Input video path")
    source_group = mix.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--bgm-path", help="Explicit BGM audio path")
    source_group.add_argument("--bgm-name", help="Preset BGM track name from the library directory")
    source_group.add_argument("--random", action="store_true", help="Randomly select a preset BGM track from the library")
    mix.add_argument("--output", help="Output video path")
    mix.add_argument("--library-dir", help="Optional override for the BGM library directory")
    mix.add_argument("--volume", type=float, default=0.35, help="BGM volume multiplier")
    mix.add_argument("--original-volume", type=float, default=1.0, help="Original video audio volume multiplier")
    mix.add_argument("--fade-out", type=float, default=0.0, help="Fade out the BGM in the last N seconds")
    mix.add_argument("--no-loop", dest="loop_bgm", action="store_false", help="Do not loop the BGM track")
    mix.add_argument("--ffmpeg-bin", help="ffmpeg executable path")
    mix.add_argument("--ffprobe-bin", help="ffprobe executable path")
    mix.set_defaults(loop_bgm=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        BgmEngine = _load_bgm_engine()
        engine = BgmEngine(
            library_dir=args.library_dir,
            ffmpeg_bin=getattr(args, "ffmpeg_bin", None),
            ffprobe_bin=getattr(args, "ffprobe_bin", None),
        )

        if args.command == "list":
            tracks = engine.list_library_tracks()
            for track in tracks:
                print("%s\t%s" % (track.name, track.path))
            print("count=%d" % len(tracks))
            return

        result = engine.mix(
            video_path=args.video,
            bgm_path=args.bgm_path,
            bgm_name=args.bgm_name,
            output_path=args.output,
            bgm_volume=args.volume,
            original_volume=args.original_volume,
            random_choice=args.random,
            loop_bgm=args.loop_bgm,
            fade_out_seconds=args.fade_out,
        )
        print(result.output_path)
        print("track=%s" % result.track_name)
        print("track_path=%s" % result.track_path)
        print("used_library_track=%s" % ("true" if result.used_library_track else "false"))
        print("had_original_audio=%s" % ("true" if result.had_original_audio else "false"))
        print("volume=%s" % result.volume)
        print("original_volume=%s" % result.original_volume)
        print("fade_out_seconds=%s" % result.fade_out_seconds)
        print("loop_enabled=%s" % ("true" if result.loop_enabled else "false"))
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, "%s\n" % exc)


if __name__ == "__main__":
    main()
