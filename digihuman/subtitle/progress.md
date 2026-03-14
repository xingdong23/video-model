# Progress

## 2026-03-08

- Read the existing my-video submodule structure and matched the new subtitle module to that layout.
- Extracted the Whisper-based SRT generation logic into a standalone `SubtitleEngine`.
- Added standalone ffmpeg subtitle burn-in support with font styling options.
- Added CLI, README, requirements, and local planning files under `my-video/subtitle`.
- Verified `py_compile`, `python -m subtitle --help`, and `python -m subtitle.cli generate --help`.
- Ran a real smoke transcription on `../10s.mp4` with `--model tiny`, producing `subtitle/output/generated_smoke.srt`.
- Confirmed burn-in currently exits with a clear environment error because the local ffmpeg build lacks the `subtitles` filter.
- Added standalone OpenAI-compatible subtitle correction support as both `generate --correct` and a dedicated `correct` command.
- Removed the need for the main project's `config.ini` and `openai` SDK by using direct HTTP requests for correction.
- Refactored the CLI and engine for lazy dependency loading so `correct` and `burn` do not require Whisper runtime imports.
- Verified `python -m subtitle --help`, `python -m subtitle.cli correct --help`, and `python -m subtitle.cli generate --help`.
- Verified correction logic with a mocked HTTP response at the engine level.
- Verified the `subtitle correct` CLI end-to-end against a temporary local fake API server.
- Re-ran `subtitle generate` successfully outside the sandbox because the sandbox blocks the shared-memory/OpenMP resources required by the local Whisper runtime.
- Added a Python subtitle burn fallback for hosts whose ffmpeg lacks the `subtitles` filter.
- Fixed the fallback to support the installed `moviepy 2.1.2` API instead of only the older `moviepy.editor` import path.
- Adjusted the fallback to render video-only frames first, then mux the original audio stream back with ffmpeg so audio stays at `16000 Hz` mono instead of being resampled to `44100 Hz` stereo.
- Verified `subtitle burn` end-to-end on a reduced sample video, producing `subtitle/output/test_assets/burn_output_small.mp4` with visible subtitles and preserved source audio properties.
