# Progress

## 2026-03-08

- Started planning the `my-video` root-level integration of `voice`, `digital_human`, and `subtitle`.
- Confirmed the current codebase already exposes reusable engine APIs suitable for orchestration.
- Added a new `workflow/` module that orchestrates `voice`, `digital_human`, and `subtitle` without coupling their internals.
- Added integrated CLI support for two validated input modes: existing audio and text plus speaker TTS.
- Removed the `torchvision` dependency from `digital_human/pipeline.py` and replaced its resize calls with `torch.nn.functional.interpolate` to restore compatibility with the current local `torch` install.
- Verified `python -m workflow --help` and `python -m workflow run --help`.
- Verified an end-to-end workflow run from existing audio plus subtitles using:
  `workflow/output/test_assets/audio_short.wav` +
  `workflow/output/test_assets/ref_short_small.mp4`
  -> `workflow/output/test_assets/integration_short.mp4`
- Verified a text-to-speech workflow run without subtitles using:
  `--text "你好，这是工作流测试。"` +
  `--speaker jok老师`
  -> `workflow/output/test_assets/integration_tts.mp4`
- Updated `my-video`, `workflow`, and `subtitle` README files with plain-language explanations of burn-in, ffmpeg filters, fallback behavior, and the overall video-plus-subtitle pipeline.
- Started planning a new standalone `bgm/` module under `my-video` to extract the legacy background music mixing flow without depending on the old project structure.
- Confirmed the new module should follow the same standalone pattern as `subtitle/`, with its own config, engine, CLI, output directory, and copyable README.
- Added a new `bgm/` module with standalone `config.py`, `engine.py`, `cli.py`, `README.md`, and package entrypoints.
- Added optional BGM flags to `workflow run`, so the orchestration layer can now apply background music as a final step without importing legacy project code.
- Verified `python3 -m bgm --help`, `python3 -m bgm list`, and `python3 -m workflow run --help`.
- Verified direct BGM mixing on a video that already had audio:
  `workflow/output/test_assets/integration_short.mp4` +
  `workflow/output/test_assets/audio_short.wav`
  -> `bgm/output/integration_short_bgm.mp4`
- Verified direct BGM mixing on a video with no audio using library selection:
  `workflow/output/test_assets/ref_short_small.mp4` +
  `--library-dir workflow/output/test_assets --bgm-name audio_short`
  -> `bgm/output/ref_short_small_bgm.mp4`
