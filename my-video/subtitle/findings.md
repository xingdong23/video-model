# Findings

- The original project's subtitle generation is separate from digital human video generation; the video step only creates lip-synced frames, while the subtitle step creates timed text.
- The real subtitle timing comes from `faster-whisper` word timestamps, not from the source copywriting text.
- The original correction flow only changes subtitle text and preserves timing by rewriting each existing SRT entry with corrected text.
- The original project coupled correction to `config.ini` and the `openai` Python SDK, which is unnecessary for the standalone module.
- The standalone CLI should lazy-load Whisper and requests-related dependencies; otherwise even `--help` or correction-only flows pull in unnecessary runtime requirements.
- Burning subtitles is a second step implemented with `ffmpeg subtitles=...:force_style=...`.
- A better standalone correction path is direct OpenAI-compatible HTTP with env/CLI configuration, keeping the feature but removing the main-project dependency.
- The local `/opt/homebrew/bin/ffmpeg` build on this machine does not include the `subtitles` filter, so a standalone module needs a Python fallback instead of assuming libass support.
- `moviepy` 2.x removed `moviepy.editor` and prefers root-level imports plus `transform(...)`; the fallback burn path must support both 1.x and 2.x APIs.
- A robust fallback burn path is: render subtitle frames with `moviepy + Pillow`, then mux the original audio stream back with `ffmpeg` to avoid changing sample rate and channels.
