# Task Plan

## Goal
Extract the main project's subtitle generation area into `my-video/subtitle` so it can generate SRT subtitles, optionally correct them with an LLM, and burn them into video without importing the main project.

## Phases
- [x] Define the standalone module boundary and CLI shape.
- [x] Implement subtitle generation from audio or video with Whisper word timestamps.
- [x] Implement subtitle burn-in with ffmpeg style parameters.
- [x] Add optional OpenAI-compatible subtitle correction support.
- [x] Verify generation, correction, and burn-related CLI behavior with the local `my-video/.venv`, including the no-libass burn fallback.

## Decisions
- Keep the core pipeline as three optional steps: ASR SRT generation, LLM correction, and ffmpeg burn-in.
- Remove the old Windows-only subprocess wrapper and fixed root-level `subtitle.srt`.
- Make video input first extract mono 16k audio with ffmpeg, then run Whisper.
- Migrate subtitle correction without reintroducing the main project's `config.ini` or `openai` SDK dependency.
- Use OpenAI-compatible HTTP requests plus env/CLI configuration for correction settings.
- When ffmpeg lacks the `subtitles` filter, fall back to `moviepy + Pillow` for frame rendering and remux the original audio with ffmpeg.

## Risks
- First transcription may download a Whisper model if none is cached locally.
- Font rendering depends on the local machine having the specified font or font file.
- Burn fallback still re-encodes video frames, so it is slower than native ffmpeg subtitle burn-in on larger clips.
