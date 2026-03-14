# Task Plan

## Goal
Integrate the standalone subtitle module into the `my-video` main workflow so users can produce a final subtitled digital human video from either existing audio or fresh TTS input.

## 2026-03-08 BGM Extraction Goal
Extract the background music mixing flow into a new standalone `bgm/` module under `my-video` so it can be copied into other projects without depending on the legacy Gradio app or fixed project paths.

## Phases
- [x] Inspect existing `voice`, `digital_human`, and `subtitle` module boundaries and choose the integration shape.
- [x] Implement an integrated workflow entry point inside `my-video`.
- [x] Verify the integrated flow with local sample assets and update docs.
- [x] Inspect `my-video` module conventions and define the standalone `bgm/` module boundary.
- [x] Implement the standalone `bgm/` engine, config, CLI, and docs.
- [x] Optionally wire `workflow/` to the new `bgm/` module if the integration stays thin.
- [x] Verify local CLI behavior with a real ffmpeg mix run.

## Decisions
- Keep `voice`, `digital_human`, and `subtitle` as independent modules; add a thin orchestration layer instead of tightly coupling their engines.
- Support both existing-audio input and text-to-speech input in the integrated flow.
- Reuse the extracted subtitle engine directly, including optional correction and burn-in.
- Remove the `torchvision` dependency from `digital_human` and use `torch.nn.functional.interpolate` instead, because the current local `torch` and `torchvision` versions are not import-compatible.
- Model the new `bgm/` directory after `subtitle/`: a self-contained engine plus CLI, with no runtime dependency on the old HD_HUMAN Gradio layer.
- Improve on the legacy BGM behavior by looping library audio to the video duration, instead of risking a truncated result when the BGM file is shorter than the video.

## Risks
- A full end-to-end run can be slow on CPU, especially the digital human and subtitle burn fallback path.
- The integrated flow depends on the union of dependencies from `voice`, `digital_human`, and `subtitle`.
- The current CPU sample workflow is practical only with shorter reference videos; longer presets are still expensive to validate locally.
- ffmpeg filter behavior differs across local installs, so the standalone BGM module needs explicit ffmpeg and ffprobe resolution plus clear error messages.
