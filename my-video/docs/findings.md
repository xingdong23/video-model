# Findings

- `voice`, `digital_human`, and `subtitle` are already cleanly separated and expose reusable engine classes, so the lowest-risk integration path is a new orchestrator module rather than modifying each module's core behavior.
- The local `.venv` has a `torch`/`torchvision` mismatch: importing `torchvision` raises `AttributeError: module 'torch.library' has no attribute 'register_fake'`.
- `digital_human/pipeline.py` only used `torchvision` for bicubic tensor resizing, so the dependency can be safely removed in favor of `torch.nn.functional.interpolate`.
- A short custom reference video is the practical way to verify the integrated CPU workflow; long preset face videos make smoke tests unnecessarily expensive.
- `my-video` already uses a repeatable standalone-module shape (`config.py`, `engine.py`, `cli.py`, `README.md`, `requirements.txt`), so `bgm/` should follow that pattern instead of embedding itself inside `workflow/`.
- The legacy BGM implementation in `utils/voice_processor.py` is simple ffmpeg mixing, but it can shorten the final video when the BGM source is shorter than the video and the input video has no existing audio track.
- A portable standalone BGM module should resolve both `ffmpeg` and `ffprobe`, support direct file paths and a library directory, and treat random-choice selection as a library concern instead of a UI concern.
- Some sample videos in `workflow/output/test_assets/` have mismatched stream durations: for example `integration_short.mp4` has a 1.72s video stream but a 1.83s container duration because its audio stream is longer. The standalone `bgm/` mixer therefore preserves the actual video stream length, not the longer container-reported duration.
