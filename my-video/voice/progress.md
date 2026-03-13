# Progress Log

## 2026-03-08

- 完成原项目语音合成链路定位。
- 确认需要抽离的最小模块包括：入口、配置、TTS 服务、CosyVoice vendor 源码、默认资源查找逻辑。
- 创建 `voice/config.py`、`voice/engine.py`、`voice/server.py`、`voice/cli.py`、`voice/requirements.txt`、`voice/README.md`。
- vendored `CosyVoice` 和 `Matcha-TTS` 已做最小兼容修补，避免训练/下载依赖阻塞推理。
- 安装并验证了独立运行所需依赖，处理了 `setuptools`、`ruamel.yaml`、`scipy`、`pyarrow`、`librosa` 等兼容问题。
- 已将 `CosyVoice2-0.5B` 复制到 `my-video/voice/models/CosyVoice2-0.5B`。
- CLI 实测成功生成 `voice/output/test.wav`。
- HTTP 服务实测通过 `GET /health` 和 `POST /tts_to_audio`，生成 `voice/output/api_test.wav`。
- 新增 `python -m voice.cli zero-shot`，实测生成 `voice/output/zero_shot_demo.wav`。
- 新增 `python -m voice.cli export-voice`，实测生成 `voice/voices/codex试听.pt`。
- 使用新导出的 `codex试听.pt` 再次调用 `python -m voice.cli synthesize`，实测生成 `voice/output/codex_voice_demo.wav`。
- HTTP 新接口实测通过：
  - `POST /inference_zero_shot` -> `voice/output/http_zero_shot_demo.wav`
  - `POST /voices/export_pt` -> `voice/voices/http试听.pt`
- 入口懒加载修复完成：`import voice` 与 `python -m voice.cli --help` 已不再触发重依赖导入。
- HTTP 非法 `speed` 参数已验证返回 400，不再抛 500。
- 修复 `my-video/.venv` 中 `torch 2.8.0` / `torchaudio 2.3.1` 版本漂移，现已对齐到 `2.3.1 / 2.3.1`。
- 修复后重新实测通过：
  - `python -m voice.cli zero-shot` -> `voice/output/selftest_after_fix.wav`
  - `python -m voice.cli export-voice` -> `voice/voices/修复后音色.pt`
  - `python -m voice.cli synthesize --speaker 修复后音色` -> `voice/output/selftest_pt_after_fix.wav`
