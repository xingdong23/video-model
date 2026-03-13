# Voice Extraction Plan

## Goal

将原项目的“音频生成”逻辑独立抽离到 `my-video/voice`，做到代码不再 import 原项目业务模块，并提供可运行入口。

## Phases

- [x] 梳理原始 UI -> voice_processor -> CosyVoice API -> 模型 推理链路
- [x] 设计 `my-video/voice` 的目录结构、运行入口、配置方式
- [x] 复制并适配必要源码
- [x] 在当前机器完成一次最小可运行验证
- [x] 补充 README 和依赖说明
- [x] 增加 zero-shot 直接合成能力
- [x] 增加参考音频导出 `.pt` 音色能力
- [x] 验证新 `.pt` 能通过现有 `synthesize` 链路复用
- [x] 验证新增 HTTP 接口可用
- [x] 修复 CLI / package 导入期的重依赖问题
- [x] 修复 HTTP 非法 `speed` 返回 500 的问题
- [x] 修复 `my-video/.venv` 中 `torch/torchaudio` 版本漂移导致的 native 加载失败

## Constraints

- 不修改原项目现有代码
- 新代码放在 `my-video/voice`
- 尽量保持和原始实现行为一致
- 如模型文件过大，不做 12G 级别无意义复制，改为显式配置

## Open Questions

- 已将 `CosyVoice2-0.5B` 物理复制到 `my-video/voice/models`
- 当前 macOS 环境已跑通 CLI 和 HTTP 服务

## Resolved Issues

- `openai-whisper` 依赖老版 `pkg_resources`，需将 `setuptools` 降到 `<81`
- `hyperpyyaml` 需搭配 `ruamel.yaml<0.18`
- `ttsfrd` / `WeTextProcessing` 不可用时，已回退到轻量文本正规化
- `Matcha-TTS` 的训练态依赖通过 vendored patch 避免影响推理启动
- `inference_zero_shot()` 原实现会把中间 `model_input` 落到当前目录，独立模块改为在 `engine` 内部直接复用 `frontend_zero_shot + model.tts`，避免生成多余 `output.pt`
