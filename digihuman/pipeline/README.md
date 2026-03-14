# DigiHuman Pipeline

`pipeline/` 是 `digihuman` 的编排层，用来把下面三段链路串成一次命令：

- `tts/`: 需要时先做 TTS
- `avatar/`: 用音频驱动数字人视频
- `subtitle/`: 可选生成字幕并烧录到最终视频
- `audio_mixer/`: 可选给最终视频叠加背景音乐

它不复制各模块内部实现，只复用它们已经抽离好的 engine。

## 用人话说

这条工作流做的事可以直接理解成：

1. 如果你给的是文字，就先把文字变成声音
2. 再拿这段声音去驱动数字人口型，生成“会说话的视频”
3. 如果你勾上字幕，就再去听这段声音，生成字幕
4. 如果你提供 BGM，就在最终视频上叠加背景音乐
5. 最后导出最终成片

所以这里不是一个神秘的大模型一步出视频，而是把几段明确的小流程串起来：

`文字或音频 -> 声音 -> 数字人视频 -> 可选字幕 -> 可选BGM -> 最终视频`

## 为什么字幕不是在数字人阶段顺便完成

因为“数字人视频”和“字幕字层”其实是两件不同的事：

- 数字人阶段解决的是：人物怎么张嘴、音画怎么同步
- 字幕阶段解决的是：屏幕上显示什么字、什么时候出现、用什么样式显示

所以即使视频已经生成出来了，字幕仍然需要单独再做一步。

## 这条链路为什么看起来复杂

原理本身不复杂，复杂主要来自工程兼容：

- 不同机器的 `ffmpeg` 能力不一样
- 不同 Python 依赖版本可能不兼容
- CPU 跑数字人和字幕回退会比较慢
- 长参考视频会明显增加验证成本

但从使用角度看，核心思路一直没变：

- 先有声音
- 再有会说话的视频
- 最后把字贴上去

## 安装

```bash
cd /path/to/digihuman
.venv/bin/pip install -r pipeline/requirements.txt
```

## 运行

已有音频，直接生成带字幕数字人视频：

```bash
cd /path/to/digihuman
.venv/bin/python -m pipeline run \
  --audio tts/output/api_test.wav \
  --face test-video.mp4 \
  --runtime cpu \
  --with-subtitles \
  --subtitle-font-path /path/to/font/AlibabaPuHuiTi-3-55-Regular.ttf \
  --output pipeline/output/api_test_final.mp4
```

已有音频，直接给最终视频叠加 BGM：

```bash
cd /path/to/digihuman
.venv/bin/python -m pipeline run \
  --audio tts/output/api_test.wav \
  --face test-video.mp4 \
  --runtime cpu \
  --bgm-path pipeline/output/test_assets/audio_short.wav \
  --bgm-volume 0.25 \
  --output pipeline/output/api_test_with_bgm.mp4
```

从文本开始，先 TTS 再生成最终视频：

```bash
cd /path/to/digihuman
.venv/bin/python -m pipeline run \
  --text "你好，这是一条一键链路测试。" \
  --speaker jok-teacher \
  --face test-video.mp4 \
  --runtime cpu \
  --with-subtitles
```

zero-shot 模式：

```bash
cd /path/to/digihuman
.venv/bin/python -m pipeline run \
  --text "你好，这是 zero-shot 一键视频测试。" \
  --prompt-text "希望你以后能够做的比我还好呦。" \
  --prompt-audio tts/asset/zero_shot_prompt.wav \
  --face test-video.mp4 \
  --runtime cpu \
  --with-subtitles
```

## 输入模式

三选一：

- `--audio`
- `--text` + `--speaker`
- `--text` + `--prompt-text` + `--prompt-audio`

## BGM 输入模式

三选一：

- `--bgm-path`
- `--bgm-name`
- `--bgm-random`

可调参数：

- `--bgm-volume`
- `--bgm-original-volume`
- `--bgm-fade-out`
- `--no-bgm-loop`
- `--bgm-library-dir`

## 结果

命令会输出：

- `final_video_path`
- `raw_video`
- `audio`
- `subtitle`（如果启用字幕）
- `bgm_track`（如果启用 BGM）
- 数字人推理 runtime 和耗时
