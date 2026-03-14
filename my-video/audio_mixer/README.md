# DigiHuman Audio Mixer

从旧项目里抽出来的独立背景音乐模块，负责一件事：

- 把一段 BGM 音频叠加到现有视频上

它不依赖旧项目里的 Gradio、`utils/voice_processor.py`、固定 `social-auto-upload-main/videos/output.mp4` 路径，也不要求你保留原有工程目录结构。复制 `digihuman/audio_mixer` 到别的项目后，只要机器上有 `ffmpeg` 和 `ffprobe`，就能直接使用。

## 用人话说

这块做的事情其实就是：

1. 先看视频里本来有没有声音
2. 再选一首背景音乐
3. 最后用 `ffmpeg` 把原声音和背景音乐混在一起

如果原视频有声音：

- 保留原声音
- 再把 BGM 叠上去

如果原视频没声音：

- 直接把 BGM 作为视频音轨

## 和旧实现的差别

旧项目里的实现核心是对的，但有两个耦合点不适合复用：

- 它依赖旧项目的固定 `bgm/` 和 `social-auto-upload-main/videos/output.mp4`
- 当视频本身没有音轨且 BGM 比视频短时，结果可能被截短

这个独立模块做了两件事来解决这些问题：

- 把素材库目录和输出目录都收回模块内部
- 默认循环 BGM 直到视频结束，避免成片长度被 BGM 文件拖短

## 技术原理

核心原理就是：

`视频 + BGM 音频 -> ffprobe 检查音轨 -> ffmpeg 混音 -> 新视频`

具体分成三步：

1. `ffprobe` 检查输入视频
   - 读取视频时长
   - 判断是否存在原始音轨

2. 选择 BGM 来源
   - 直接传文件路径
   - 从 `library/` 里按名字选
   - 从 `library/` 里随机挑一首

3. `ffmpeg` 输出成片
   - 有原音轨时：`amix`
   - 无原音轨时：直接把 BGM 作为目标音轨
   - 默认循环 BGM 到视频时长
   - 最终输出一个新视频文件

## 目录

- `config.py`: 路径、ffmpeg、ffprobe 解析
- `engine.py`: 素材库扫描、探测视频、混音输出
- `cli.py`: 命令行入口
- `library/`: 预设 BGM 素材目录，默认可为空
- `output/`: 默认输出目录

## 安装

这个模块没有 Python 第三方依赖，但需要系统中能调用：

- `ffmpeg`
- `ffprobe`

如果它们不在环境变量里，可以传：

- `--ffmpeg-bin`
- `--ffprobe-bin`

或者设置环境变量：

- `BGM_FFMPEG_BIN`
- `BGM_FFPROBE_BIN`

## 运行

先查看素材库里的预设 BGM：

```bash
cd /path/to/digihuman
python -m audio_mixer list
```

使用显式音频文件混音：

```bash
cd /path/to/digihuman
python -m audio_mixer mix \
  --video /absolute/path/to/input.mp4 \
  --bgm-path /absolute/path/to/music.wav \
  --output /absolute/path/to/output_bgm.mp4
```

使用素材库里的预设 BGM：

```bash
cd /path/to/digihuman
python -m audio_mixer mix \
  --video /absolute/path/to/input.mp4 \
  --bgm-name calm/demo
```

随机挑一首预设 BGM：

```bash
cd /path/to/digihuman
python -m audio_mixer mix \
  --video /absolute/path/to/input.mp4 \
  --random
```

调低 BGM，保留更明显的人声：

```bash
cd /path/to/digihuman
python -m audio_mixer mix \
  --video /absolute/path/to/input.mp4 \
  --bgm-path /absolute/path/to/music.wav \
  --volume 0.25 \
  --original-volume 1.0
```

让 BGM 结尾淡出：

```bash
cd /path/to/digihuman
python -m audio_mixer mix \
  --video /absolute/path/to/input.mp4 \
  --bgm-path /absolute/path/to/music.wav \
  --fade-out 2.0
```

如果你不想循环 BGM：

```bash
cd /path/to/digihuman
python -m audio_mixer mix \
  --video /absolute/path/to/input.mp4 \
  --bgm-path /absolute/path/to/music.wav \
  --no-loop
```

`--no-loop` 时，这个模块仍然会尽量保证最终视频长度不被缩短；短掉的那一段会补静音，而不是直接把视频截断。

## 复制到其他工程

如果你要把它复制到别的项目，最少带走这些文件：

- `audio_mixer/config.py`
- `audio_mixer/engine.py`
- `audio_mixer/cli.py`
- `audio_mixer/__init__.py`
- `audio_mixer/__main__.py`

然后保证：

- 新环境能运行 `python -m audio_mixer`
- 新机器安装了 `ffmpeg` 和 `ffprobe`
- 你自己往 `library/` 放预设 BGM，或者直接传 `--bgm-path`

## 输出行为

默认输出到：

- `audio_mixer/output/<原视频名>_bgm.<后缀>`

你也可以显式传 `--output` 改成任何路径。
