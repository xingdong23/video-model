# DigiHuman Subtitle

从主项目抽离出来的独立字幕模块，负责三件事：

- 用 `faster-whisper` 把音频或视频转成 `SRT`
- 用 OpenAI 兼容接口可选纠正字幕中的错别字和多音字
- 优先用 `ffmpeg` 烧录字幕，缺少 `subtitles` 过滤器时自动回退到 `moviepy + Pillow`

它不再依赖主项目里的 Gradio、固定 `subtitle.srt` 路径、Windows `python.exe` 子进程，复制 `digihuman/subtitle` 到别的工程后也可以单独运行。

## 用人话说

这块功能做的事情其实很简单：

1. 先听音频里说了什么
2. 再记下每句话大概是从几秒到几秒
3. 最后把这些字按时间画到视频底部

所以它不是“直接把文案贴上去”，而是“先听声音，再生成带时间轴的字幕”。

## 常见词解释

`烧录`

- 就是把字幕直接画进视频画面
- 导出后字幕和视频是一体的，不能像播放器外挂字幕那样随时开关

`过滤器`

- 这里是 `ffmpeg` 处理视频时的一个加工步骤
- 你可以把它理解成“视频加工功能块”
- 例如缩放、裁剪、加水印、加字幕，都属于过滤器

`字幕过滤器`

- 就是专门负责“把字幕画到画面上”的那一步

`回退链路`

- 就是备用方案
- 正常先走最快、最标准的方案
- 如果当前机器不支持，就自动换第二种做法继续完成任务

在这个模块里具体就是：

- 优先：`ffmpeg` 自己加字幕
- 备用：用 `moviepy + Pillow` 逐帧把字画上去，再导出视频

## 技术原理

如果只看核心原理，这块其实就是：

`音频 -> 语音识别 -> 带时间轴的字幕 -> 画到视频上`

原项目里这一块实际分成两段：

1. `ASR 转写`
   - 读取音频
   - Whisper 生成词级时间戳
   - 按标点或最大字数切成字幕行
   - 写出 `SRT`

2. `字幕纠错`
   - 读取 `SRT`
   - 保留原时间轴和条目数量
   - 只修正每条字幕文本
   - 回写原 `SRT`，同时创建 `.backup`

3. `字幕烧录`
   - 读取视频和 `SRT`
   - 优先用 `ffmpeg subtitles=...:force_style=...` 渲染字幕
   - 如果本机 `ffmpeg` 没编译 `subtitles/libass`，自动回退到 Python 渲染
   - 输出带字幕的新视频

这个独立模块保留的就是这三段核心逻辑。

## 字幕加到视频上的原理

这里再单独讲一次“字幕到底怎么加上去”的原理：

1. 先拿到一份 `SRT`
   - 里面有字幕文字
   - 也有每条字幕出现和消失的时间

2. 再按时间去处理视频帧
   - 到了某一秒，该出现哪句字幕，就取哪句
   - 把这句字画到当前视频帧底部

3. 所有帧处理完之后，重新导出视频
   - 同时把原来的音频重新封装回去

所以你可以把“烧录字幕”理解成：

- 给视频的每一帧都做一次“是否需要画字”的判断
- 需要就画，不需要就原样保留

## 目录

- `cli.py`: 命令行入口
- `config.py`: 路径、Whisper 运行时、ffmpeg 定位
- `engine.py`: 字幕生成、纠错与烧录实现
- `output/`: 默认输出目录

## 安装

```bash
cd /path/to/digihuman
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r subtitle/requirements.txt
```

需要系统可用的 `ffmpeg`。如果 `ffmpeg` 不在环境变量里，可以传 `--ffmpeg-bin`，或者设置环境变量 `SUBTITLE_FFMPEG_BIN`。

如果 `ffmpeg` 本身带有 `subtitles` 过滤器，会优先使用原生烧录。没有这个过滤器时，模块会自动回退到 `moviepy + Pillow` 绘制字幕，再用 `ffmpeg` 把原音轨无损 mux 回结果视频。

如果要启用字幕纠错，还需要提供下面三项之一：

- `--api-key`
- `--api-base`
- `--llm-model`

或者设置环境变量：

- `SUBTITLE_LLM_API_KEY`
- `SUBTITLE_LLM_API_BASE`
- `SUBTITLE_LLM_MODEL`
- `SUBTITLE_LLM_TIMEOUT`

## 运行

从音频生成字幕：

```bash
cd /path/to/digihuman
.venv/bin/python -m subtitle generate \
  --input /absolute/path/to/test.wav \
  --output subtitle/output/test.srt
```

从视频直接生成字幕：

```bash
cd /path/to/digihuman
.venv/bin/python -m subtitle generate \
  --input /absolute/path/to/test.mp4 \
  --output subtitle/output/test.srt
```

生成后直接纠错：

```bash
cd /path/to/digihuman
.venv/bin/python -m subtitle generate \
  --input /absolute/path/to/test.mp4 \
  --output subtitle/output/test.srt \
  --correct \
  --api-key "$SUBTITLE_LLM_API_KEY" \
  --api-base "$SUBTITLE_LLM_API_BASE" \
  --llm-model "$SUBTITLE_LLM_MODEL"
```

对已有字幕单独纠错：

```bash
cd /path/to/digihuman
.venv/bin/python -m subtitle correct \
  --subtitle subtitle/output/test.srt \
  --api-key "$SUBTITLE_LLM_API_KEY" \
  --api-base "$SUBTITLE_LLM_API_BASE" \
  --llm-model "$SUBTITLE_LLM_MODEL"
```

烧录字幕：

```bash
cd /path/to/digihuman
.venv/bin/python -m subtitle burn \
  --video /absolute/path/to/test.mp4 \
  --subtitle subtitle/output/test.srt \
  --output subtitle/output/test_subtitled.mp4
```

指定字体文件和样式：

```bash
cd /path/to/digihuman
.venv/bin/python -m subtitle burn \
  --video /absolute/path/to/test.mp4 \
  --subtitle subtitle/output/test.srt \
  --font-path /absolute/path/to/font.ttf \
  --font-size 28 \
  --font-color "#FFFFFF" \
  --outline-color "#000000" \
  --bottom-margin 60
```

## 可调参数

`generate` 支持：

- `--language zh|en|auto`
- `--model tiny|base|small|medium|large-v3|/local/model/path`
- `--device auto|cpu|cuda`
- `--compute-type auto|int8|float16|float32`
- `--max-chars`
- `--beam-size`
- `--best-of`
- `--no-vad-filter`
- `--correct`
- `--api-key`
- `--api-base`
- `--llm-model`

`correct` 支持：

- `--subtitle`
- `--api-key`
- `--api-base`
- `--llm-model`
- `--request-timeout`

`burn` 支持：

- `--font-path`
- `--font-name`
- `--font-index`
- `--font-size`
- `--font-color`
- `--outline-color`
- `--outline`
- `--wrap-style`
- `--bottom-margin`

## 与原项目的差异

- 保留了字幕生成和烧录的核心流程
- 补回了原项目里可选的字幕 LLM 纠错能力
- 去掉了主项目里对 `cosyvoice/generate_srt.py` 的 Windows 子进程封装
- 去掉了固定写入项目根目录 `subtitle.srt` 的行为
- 去掉了对主项目 `config.ini` 和 `openai` SDK 的依赖，改成 OpenAI 兼容 HTTP 接口
- 增加了无 `ffmpeg subtitles/libass` 环境下的 Python 烧录回退，并保留原视频音轨

## 可独立复制的最小清单

如果你要把这个模块单独拷走，至少要带上：

- `subtitle/__init__.py`
- `subtitle/__main__.py`
- `subtitle/cli.py`
- `subtitle/config.py`
- `subtitle/engine.py`
- `subtitle/requirements.txt`

同时还需要：

- 一个可用的 Python 环境
- 安装 `requirements.txt`
- 系统级 `ffmpeg`
- 待转写的音频或视频文件

## 本地验证

2026-03-08 已在当前机器完成以下验证：

- `py_compile`
- `python -m subtitle --help`
- `python -m subtitle generate --help`
- `python -m subtitle correct --help`
- `python -m subtitle generate --input ../10s.mp4 --model tiny --device cpu`
- `python -m subtitle correct` 对接本地 fake API
- `python -m subtitle burn` 在本机缺少 `ffmpeg subtitles` 过滤器时自动走 Python 回退，并成功生成
  [burn_output_small.mp4](output/test_assets/burn_output_small.mp4)
