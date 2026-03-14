# DigiHuman TTS

独立版 CosyVoice 语音合成模块，只依赖 `digihuman/tts` 下的代码和模型资源。

## 目录

- `vendor/`: vendored CosyVoice 源码和第三方依赖
- `speakers/`: 自定义音色 `.pt`
- `models/CosyVoice2-0.5B/`: CosyVoice 模型目录
- `output/`: 默认输出目录

## 安装

```bash
cd /path/to/digihuman
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install --no-build-isolation -r tts/requirements.txt
```

如果本机没有 `ffmpeg`，先安装系统级 `ffmpeg`。
如果没有 `ttsfrd` / `WeTextProcessing`，当前模块会自动回退到轻量文本正规化，不影响基本语音生成。
如果你把 `digihuman` 复制到别的位置，不要连旧 `.venv` 一起搬；虚拟环境脚本的 shebang 会保留旧路径，应该在新目录里重新创建 `.venv`。

## 运行

列出音色：

```bash
cd /path/to/digihuman
.venv/bin/python -m tts.cli speakers
```

生成音频：

```bash
cd /path/to/digihuman
.venv/bin/python -m tts.cli synthesize \
  --text "你好，这是一段测试语音。" \
  --speaker "jok-teacher" \
  --output tts/output/test.wav
```

zero-shot 直接试音：

```bash
cd /path/to/digihuman
.venv/bin/python -m tts.cli zero-shot \
  --text "你好，这是用参考音频直接生成的新语音。" \
  --prompt-text "希望你以后能够做的比我还好呦。" \
  --prompt-audio tts/asset/zero_shot_prompt.wav \
  --output tts/output/zero_shot_demo.wav
```

导出可复用 `.pt` 音色：

```bash
cd /path/to/digihuman
.venv/bin/python -m tts.cli export-voice \
  --voice-name "我的音色" \
  --prompt-text "希望你以后能够做的比我还好呦。" \
  --prompt-audio /绝对路径/你的参考音频.wav
```

导出后，就可以直接像普通音色一样继续用：

```bash
cd /path/to/digihuman
.venv/bin/python -m tts.cli synthesize \
  --text "你好，这是通过导出的 pt 音色再次合成的测试。" \
  --speaker "我的音色" \
  --output tts/output/my_voice_demo.wav
```

启动 HTTP 服务：

```bash
cd /path/to/digihuman
.venv/bin/python -m tts.cli serve --host 127.0.0.1 --port 9880
```

接口兼容原项目：

- `GET /health`
- `GET /speakers`
- `GET /speakers_list`
- `POST /tts_to_audio`
- `POST /inference_zero_shot`
- `POST /voices/export_pt`

`POST /tts_to_audio` 请求体示例：

```json
{
  "text": "你好，这是一段测试语音。",
  "speaker": "jok老师",
  "speed": 1.0
}
```

`POST /inference_zero_shot` 表单示例：

```bash
curl -X POST \
  -F "tts_text=你好，这是 HTTP zero-shot 接口测试。" \
  -F "prompt_text=希望你以后能够做的比我还好呦。" \
  -F "prompt_wav=@/path/to/digihuman/tts/asset/zero_shot_prompt.wav" \
  http://127.0.0.1:9880/inference_zero_shot \
  -o tts/output/http_zero_shot_demo.wav
```

`POST /voices/export_pt` 表单示例：

```bash
curl -X POST \
  -F "voice_name=我的音色" \
  -F "prompt_text=希望你以后能够做的比我还好呦。" \
  -F "prompt_wav=@/绝对路径/你的参考音频.wav" \
  http://127.0.0.1:9880/voices/export_pt
```

## 参考音频要求

- 参考音频和 `prompt_text` 必须一一对应，文本要和音频里实际说的话一致。
- 建议时长 `10-30` 秒。
- 只保留清晰人声，尽量不要有背景音乐和混响。
- 采样率建议 `16kHz` 及以上。

## 模型

默认从 `digihuman/tts/models/CosyVoice2-0.5B` 加载模型。也可以通过 `--model-dir` 指向别的本地模型目录。
