# DigiHuman HTTP API 文档

Base URL: `http://<host>:<port>`

所有业务接口前缀 `/api/v1`，需要在请求头带 `X-API-Key`（如果服务端配置了 `api_key`）。

---

## 通用约定

### 响应格式

所有接口返回统一 JSON 信封：

```json
{
  "success": true,
  "data": { ... },
  "error": null
}
```

失败时：

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "ENGINE_ERROR",
    "message": "..."
  }
}
```

### 请求追踪

- 请求头可携带 `X-Request-ID`，服务端原样返回
- 若未携带，服务端自动生成并在响应头返回
- 异步任务中 `request_id` 字段记录此值

### 异步任务模式

耗时操作（语音合成、数字人生成、字幕识别等）采用 **异步任务** 模式：

1. 提交请求 → 立即返回 `202` + `task_id`
2. 通过以下三种方式之一获取结果

#### 方式一：Webhook 回调（推荐，server-to-server）

提交任务时带 `callback_url`，任务完成/失败后 DigiHuman 会 POST 到该 URL：

```
Web项目                            DigiHuman (GPU)
  │                                    │
  │  POST /avatar/generate              │
  │  { ..., callback_url: "http://     │
  │    localhost:3000/webhook/task" }   │
  │───────────────────────────────────→ │
  │       202 { task_id }              │
  │←─────────────────────────────────── │
  │                                    │
  │       (GPU 推理 3~10 分钟...)       │
  │                                    │
  │       POST callback_url            │
  │  { task_id, status, result, ... }  │
  │←─────────────────────────────────── │
```

- 回调重试策略：最多 3 次，间隔 1s / 3s / 10s
- 回调在独立线程执行，不阻塞 worker
- 回调失败不影响任务状态，Web 项目可降级到轮询

#### 方式二：SSE 实时推送（适合前端进度条）

```
GET /api/v1/tasks/{task_id}/stream
```

返回 `text/event-stream`，每秒推送一次状态变化：

```
event: progress
data: {"task_id":"...","status":"processing","progress":55,"message":"推理中 147/267"}

event: completed
data: {"task_id":"...","status":"completed","result":{...}}
```

连接在任务到达终态后自动关闭。

#### 方式三：轮询（最简单）

```
GET /api/v1/tasks/{task_id}
```

建议间隔 2-5 秒，检查 `status` 是否为终态（completed / failed / cancelled）。

---

## 任务管理

### GET /api/v1/tasks/{task_id}

查询任务状态和进度。

**响应示例**：

```json
{
  "success": true,
  "data": {
    "task_id": "a1b2c3d4e5f67890",
    "task_type": "avatar",
    "status": "processing",
    "step": "inference",
    "progress": 55,
    "message": "推理中 147/267",
    "result": null,
    "error": null,
    "created_at": 1710352800.0,
    "updated_at": 1710352850.0,
    "request_id": "req-xxx"
  }
}
```

**status 取值**：

| 状态 | 说明 |
|------|------|
| `queued` | 排队中，等待 worker |
| `processing` | 执行中，可查看 `progress` / `step` / `message` |
| `completed` | 完成，`result` 包含结果 |
| `failed` | 失败，`error` 包含错误信息 |
| `cancelled` | 已取消 |

### DELETE /api/v1/tasks/{task_id}

取消排队中的任务。仅 `queued` 状态可取消。

**成功响应** (200)：

```json
{
  "success": true,
  "data": { "task_id": "...", "status": "cancelled" }
}
```

**失败响应** (409)：任务已在执行中或已完成，无法取消。

### GET /api/v1/tasks/{task_id}/stream

SSE 实时推送任务进度。连接建立后每秒检查状态变化并推送事件，到达终态自动关闭。

**事件类型**：

| event | 说明 |
|-------|------|
| `progress` | 进度更新，`data` 中包含 progress/step/message |
| `completed` | 任务完成，`data.result` 包含结果 |
| `failed` | 任务失败，`data.error` 包含错误信息 |
| `cancelled` | 任务已取消 |
| `error` | 任务不存在 |

**前端使用示例**：

```javascript
const es = new EventSource("/api/v1/tasks/abc123/stream");

es.addEventListener("progress", (e) => {
  const data = JSON.parse(e.data);
  console.log(`[${data.progress}%] ${data.message}`);
});

es.addEventListener("completed", (e) => {
  const data = JSON.parse(e.data);
  console.log("完成:", data.result);
  es.close();
});

es.addEventListener("failed", (e) => {
  const data = JSON.parse(e.data);
  console.error("失败:", data.error);
  es.close();
});
```

### Webhook 回调

所有异步接口的请求体均支持可选字段 `callback_url`。任务到达终态后，DigiHuman 会向该 URL 发送 POST 请求：

**回调请求**：

```http
POST {callback_url}
Content-Type: application/json

{
  "task_id": "a1b2c3d4e5f67890",
  "task_type": "avatar",
  "status": "completed",
  "step": "completed",
  "progress": 100,
  "message": "完成",
  "result": {
    "file_id": "xxx",
    "download_url": "/api/v1/files/xxx",
    "elapsed_seconds": 180.5
  },
  "error": null,
  "created_at": 1710352800.0,
  "updated_at": 1710352980.0,
  "request_id": "order-12345"
}
```

**重试策略**：

- 最多 3 次，间隔 1s → 3s → 10s
- HTTP 状态码 < 400 视为成功
- 回调在独立线程执行，不阻塞 GPU/CPU worker
- 所有重试失败后仅记录日志，不影响任务状态

**Web 项目回调接口示例** (接收端)：

```python
# Web 项目（如 Flask / Django / FastAPI）
@app.post("/webhook/digihuman")
async def handle_callback(payload: dict):
    task_id = payload["task_id"]
    status = payload["status"]
    request_id = payload["request_id"]  # 对应业务订单号

    if status == "completed":
        file_id = payload["result"]["file_id"]
        download_url = f"http://gpu-server:8000{payload['result']['download_url']}"
        # 更新数据库、通知用户...
    elif status == "failed":
        error = payload["error"]
        # 记录失败、退还积分...

    return {"ok": True}
```

---

## Digital Human (数字人)

### GET /api/v1/digital-human/faces

列出可用的人脸预设视频。**同步**。

**响应**：

```json
{
  "success": true,
  "data": ["face1.mp4", "face2.mp4"]
}
```

### POST /api/v1/digital-human/generate

生成数字人视频。**异步 (GPU)**，返回 `202`。

**请求体**：

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `audio_file_id` | string | Y | - | 音频文件 ID |
| `face` | string | N | null | 人脸预设名称 |
| `video_file_id` | string | N | null | 自定义参考视频文件 ID |
| `batch_size` | int | N | 4 | 推理批大小 |
| `sync_offset` | int | N | 0 | 音画同步偏移 |
| `scale_h` | float | N | 1.6 | 融合区域纵向比例 |
| `scale_w` | float | N | 3.6 | 融合区域横向比例 |
| `compress_inference` | bool | N | false | 是否压缩推理 |
| `beautify_teeth` | bool | N | false | 是否美化牙齿 |
| `runtime` | string | N | null | 运行时 (auto/cuda/cpu) |
| `callback_url` | string | N | null | 任务完成后回调的 URL |

**响应** (202)：

```json
{
  "success": true,
  "data": { "task_id": "a1b2c3d4e5f67890" }
}
```

**任务完成后 result**：

```json
{
  "file_id": "xxx",
  "download_url": "/api/v1/files/xxx",
  "elapsed_seconds": 180.5,
  "runtime": "cuda",
  "runtime_description": "NVIDIA GeForce RTX 4090"
}
```

**进度步骤**：

| step | progress | 说明 |
|------|----------|------|
| `preprocessing` | 5 | 视频预处理 |
| `audio_features` | 20 | 提取音频特征 (HuBERT) |
| `face_detection` | 35 | 人脸检测 |
| `inference` | 40-90 | ONNX 模型推理 |
| `compositing` | 92 | FFmpeg 合成 |
| `completed` | 100 | 完成 |

---

## Voice (语音)

### GET /api/v1/voice/speakers

列出可用的说话人。**同步**。

**响应**：

```json
{
  "success": true,
  "data": [
    { "name": "中文女", "vid": 1 },
    { "name": "中文男", "vid": 2 }
  ]
}
```

### POST /api/v1/voice/synthesize

文本转语音。**异步 (GPU)**，返回 `202`。

**请求体**：

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `text` | string | Y | - | 要合成的文本 |
| `speaker` | string | Y | - | 说话人名称 |
| `speed` | float | N | 1.0 | 语速倍率 |
| `callback_url` | string | N | null | 任务完成后回调的 URL |

**任务完成后 result**：

```json
{
  "file_id": "xxx",
  "download_url": "/api/v1/files/xxx"
}
```

### POST /api/v1/voice/synthesize/stream

流式语音合成，直接返回 WAV 音频流。**同步**。

请求体同 `/synthesize`。响应 `Content-Type: audio/wav`。

### POST /api/v1/voice/zero-shot

零样本语音克隆。**异步 (GPU)**，返回 `202`。

**请求** (`multipart/form-data`)：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | string | Y | 要合成的文本 |
| `prompt_text` | string | Y | 参考音频对应文本 |
| `speed` | float | N | 语速倍率，默认 1.0 |
| `prompt_wav` | file | Y | 参考音频文件 |
| `callback_url` | string | N | 任务完成后回调的 URL |

**任务完成后 result**：

```json
{
  "file_id": "xxx",
  "download_url": "/api/v1/files/xxx"
}
```

### POST /api/v1/voice/voices/export

导出自定义音色。**同步**。

**请求** (`multipart/form-data`)：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `voice_name` | string | Y | 音色名称 |
| `prompt_text` | string | Y | 参考文本 |
| `prompt_wav` | file | Y | 参考音频 |

---

## 字幕

### POST /api/v1/subtitle/generate-srt

语音识别生成 SRT 字幕。**异步 (GPU)**，返回 `202`。

**请求体**：

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `audio_file_id` | string | Y | - | 音频/视频文件 ID |
| `language` | string | N | "zh" | 识别语言 |
| `max_chars` | int | N | 20 | 每行最大字数 |
| `beam_size` | int | N | 10 | beam search 宽度 |
| `best_of` | int | N | 5 | 候选数 |
| `vad_filter` | bool | N | true | VAD 过滤 |
| `vad_min_silence_ms` | int | N | 1000 | VAD 最小静音时长 (ms) |
| `speech_pad_ms` | int | N | 300 | 语音填充 (ms) |
| `apply_correction` | bool | N | false | 是否 LLM 纠错 |
| `correction_api_key` | string | N | null | LLM API Key |
| `correction_api_base` | string | N | null | LLM API Base URL |
| `correction_model_name` | string | N | null | LLM 模型名 |
| `correction_timeout` | int | N | null | 超时秒数 |
| `callback_url` | string | N | null | 任务完成后回调的 URL |

**任务完成后 result**：

```json
{
  "file_id": "xxx",
  "download_url": "/api/v1/files/xxx",
  "entries_count": 42,
  "detected_language": "zh",
  "correction_applied": false
}
```

### POST /api/v1/subtitle/correct

LLM 纠错已有字幕。**同步**（调用外部 LLM API）。

**请求体**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `srt_file_id` | string | Y | SRT 文件 ID |
| `api_key` | string | N | LLM API Key |
| `api_base` | string | N | LLM API Base |
| `model_name` | string | N | 模型名称 |
| `request_timeout` | int | N | 超时秒数 |

### POST /api/v1/subtitle/burn

将字幕烧录到视频。**异步 (CPU)**，返回 `202`。

**请求体**：

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `video_file_id` | string | Y | - | 视频文件 ID |
| `srt_file_id` | string | Y | - | SRT 文件 ID |
| `font_path` | string | N | null | 自定义字体路径 |
| `font_name` | string | N | null | 字体名称 |
| `font_index` | int | N | 0 | 字体索引 |
| `font_size` | int | N | 24 | 字号 |
| `font_color` | string | N | "#FFFFFF" | 字体颜色 |
| `outline_color` | string | N | "#000000" | 描边颜色 |
| `outline` | int | N | 1 | 描边宽度 |
| `wrap_style` | int | N | 2 | 换行模式 |
| `bottom_margin` | int | N | 30 | 底部边距 |
| `callback_url` | string | N | null | 任务完成后回调的 URL |

**任务完成后 result**：

```json
{
  "file_id": "xxx",
  "download_url": "/api/v1/files/xxx"
}
```

---

## BGM (背景音乐)

### GET /api/v1/bgm/tracks

列出 BGM 素材库。**同步**。

**响应**：

```json
{
  "success": true,
  "data": [
    { "name": "轻快钢琴.mp3", "relative_path": "piano/轻快钢琴.mp3" }
  ]
}
```

### POST /api/v1/bgm/mix

混合背景音乐到视频。**异步 (CPU)**，返回 `202`。

**请求体**：

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `video_file_id` | string | Y | - | 视频文件 ID |
| `bgm_path` | string | N | null | BGM 文件路径 |
| `bgm_name` | string | N | null | BGM 素材名 |
| `random_choice` | bool | N | false | 随机选择 |
| `bgm_volume` | float | N | 0.35 | BGM 音量 |
| `original_volume` | float | N | 1.0 | 原始音量 |
| `loop_bgm` | bool | N | true | BGM 循环 |
| `fade_out_seconds` | float | N | 0.0 | 淡出时长 |
| `callback_url` | string | N | null | 任务完成后回调的 URL |

**任务完成后 result**：

```json
{
  "file_id": "xxx",
  "download_url": "/api/v1/files/xxx",
  "track_name": "轻快钢琴.mp3",
  "had_original_audio": true
}
```

---

## Rewrite (文案改写)

### POST /api/v1/rewrite/auto

自动改写文案。**同步**。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | string | Y | 原始文本 |

**响应**：`{ "rewritten_text": "..." }`

### POST /api/v1/rewrite/instruction

按指令改写。**同步**。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | string | Y | 原始文本 |
| `instruction` | string | Y | 改写指令 |

---

## Douyin (视频采集)

### POST /api/v1/douyin/transcribe

下载抖音视频并转录文案。**异步 (CPU)**，返回 `202`。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `share_link` | string | Y | 抖音分享链接 |
| `callback_url` | string | N | 任务完成后回调的 URL |

**任务完成后 result**：

```json
{
  "transcript": "转录文本...",
  "files": [
    { "file_id": "xxx", "filename": "video.mp4", "download_url": "/api/v1/files/xxx" }
  ]
}
```

### POST /api/v1/douyin/download

仅下载抖音视频（不转录）。**异步 (CPU)**，返回 `202`。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `share_link` | string | Y | 抖音分享链接 |
| `callback_url` | string | N | 任务完成后回调的 URL |

**任务完成后 result**：

```json
{
  "file_id": "xxx",
  "download_url": "/api/v1/files/xxx"
}
```

---

## Pipeline (一键生成)

### POST /api/v1/pipeline/run

串联多个步骤的一键工作流。**同步**（内部阻塞直到全部完成）。

**请求体**（嵌套配置组）：

```json
{
  "audio": {
    "file_id": null,
    "text": "你好，欢迎收看今天的节目",
    "speaker": "中文女",
    "prompt_text": null,
    "prompt_audio_file_id": null,
    "speed": 1.0
  },
  "avatar": {
    "face": "anchor1.mp4",
    "video_file_id": null,
    "batch_size": 4,
    "sync_offset": 0,
    "scale_h": 1.6,
    "scale_w": 3.6,
    "compress_inference": false,
    "beautify_teeth": false,
    "runtime": null
  },
  "subtitle": {
    "language": "zh",
    "max_chars": 20,
    "beam_size": 10,
    "best_of": 5,
    "vad_filter": true,
    "vad_min_silence_ms": 1000,
    "speech_pad_ms": 300,
    "correct": false,
    "api_key": null,
    "api_base": null,
    "llm_model": null,
    "request_timeout": null
  },
  "subtitle_style": {
    "font_path": null,
    "font_name": null,
    "font_index": 0,
    "font_size": 24,
    "font_color": "#FFFFFF",
    "outline_color": "#000000",
    "outline": 1,
    "wrap_style": 2,
    "bottom_margin": 30
  },
  "bgm": {
    "path": null,
    "name": "轻快钢琴.mp3",
    "random": false,
    "volume": 0.35,
    "original_volume": 1.0,
    "fade_out": 0.0,
    "loop": true
  }
}
```

**字段说明**：

- `audio` (必填) — 音频来源，提供 `file_id` 直接使用现有音频，或提供 `text` + `speaker` 走 TTS
- `avatar` (可选) — 数字人参数，全部有默认值
- `subtitle` (可选) — 传 `null` 则不生成字幕，传对象则生成
- `subtitle_style` (可选) — 字幕样式，仅在 `subtitle` 非空时生效
- `bgm` (可选) — 传 `null` 则不加 BGM

**响应**：

```json
{
  "success": true,
  "data": {
    "audio_generated": true,
    "subtitle_generated": true,
    "subtitle_burned": true,
    "bgm_applied": true,
    "video_runtime": "cuda",
    "video_runtime_description": "NVIDIA GeForce RTX 4090",
    "video_elapsed_seconds": 195.3,
    "final_video": { "file_id": "xxx", "download_url": "/api/v1/files/xxx" },
    "audio": { "file_id": "xxx", "download_url": "/api/v1/files/xxx" },
    "raw_video": { "file_id": "xxx", "download_url": "/api/v1/files/xxx" },
    "subtitle": { "file_id": "xxx", "download_url": "/api/v1/files/xxx" }
  }
}
```

---

## 文件管理

### POST /api/v1/files/upload

上传文件。**同步**。

**请求** (`multipart/form-data`)：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | file | Y | 文件 |
| `category` | string | N | 分类，默认 "upload" |

**响应**：

```json
{
  "success": true,
  "data": {
    "file_id": "xxx",
    "filename": "audio.wav",
    "download_url": "/api/v1/files/xxx"
  }
}
```

### GET /api/v1/files/{file_id}

下载文件。返回文件流。

### GET /api/v1/files/{file_id}/info

查询文件元信息。

```json
{
  "success": true,
  "data": {
    "file_id": "xxx",
    "original_name": "audio.wav",
    "category": "tts",
    "size_bytes": 1234567,
    "created_at": 1710352800.0,
    "download_url": "/api/v1/files/xxx"
  }
}
```

### DELETE /api/v1/files/{file_id}

删除文件。

---

## 健康检查

### GET /health

```json
{ "status": "ok" }
```

### GET /health/gpu

```json
{
  "cuda_available": true,
  "device_count": 1,
  "devices": [
    { "index": 0, "name": "NVIDIA GeForce RTX 4090", "total_memory_gb": 24.0 }
  ]
}
```

---

## 队列分类

| 队列 | 并发 | 包含接口 |
|------|------|----------|
| GPU | 串行 (1 worker) | digital-human/generate, voice/synthesize, voice/zero-shot, subtitle/generate-srt |
| CPU | 2 并发 workers | subtitle/burn, bgm/mix, douyin/transcribe, douyin/download |
| 同步 (无队列) | 不限 | voice/speakers, voice/synthesize/stream, voice/voices/export, digital-human/faces, subtitle/correct, bgm/tracks, rewrite/*, files/*, pipeline/run |

---

## Web 项目对接示例

### 方式一：Webhook 回调（推荐）

**发送端**（Web 项目提交任务）：

```python
import httpx

GPU_API = "http://gpu-server:8000"
HEADERS = {"X-API-Key": "your-key", "X-Request-ID": "order-12345"}

# 1. 上传音频
with open("audio.wav", "rb") as f:
    r = httpx.post(f"{GPU_API}/api/v1/files/upload", files={"file": f}, headers=HEADERS)
audio_file_id = r.json()["data"]["file_id"]

# 2. 提交数字人生成，带 callback_url
r = httpx.post(f"{GPU_API}/api/v1/digital-human/generate", json={
    "audio_file_id": audio_file_id,
    "face": "anchor1.mp4",
    "callback_url": "http://localhost:3000/webhook/digihuman",
}, headers=HEADERS)
task_id = r.json()["data"]["task_id"]
# 记录 task_id 到数据库，关联业务订单
```

**接收端**（Web 项目回调接口）：

```python
@app.post("/webhook/digihuman")
async def handle_callback(payload: dict):
    task_id = payload["task_id"]
    status = payload["status"]
    request_id = payload["request_id"]  # 对应业务订单号

    if status == "completed":
        file_id = payload["result"]["file_id"]
        download_url = f"http://gpu-server:8000{payload['result']['download_url']}"
        # 更新数据库、通知前端、扣减积分...
    elif status == "failed":
        error = payload["error"]
        # 记录失败、退还积分...

    return {"ok": True}
```

### 方式二：轮询（简单场景 / 回调降级）

```python
import httpx, time

GPU_API = "http://gpu-server:8000"
HEADERS = {"X-API-Key": "your-key", "X-Request-ID": "order-12345"}

# 1. 上传 + 提交（同上，省略 callback_url）
task_id = "..."

# 2. 轮询进度
while True:
    r = httpx.get(f"{GPU_API}/api/v1/tasks/{task_id}", headers=HEADERS)
    task = r.json()["data"]
    print(f"[{task['progress']}%] {task['message']}")
    if task["status"] in ("completed", "failed", "cancelled"):
        break
    time.sleep(2)

# 3. 获取结果
if task["status"] == "completed":
    file_id = task["result"]["file_id"]
    video = httpx.get(f"{GPU_API}/api/v1/files/{file_id}", headers=HEADERS)
    with open("output.mp4", "wb") as f:
        f.write(video.content)
```

### 方式三：SSE 实时进度（前端进度条）

```javascript
// 前端直接连 GPU 服务，或通过 Web 项目代理
const taskId = "a1b2c3d4e5f67890";
const es = new EventSource(`http://gpu-server:8000/api/v1/tasks/${taskId}/stream`);

es.addEventListener("progress", (e) => {
  const { progress, message } = JSON.parse(e.data);
  updateProgressBar(progress, message);
});

es.addEventListener("completed", (e) => {
  const { result } = JSON.parse(e.data);
  showVideo(result.download_url);
  es.close();
});

es.addEventListener("failed", (e) => {
  const { error } = JSON.parse(e.data);
  showError(error);
  es.close();
});
```

---

## 错误码

| code | 说明 |
|------|------|
| `VALIDATION_ERROR` | 参数校验失败 (400) |
| `AUTH_ERROR` | API Key 无效 (401) |
| `FILE_NOT_FOUND` | 文件/任务不存在 (404) |
| `ENGINE_ERROR` | 引擎执行失败 (500) |
| `ENGINE_BUSY` | 引擎繁忙 (503) |
| `FILE_UPLOAD_ERROR` | 文件上传失败 (500) |
| `INTERNAL_ERROR` | 未知内部错误 (500) |
