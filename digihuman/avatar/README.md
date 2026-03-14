# DigiHuman Avatar

独立版 TuiliONNX 数字人视频生成模块，复刻主项目当前可验证的核心流程。

当前包含两条运行链路：

- `CUDA/NVIDIA`：保留原有高性能链路
- `CPU experimental`：为 Mac 和无 CUDA 机器增加的实验回退链路

- 输入音频
- 选择预设人物视频或自定义参考视频
- 用音频驱动参考视频中的脸部区域
- 输出口型同步后的 MP4

## 技术原理

当前迁移的是主项目里正在使用的 `TuiliONNX` 链路，不是旧版 `digit_human` 训练推理链。

一句话概括：

`输入音频 + 参考人物视频 -> 提取语音特征 -> 重绘参考视频中的脸部口型 -> 输出带原音频的 mp4`

本质上这是“音频驱动的参考视频重绘”，不是从零生成整个人物视频。

### 输入和输出

输入包括三类：

- 一段已经生成好的音频
- 一段参考人物视频
- 一套提前训练好的通用数字人模型

输出是一段新视频：

- 背景、身体动作、头部姿态主要沿用参考视频
- 嘴型和局部脸部细节根据输入音频重新生成
- 最终重新封装成带音轨的 `mp4`

### 为什么必须有参考视频

这条链路是“免训练”，不是“无素材”。

- 免训练：当前人物不需要再单独训练一个专属 checkpoint
- 无素材：做不到

参考视频负责提供：

- 这个人长什么样
- 头部怎么转动
- 身体如何微动
- 原始背景和构图

所以你可以不上传自己的视频，但系统至少要能拿到一个预置参考视频，比如 `faces/` 里的素材；否则模型没有“底板”可以重绘。

### 处理流程

1. 标准化输入  
   参考视频先统一到 `25fps`，音频转成 `16k` 单声道 wav。对应 `pipeline.py:445` 和 `pipeline.py:489`。

2. 提取音频驱动特征  
   使用 `HuBERT` 把音频转成连续语音特征，再切成一小段一小段的 `rep_chunks`，用于和视频帧对齐。对应 `pipeline.py:503`。

3. 逐帧检测和对齐人脸  
   使用 `insightface` 找到人脸和 `106` 个关键点，再用双眼和鼻子三点做仿射对齐，把脸裁成固定尺寸。对应 `pipeline.py:205` 和 `pipeline.py:283`。

4. 构造模型输入  
   每一帧不会直接把整张图送进模型，而是构造成：
   - 原始对齐后的人脸
   - 被遮掉口部的人脸
   - 固定遮罩
   - 当前时刻的音频特征
   - LSTM 隐状态 `hn/cn`  
   对应 `pipeline.py:406` 和 `pipeline.py:542`。

5. 生成新的嘴型和局部脸部结果  
   ONNX 数字人模型根据音频特征推理出新的脸部区域，重点是嘴型同步，不会重新生成背景和身体。对应 `pipeline.py:551`。

6. 把生成结果贴回原视频帧  
   生成的人脸结果会按仿射矩阵缩放、变换，再融合回原始视频帧。对应 `pipeline.py:563`。

7. 合成最终视频  
   先写临时视频，再用 `ffmpeg` 把生成后的画面和输入音频重新封装成最终 `mp4`。对应 `pipeline.py:582`。

### 这条链路不做什么

为了避免理解偏差，这里明确说明它不负责的部分：

- 不从文本直接生成视频
- 不重新生成完整人物动作
- 不凭空生成新背景
- 不替你训练某个人的专属数字人模型

文本转语音发生在前一阶段；这一阶段接收的是已经存在的音频文件。

### 效果特征和限制

- 结果强依赖参考视频质量。人脸越清晰、姿态越稳定，效果通常越好。
- 如果音频比参考视频更长，系统会来回复用参考帧，而不是生成全新的长动作序列。对应 `pipeline.py:383`。
- 这条方案本质是“脸部重绘”，因此最明显的变化通常集中在口型区域。
- `CPU experimental` 和 `CUDA/NVIDIA` 的算法流程相同，区别主要在推理设备和执行速度。

## 目录

- `cli.py`: 命令行入口
- `engine.py`: 可复用接口
- `pipeline.py`: 已迁移到 `digihuman` 的 TuiliONNX 推理逻辑
- `faces/`: 可选的本地预设人物视频
- `output/`: 默认输出目录

## 依赖

- 本地 `ffmpeg`
- `torch`
- `insightface`
- `onnxruntime` 或 `onnxruntime-gpu`
- `digihuman/avatar/models/tuilionnx/checkpoints`
- `digihuman/avatar/faces`

默认目录：

- `/path/to/digihuman/avatar/models/tuilionnx/checkpoints`
- `/path/to/digihuman/avatar/faces`

如果你想改成别的模型目录，可以传 `--tuilionnx-dir` 或设置环境变量 `DIGITAL_HUMAN_TUILIONNX_DIR`。

## 运行时选择

默认是 `--runtime auto`：

- 有可用 NVIDIA CUDA 和 `CUDAExecutionProvider` 时，自动走原来的 CUDA 链路
- 否则自动退回 `CPU experimental`

也可以手动指定：

- `--runtime cuda`
- `--runtime cpu`

## 安装

```bash
cd /path/to/digihuman
.venv/bin/pip install -r avatar/requirements.txt
```

并确保系统里能找到 `ffmpeg`，或者通过 `--ffmpeg-bin` / `DIGITAL_HUMAN_FFMPEG_BIN` 指定可执行文件。

## 运行

列出可用人物视频：

```bash
cd /path/to/digihuman
.venv/bin/python -m avatar.cli faces
```

显示当前机器会使用什么链路：

```bash
cd /path/to/digihuman
.venv/bin/python -m avatar.cli generate \
  --audio voice/output/test.wav \
  --face test-video.mp4 \
  --runtime auto
```

使用预设人物视频生成：

```bash
cd /path/to/digihuman
.venv/bin/python -m avatar.cli generate \
  --audio ../post_request_test.wav \
  --face test-video.mp4 \
  --runtime auto \
  --output avatar/output/output.mp4
```

使用自定义参考视频生成：

```bash
cd /path/to/digihuman
.venv/bin/python -m avatar.cli generate \
  --audio ../post_request_test.wav \
  --runtime cpu \
  --video /absolute/path/to/reference.mp4
```

可调参数：

- `--batch-size`
- `--sync-offset`
- `--scale-h`
- `--scale-w`
- `--compress-inference`
- `--beautify-teeth`

## 独立性

- 运行时不导入主项目任何 Python 模块
- 默认只读取 `digihuman/avatar` 自己目录下的模型和人物素材
- 未迁移：Gradio UI、OSS 上传、发布链路、特效字幕包装

### 可独立复制的最小清单

如果要把这个模块单独拷走，不能只拷 `cli.py` 或几个源码文件，至少要包含：

- `avatar/cli.py`
- `avatar/engine.py`
- `avatar/config.py`
- `avatar/pipeline.py`
- `avatar/__init__.py`
- `avatar/models/tuilionnx/checkpoints/`
- `avatar/faces/`

同时还需要：

- 已安装 `requirements.txt` 中的 Python 依赖
- 系统里可用的 `ffmpeg`
- 一段输入音频

只要整包复制完整，并且模型目录一并带走，当前实现不需要再回到原项目里导入其他模块。

拷走后的推荐启动方式：

```bash
cd /path/to/parent/of/avatar
python -m avatar.cli --help
```

如果你不在 `avatar` 的父目录下执行，也可以显式指定：

```bash
PYTHONPATH=/path/to/parent/of/avatar python -m avatar.cli --help
```

## 注意

- `CPU experimental` 主要是为了在 Mac 或无 CUDA 机器上验证链路可跑通。
- 这条链路会比 NVIDIA CUDA 慢很多，长音频或长参考视频耗时会明显增加。
- 若当前环境只有 `onnxruntime` CPU 版，`--runtime auto` 会自动退回 CPU。

如果后面需要，我可以继续把“字幕/BGM/封面/发布”也按 `digihuman` 模块化方式拆出来。
