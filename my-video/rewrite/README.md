# my-video Rewrite

独立版文案仿写模块，复用原项目的 DeepSeek prompt 和模式分支，适合单独调用或后续接入 `my-video` 其他流程。

## 能力

- `auto`: 复用原项目“自动仿写”长 prompt
- `prompt`: 复用原项目“根据指令仿写”链路
- `execute`: 兼容原项目模式名，如 `AI自动仿写`、`根据指令仿写`

## 配置来源

模块按以下优先级读取配置：

1. CLI 显式传参
2. 环境变量
   - `REWRITE_API_KEY`
   - `DASHSCOPE_API_KEY`
   - `DEEPSEEK_API_KEY`
   - `OPENAI_API_KEY`
   - `REWRITE_API_BASE`
   - `REWRITE_MODEL`
3. `rewrite/config.ini`
4. 仓库根目录 `config.ini`

如果沿用当前仓库配置，默认会复用根目录里的：

- `openai_api_base = https://dashscope.aliyuncs.com/compatible-mode/v1`
- `default_model = deepseek-v3`

## 安装

```bash
cd /Users/xingdong/workspace/HD_HUMAN/my-video
.venv/bin/pip install -r rewrite/requirements.txt
```

## 运行

自动仿写：

```bash
cd /Users/xingdong/workspace/HD_HUMAN/my-video
.venv/bin/python -m rewrite auto \
  --text "这是原始文案" \
  --api-key "$DASHSCOPE_API_KEY"
```

根据指令仿写：

```bash
cd /Users/xingdong/workspace/HD_HUMAN/my-video
.venv/bin/python -m rewrite prompt \
  --text "这是原始文案" \
  --instruction "请用更口语化、更有悬念的方式改写"
```

兼容原项目模式名：

```bash
cd /Users/xingdong/workspace/HD_HUMAN/my-video
.venv/bin/python -m rewrite execute \
  --mode "AI自动仿写" \
  --input-file douyin/output/transcript.txt
```

写入文件：

```bash
cd /Users/xingdong/workspace/HD_HUMAN/my-video
.venv/bin/python -m rewrite auto \
  --input-file douyin/output/transcript.txt \
  --output rewrite/output/output_txt.txt
```

## 代码入口

- `rewrite/engine.py`: 可复用接口
- `rewrite/prompts.py`: 原项目 prompt 常量
- `rewrite/cli.py`: 命令行入口
