# douyin

`my-video` 下抖音文案提取的独立功能目录。

## 特点

- 适合 Linux 云主机部署
- 纯服务端 `requests` 链路
- 不依赖 Playwright、Chrome 或桌面浏览器
- 支持抖音短链、直链、整段分享文案
- 支持通过 `DOUYIN_COOKIE` 或 `--cookie` 补 Cookie
- 默认下载目录: `douyin/downloads`
- 推荐输出目录: `douyin/output`

## 示例

```bash
pip install -r douyin/requirements.txt
python3 douyin/douyin_transcript.py "https://v.douyin.com/xxxxxx/"
python3 douyin/douyin_transcript.py "https://www.douyin.com/video/1234567890123456789"
```

如果本地 Whisper 模型不在 `douyin/models`，可以显式指定：

```bash
python3 douyin/douyin_transcript.py "https://v.douyin.com/xxxxxx/" \
  --model-dir ../cosyvoice/models
```
