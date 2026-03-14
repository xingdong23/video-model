# DigiHuman

`digihuman` 目录按功能拆分维护。

## 子目录

- `scraper/`: 视频文案提取
- `copywriter/`: 文案仿写
- `subtitle/`: 字幕生成与烧录
- `audio_mixer/`: 背景音乐素材库与视频混音
- `tts/`: 语音相关功能
- `avatar/`: 数字人视频生成
- `pipeline/`: 串联语音、数字人、字幕的一键工作流

各功能使用各自目录下的 `README.md` 和 `requirements.txt`，不再共用根目录入口。

## 用人话说

如果只想理解 `digihuman` 这套东西在干什么，可以把它看成四步：

1. 先准备一段声音
   - 可以直接给现成音频
   - 也可以先用 `tts/` 把文字转成音频
2. 再让 `avatar/` 用这段声音驱动人物视频里的嘴型
3. 如果需要字幕，就让 `subtitle/` 再去听这段声音，生成带时间轴的字幕
4. 如果需要背景音乐，就让 `audio_mixer/` 把 BGM 音频叠到视频上
5. 最后导出成片

所以最终成片其实是四层东西叠在一起：

- 视频层：人物画面
- 音频层：说话声音
- BGM 层：背景音乐
- 字幕层：屏幕上的字

如果你想看整条链路怎么串起来，优先看 `pipeline/README.md`。
如果你想看字幕这块的术语解释，优先看 `subtitle/README.md`。
如果你想单独复用背景音乐能力，优先看 `audio_mixer/README.md`。
