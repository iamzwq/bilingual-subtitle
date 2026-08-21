# 双语字幕烧录流水线 · 操作手册

把一段外语视频(YouTube 链接或本地文件)自动做成 **中文在上、原文在下** 的双语硬字幕视频。

流程:取视频 → faster-whisper 转录原文 → LLM 翻译成中文 → 生成 ASS 双语字幕 → ffmpeg 硬烧录。

---

## 一、准备工作(每台电脑一次)

只有 **两步**：装 ffmpeg（一个安装包）+ 跑一条 pip 命令。其余全部由 pip 一次装好。

### 第 1 步：装 ffmpeg（唯一需要单独装的外部工具）

用系统包管理器一条命令装好（yt-dlp 合并音视频、字幕烧录都靠它）：

```bash
# Windows（PowerShell）
winget install --id Gyan.FFmpeg -e

# macOS（需先装 Homebrew：https://brew.sh）
brew install ffmpeg
```

装完确认能在命令行调用（装完可能需要**重开终端**让 PATH 生效）：

```bash
ffmpeg -version
python --version    # Mac 上用 python3 --version（需 Python 3.11+）
```

> 也可从 https://ffmpeg.org/download.html 手动下载。

### 第 2 步：一条命令装好所有 Python 依赖

```bash
cd bilingual-subtitle
pip install -r requirements.txt
```

这一条就把 `openai`、`yt-dlp`、`faster-whisper`（转录）、`srt_equalizer`（长句重分段）全装好了，**不用再一个个装**。

> **关于模型下载**：faster-whisper 首次转录会自动下载模型（约 1.6GB，联网即可，之后离线可用）。下载位置是 HuggingFace 缓存目录：
>
> - Windows：`C:\Users\<你的用户名>\.cache\huggingface\hub`
> - macOS / Linux：`~/.cache/huggingface/hub`
>
> 想换到别的盘/目录，设置环境变量 `HF_HOME` 指向新路径即可（例如 Windows：`setx HF_HOME D:\hf-cache`）。

---

## 二、配置

### 1. 生成配置文件

复制示例并按需修改:

```bash
cp config.example.toml config.toml
```

### 2. 填写 LLM 信息

直接在 `config.toml` 的 `[llm]` 里填：

```toml
[llm]
base_url = "https://token-plan-cn.xiaomimimo.com/v1"
api_key = "tp-你的key"
model = "模型名"      # 填你平台实际的模型名，不确定时查平台文档或调 GET /v1/models
```

### 3. 关键配置项说明

`model_size`(留空默认 `large-v3-turbo`):

| 档位                        | 速度 | 准确度     | 说明            |
| --------------------------- | ---- | ---------- | --------------- |
| tiny / base / small         | 快   | 低~中      | 快速测试        |
| medium                      | 中   | 高         | CPU 折中        |
| large / large-v2 / large-v3 | 慢   | 最高       | 需较强硬件      |
| **large-v3-turbo(默认)**    | 快   | 接近 large | 体积小,通用推荐 |

`[transcribe]` 转录引擎与后处理:

| 选项                                  | 说明                                           |
| ------------------------------------- | ---------------------------------------------- |
| `engine`                              | `fasterwhisper`(默认) / `whispercpp`           |
| `fallback`                            | 主引擎失败时自动改用的引擎;留空则不回退        |
| `equalize_max_chars`                  | 长句自动重分段的每条最大字符数,`0` 关闭        |
| `device` / `compute_type`             | faster-whisper 专用,`auto` 会按有无 GPU 自动选 |
| `whispercpp_bin` / `whispercpp_model` | whisper.cpp 专用:可执行文件与 ggml 模型路径    |

> **长句一屏放不下**:设 `equalize_max_chars = 42` 之类,会用 `srt_equalizer` 把过长的字幕条重新切分(改的是分段本身,和显示折行不同)。

#### 改用 whisper.cpp 转录

默认用 faster-whisper（pip 装好即用）。想换成 whisper.cpp，按三步走：

1. **装 whisper.cpp（拿到 `whisper-cli` 命令）**
   - macOS：`brew install whisper-cpp`（装完就有 `whisper-cli`）
   - Windows：到 [whisper.cpp releases](https://github.com/ggml-org/whisper.cpp/releases) 下载预编译包，解压后有 `whisper-cli.exe`（记住完整路径）
2. **下载 ggml 模型**：从 [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp/tree/main) 下一个 `.bin`，如 `ggml-large-v3-turbo.bin`，存到本地任意目录。
3. **改 `config.toml` 的 `[transcribe]`**：

   ```toml
   [transcribe]
   engine = "whispercpp"
   whispercpp_bin = "whisper-cli"                       # Mac 写命令名；Windows 写 .exe 完整路径
   whispercpp_model = "D:/models/ggml-large-v3-turbo.bin"
   ```

   改完直接运行即可，脚本会自动：抽取音频 → 调 `whisper-cli` 出 SRT → 继续翻译烧录。

> whisper.cpp 模式**不看** `model_size`（用你指定的 `whispercpp_model` 文件），也不需要 `device/compute_type`。也可保留 `engine = "fasterwhisper"` 再加 `fallback = "whispercpp"`，让 faster-whisper 失败时自动兜底。

`[subtitle]` 字幕样式(相对 1920×1080):中文字号 60、原文字号 36、白字黑底框、底部居中、中文每行 ~28 字换行(带避头尾)、原文按标点软换行。可自行调整。

---

## 三、运行

### 基本用法

Mac 用 `python3` 代替下面的 `python`。

```bash
# YouTube 链接
python main.py "https://www.youtube.com/watch?v=xxxx"

# 本地视频
python main.py "D:/videos/talk.mp4"

# 一次处理多个（链接/本地文件可混用，按顺序逐个跑）
python main.py "https://youtu.be/aaa" "https://youtu.be/bbb" "D:/videos/talk.mp4"
```

### 常用参数

| 参数             | 说明                                                          |
| ---------------- | ------------------------------------------------------------- |
| `--name 名称`    | 指定工作目录/输出文件名(URL 输入建议加上)                     |
| `--config 路径`  | 指定配置文件(默认 `config.toml`)                              |
| `--output 路径`  | 指定输出视频路径                                              |
| `--workdir 目录` | 产物根目录(默认 `output`，中间与最终都在 `<workdir>/<名称>/`) |
| `--title 标题`   | 本地文件可选：视频标题，用作术语表/翻译上下文                 |
| `--desc 简介`    | 本地文件可选：视频简介，用作术语表/翻译上下文                 |
| `--no-resume`    | 忽略已有中间产物,全部重跑                                     |

> URL 输入会自动用 yt-dlp 抓取标题与简介，无需手填 `--title/--desc`。
>
> **传入多个输入时**：`--name/--output/--title/--desc` 会被忽略（改为按各视频自动命名：URL 取视频号、本地文件取文件名）；单个视频失败不中断整批，结束时汇总成败，若有失败则以非零码退出。

示例:

```bash
python main.py "https://youtu.be/xxxx" --name lecture01
```

### 运行时能看到进度吗？

能。控制台会打印当前处于哪个阶段与进度：

- `[1/4] 下载视频…` — yt-dlp 自带百分比/速度进度条。
- `[2/4] 转录中… 已用时 NNs` — 每 2 秒刷新已用时心跳。
- `[3/4] 翻译进度 X/Y 条` — 按批实时刷新已完成条数。
- `[4/4] ffmpeg 硬烧录…` — ffmpeg 自带帧号/速度/已编码时长输出。

全部完成后会打印 `完成：<输出路径>`。总耗时取决于视频时长、机器算力和网络（首次还要下模型），脚本不预估具体 ETA。

### 输出位置

中间产物与最终视频都放在同一目录 `output/<名称>/`：

- 最终视频:`output/<名称>/<名称>_[视频id].mp4`（或 `--output` 指定）。视频 id：YouTube 链接取视频号，本地文件取文件名。
  - 例：`python main.py "https://youtu.be/dQw4w9WgXcQ" --name lecture01` → `output/lecture01/lecture01_[dQw4w9WgXcQ].mp4`
- 中间产物（同目录）:
  - `source.mp4`(URL 下载的视频)
  - `source.srt`(转录出的原文字幕,统一命名)
  - `audio.wav`(faster-whisper / whisper.cpp 引擎抽取的音频)
  - `meta.json`(视频标题/简介缓存，URL 输入时)
  - `glossary.json`(LLM 生成的术语表)
  - `bilingual.ass`(双语字幕)

### 断点续跑

默认开启:已完成的阶段会自动跳过(检测产物是否存在)。想全部重来加 `--no-resume`。例如只想重做翻译:删掉 `output/<名称>/bilingual.ass` 再运行即可(想同时重建术语表则一并删掉 `glossary.json`)。

---

## 四、四个阶段说明

| 阶段          | 做什么                                                                                               | 产物                             |
| ------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------- |
| 1/4 取视频    | URL 用 yt-dlp 下载最佳画质;本地文件直接用                                                            | `source.mp4`                     |
| 2/4 转录      | 按 `engine` 转录原文(faster-whisper / whisper.cpp),支持失败回退与长句重分段                          | `source.srt`                     |
| 3/4 翻译+合成 | 取标题/简介 → 生成术语表 → LLM 分批翻译(条数校验、超时重试3次、失败逐条兜底) → 避头尾换行 → 生成 ASS | `glossary.json`、`bilingual.ass` |
| 4/4 烧录      | ffmpeg 把字幕硬烧进画面(NVIDIA 用 nvenc,否则 libx264)                                                | `<名称>_[id].mp4`                |

---

## 五、常见问题

**Q: 提示「缺少 api_key / model」?**
在 config.toml 的 `[llm]` 填好 `api_key` / `model` / `base_url`。

**Q: 转录报错或没生成字幕?**
确认 ffmpeg 已装好(能 `ffmpeg -version`);首次运行可能在下载模型,请耐心等待并保持联网。若 GPU 报错，脚本会自动改用 CPU 重试。

**Q: 转录很慢?**
Intel Mac 是纯 CPU,长视频较慢;可把 `model_size` 降到 `medium` 或 `small`。Windows 确认 NVIDIA 驱动/CUDA 12 已装,后端才会用 GPU。

**Q: 字幕字体不对?**
安装「霞鹜文楷等宽」并确认系统字体族名一致;未装会 fallback 到默认字体(可接受)。

**Q: 翻译串行/错位?**
脚本对每批做条数校验,失败自动降级逐条翻译并以原文兜底,保证时间戳不错位。若整体质量不佳,可在 config 调小 `batch_size` 或换更强的 `model`。

**Q: Windows 路径带盘符冒号,ffmpeg 报错?**
脚本已在字幕文件所在目录运行 ffmpeg 并只传文件名规避该问题,通常无需处理。
