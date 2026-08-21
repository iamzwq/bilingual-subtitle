"""双语字幕流水线编排入口：取视频 → Buzz 转录 → LLM 翻译 → 生成 ASS → ffmpeg 硬烧录。

用法：
    python main.py <视频文件或URL> [--name 名称] [--config config.toml] [--output 输出.mp4] [--no-resume]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import ffmpeg_burn
import transcriber
import translate
from env_detect import detect_environment


def is_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return bool(parsed.scheme and parsed.netloc)


def load_config(path: Path) -> dict:
    if not path.is_file():
        sys.exit(
            f"未找到配置文件 {path}，请复制 config.example.toml 为 {path.name} 并填写。"
        )
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve_llm(cfg: dict) -> tuple[str, str, str]:
    """从 config.toml 的 [llm] 读取 base_url / api_key / model。"""
    llm = cfg.get("llm", {})
    base_url = llm.get("base_url", "")
    api_key = llm.get("api_key", "")
    model = llm.get("model", "")
    if not base_url:
        sys.exit("缺少 base_url：请在 config.toml 的 [llm] 填写 base_url。")
    if not api_key:
        sys.exit("缺少 api_key：请在 config.toml 的 [llm] 填写 api_key。")
    if not model:
        sys.exit("缺少 model：请在 config.toml 的 [llm] 填写 model。")
    return base_url, api_key, model


def extract_video_id(source: str, is_url_input: bool) -> str:
    """YouTube URL 取视频号；本地文件取文件名（无扩展名）。"""
    if not is_url_input:
        return Path(source).stem
    parsed = urllib.parse.urlparse(source)
    if "youtu.be" in parsed.netloc.lower():
        vid = parsed.path.lstrip("/").split("/")[0]
        if vid:
            return vid
    qs = urllib.parse.parse_qs(parsed.query)
    if qs.get("v"):
        return qs["v"][0]
    segments = [p for p in parsed.path.split("/") if p]
    return segments[-1] if segments else "video"


def download_video(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "yt-dlp",
        "-f",
        "bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "-o",
        str(dest),
        url,
    ]
    result = subprocess.run(args)
    if result.returncode != 0 or not dest.exists():
        sys.exit(f"yt-dlp 下载失败，退出码 {result.returncode}")
    return dest


def fetch_metadata(url: str, cache: Path) -> tuple[str, str]:
    """用 yt-dlp 取视频标题与简介（结果缓存到 cache）；失败返回空串。"""
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            return data.get("title", ""), data.get("description", "")
        except Exception:
            pass
    try:
        result = subprocess.run(
            ["yt-dlp", "--skip-download", "--dump-json", url],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return "", ""
        info = json.loads(result.stdout)
        title = info.get("title", "") or ""
        description = info.get("description", "") or ""
        cache.write_text(
            json.dumps({"title": title, "description": description}, ensure_ascii=False),
            encoding="utf-8",
        )
        return title, description
    except Exception:
        return "", ""


def main() -> None:
    parser = argparse.ArgumentParser(description="YouTube 外语视频 → 中外双语硬字幕")
    parser.add_argument("input", help="本地视频路径或视频 URL")
    parser.add_argument("--name", help="工作/输出目录名，缺省用文件名或 video")
    parser.add_argument("--config", default="config.toml", help="配置文件路径")
    parser.add_argument(
        "--workdir", default="output", help="产物根目录（中间与最终视频都在 <workdir>/<名称>/）"
    )
    parser.add_argument("--output", help="输出视频路径（缺省在工作目录内）")
    parser.add_argument("--title", default="", help="视频标题（本地文件可选，用作术语表/翻译上下文）")
    parser.add_argument("--desc", default="", help="视频简介（本地文件可选）")
    parser.add_argument(
        "--no-resume", action="store_true", help="忽略已有中间产物，全部重跑"
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    base_url, api_key, model = resolve_llm(cfg)
    resume = not args.no_resume

    url_input = is_url(args.input)
    name = args.name or ("video" if url_input else Path(args.input).stem)
    video_id = extract_video_id(args.input, url_input)
    workdir = Path(args.workdir) / name
    workdir.mkdir(parents=True, exist_ok=True)

    env = detect_environment(cfg.get("buzz", {}) | cfg.get("ffmpeg", {}))
    tcfg = cfg.get("transcribe", {})
    engine = tcfg.get("engine", "fasterwhisper")
    model_size = tcfg.get("model_size") or env.buzz_model_size
    print(
        f"[环境] GPU={env.has_nvidia} 引擎={engine} "
        f"模型={model_size} 编码器={env.ffmpeg_encoder}"
    )

    # 阶段1 取视频
    if url_input:
        source_video = workdir / "source.mp4"
        if resume and source_video.exists():
            print("[1/4] 视频已存在，跳过下载")
        else:
            print("[1/4] 下载视频…")
            download_video(args.input, source_video)
    else:
        source_video = Path(args.input).resolve()
        if not source_video.exists():
            sys.exit(f"输入视频不存在：{source_video}")
        print("[1/4] 使用本地视频")

    # 阶段2 转录（引擎调度 + 可选重分段，内部处理 resume）
    srt_path = transcriber.transcribe(str(source_video), workdir, env, cfg, resume=resume)

    # 阶段3 翻译 + 生成 ASS
    ass_path = workdir / "bilingual.ass"
    if resume and ass_path.exists():
        print("[3/4] ASS 已存在，跳过翻译")
    else:
        print("[3/4] 解析字幕并调用 LLM 翻译…")
        segments = translate.parse_srt(str(srt_path))
        llm = cfg.get("llm", {})
        target_lang = llm.get("target_language", "简体中文")

        # 视频元信息（标题/简介）：URL 用 yt-dlp 取，本地文件用 --title/--desc
        if url_input:
            title, description = fetch_metadata(args.input, workdir / "meta.json")
        else:
            title, description = "", ""
        title = args.title or title
        description = args.desc or description

        # 术语表（缓存），保证全片译名一致
        glossary_path = workdir / "glossary.json"
        if resume and glossary_path.exists():
            glossary = json.loads(glossary_path.read_text(encoding="utf-8"))
            print(f"       复用术语表（{len(glossary)} 条）")
        else:
            print("       生成术语表…")
            glossary = translate.build_glossary(
                segments,
                base_url=base_url,
                api_key=api_key,
                model=model,
                title=title,
                description=description,
                target_language=target_lang,
                temperature=llm.get("temperature", 0.3),
            )
            glossary_path.write_text(
                json.dumps(glossary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"       术语表 {len(glossary)} 条")

        def _on_progress(done: int, total: int) -> None:
            print(f"       翻译进度 {done}/{total} 条", end="\r", flush=True)

        translations = translate.translate_segments(
            segments,
            base_url=base_url,
            api_key=api_key,
            model=model,
            target_language=target_lang,
            batch_size=llm.get("batch_size", 30),
            temperature=llm.get("temperature", 0.3),
            title=title,
            description=description,
            glossary=glossary,
            on_progress=_on_progress,
        )
        print()  # 结束进度行
        sub = cfg.get("subtitle", {})
        translate.build_ass(
            segments,
            translations,
            str(ass_path),
            font_name=sub.get("font_name", "霞鹜文楷等宽"),
            cjk_font_size=sub.get("cjk_font_size", 60),
            src_font_size=sub.get("src_font_size", 36),
            cjk_max_chars=sub.get("cjk_max_chars", 28),
            src_wrap_chars=sub.get("src_wrap_chars", 60),
            margin_v=sub.get("margin_v", 60),
        )
        print(f"       生成：{ass_path.name}（{len(segments)} 条）")

    # 阶段4 硬烧录
    output = (
        Path(args.output)
        if args.output
        else workdir / f"{name}_[{video_id}].mp4"
    )
    print("[4/4] ffmpeg 硬烧录…")
    ff = cfg.get("ffmpeg", {})
    ffmpeg_burn.burn(
        source_video,
        ass_path,
        output,
        env,
        crf=ff.get("crf", 18),
        preset=ff.get("preset", "medium"),
        videotoolbox_bitrate=ff.get("videotoolbox_bitrate", "8M"),
    )
    print(f"完成：{output}")


if __name__ == "__main__":
    main()
