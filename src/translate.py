"""SRT 解析、LLM 批量翻译（条数校验兜底）、中文/原文换行、双语 ASS 生成。"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass


@dataclass
class Segment:
    index: int
    start: str  # SRT 时间戳 HH:MM:SS,mmm
    end: str
    text: str


_TIME_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})"
)


def parse_srt(path: str) -> list[Segment]:
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    segments: list[Segment] = []
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if len(lines) < 2:
            continue
        time_line_idx = 0
        # 第一行可能是序号，找到时间戳行
        for i, ln in enumerate(lines):
            if _TIME_RE.search(ln):
                time_line_idx = i
                break
        else:
            continue
        m = _TIME_RE.search(lines[time_line_idx])
        text = " ".join(lines[time_line_idx + 1 :]).strip()
        segments.append(
            Segment(
                index=len(segments) + 1,
                start=m.group(1),
                end=m.group(2),
                text=text,
            )
        )
    return segments


# ----------------------------- 翻译 -----------------------------


_TRANSLATE_SYSTEM = (
    "你是专业的视频字幕翻译。将用户给出的每一条字幕翻译成{lang}。\n"
    "要求：逐条翻译，不合并、不拆分、不增删条目；保留数字、专有名词、人名；"
    "口语自然、符合中文表达习惯。\n"
    "只返回一个 JSON 数组，元素为每条对应的译文字符串，"
    "数组长度必须与输入完全一致，不要输出任何多余内容。"
)

_GLOSSARY_SYSTEM = (
    "你在为一个视频字幕翻译项目准备术语表。根据给出的标题、简介和字幕全文，"
    "提取 10~40 个关键术语、专有名词、人名、产品名或高频概念，并给出统一的{lang}译名。\n"
    "只返回一个 JSON 对象，键为原文术语，值为{lang}译名，不要输出任何多余内容。"
)


def _chat_with_retry(client, *, retries: int = 3, backoff: float = 2.0, **kwargs):
    """调用 chat.completions，失败（超时等）指数退避重试，默认 3 次。"""
    last_exc = None
    for attempt in range(retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # 超时/网络/限流等统一重试
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
    raise last_exc


def _truncate(text: str, limit: int) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def _build_context_block(title: str, description: str, glossary: dict) -> str:
    parts = []
    if title:
        parts.append(f"视频标题：{title}")
    if description:
        parts.append(f"视频简介：{_truncate(description, 500)}")
    if glossary:
        pairs = "；".join(f"{k}→{v}" for k, v in glossary.items())
        parts.append(f"术语对照（务必统一沿用）：{pairs}")
    if not parts:
        return ""
    return "以下是本视频的背景信息，帮助你翻译得更准确：\n" + "\n".join(parts) + "\n\n"


def _parse_glossary_obj(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
    lb, rb = raw.find("{"), raw.rfind("}")
    if lb != -1 and rb != -1 and rb > lb:
        raw = raw[lb : rb + 1]
    data = json.loads(raw)
    return {str(k): str(v) for k, v in data.items()}


def build_glossary(
    segments: list[Segment],
    *,
    base_url: str,
    api_key: str,
    model: str,
    title: str = "",
    description: str = "",
    target_language: str = "简体中文",
    temperature: float = 0.3,
) -> dict:
    """让 LLM 基于标题/简介/字幕全文提炼 10~40 个术语的统一译名；失败返回空表。"""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    transcript = _truncate(" ".join(s.text for s in segments), 8000)
    ctx = []
    if title:
        ctx.append(f"标题：{title}")
    if description:
        ctx.append(f"简介：{_truncate(description, 500)}")
    ctx.append("字幕全文：" + transcript)
    try:
        resp = _chat_with_retry(
            client,
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": _GLOSSARY_SYSTEM.format(lang=target_language)},
                {"role": "user", "content": "\n".join(ctx)},
            ],
        )
        return _parse_glossary_obj(resp.choices[0].message.content or "")
    except Exception:
        return {}


def _translate_batch(
    client,
    model: str,
    texts: list[str],
    target_language: str,
    temperature: float,
    context: str,
) -> list[str]:
    payload = json.dumps(
        [{"id": i, "text": t} for i, t in enumerate(texts)],
        ensure_ascii=False,
    )
    resp = _chat_with_retry(
        client,
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": context + _TRANSLATE_SYSTEM.format(lang=target_language),
            },
            {"role": "user", "content": payload},
        ],
    )
    raw = resp.choices[0].message.content or ""
    return _parse_translation_array(raw)


def _parse_translation_array(raw: str) -> list[str]:
    raw = raw.strip()
    # 去掉可能的 ```json 包裹
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
    # 截取最外层数组
    lb, rb = raw.find("["), raw.rfind("]")
    if lb != -1 and rb != -1 and rb > lb:
        raw = raw[lb : rb + 1]
    data = json.loads(raw)
    result: list[str] = []
    for item in data:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            result.append(str(item.get("text", "")))
        else:
            result.append(str(item))
    return result


def translate_segments(
    segments: list[Segment],
    *,
    base_url: str,
    api_key: str,
    model: str,
    target_language: str,
    batch_size: int = 30,
    temperature: float = 0.3,
    title: str = "",
    description: str = "",
    glossary: dict | None = None,
    on_progress=None,
) -> list[str]:
    """返回与 segments 等长的译文列表；批量失败则对该批降级逐条翻译。"""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    context = _build_context_block(title, description, glossary or {})
    translations: list[str] = []
    total = len(segments)

    for i in range(0, len(segments), batch_size):
        batch = segments[i : i + batch_size]
        texts = [s.text for s in batch]
        try:
            out = _translate_batch(
                client, model, texts, target_language, temperature, context
            )
            if len(out) != len(batch):
                raise ValueError(f"条数不匹配: 期望 {len(batch)} 得到 {len(out)}")
            translations.extend(out)
        except Exception:
            for text in texts:
                try:
                    one = _translate_batch(
                        client, model, [text], target_language, temperature, context
                    )
                    translations.append(one[0] if one else text)
                except Exception:
                    translations.append(text)  # 兜底：保留原文，绝不错位
        if on_progress is not None:
            on_progress(min(len(translations), total), total)

    assert len(translations) == len(segments)
    return translations


# ----------------------------- 换行 -----------------------------

# 避头尾（禁则）：不可出现在行首/行尾的标点
_NO_LINE_START = set("。，、！？；：）】》」』’”%·.,!?;:)")
_NO_LINE_END = set("（【《「『‘“(")


def wrap_cjk(text: str, max_chars: int) -> str:
    """中文按最大字符数换行，并做避头尾（禁则）处理，返回以 \\n 分隔的多行。"""
    text = text.replace("\n", "")
    if len(text) <= max_chars:
        return text
    lines: list[str] = []
    i, n = 0, len(text)
    while i < n:
        end = min(i + max_chars, n)
        # 避头：不让禁则标点落在下一行行首，把它并入本行
        while end < n and text[end] in _NO_LINE_START:
            end += 1
        # 避尾：不让禁则标点（如开引号/括号）落在本行行尾，回退一位
        while end - 1 > i and text[end - 1] in _NO_LINE_END:
            end -= 1
        lines.append(text[i:end])
        i = end
    return "\n".join(lines)


_PUNCT = tuple(".,!?;:，。！？；：、")


def wrap_src_by_punct(text: str, max_chars: int) -> str:
    """原文单行过长时，在最接近中点的标点处软换行（递归处理）。"""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text

    mid = len(text) // 2
    best = -1
    for i, ch in enumerate(text):
        if ch in _PUNCT and abs(i - mid) < abs(best - mid):
            best = i
    # 切分点必须能让两侧都真正变短；否则退化到空格处，再退化为硬断
    if best <= 0 or best >= len(text) - 1:
        best = text.rfind(" ", 0, max_chars)
        if best <= 0:
            best = max_chars - 1

    left = text[: best + 1].strip()
    right = text[best + 1 :].strip()
    if not left or not right:  # 兜底硬切，杜绝无限递归
        left, right = text[:max_chars].strip(), text[max_chars:].strip()
    return wrap_src_by_punct(left, max_chars) + "\n" + wrap_src_by_punct(right, max_chars)


# ----------------------------- ASS 生成 -----------------------------


def _srt_time_to_ass(t: str) -> str:
    """HH:MM:SS,mmm -> H:MM:SS.cc（厘秒）。"""
    hms, ms = t.split(",")
    h, m, s = hms.split(":")
    cs = int(ms) // 10
    return f"{int(h)}:{m}:{s}.{cs:02d}"


def _escape_ass(text: str) -> str:
    return text.replace("{", "(").replace("}", ")")


def _to_ass_multiline(text: str) -> str:
    """把 \\n 换成 ASS 硬换行标记 \\N。"""
    return _escape_ass(text).replace("\n", r"\N")


def build_ass_header(
    *,
    font_name: str,
    cjk_font_size: int,
    margin_v: int,
) -> str:
    # BorderStyle=3 不透明背景框；BackColour/OutlineColour 黑；Primary 白；底部居中 Alignment=2
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Bi,{font_name},{cjk_font_size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,3,2,0,2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def build_ass(
    segments: list[Segment],
    translations: list[str],
    out_path: str,
    *,
    font_name: str,
    cjk_font_size: int,
    src_font_size: int,
    cjk_max_chars: int,
    src_wrap_chars: int,
    margin_v: int,
) -> None:
    header = build_ass_header(
        font_name=font_name,
        cjk_font_size=cjk_font_size,
        margin_v=margin_v,
    )
    lines = [header]
    for seg, zh in zip(segments, translations):
        cjk = _to_ass_multiline(wrap_cjk(zh, cjk_max_chars))
        src = _to_ass_multiline(wrap_src_by_punct(seg.text, src_wrap_chars))
        # 中文在上（样式默认字号），原文在下（内联缩小字号）
        text = f"{cjk}\\N{{\\fs{src_font_size}}}{src}"
        start = _srt_time_to_ass(seg.start)
        end = _srt_time_to_ass(seg.end)
        lines.append(f"Dialogue: 0,{start},{end},Bi,,0,0,0,,{text}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
