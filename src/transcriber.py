"""转录调度：buzz / faster-whisper / whisper.cpp，支持失败回退，并可选 srt_equalizer 重分段。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import buzz_runner
from env_detect import Environment


class TranscribeError(RuntimeError):
    pass


# 供 faster-whisper 使用的短名 → HF 仓库映射（其余短名直接传给 WhisperModel）
_FW_MODEL_MAP = {
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "large": "large-v1",
}


def _srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_srt(cues, out_path: Path) -> None:
    """cues: 可迭代的 (start_sec, end_sec, text)。"""
    lines = []
    for i, (start, end, text) in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(start)} --> {_srt_timestamp(end)}")
        lines.append(text.strip())
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _extract_audio(source: str, out_dir: Path) -> Path:
    """用 ffmpeg 抽取 16k 单声道 wav（faster-whisper / whisper.cpp 需要）。"""
    wav = out_dir / "audio.wav"
    if wav.exists():
        return wav
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", source, "-vn", "-ac", "1", "-ar", "16000", str(wav)]
    )
    if result.returncode != 0 or not wav.exists():
        raise TranscribeError(f"ffmpeg 抽取音频失败，退出码 {result.returncode}")
    return wav


# ----------------------------- 各引擎 -----------------------------


def _run_buzz(source: str, out_dir: Path, env: Environment, cfg: dict) -> Path:
    buzz_cfg = cfg.get("buzz", {})
    return buzz_runner.transcribe(
        source,
        out_dir,
        env,
        extract_speech=buzz_cfg.get("extract_speech", True),
        timeout_seconds=buzz_cfg.get("timeout_seconds", 7200),
    )


def _fw_transcribe(model_id: str, device: str, compute_type: str, audio: Path, srt_path: Path) -> None:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_id, device=device, compute_type=compute_type)
    # vad_filter 过滤静音，减少静音段的幻觉/重复字幕
    segments, _info = model.transcribe(str(audio), vad_filter=True)
    _write_srt(((s.start, s.end, s.text) for s in segments), srt_path)


def _run_faster_whisper(source: str, out_dir: Path, env: Environment, cfg: dict) -> Path:
    try:
        import faster_whisper  # noqa: F401
    except ImportError as exc:
        raise TranscribeError(
            "未安装 faster-whisper，请先 `pip install -r requirements.txt`。"
        ) from exc

    tcfg = cfg.get("transcribe", {})
    size = tcfg.get("model_size") or env.buzz_model_size
    model_id = _FW_MODEL_MAP.get(size, size)

    device = tcfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if env.has_nvidia else "cpu"
    compute_type = tcfg.get("compute_type", "auto")
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    audio = _extract_audio(source, out_dir)
    srt_path = out_dir / "fasterwhisper.srt"
    try:
        _fw_transcribe(model_id, device, compute_type, audio, srt_path)
    except Exception as exc:
        # GPU 环境不全（缺 cuDNN 等）时自动降级到 CPU，保证能跑通
        if device == "cuda":
            print(f"       GPU 转录失败（{exc}），自动改用 CPU（会慢一些）…")
            _fw_transcribe(model_id, "cpu", "int8", audio, srt_path)
        else:
            raise
    return srt_path


def _run_whispercpp(source: str, out_dir: Path, env: Environment, cfg: dict) -> Path:
    tcfg = cfg.get("transcribe", {})
    binary = tcfg.get("whispercpp_bin", "")
    model = tcfg.get("whispercpp_model", "")
    if not binary or not model:
        raise TranscribeError(
            "whisper.cpp 需在 config 的 [transcribe] 填写 whispercpp_bin 与 whispercpp_model。"
        )

    audio = _extract_audio(source, out_dir)
    out_stem = out_dir / "whispercpp"
    result = subprocess.run(
        [binary, "-m", model, "-f", str(audio), "-osrt", "-of", str(out_stem)]
    )
    srt_path = out_stem.with_suffix(".srt")
    if result.returncode != 0 or not srt_path.exists():
        raise TranscribeError(f"whisper.cpp 转录失败，退出码 {result.returncode}")
    return srt_path


_ENGINES = {
    "buzz": _run_buzz,
    "fasterwhisper": _run_faster_whisper,
    "whispercpp": _run_whispercpp,
}


def _dispatch(engine: str, source: str, out_dir: Path, env: Environment, cfg: dict) -> Path:
    fn = _ENGINES.get(engine)
    if fn is None:
        raise TranscribeError(f"未知转录引擎：{engine}（可选 {', '.join(_ENGINES)}）")
    return fn(source, out_dir, env, cfg)


# ----------------------------- 重分段 -----------------------------


def _equalize(src_srt: Path, dst_srt: Path, max_chars: int) -> None:
    try:
        from srt_equalizer import srt_equalizer
    except ImportError as exc:
        raise TranscribeError(
            "未安装 srt_equalizer，请先 `pip install srt_equalizer`（或把 equalize_max_chars 设为 0 关闭）。"
        ) from exc
    srt_equalizer.equalize_srt_file(str(src_srt), str(dst_srt), max_chars)


# ----------------------------- 对外入口 -----------------------------


def transcribe(
    source: str,
    out_dir: Path,
    env: Environment,
    cfg: dict,
    *,
    resume: bool = True,
) -> Path:
    """转录并（可选）重分段，返回统一命名的 source.srt。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    final_srt = out_dir / "source.srt"
    if resume and final_srt.exists():
        print(f"[2/4] 已有字幕，跳过转录：{final_srt.name}")
        return final_srt

    tcfg = cfg.get("transcribe", {})
    engine = tcfg.get("engine", "fasterwhisper")
    fallback = tcfg.get("fallback", "")

    print(f"[2/4] 转录中（引擎 {engine}）…")
    try:
        raw_srt = _dispatch(engine, source, out_dir, env, cfg)
    except Exception as exc:
        if fallback and fallback != engine:
            print(f"       引擎 {engine} 失败（{exc}），改用回退引擎 {fallback}…")
            raw_srt = _dispatch(fallback, source, out_dir, env, cfg)
        else:
            raise

    max_chars = int(tcfg.get("equalize_max_chars", 0) or 0)
    if max_chars > 0:
        print(f"       重分段（每条 ≤ {max_chars} 字符）…")
        _equalize(raw_srt, final_srt, max_chars)
    elif raw_srt.resolve() != final_srt.resolve():
        shutil.copyfile(raw_srt, final_srt)

    print(f"       生成：{final_srt.name}")
    return final_srt
