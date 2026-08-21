"""调用 Buzz CLI 转录视频，并轮询等待 SRT 生成 + 进程退出（双条件）。"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from env_detect import Environment


class BuzzError(RuntimeError):
    pass


def _existing_srts(out_dir: Path) -> set[Path]:
    return set(out_dir.glob("*.srt"))


def transcribe(
    source: str,
    out_dir: Path,
    env: Environment,
    *,
    extract_speech: bool = True,
    poll_interval: float = 2.0,
    timeout_seconds: int = 7200,
) -> Path:
    """转录 source（本地路径或 URL），返回生成的 SRT 路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    before = _existing_srts(out_dir)

    args = list(env.buzz_command) + [
        "add",
        "--task",
        "transcribe",
        "--model-type",
        env.buzz_backend,
        "--model-size",
        env.buzz_model_size,
        "--word-timestamps",
        "--srt",
        "--hide-gui",
        "--output-directory",
        str(out_dir),
    ]
    if extract_speech:
        args.append("--extract-speech")
    args.append(source)

    proc = subprocess.Popen(args)

    start = time.monotonic()
    new_srt: Path | None = None
    while True:
        exited = proc.poll() is not None
        appeared = _existing_srts(out_dir) - before
        if appeared:
            new_srt = max(appeared, key=lambda p: p.stat().st_mtime)

        # 双条件：目标 SRT 已生成且进程已退出
        if new_srt is not None and exited:
            print()  # 结束心跳行
            if proc.returncode not in (0, None):
                raise BuzzError(f"Buzz 退出码 {proc.returncode}")
            return new_srt

        if exited and new_srt is None:
            print()
            raise BuzzError(
                f"Buzz 进程已退出（码 {proc.returncode}）但未生成 SRT 文件"
            )

        if time.monotonic() - start > timeout_seconds:
            proc.kill()
            raise BuzzError("Buzz 转录超时")

        elapsed = int(time.monotonic() - start)
        print(f"       Buzz 转录中… 已用时 {elapsed}s", end="\r", flush=True)
        time.sleep(poll_interval)
