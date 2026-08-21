"""检测运行环境（OS / NVIDIA GPU）并给出 Buzz 后端、模型大小、ffmpeg 编码器与 Buzz 调用命令。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Environment:
    is_windows: bool
    is_mac: bool
    has_nvidia: bool
    buzz_backend: str
    buzz_model_size: str
    ffmpeg_encoder: str
    buzz_command: list[str]


def _has_nvidia_gpu() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        result = subprocess.run(
            [exe, "-L"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and "GPU" in result.stdout


def _detect_buzz_command() -> list[str]:
    """按平台探测 Buzz 可执行入口，找不到则回退到 `python -m buzz`。"""
    if sys.platform == "win32":
        candidates = [
            os.path.expandvars(r"%ProgramFiles%\Buzz\Buzz.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Buzz\Buzz.exe"),
        ]
    elif sys.platform == "darwin":
        candidates = ["/Applications/Buzz.app/Contents/MacOS/Buzz"]
    else:
        candidates = []

    for path in candidates:
        if path and os.path.isfile(path):
            return [path]

    on_path = shutil.which("buzz")
    if on_path:
        return [on_path]

    return [sys.executable, "-m", "buzz"]


def detect_environment(overrides: dict | None = None) -> Environment:
    overrides = overrides or {}
    has_nvidia = _has_nvidia_gpu()

    backend = overrides.get("backend") or (
        "fasterwhisper" if has_nvidia else "whispercpp"
    )
    model_size = overrides.get("model_size") or "large-v3-turbo"
    if overrides.get("encoder"):
        encoder = overrides["encoder"]
    elif has_nvidia:
        encoder = "h264_nvenc"          # NVIDIA 硬件加速
    elif sys.platform == "darwin":
        encoder = "h264_videotoolbox"   # macOS 硬件加速
    else:
        encoder = "libx264"             # 纯 CPU 软件编码

    command_override = overrides.get("command")
    if command_override:
        buzz_command = (
            command_override.split()
            if isinstance(command_override, str)
            else list(command_override)
        )
    else:
        buzz_command = _detect_buzz_command()

    return Environment(
        is_windows=sys.platform == "win32",
        is_mac=sys.platform == "darwin",
        has_nvidia=has_nvidia,
        buzz_backend=backend,
        buzz_model_size=model_size,
        ffmpeg_encoder=encoder,
        buzz_command=buzz_command,
    )
