"""用 ffmpeg 的 ass 滤镜把双语字幕硬烧录进视频。"""

from __future__ import annotations

import subprocess
from pathlib import Path

from env_detect import Environment


class FfmpegError(RuntimeError):
    pass


def burn(
    video: Path,
    ass_path: Path,
    out_path: Path,
    env: Environment,
    *,
    crf: int = 18,
    preset: str = "medium",
    videotoolbox_bitrate: str = "8M",
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 在 ass 所在目录运行，滤镜只传文件名，规避 Windows 盘符冒号转义问题
    ass_dir = ass_path.parent
    vf = f"ass={ass_path.name}"

    def _encoder_args(encoder: str) -> list[str]:
        if encoder == "h264_nvenc":  # NVIDIA 硬件加速
            return ["-rc", "vbr", "-cq", str(crf), "-preset", "p4"]
        if encoder == "h264_videotoolbox":  # macOS 硬件加速（用码率控质量）
            return ["-b:v", videotoolbox_bitrate]
        return ["-crf", str(crf), "-preset", preset]  # libx264 软件编码

    def _run(encoder: str):
        args = [
            "ffmpeg",
            "-y",
            "-i",
            str(video.resolve()),
            "-vf",
            vf,
            "-c:v",
            encoder,
            *_encoder_args(encoder),
            "-c:a",
            "copy",
            str(out_path.resolve()),
        ]
        return subprocess.run(args, cwd=str(ass_dir))

    result = _run(env.ffmpeg_encoder)
    # 硬件编码器不可用时自动退回软件编码 libx264
    if result.returncode != 0 and env.ffmpeg_encoder != "libx264":
        print(f"       {env.ffmpeg_encoder} 编码失败，自动改用 libx264…")
        result = _run("libx264")
    if result.returncode != 0:
        raise FfmpegError(f"ffmpeg 烧录失败，退出码 {result.returncode}")
    return out_path
