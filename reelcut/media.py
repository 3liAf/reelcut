"""Thin ffmpeg/ffprobe helpers: probing, silence detection, contact sheets."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise FFmpegError(f"{tool} not found on PATH. Install ffmpeg first (https://ffmpeg.org/download.html).")


def run(cmd: list[str], cwd: str | Path | None = None) -> str:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        raise FFmpegError(f"command failed ({r.returncode}): {' '.join(cmd[:6])} ...\n{r.stderr[-3000:]}")
    return r.stderr + r.stdout


@dataclass
class VideoInfo:
    width: int      # display width, after rotation metadata is applied
    height: int     # display height
    duration: float
    fps: float


def probe(path: str | Path) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate:stream_side_data=rotation:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    st = data["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    rot = 0
    for sd in st.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    if abs(rot) % 180 == 90:
        w, h = h, w
    num, _, den = st.get("r_frame_rate", "30/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    return VideoInfo(w, h, float(data["format"]["duration"]), fps)


def silences(path: str | Path, noise_db: float = -35, min_len: float = 0.35) -> list[tuple[float, float]]:
    """Silent stretches as (start, end) pairs, from ffmpeg's silencedetect."""
    log = run(["ffmpeg", "-hide_banner", "-i", str(path), "-vn", "-af",
               f"silencedetect=n={noise_db}dB:d={min_len}", "-f", "null", "-"])
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
    return list(zip(starts, ends))


def extract_audio(src: str | Path, dst: str | Path) -> None:
    run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(dst)])


def find_font_file() -> str | None:
    """A bold TTF for burned-in labels on contact sheets (drawtext needs a file path)."""
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]
    return next((c for c in candidates if os.path.exists(c)), None)


def filter_path(p: str) -> str:
    """Escape a filesystem path for use inside an ffmpeg filter argument."""
    return p.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def contact_sheets(src: str | Path, out_dir: Path, every: float, duration: float,
                   per_sheet: int = 60, cols: int = 10) -> list[tuple[Path, float, float]]:
    """Grids of thumbnails, one every `every` seconds, each stamped with its source second.

    Returns (image, first_second, last_second) per sheet.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = per_sheet // cols
    span = every * per_sheet
    font = find_font_file()
    sheets = []
    t0 = 0.0
    while t0 < duration:
        label = ""
        if font:
            label = (f",drawtext=fontfile='{filter_path(font)}':text='%{{eif\\:t+{t0:.0f}\\:d}}':"
                     "x=4:y=4:fontsize=26:fontcolor=yellow:box=1:boxcolor=black")
        img = out_dir / f"sheet_{int(t0):05d}.jpg"
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0}", "-t", f"{span}", "-i", str(src),
             "-vf", f"fps=1/{every},scale=150:-2{label},tile={cols}x{rows}", "-frames:v", "1",
             "-q:v", "4", str(img)])
        sheets.append((img, t0, min(t0 + span, duration)))
        t0 += span
    return sheets


def windows_fonts_dir() -> str | None:
    return r"C:\Windows\Fonts" if sys.platform == "win32" and os.path.isdir(r"C:\Windows\Fonts") else None
