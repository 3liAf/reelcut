"""Render a validated plan into a 1080x1920 reel with ffmpeg."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .captions import H, W, Segment, build_ass, build_timeline
from .media import VideoInfo, filter_path, run, windows_fonts_dir

FPS = 30

# Applied once in the final pass, under the captions so text stays crisp.
GRADES = {
    "vintage": ("curves=r='0/0.06 0.5/0.55 1/0.98':g='0/0.05 0.5/0.51 1/0.95':b='0/0.09 0.5/0.48 1/0.9',"
                "colorbalance=rs=-0.03:bs=0.04:rm=0.04:gm=-0.01:bm=-0.03:rh=0.04:bh=-0.03,"
                "eq=saturation=0.95:contrast=1.05,noise=alls=6:allf=t,vignette=angle=PI/3.3"),
    "punchy": "eq=contrast=1.1:saturation=1.25,unsharp=5:5:0.5,vignette=angle=PI/4.5",
    "clean": "eq=contrast=1.04:saturation=1.08,vignette=angle=PI/6",
    "none": "null",
}

POP_SFX = "aevalsrc='0.9*sin(2*PI*t*(220+1800*exp(-45*t)))*exp(-28*t)':s=48000:d=0.18"
TING_SFX = ("aevalsrc='0.55*sin(2*PI*2093*t)*exp(-5*t)+0.3*sin(2*PI*3136*t)*exp(-7*t)"
            "+0.15*sin(2*PI*5274*t)*exp(-11*t)':s=48000:d=0.9")


def _framing_at(plan: dict, t: float) -> dict:
    cur = plan["framing"][0]
    for f in plan["framing"]:
        if f["start"] <= t:
            cur = f
    return cur


def crop_box(info: VideoInfo, face_x: float, face_y: float, zoom: float) -> tuple[int, int, int, int]:
    """Largest 9:16 window in the source, shrunk by `zoom`, placed so the face sits ~36% from the top."""
    aspect = W / H
    if info.width / info.height > aspect:
        ch = info.height
        cw = ch * aspect
    else:
        cw = info.width
        ch = cw / aspect
    cw, ch = cw / zoom, ch / zoom
    cx = min(max(face_x * info.width - cw / 2, 0), info.width - cw)
    cy = min(max(face_y * info.height - 0.36 * ch, 0), info.height - ch)
    even = lambda v: int(v) // 2 * 2
    return even(cw), even(ch), even(cx), even(cy)


def _segment_cmd(src: Path, info: VideoInfo, plan: dict, seg: Segment, out: Path) -> list[str]:
    f = _framing_at(plan, seg.start)
    base = {"small": 1.3, "medium": 1.0, "large": 1.0}[f["subject_scale"]]
    zoom = base * (1.28 if seg.punch else 1.0)
    cw, ch, cx, cy = crop_box(info, f["face_x"], f["face_y"], zoom)
    push = 0.05 if seg.dur > 2.5 else 0.0  # slow push-in keeps long takes alive
    k = f"(1+{push}*t/{seg.dur:.3f}" + ("+0.22*pow(max(0,1-t/0.22),2))" if seg.pop else ")")
    vf = (f"crop={cw}:{ch}:{cx}:{cy},"
          f"scale=w='trunc({W}*{k}/2)*2':h='trunc({H}*{k}/2)*2':eval=frame:flags=lanczos,"
          f"crop={W}:{H}:(iw-{W})/2:(ih-{H})*0.4,eq=contrast=1.05:saturation=1.1,setsar=1,fps={FPS}")
    if seg.pop:
        vf += ",fade=t=in:d=0.07:color=white"
    nfr = max(1, round(seg.dur * FPS))
    return ["ffmpeg", "-v", "error", "-y", "-ss", f"{seg.start:.3f}", "-i", str(src), "-t", f"{seg.dur:.4f}",
            "-vf", vf, "-af", f"afade=t=in:d=0.012,afade=t=out:st={max(seg.dur - 0.02, 0):.3f}:d=0.02",
            "-frames:v", str(nfr), "-c:v", "libx264", "-preset", "fast", "-crf", "15", "-pix_fmt", "yuv420p",
            "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", str(out)]


def render(src: Path, info: VideoInfo, plan: dict, words: list[dict], workdir: Path, out: Path,
           grade: str = "vintage", font: str = "Arial Black", pop_gap: float = 3.0,
           jobs: int = 0, log=print) -> Path:
    segdir = workdir / "segments"
    segdir.mkdir(parents=True, exist_ok=True)
    segs = build_timeline(plan, FPS, pop_gap)

    # Segment files are cached by the exact command that makes them, so re-rendering
    # after a plan edit only redoes the cuts that changed.
    cmds, files = [], []
    for seg in segs:
        tmp = _segment_cmd(src, info, plan, seg, Path("X"))
        key = hashlib.sha1(json.dumps(tmp[:-1]).encode()).hexdigest()[:12]
        f = segdir / f"{seg.index:04d}_{key}.mov"
        files.append(f)
        if not f.exists():
            cmds.append(tmp[:-1] + [str(f)])
    log(f"cutting {len(segs)} segments ({len(cmds)} new)")
    with ThreadPoolExecutor(max_workers=jobs or max(1, min(4, (os.cpu_count() or 2) // 2))) as ex:
        list(ex.map(run, cmds))
    (segdir / "list.txt").write_text("".join(f"file '{f.name}'\n" for f in files), encoding="utf-8")

    ass, tings = build_ass(plan, segs, words, font)
    (workdir / "captions.ass").write_text(ass, encoding="utf-8")
    run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", POP_SFX, "-ac", "2", str(workdir / "sfx_pop.wav")])
    run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", TING_SFX, "-ac", "2", str(workdir / "sfx_ting.wav")])
    sfx = [("sfx_pop.wav", s.out_start, 0.9) for s in segs if s.pop] + [("sfx_ting.wav", t, 0.5) for t in tings]

    inputs, chains, labels = [], [], []
    for n, (name, t0, vol) in enumerate(sfx, start=1):
        inputs += ["-i", name]
        ms = int(t0 * 1000)
        chains.append(f"[{n}:a]volume={vol},adelay={ms}|{ms}[s{n}]")
        labels.append(f"[s{n}]")
    fonts = windows_fonts_dir()
    subs = "subtitles=captions.ass" + (f":fontsdir='{filter_path(fonts)}'" if fonts else "")
    fc = ("[0:a]highpass=f=70,acompressor=threshold=-20dB:ratio=3:attack=5:release=80[v];"
          + "".join(c + ";" for c in chains)
          + f"[v]{''.join(labels)}amix=inputs={len(sfx) + 1}:normalize=0:duration=first,"
          + "loudnorm=I=-14:TP=-1.5:LRA=9[a];"
          + f"[0:v]{GRADES.get(grade, grade)},{subs}[vo]")
    log(f"final pass: grade={grade}, {len(sfx)} sound effects")
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", "segments/list.txt", *inputs,
         "-filter_complex", fc, "-map", "[vo]", "-map", "[a]",
         "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-maxrate", "14M", "-bufsize", "28M",
         "-profile:v", "high", "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(out.resolve())],
        cwd=workdir)
    return out
