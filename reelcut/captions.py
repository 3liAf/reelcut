"""Timeline mapping and ASS caption generation. Pure functions, no ffmpeg."""
from __future__ import annotations

import re
from dataclasses import dataclass

W, H = 1080, 1920
CAPTION_Y = 1330          # caption centre: below the face, above Instagram's bottom UI
YELLOW, GREEN, WHITE = "&H0000E5FF&", "&H006EFF3D&", "&H00FFFFFF&"


@dataclass
class Segment:
    index: int
    start: float      # source seconds
    end: float
    text: str
    out_start: float  # seconds on the output timeline
    dur: float
    pop: bool         # zoom-pop entrance (speaker walked off before this cut)
    punch: bool       # tighter framing (alternates cut to cut)


def build_timeline(plan: dict, fps: int = 30, pop_gap: float = 3.0) -> list[Segment]:
    segs, t, punch, prev_end = [], 0.0, False, None
    for i, c in enumerate(plan["cuts"]):
        dur = round((c["end"] - c["start"]) * fps) / fps
        if dur <= 0:
            continue
        pop = prev_end is not None and c["start"] - prev_end > pop_gap
        segs.append(Segment(i, c["start"], c["end"], c["text"], t, dur, pop, punch))
        t += dur
        punch = not punch
        prev_end = c["end"]
    return segs


def to_out(segs: list[Segment], t: float) -> float | None:
    for s in segs:
        if s.start - 0.05 <= t <= s.end + 0.05:
            return s.out_start + min(max(t - s.start, 0.0), s.dur)
    return None


def _ts(t: float) -> str:
    t = max(t, 0.0)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def _norm(w: str) -> str:
    return re.sub(r"[^A-Z0-9\-']", "", w.upper())


def _esc(text: str) -> str:
    return text.replace("{", "(").replace("}", ")").replace("\\", "/")


def word_timings(seg: Segment, words: list[dict]) -> list[tuple[float, float]]:
    """Per-token (start, end) in source seconds for the segment's caption text.

    Uses the ASR word times when the token count matches the words inside the
    cut; otherwise spreads tokens across the speech span by character length.
    """
    toks = seg.text.split()
    inside = [w for w in words if seg.start <= (w["start"] + w["end"]) / 2 <= seg.end]
    if len(inside) == len(toks):
        return [(max(w["start"], seg.start), min(w["end"], seg.end)) for w in inside]
    a = max(inside[0]["start"], seg.start) if inside else seg.start
    b = min(inside[-1]["end"], seg.end) if inside else seg.end
    total = sum(len(x) + 2 for x in toks) or 1
    out, acc = [], a
    for x in toks:
        d = (b - a) * (len(x) + 2) / total
        out.append((acc, acc + d))
        acc += d
    return out


def chunk(tokens: list[str], max_words: int = 3) -> list[list[int]]:
    chunks, cur = [], []
    for k, x in enumerate(tokens):
        cur.append(k)
        if len(cur) == max_words or re.search(r"[,.?!:;]$", x) or (len(cur) == 2 and len(x) > 9):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


class Ass:
    def __init__(self):
        self.events: list[str] = []

    def add(self, st: float, en: float, style: str, text: str, layer: int = 0):
        if en - st > 0.02:
            self.events.append(f"Dialogue: {layer},{_ts(st)},{_ts(en)},{style},,0,0,0,,{text}")


def build_ass(plan: dict, segs: list[Segment], words: list[dict], font: str = "Arial Black") -> tuple[str, list[float]]:
    """Return (ASS document, output times of counter increments for the 'ting' sound)."""
    ass = Ass()
    keywords = set(plan.get("keywords") or [])
    total = segs[-1].out_start + segs[-1].dur

    for seg in segs:
        toks = seg.text.split()
        if not toks:
            continue
        tim = word_timings(seg, words)
        chunks = chunk(toks)
        for ci, c in enumerate(chunks):
            c0 = seg.out_start if ci == 0 else seg.out_start + tim[c[0]][0] - seg.start
            cend = (seg.out_start + tim[chunks[ci + 1][0]][0] - seg.start) if ci + 1 < len(chunks) \
                else seg.out_start + seg.dur
            for j, k in enumerate(c):
                st = c0 if j == 0 else seg.out_start + tim[k][0] - seg.start
                en = cend if j == len(c) - 1 else seg.out_start + tim[c[j + 1]][0] - seg.start
                parts = []
                for kk in c:
                    wd = _esc(toks[kk].upper().rstrip(","))
                    if kk == k:
                        parts.append(f"{{\\c{YELLOW}\\fscx108\\fscy108}}{wd}{{\\c{WHITE}\\fscx100\\fscy100}}")
                    elif _norm(wd) in keywords:
                        parts.append(f"{{\\c{GREEN}}}{wd}{{\\c{WHITE}}}")
                    else:
                        parts.append(wd)
                pop = "\\fscx70\\fscy70\\t(0,90,\\fscx100\\fscy100)" if j == 0 else ""
                ass.add(st, en, "Cap", f"{{\\an5\\pos({W // 2},{CAPTION_Y}){pop}}}" + " ".join(parts))

    hook = plan.get("hook") or {}
    if hook.get("line1"):
        ass.add(0, 2.6, "Hook", "{\\an5\\pos(540,330)\\frz-3\\fscx60\\fscy60\\t(0,140,\\fscx100\\fscy100)}"
                + _esc(hook["line1"]))
    if hook.get("line2"):
        ass.add(0.25, 2.6, "HookBox", "{\\an5\\pos(540,460)\\frz-3\\fscx60\\fscy60\\t(0,140,\\fscx100\\fscy100)}"
                + _esc(hook["line2"]))

    for st in plan.get("stamps") or []:
        o = to_out(segs, st["time"])
        if o is None:
            continue
        seg_end = next(s.out_start + s.dur for s in segs if s.out_start <= o <= s.out_start + s.dur)
        ass.add(o, min(seg_end + 0.5, total), "Stamp",
                "{\\an5\\pos(540,560)\\frz7\\fscx220\\fscy220\\alpha&HFF&\\t(0,120,\\fscx100\\fscy100\\alpha&H00&)}"
                + _esc(st["text"].upper()), 2)

    tings = []
    counter = plan.get("counter") or {}
    outro = plan.get("outro") or {}
    outro_len = 2.6 if (outro.get("line1") or outro.get("line2")) else 0.0
    evs = [(to_out(segs, e["time"]), e) for e in counter.get("events") or []]
    evs = [(o, e) for o, e in evs if o is not None]
    for i, (o, e) in enumerate(evs):
        end = evs[i + 1][0] if i + 1 < len(evs) else total - outro_len
        tings.append(o)
        ass.add(o, end, "Count", "{\\an5\\pos(540,250)\\t(0,80,\\fscx118\\fscy118)\\t(80,200,\\fscx100\\fscy100)}"
                + _esc(f"{counter.get('label', '')}: {e['value']}".upper()), 1)
        if e.get("label"):
            ass.add(o, min(o + 1.3, end), "Plus", "{\\an5\\move(540,400,540,365,0,200)\\fad(80,200)}"
                    + _esc(e["label"].upper()), 1)

    if outro.get("line1"):
        ass.add(total - 2.6, total, "Hook", "{\\an5\\pos(540,420)\\fad(150,0)\\fscx60\\fscy60\\t(0,140,\\fscx100\\fscy100)}"
                + _esc(outro["line1"]))
    if outro.get("line2"):
        ass.add(total - 1.9, total, "HookBox", "{\\an5\\pos(540,1330)\\fad(150,0)\\q0}" + _esc(outro["line2"]))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},78,&H00FFFFFF,&H00FFFFFF,&H00000000,&H99000000,-1,0,0,0,100,100,1,0,1,7,4,5,70,70,0,1
Style: Hook,{font},96,&H00FFFFFF,&H00FFFFFF,&H00000000,&H99000000,-1,0,0,0,100,100,1,0,1,8,5,5,60,60,0,1
Style: HookBox,{font},96,&H00000000,&H00000000,&H0000E5FF,&H0000E5FF,-1,0,0,0,100,100,1,0,3,18,0,5,60,60,0,1
Style: Stamp,{font},100,&H00FFFFFF,&H00FFFFFF,&H002A2AE8,&H002A2AE8,-1,0,0,0,100,100,2,0,3,20,0,5,60,60,0,1
Style: Count,{font},60,&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,1,0,3,14,0,5,60,60,0,1
Style: Plus,{font},54,&H006EFF3D,&H006EFF3D,&H00000000,&H99000000,-1,0,0,0,100,100,1,0,1,6,2,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    return header + "\n".join(ass.events) + "\n", tings
