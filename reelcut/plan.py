"""The edit plan: its JSON schema, the prompt that asks a model for one, validation,
and an offline heuristic planner used when no model is configured.

A plan is plain JSON (see examples/plan.example.json). Every time is in SOURCE
seconds; the renderer maps them onto the cut timeline.
"""
from __future__ import annotations

import json
import re

_STR = {"type": "string"}
_NUM = {"type": "number"}


def _obj(props: dict) -> dict:
    # additionalProperties:false + every property required = valid for OpenAI and Claude strict modes
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


PLAN_SCHEMA = _obj({
    "hook": _obj({"line1": _STR, "line2": _STR}),
    "cuts": {"type": "array", "items": _obj({"start": _NUM, "end": _NUM, "text": _STR})},
    "keywords": {"type": "array", "items": _STR},
    "counter": _obj({
        "label": _STR,
        "events": {"type": "array", "items": _obj({"time": _NUM, "value": _STR, "label": _STR})},
    }),
    "stamps": {"type": "array", "items": _obj({"time": _NUM, "text": _STR})},
    "outro": _obj({"line1": _STR, "line2": _STR}),
    "framing": {"type": "array", "items": _obj({
        "start": _NUM, "end": _NUM, "face_x": _NUM, "face_y": _NUM,
        "subject_scale": {"type": "string", "enum": ["small", "medium", "large"]},
    })},
})

SYSTEM_PROMPT = """You are a short-form video editor who cuts raw talking-head footage into \
Instagram Reels / TikToks in the style of top creators: tight jump cuts, no dead air, a hook in \
the first second, bold word-by-word captions, and one running visual gag when the content allows it.

You never invent speech. Every caption must be words the speaker actually says inside that cut. \
You return only the JSON edit plan."""

PROMPT_TEMPLATE = """Plan the edit for this clip.

SOURCE
- duration: {duration:.2f}s, frame {width}x{height} (output is 9:16 vertical)
- word-level transcript (source seconds, automatic speech recognition, may contain mishearings):
{words}
- silent stretches (start-end): {silences}
- contact sheets: {sheets}

RULES FOR "cuts" (the most important part)
1. Each cut is a contiguous source range [start, end] that will be kept. Cuts must be in order and not overlap.
2. Start ~0.05s before the first word you keep and end ~0.08s after the last; never slice through a word.
3. Remove: silences, stutters and restarted sentences (keep only the clean take), filler like
   "okay"/"um"/"so" at sentence starts, and stretches where the speaker walks away or nothing happens.
   Keep self-corrections or asides only when they are funny.
4. Prefer several short cuts (0.5-5s) over long ones; split long sentences at natural pauses so the
   renderer can alternate zoom levels between cuts.
5. "text" = the exact words spoken in that range, with ASR mistakes corrected from context
   (names, products, slang). Keep it verbatim otherwise. Use "" only for a cut with no speech.
6. When the speaker leaves and comes back (a gap of a few seconds between cuts), the renderer adds a
   quick zoom-pop transition automatically. Do not keep the walking.
7. Total length is usually 40-70% of the source.

OTHER FIELDS
- hook: two short ALL-CAPS lines (2-4 words each) that tease the topic, shown over the first 2.5s.
- keywords: 10-40 single UPPERCASE words from the captions to highlight (topic nouns, numbers, punchlines).
- counter: if the speaker lists or shows items one by one, a running counter gag. label like
  "TECH COUNT" / "MISTAKES" / "DAYS"; events at the source second each item is named, value is the
  running total as text (a final "TOO MANY" is fine), label like "+1 CAMERA". Otherwise label "" and no events.
- stamps: 0-3 rubber-stamp moments for punchlines or repeated denials, e.g. "NOT A FLEX"; time = source second.
- outro: line1 echoes the ending punchline (or ""), line2 is a comment prompt, e.g. "TELL ME I'M NOT THE ONLY ONE".
- framing: cover the whole clip with ranges giving where the speaker's face is in the source frame
  (face_x, face_y from 0 to 1, read them off the contact sheets) and subject_scale: "small" if the person
  fills less than about a third of the frame height, "large" for a close-up, else "medium".
"""


def build_prompt(words: list[dict], silences: list[tuple[float, float]], duration: float,
                 width: int, height: int, sheets: list[tuple[float, float]], every: float) -> str:
    wl = "\n".join(f"  [{w['start']:.2f}-{w['end']:.2f}] {w['text']}" for w in words) or "  (no speech detected)"
    sl = ", ".join(f"{a:.2f}-{b:.2f}" for a, b in silences) or "none"
    if sheets:
        sh = (f"{len(sheets)} image(s) attached in order; each is a 10-column grid with one thumbnail every "
              f"{every:g}s read left-to-right, top-to-bottom, stamped with its source second. Ranges: "
              + ", ".join(f"{a:.0f}-{b:.0f}s" for a, b in sheets))
    else:
        sh = "none attached (estimate framing as centered, face_y 0.35)"
    return PROMPT_TEMPLATE.format(duration=duration, width=width, height=height, words=wl, silences=sl, sheets=sh)


class PlanError(ValueError):
    pass


def _num(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def validate_plan(raw: dict, duration: float) -> dict:
    """Clamp, sort and de-overlap a model's plan so the renderer can trust it."""
    if not isinstance(raw, dict):
        raise PlanError("plan must be a JSON object")
    cuts = []
    for c in raw.get("cuts") or []:
        s = min(max(_num(c.get("start")), 0.0), duration)
        e = min(max(_num(c.get("end")), 0.0), duration)
        if e - s >= 0.15:
            cuts.append({"start": round(s, 3), "end": round(e, 3), "text": str(c.get("text") or "").strip()})
    cuts.sort(key=lambda c: c["start"])
    clean = []
    for c in cuts:
        if clean and c["start"] < clean[-1]["end"]:
            c["start"] = clean[-1]["end"]
            if c["end"] - c["start"] < 0.15:
                continue
        clean.append(c)
    if not clean:
        raise PlanError("plan has no usable cuts")

    def inside(t: float) -> bool:
        return any(c["start"] - 0.05 <= t <= c["end"] + 0.05 for c in clean)

    counter = raw.get("counter") or {}
    events = [{"time": _num(ev.get("time")), "value": str(ev.get("value", "")).strip(),
               "label": str(ev.get("label", "")).strip()} for ev in counter.get("events") or []]
    events = sorted([ev for ev in events if ev["value"] and inside(ev["time"])], key=lambda ev: ev["time"])
    stamps = [{"time": _num(s.get("time")), "text": str(s.get("text", "")).strip()} for s in raw.get("stamps") or []]
    stamps = [s for s in stamps if s["text"] and inside(s["time"])]

    framing = []
    for f in raw.get("framing") or []:
        scale = f.get("subject_scale") if f.get("subject_scale") in ("small", "medium", "large") else "medium"
        framing.append({"start": _num(f.get("start")), "end": _num(f.get("end"), duration),
                        "face_x": min(max(_num(f.get("face_x"), 0.5), 0.0), 1.0),
                        "face_y": min(max(_num(f.get("face_y"), 0.35), 0.0), 1.0),
                        "subject_scale": scale})
    if not framing:
        framing = [{"start": 0.0, "end": duration, "face_x": 0.5, "face_y": 0.35, "subject_scale": "medium"}]

    hook, outro = raw.get("hook") or {}, raw.get("outro") or {}
    return {
        "hook": {"line1": str(hook.get("line1", "")).upper(), "line2": str(hook.get("line2", "")).upper()},
        "cuts": clean,
        "keywords": sorted({re.sub(r"[^A-Z0-9\-']", "", str(k).upper()) for k in raw.get("keywords") or []} - {""}),
        "counter": {"label": str(counter.get("label", "")).upper() if events else "", "events": events},
        "stamps": stamps,
        "outro": {"line1": str(outro.get("line1", "")).upper(), "line2": str(outro.get("line2", "")).upper()},
        "framing": sorted(framing, key=lambda f: f["start"]),
    }


FILLERS = {"okay", "ok", "um", "uh", "uhm", "erm", "hmm", "like", "so", "yeah", "right"}


def heuristic_plan(words: list[dict], duration: float, gap: float = 0.45) -> dict:
    """No-model fallback: keep speech, drop pauses longer than `gap` and filler-only phrases."""
    phrases: list[list[dict]] = []
    for w in words:
        if phrases and w["start"] - phrases[-1][-1]["end"] <= gap:
            phrases[-1].append(w)
        else:
            phrases.append([w])
    cuts = []
    for ph in phrases:
        if all(re.sub(r"[^a-z]", "", w["text"].lower()) in FILLERS for w in ph):
            continue
        cuts.append({"start": max(ph[0]["start"] - 0.06, 0), "end": min(ph[-1]["end"] + 0.1, duration),
                     "text": " ".join(w["text"] for w in ph)})
    if not cuts:
        cuts = [{"start": 0.0, "end": duration, "text": ""}]
    return validate_plan({"cuts": cuts}, duration)


def load_plan(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
