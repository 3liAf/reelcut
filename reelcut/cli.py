"""reelcut command line.

  reelcut make clip.mp4                       # plan with whichever API key is set, then render
  reelcut make clip.mp4 --provider gemini
  reelcut make clip.mp4 --provider openai --base-url http://localhost:1234/v1 --model qwen2.5-vl
  reelcut plan clip.mp4                       # stop after writing plan.json, edit it by hand
  reelcut render clip.mp4 --plan my_plan.json # render an edited plan
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .media import FFmpegError, contact_sheets, extract_audio, probe, require_ffmpeg, silences
from .plan import PLAN_SCHEMA, SYSTEM_PROMPT, PlanError, build_prompt, heuristic_plan, load_plan, validate_plan
from .providers import ProviderError, get_provider
from .render import GRADES, render
from .transcribe import transcribe


def _workdir(src: Path, given: str | None) -> Path:
    wd = Path(given) if given else src.with_name(src.stem + ".reelcut")
    wd.mkdir(parents=True, exist_ok=True)
    return wd


def _prepare(src: Path, wd: Path, args) -> tuple:
    info = probe(src)
    audio = wd / "audio.wav"
    if not audio.exists():
        print("extracting audio")
        extract_audio(src, audio)
    print(f"transcribing with whisper '{args.whisper_model}' (cached after the first run)")
    words = transcribe(audio, wd / "transcript.json", args.whisper_model, args.language)
    return info, words


def _make_plan(src: Path, wd: Path, info, words, args) -> dict:
    provider = get_provider(args.provider, args.model, args.base_url)
    if provider is None:
        print("no model configured: using the offline heuristic planner (silence cutting + captions)")
        plan = heuristic_plan(words, info.duration)
    else:
        sil = silences(wd / "audio.wav")
        every = 2.0 if info.duration <= 300 else 4.0
        sheets = [] if args.no_images else contact_sheets(src, wd / "sheets", every, info.duration)
        prompt = build_prompt(words, sil, info.duration, info.width, info.height,
                              [(a, b) for _, a, b in sheets], every)
        images = [p.read_bytes() for p, _, _ in sheets]
        print(f"planning the edit with {provider.name} ({provider.model}), {len(images)} contact sheet(s)")
        raw = provider.complete_json(SYSTEM_PROMPT, prompt, images, PLAN_SCHEMA)
        (wd / "plan.raw.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
        plan = validate_plan(raw, info.duration)
    out = wd / "plan.json"
    out.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    kept = sum(c["end"] - c["start"] for c in plan["cuts"])
    print(f"plan: {len(plan['cuts'])} cuts, {kept:.1f}s kept of {info.duration:.1f}s -> {out}")
    return plan


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="reelcut", description="Raw talking-head clip -> captioned vertical reel.")
    p.add_argument("--version", action="version", version=f"reelcut {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("video", type=Path)
        sp.add_argument("--workdir", help="cache folder (default: <video>.reelcut next to the video)")
        sp.add_argument("--whisper-model", default="small", help="faster-whisper size: tiny/base/small/medium/large-v3")
        sp.add_argument("--language", help="spoken language code, e.g. en, de (default: auto-detect)")

    def planning(sp):
        sp.add_argument("--provider", default="auto", help="auto | claude | openai | gemini | none (default: auto)")
        sp.add_argument("--model", help="model id (default per provider, or $REELCUT_MODEL)")
        sp.add_argument("--base-url", help="custom endpoint, e.g. an OpenAI-compatible local server")
        sp.add_argument("--no-images", action="store_true", help="don't send contact sheets (text-only models)")

    def rendering(sp):
        sp.add_argument("-o", "--output", type=Path, help="output .mp4 (default: <video>_reel.mp4)")
        sp.add_argument("--grade", default="vintage", help=f"{' | '.join(GRADES)} or a raw ffmpeg filter chain")
        sp.add_argument("--font", default="Arial Black", help="caption font family installed on this system")
        sp.add_argument("--pop-gap", type=float, default=3.0,
                        help="seconds skipped between cuts that trigger a zoom-pop + pop sound")

    mk = sub.add_parser("make", help="transcribe, plan and render")
    common(mk); planning(mk); rendering(mk)
    pl = sub.add_parser("plan", help="transcribe and plan only; writes plan.json for review")
    common(pl); planning(pl)
    rd = sub.add_parser("render", help="render from an existing plan.json")
    common(rd); rendering(rd)
    rd.add_argument("--plan", type=Path, help="plan file (default: <workdir>/plan.json)")

    args = p.parse_args(argv)
    try:
        require_ffmpeg()
        src = args.video.resolve()
        if not src.exists():
            p.error(f"no such file: {src}")
        wd = _workdir(src, args.workdir)
        info, words = _prepare(src, wd, args)
        if args.cmd == "render":
            plan_path = args.plan or wd / "plan.json"
            if not plan_path.exists():
                p.error(f"no plan at {plan_path}; run 'reelcut plan' first")
            plan = validate_plan(load_plan(plan_path), info.duration)
        else:
            plan = _make_plan(src, wd, info, words, args)
        if args.cmd == "plan":
            return 0
        out = args.output or src.with_name(src.stem + "_reel.mp4")
        render(src, info, plan, words, wd, out, args.grade, args.font, args.pop_gap)
        print(f"done -> {out}")
        return 0
    except (FFmpegError, ProviderError, PlanError) as e:
        print(f"reelcut: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
