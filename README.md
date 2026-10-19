# reelcut

Turn a raw talking-head clip into a captioned, jump-cut vertical reel for Instagram, TikTok or YouTube Shorts. Any LLM plans the edit (Claude, OpenAI, Gemini, or a local model) and ffmpeg renders it.

```
raw 4-min phone clip  ──►  reelcut make clip.mp4  ──►  1:50 reel, 1080×1920
```

What you get, in the style of fast-cut creator reels:

- **Jump cuts.** Pauses, stutters, restarted sentences and walk-offs are removed.
- **Punch-in zooms** alternate on every cut, with a slow push-in on longer takes.
- **Word-by-word captions.** Bold text, 2–3 words at a time; the spoken word lights up yellow and keywords are green.
- **A hook title** over the first 2.5 seconds.
- **Zoom-pop transitions** with a pop sound wherever footage was skipped (for example, you left the frame to fetch something).
- **A running counter gag** (`TECH COUNT: 7`, `+1 MACBOOK`) with a bell "ting" on every increment, when the video lists things.
- **Stamps** for punchlines (`NOT A FLEX`) and an **outro** with a comment prompt.
- **A colour grade** (`vintage`, `punchy`, `clean`), a vignette, and loudness normalised to −14 LUFS.

## How it works

1. **Transcribe.** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) runs locally and gives word-level timestamps. No API key is needed.
2. **Plan.** The transcript, the silence map and contact sheets (a thumbnail every 2 s) go to the model you choose. It returns a JSON **edit plan**: cuts, corrected caption text, hook, keywords, counter events, stamps, outro and face position. The plan is validated (clamped, sorted, overlaps removed) and saved as `plan.json`.
3. **Render.** ffmpeg cuts each segment, frames the face in 9:16, adds the zoom and pop effects, burns in the captions, mixes in the sound effects, applies the grade and normalises the loudness.

The model only writes a plan; it never touches the video. That makes the result reproducible and lets you fix the plan by hand.

## Install

You need **Python 3.10+** and **ffmpeg** on your PATH.

```bash
pip install "reelcut[all] @ git+https://github.com/<you>/reelcut"
```

Or install only the SDK you'll use: `reelcut[claude]`, `reelcut[openai]` or `reelcut[gemini]`.

## Use

```bash
# Uses whichever key is set: ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY / GOOGLE_API_KEY
reelcut make clip.mp4

# Choose a provider and model explicitly
reelcut make clip.mp4 --provider claude
reelcut make clip.mp4 --provider openai --model gpt-5
reelcut make clip.mp4 --provider gemini --model gemini-2.5-pro

# A local model through any OpenAI-compatible server (LM Studio, Ollama, vLLM)
reelcut make clip.mp4 --provider openai --base-url http://localhost:1234/v1 --model qwen2.5-vl-7b
#   add --no-images if the model can't read images

# No model at all: silence cutting and captions only
reelcut make clip.mp4 --provider none
```

### Review the plan before rendering

```bash
reelcut plan clip.mp4                 # writes clip.reelcut/plan.json
# edit the JSON: fix a caption, drop a cut, change the hook...
reelcut render clip.mp4               # renders clip.reelcut/plan.json
```

Re-renders only redo the cuts you changed. See [`examples/plan.example.json`](examples/plan.example.json) for the format.

### Options

| flag | default | |
|---|---|---|
| `--provider` | `auto` | `auto`, `claude`, `openai`, `gemini` or `none` |
| `--model` | per provider | also `$REELCUT_MODEL` |
| `--base-url` | | custom or local endpoint |
| `--grade` | `vintage` | `vintage`, `punchy`, `clean`, `none`, or any ffmpeg filter chain |
| `--font` | `Arial Black` | caption font; must be installed |
| `--pop-gap` | `3.0` | skipped seconds that trigger a zoom-pop |
| `--whisper-model` | `small` | `medium` or `large-v3` is more accurate but slower |
| `--language` | auto | e.g. `en`, `de` |
| `-o` | `<clip>_reel.mp4` | |

## Providers

| provider | SDK | default model | notes |
|---|---|---|---|
| `claude` | `anthropic` | `claude-opus-5-5` | structured JSON output; server-side refusal fallback is turned on |
| `openai` | `openai` | `gpt-5` | Chat Completions with a strict JSON schema; also works with OpenAI-compatible servers |
| `gemini` | `google-genai` | `gemini-2.5-pro` | JSON mode; the schema is sent in the prompt and the reply is checked by reelcut |

Model names change often. If a default has been retired, pass `--model`. To add a provider, implement one `complete_json(system, prompt, images, schema) -> dict` method in [`reelcut/providers.py`](reelcut/providers.py).

## Tips for footage that cuts well

- Film vertically, with the camera at chest height and your face in the upper third.
- Pause for a beat between thoughts. Clean gaps make clean cuts.
- When you fetch something, leave the frame and come back. reelcut turns that into a pop transition.

## Development

```bash
pip install -e ".[all,dev]"
pytest
```

The tests cover plan validation, the timeline and captions, cropping, and provider selection. They need no network and no ffmpeg.

## License

MIT
