"""Local word-level transcription with faster-whisper (no API key needed)."""
from __future__ import annotations

import json
from pathlib import Path


def transcribe(audio: Path, out_json: Path, model: str = "small", language: str | None = None) -> list[dict]:
    """Write and return a flat word list: [{"text", "start", "end"}, ...]."""
    if out_json.exists():
        return json.loads(out_json.read_text(encoding="utf-8"))
    from faster_whisper import WhisperModel

    m = WhisperModel(model, device="auto", compute_type="int8")
    segments, _ = m.transcribe(str(audio), word_timestamps=True, language=language)
    words = [{"text": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)}
             for s in segments for w in (s.words or []) if w.word.strip()]
    out_json.write_text(json.dumps(words, ensure_ascii=False, indent=0), encoding="utf-8")
    return words
