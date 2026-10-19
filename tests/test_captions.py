import pytest

from reelcut.captions import build_ass, build_timeline, chunk, to_out, word_timings
from reelcut.plan import validate_plan
from reelcut.render import crop_box
from reelcut.media import VideoInfo


def _plan(**extra):
    raw = {"cuts": [{"start": 1.0, "end": 2.0, "text": "I have tech."},
                    {"start": 2.2, "end": 3.0, "text": "Lots."},
                    {"start": 10.0, "end": 11.0, "text": "A camera."}]}
    raw.update(extra)
    return validate_plan(raw, duration=20)


WORDS = [{"text": "I", "start": 1.0, "end": 1.2}, {"text": "have", "start": 1.2, "end": 1.5},
         {"text": "tech.", "start": 1.5, "end": 1.9}, {"text": "Lots.", "start": 2.3, "end": 2.8},
         {"text": "A", "start": 10.1, "end": 10.2}, {"text": "camera.", "start": 10.2, "end": 10.8}]


def test_timeline_is_contiguous_and_pops_after_long_gap():
    segs = build_timeline(_plan(), fps=30, pop_gap=3.0)
    assert [s.out_start for s in segs] == pytest.approx([0.0, 1.0, 1.8])
    assert [s.pop for s in segs] == [False, False, True]
    assert [s.punch for s in segs] == [False, True, False]


def test_to_out_maps_source_time_and_skips_removed_footage():
    segs = build_timeline(_plan())
    assert to_out(segs, 10.5) == pytest.approx(2.3)
    assert to_out(segs, 6.0) is None


def test_word_timings_use_asr_times_when_counts_match():
    seg = build_timeline(_plan())[0]
    assert word_timings(seg, WORDS)[1] == (1.2, 1.5)


def test_word_timings_spread_when_text_was_corrected():
    seg = build_timeline(validate_plan({"cuts": [{"start": 1, "end": 2, "text": "I've got tech"}]}, 5))[0]
    t = word_timings(seg, WORDS)
    assert len(t) == 3 and t[0][0] == pytest.approx(1.0) and t[-1][1] == pytest.approx(1.9)


def test_chunks_break_on_punctuation_and_three_words():
    assert chunk("one two three four, five".split()) == [[0, 1, 2], [3], [4]]


def test_ass_contains_captions_counter_and_ting_times():
    plan = _plan(keywords=["tech"], counter={"label": "Tech count", "events": [
        {"time": 10.3, "value": "1", "label": "+1 camera"}]}, hook={"line1": "I have a", "line2": "tech problem"})
    ass, tings = build_ass(plan, build_timeline(plan), WORDS)
    assert "TECH COUNT: 1" in ass and "+1 CAMERA" in ass and "TECH PROBLEM" in ass
    assert "{\\c&H006EFF3D&}TECH." in ass  # keyword highlighted; match ignores the full stop
    assert tings == pytest.approx([2.1])
    assert sum(line.startswith("Style: ") for line in ass.splitlines()) == 6


def test_braces_in_text_cannot_inject_ass_overrides():
    plan = validate_plan({"cuts": [{"start": 0, "end": 1, "text": "{\\pos(0,0)}hi"}]}, 5)
    ass, _ = build_ass(plan, build_timeline(plan), [])
    assert "{\\pos(0,0)}" not in ass


def test_crop_box_landscape_source_is_9x16_and_inside_frame():
    cw, ch, cx, cy = crop_box(VideoInfo(1920, 1080, 10, 30), face_x=0.9, face_y=0.3, zoom=1.0)
    assert ch == 1080 and abs(cw / ch - 9 / 16) < 0.01
    assert 0 <= cx <= 1920 - cw and cy == 0


def test_crop_box_zoom_shrinks_window():
    full = crop_box(VideoInfo(1080, 1920, 10, 30), 0.5, 0.4, 1.0)
    tight = crop_box(VideoInfo(1080, 1920, 10, 30), 0.5, 0.4, 1.28)
    assert tight[0] < full[0] and tight[1] < full[1]
