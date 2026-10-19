import json
from pathlib import Path

import pytest

from reelcut.plan import PLAN_SCHEMA, PlanError, build_prompt, heuristic_plan, validate_plan

EXAMPLE = Path(__file__).parent.parent / "examples" / "plan.example.json"


def test_validate_sorts_clamps_and_removes_overlaps():
    plan = validate_plan({"cuts": [
        {"start": 5, "end": 7, "text": "second"},
        {"start": -1, "end": 2, "text": "first"},
        {"start": 6.5, "end": 9, "text": "overlaps"},
        {"start": 9, "end": 9.05, "text": "too short"},
        {"start": 58, "end": 99, "text": "past the end"},
    ]}, duration=60)
    assert [(c["start"], c["end"]) for c in plan["cuts"]] == [(0, 2), (5, 7), (7, 9), (58, 60)]


def test_validate_rejects_empty_plan():
    with pytest.raises(PlanError):
        validate_plan({"cuts": []}, duration=10)


def test_counter_and_stamps_outside_cuts_are_dropped():
    plan = validate_plan({
        "cuts": [{"start": 1, "end": 3, "text": "a"}],
        "counter": {"label": "tech count", "events": [
            {"time": 2, "value": "1", "label": "+1 CAMERA"},
            {"time": 20, "value": "2", "label": "+1 PHONE"}]},
        "stamps": [{"time": 2.5, "text": "NOT A FLEX"}, {"time": 30, "text": "lost"}],
    }, duration=40)
    assert [e["value"] for e in plan["counter"]["events"]] == ["1"]
    assert plan["counter"]["label"] == "TECH COUNT"
    assert [s["text"] for s in plan["stamps"]] == ["NOT A FLEX"]


def test_counter_label_cleared_when_no_events():
    plan = validate_plan({"cuts": [{"start": 0, "end": 1, "text": ""}],
                          "counter": {"label": "X", "events": []}}, duration=5)
    assert plan["counter"]["label"] == ""


def test_framing_defaults_and_clamps():
    plan = validate_plan({"cuts": [{"start": 0, "end": 1, "text": ""}],
                          "framing": [{"start": 0, "end": 1, "face_x": 3, "face_y": -1, "subject_scale": "huge"}]},
                         duration=5)
    f = plan["framing"][0]
    assert (f["face_x"], f["face_y"], f["subject_scale"]) == (1.0, 0.0, "medium")
    assert validate_plan({"cuts": [{"start": 0, "end": 1, "text": ""}]}, 5)["framing"][0]["face_x"] == 0.5


def test_heuristic_drops_pauses_and_filler():
    words = [
        {"text": "Okay,", "start": 0.0, "end": 0.4},
        {"text": "I", "start": 2.0, "end": 2.1}, {"text": "have", "start": 2.1, "end": 2.3},
        {"text": "tech.", "start": 2.3, "end": 2.6},
        {"text": "Too", "start": 6.0, "end": 6.2}, {"text": "much.", "start": 6.2, "end": 6.5},
    ]
    plan = heuristic_plan(words, duration=8)
    assert [c["text"] for c in plan["cuts"]] == ["I have tech.", "Too much."]
    assert plan["cuts"][0]["start"] == pytest.approx(1.94)


def test_schema_is_strict_mode_compatible():
    def walk(node):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
            for child in node["properties"].values():
                walk(child)
        if node.get("type") == "array":
            walk(node["items"])
    walk(PLAN_SCHEMA)


def test_example_plan_is_valid():
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    plan = validate_plan(raw, duration=60)
    assert len(plan["cuts"]) == len(raw["cuts"])


def test_prompt_mentions_every_word_and_sheet_range():
    words = [{"text": "hello", "start": 1.0, "end": 1.4}]
    prompt = build_prompt(words, [(0, 1)], 30, 1080, 1920, [(0, 30)], 2)
    assert "[1.00-1.40] hello" in prompt and "0-30s" in prompt
