import pytest

from reelcut.providers import ProviderError, detect_provider, get_provider, parse_json_text


@pytest.mark.parametrize("env,expected", [
    ({"ANTHROPIC_API_KEY": "x", "OPENAI_API_KEY": "y"}, "claude"),
    ({"OPENAI_API_KEY": "y"}, "openai"),
    ({"GEMINI_API_KEY": "z"}, "gemini"),
    ({"GOOGLE_API_KEY": "z"}, "gemini"),
    ({}, "none"),
])
def test_detect_provider(env, expected):
    assert detect_provider(env) == expected


def test_none_provider_means_heuristic():
    assert get_provider("none") is None


def test_unknown_provider_is_a_clear_error():
    with pytest.raises(ProviderError, match="unknown provider"):
        get_provider("llama-of-doom")


@pytest.mark.parametrize("text", [
    '{"cuts": []}',
    '```json\n{"cuts": []}\n```',
    'Here is the plan:\n{"cuts": []}\nHope that helps.',
])
def test_parse_json_text_tolerates_wrapping(text):
    assert parse_json_text(text) == {"cuts": []}


def test_parse_json_text_rejects_prose():
    with pytest.raises(ProviderError):
        parse_json_text("I cannot do that.")
