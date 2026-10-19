"""Model-agnostic edit planning: one interface, one adapter per vendor SDK.

Every adapter takes the same inputs (system prompt, user prompt, JPEG contact
sheets, JSON schema) and returns the parsed JSON object. SDKs are imported
lazily so you only install the one you use.
"""
from __future__ import annotations

import base64
import json
import os
import re
from typing import Protocol


class ProviderError(RuntimeError):
    pass


class Provider(Protocol):
    name: str
    model: str

    def complete_json(self, system: str, prompt: str, images: list[bytes], schema: dict) -> dict: ...


DEFAULT_MODELS = {
    "claude": "claude-opus-5-5",
    # OpenAI and Gemini model names change often; override with --model or REELCUT_MODEL.
    "openai": "gpt-5",
    "gemini": "gemini-2.5-pro",
}

# Models that accept Anthropic's server-side refusal fallback ("default" routing).
_CLAUDE_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


def parse_json_text(text: str) -> dict:
    """Parse a model's JSON reply, tolerating ``` fences or prose around the object."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a == -1 or b <= a:
            raise ProviderError(f"model did not return JSON: {text[:300]!r}")
        return json.loads(text[a:b + 1])


def _missing(pkg: str, extra: str) -> ProviderError:
    return ProviderError(f"the '{pkg}' package is not installed. Run: pip install \"reelcut[{extra}]\"")


class ClaudeProvider:
    name = "claude"

    def __init__(self, model: str | None = None, base_url: str | None = None):
        try:
            import anthropic
        except ImportError as e:
            raise _missing("anthropic", "claude") from e
        self._anthropic = anthropic
        self.model = model or DEFAULT_MODELS["claude"]
        self.client = anthropic.Anthropic(base_url=base_url) if base_url else anthropic.Anthropic()

    def complete_json(self, system, prompt, images, schema):
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                 "data": base64.standard_b64encode(img).decode()}}
                   for img in images]
        content.append({"type": "text", "text": prompt})
        kwargs = dict(
            model=self.model,
            max_tokens=64000,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": "high", "format": {"type": "json_schema", "schema": schema}},
        )
        try:
            if self.model in _CLAUDE_FALLBACK_MODELS:
                # On a safety decline the API re-runs the request on a suitable fallback model.
                with self.client.beta.messages.stream(
                        betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs) as stream:
                    msg = stream.get_final_message()
            else:
                with self.client.messages.stream(**kwargs) as stream:
                    msg = stream.get_final_message()
        except self._anthropic.AuthenticationError as e:
            raise ProviderError("Anthropic rejected the credentials. Set ANTHROPIC_API_KEY.") from e
        except self._anthropic.NotFoundError as e:
            raise ProviderError(f"unknown Claude model '{self.model}'") from e
        except self._anthropic.APIStatusError as e:
            raise ProviderError(f"Claude API error {e.status_code}: {e.message}") from e
        except self._anthropic.APIConnectionError as e:
            raise ProviderError("could not reach the Claude API") from e
        if msg.stop_reason == "refusal":
            raise ProviderError("Claude declined to plan this video")
        if msg.stop_reason == "max_tokens":
            raise ProviderError("Claude ran out of output tokens before finishing the plan")
        text = next((b.text for b in msg.content if b.type == "text"), "")
        return parse_json_text(text)


class OpenAIProvider:
    """OpenAI, or any OpenAI-compatible server (LM Studio, Ollama, vLLM, Groq...) via base_url."""
    name = "openai"

    def __init__(self, model: str | None = None, base_url: str | None = None):
        try:
            import openai
        except ImportError as e:
            raise _missing("openai", "openai") from e
        self.model = model or DEFAULT_MODELS["openai"]
        key = os.environ.get("OPENAI_API_KEY") or ("not-needed" if base_url else None)
        self.client = openai.OpenAI(base_url=base_url, api_key=key)
        self._openai = openai

    def complete_json(self, system, prompt, images, schema):
        parts = [{"type": "text", "text": prompt}]
        parts += [{"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.standard_b64encode(img).decode()}} for img in images]
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": parts}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "edit_plan", "schema": schema, "strict": True}},
            )
        except self._openai.APIStatusError as e:
            raise ProviderError(f"OpenAI-compatible API error {e.status_code}: {e.message}") from e
        except self._openai.APIConnectionError as e:
            raise ProviderError("could not reach the OpenAI-compatible endpoint") from e
        choice = resp.choices[0]
        if getattr(choice.message, "refusal", None):
            raise ProviderError(f"model declined: {choice.message.refusal}")
        return parse_json_text(choice.message.content or "")


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str | None = None, base_url: str | None = None):
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise _missing("google-genai", "gemini") from e
        self._types = types
        self.model = model or DEFAULT_MODELS["gemini"]
        self.client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY

    def complete_json(self, system, prompt, images, schema):
        t = self._types
        contents = [t.Part.from_bytes(data=img, mime_type="image/jpeg") for img in images]
        # Gemini's response schema dialect is narrower than JSON Schema, so the schema
        # rides in the prompt and the reply is validated client-side.
        contents.append(prompt + "\n\nReturn JSON matching this JSON Schema exactly:\n" + json.dumps(schema))
        try:
            resp = self.client.models.generate_content(
                model=self.model, contents=contents,
                config=t.GenerateContentConfig(system_instruction=system, response_mime_type="application/json"))
        except Exception as e:  # google-genai raises google.genai.errors.APIError subclasses
            raise ProviderError(f"Gemini API error: {e}") from e
        return parse_json_text(resp.text or "")


PROVIDERS = {"claude": ClaudeProvider, "openai": OpenAIProvider, "gemini": GeminiProvider}


def detect_provider(env: dict | None = None) -> str:
    """Pick a provider from whichever API key is set; 'none' means the offline heuristic."""
    env = os.environ if env is None else env
    if env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN"):
        return "claude"
    if env.get("OPENAI_API_KEY"):
        return "openai"
    if env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY"):
        return "gemini"
    return "none"


def get_provider(name: str, model: str | None = None, base_url: str | None = None) -> Provider | None:
    if name == "auto":
        name = detect_provider()
    if name == "none":
        return None
    if name not in PROVIDERS:
        raise ProviderError(f"unknown provider '{name}'. Choose from: auto, none, {', '.join(PROVIDERS)}")
    return PROVIDERS[name](model=model or os.environ.get("REELCUT_MODEL"), base_url=base_url)
