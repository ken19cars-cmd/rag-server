"""Optional local OpenAI-compatible client for Notion analysis.

Query and structured Q&A stay extractive. This client is built only when a
base URL and model are both set, for example a local Ollama server.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

import httpx


class LlmError(RuntimeError):
    pass


class LlmClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(self, messages: list[dict[str, str]]) -> str:
        url = chat_completions_url(self.base_url)
        try:
            response = httpx.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "temperature": 0,
                    "messages": messages,
                },
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise LlmError(f"llm request failed: {exc}") from exc
        if response.status_code >= 400:
            raise LlmError(f"llm request failed: HTTP {response.status_code}")
        try:
            payload = response.json()
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LlmError("llm response did not include a message") from exc


def chat_completions_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def parse_json_object(text: str) -> dict | None:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if isinstance(value, dict):
        return value
    return None
