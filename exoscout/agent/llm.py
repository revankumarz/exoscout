"""Minimal OpenAI-compatible LLM client.

Works with any endpoint that speaks the OpenAI /chat/completions API and
supports tool calling: local Ollama (`/v1`), llama.cpp server, Groq, OpenRouter,
OpenAI, etc. Configured entirely by environment variables so no key is baked in:

    EXOSCOUT_LLM_BASE_URL   default http://localhost:11434/v1  (Ollama)
    EXOSCOUT_LLM_MODEL      default llama3.1
    EXOSCOUT_LLM_API_KEY    default "ollama" (placeholder; real key for hosted)

Uses `requests` only - no vendor SDK dependency.
"""

from __future__ import annotations

import os
import requests

DEFAULT_BASE = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3.2"


class LLMConfig:
    def __init__(self) -> None:
        self.base_url = os.environ.get("EXOSCOUT_LLM_BASE_URL", DEFAULT_BASE).rstrip("/")
        self.model = os.environ.get("EXOSCOUT_LLM_MODEL", DEFAULT_MODEL)
        self.api_key = os.environ.get("EXOSCOUT_LLM_API_KEY", "ollama")

    def describe(self) -> str:
        return f"{self.model} @ {self.base_url}"


class LLMClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.cfg = config or LLMConfig()

    def available(self, timeout: float = 2.0) -> bool:
        """Cheap reachability probe so the app can fall back gracefully."""
        try:
            r = requests.get(f"{self.cfg.base_url}/models",
                             headers={"Authorization": f"Bearer {self.cfg.api_key}"},
                             timeout=timeout)
            return r.status_code < 500
        except requests.RequestException:
            return False

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             timeout: float = 120.0) -> dict:
        """One chat completion. Returns {'content': str|None, 'tool_calls': list}."""
        payload = {"model": self.cfg.model, "messages": messages, "temperature": 0.1}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        r = requests.post(f"{self.cfg.base_url}/chat/completions",
                          headers={"Authorization": f"Bearer {self.cfg.api_key}",
                                   "Content-Type": "application/json"},
                          json=payload, timeout=timeout)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        return {"content": msg.get("content"), "tool_calls": msg.get("tool_calls") or [],
                "raw": msg}
