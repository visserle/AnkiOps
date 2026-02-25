"""Async OpenAI-compatible AI client for inline JSON note editing."""

from __future__ import annotations

import json
from typing import Any

import httpx

from .types import PromptConfig, RuntimeAIConfig
from .validators import normalize_batch_response


class OpenAICompatibleAsyncEditor:
    """Async OpenAI-compatible chat-completions client for inline JSON edits."""

    def __init__(self, config: RuntimeAIConfig):
        self._config = config

    async def edit_batch(
        self,
        prompt_config: PromptConfig,
        note_payloads: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        if not note_payloads:
            return {}

        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        payload = {
            "model": self._config.model,
            "temperature": prompt_config.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": prompt_config.prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": "inline_json_edit_batch",
                            "requirements": [
                                "Return JSON only.",
                                "Return every edited note keyed by note_key.",
                                "Do not change note_key values.",
                                "Return the same fields structure per note.",
                            ],
                            "notes": note_payloads,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }

        async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
            )

        if response.status_code >= 400:
            raise ValueError(
                f"AI request failed ({response.status_code}): {response.text[:200]}"
            )

        raw = response.json()
        content = raw["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("AI response content is not text")
        return normalize_batch_response(content)
