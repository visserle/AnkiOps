"""Async OpenAI-compatible AI client for inline JSON note editing."""

from __future__ import annotations

import json
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .errors import AIRequestError, AIResponseError
from .types import InlineEditedNote, InlineNotePayload, PromptConfig, RuntimeAIConfig
from .validators import normalize_batch_response

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_MIN_SECONDS = 0.25
DEFAULT_RETRY_MAX_SECONDS = 2.0


class _RetryableRequestError(RuntimeError):
    """Request failure class that should be retried."""


class OpenAICompatibleAsyncEditor:
    """Async OpenAI-compatible chat-completions client for inline JSON edits."""

    def __init__(self, config: RuntimeAIConfig):
        self._config = config

    async def edit_notes(
        self,
        prompt: PromptConfig,
        notes: list[InlineNotePayload],
    ) -> dict[str, InlineEditedNote]:
        """Edit notes in batch and return edited payloads keyed by note_key."""
        if not notes:
            return {}

        endpoint = f"{self._config.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        payload = _build_chat_payload(
            model=self._config.model,
            temperature=prompt.temperature,
            system_prompt=prompt.prompt,
            notes=notes,
        )

        raw = await self._send_with_retry(endpoint, headers, payload)

        content = _extract_assistant_content(raw)
        if not content:
            raise AIResponseError("AI response did not include assistant text content")

        return normalize_batch_response(content)

    async def _send_with_retry(
        self,
        endpoint: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(_RetryableRequestError),
            stop=stop_after_attempt(DEFAULT_MAX_ATTEMPTS),
            wait=wait_exponential(
                multiplier=DEFAULT_RETRY_MIN_SECONDS,
                min=DEFAULT_RETRY_MIN_SECONDS,
                max=DEFAULT_RETRY_MAX_SECONDS,
            ),
            reraise=True,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    return await self._send_once(endpoint, headers, payload)
            raise RuntimeError("retry loop exhausted without returning")
        except _RetryableRequestError as error:
            raise AIRequestError(str(error)) from error

    async def _send_once(
        self,
        endpoint: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                timeout=self._config.timeout_seconds
            ) as client:
                response = await client.post(endpoint, headers=headers, json=payload)
        except httpx.TimeoutException as error:
            raise _RetryableRequestError(
                f"AI request timed out after {self._config.timeout_seconds}s"
            ) from error
        except httpx.HTTPError as error:
            raise _RetryableRequestError(f"AI request failed: {error}") from error

        if response.status_code == 429 or response.status_code >= 500:
            raise _RetryableRequestError(_http_error_message(response))
        if response.status_code >= 400:
            raise AIRequestError(_http_error_message(response))

        try:
            raw = response.json()
        except ValueError as error:
            raise AIResponseError("AI response body was not valid JSON") from error
        if not isinstance(raw, dict):
            raise AIResponseError("AI response body must be a JSON object")
        return raw


def _build_chat_payload(
    *,
    model: str,
    temperature: float,
    system_prompt: str,
    notes: list[InlineNotePayload],
) -> dict[str, Any]:
    return {
        "model": model,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
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
                        "notes": [note.to_json() for note in notes],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }


def _extract_assistant_content(response: dict[str, Any]) -> str | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None

    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                text_parts.append(text)
        joined = "".join(text_parts).strip()
        return joined or None
    return None


def _http_error_message(response: httpx.Response) -> str:
    body = " ".join(response.text.split())[:200] or "<empty>"
    return f"AI request failed ({response.status_code}): {body}"
