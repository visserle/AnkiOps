"""Async OpenAI-compatible AI client for inline JSON note editing."""

from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Any

import httpx

from .errors import AIRequestError, AIResponseError
from .types import InlineEditedNote, InlineNotePayload, RuntimeAIConfig, TaskConfig
from .validators import normalize_batch_response

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_MIN_SECONDS = 0.25
DEFAULT_RETRY_MAX_SECONDS = 2.0
DEFAULT_RETRY_JITTER_FACTOR = 0.25
DEFAULT_RETRY_AFTER_MAX_SECONDS = 30.0


class _RetryableRequestError(RuntimeError):
    """Request failure class that should be retried."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
    ):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class OpenAICompatibleAsyncEditor:
    """Async OpenAI-compatible chat-completions client for inline JSON edits."""

    def __init__(self, config: RuntimeAIConfig):
        self._config = config
        self._endpoint = f"{self._config.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        self._headers = headers
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> OpenAICompatibleAsyncEditor:
        self._get_or_create_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        _ = exc_type
        _ = exc
        _ = traceback
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP client, if initialized."""
        client = self._client
        self._client = None
        if client is not None and not client.is_closed:
            await client.aclose()

    async def edit_notes(
        self,
        task_config: TaskConfig,
        notes: list[InlineNotePayload],
    ) -> dict[str, InlineEditedNote]:
        """Edit notes in batch and return edited payloads keyed by note_key."""
        if not notes:
            return {}

        payload = _build_chat_payload(
            model=self._config.model,
            task_config=task_config,
            notes=notes,
        )

        raw = await self._send_with_retry(
            client=self._get_or_create_client(),
            endpoint=self._endpoint,
            headers=self._headers,
            payload=payload,
        )

        content = _extract_assistant_content(raw)
        if not content:
            raise AIResponseError("AI response did not include assistant text content")

        return normalize_batch_response(content)

    def _get_or_create_client(self) -> httpx.AsyncClient:
        client = self._client
        if client is None or client.is_closed:
            client = httpx.AsyncClient(timeout=self._config.timeout_seconds)
            self._client = client
        return client

    async def _send_with_retry(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        last_error: _RetryableRequestError | None = None
        for attempt in range(1, DEFAULT_MAX_ATTEMPTS + 1):
            try:
                return await self._send_once(
                    client=client,
                    endpoint=endpoint,
                    headers=headers,
                    payload=payload,
                )
            except _RetryableRequestError as error:
                last_error = error
                if attempt >= DEFAULT_MAX_ATTEMPTS:
                    break
                await asyncio.sleep(
                    _retry_delay(
                        attempt,
                        retry_after_seconds=error.retry_after_seconds,
                    )
                )

        message = str(last_error) if last_error is not None else "AI request failed"
        raise AIRequestError(message) from last_error

    async def _send_once(
        self,
        *,
        client: httpx.AsyncClient,
        endpoint: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            response = await client.post(endpoint, headers=headers, json=payload)
        except httpx.TimeoutException as error:
            raise _RetryableRequestError(
                f"AI request timed out after {self._config.timeout_seconds}s"
            ) from error
        except httpx.HTTPError as error:
            raise _RetryableRequestError(f"AI request failed: {error}") from error

        if response.status_code == 429 or response.status_code >= 500:
            raise _RetryableRequestError(
                _http_error_message(response),
                retry_after_seconds=_retry_after_seconds(response),
            )
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
    task_config: TaskConfig,
    notes: list[InlineNotePayload],
) -> dict[str, Any]:
    allowed_fields = ", ".join(task_config.write_fields)
    context_fields = ", ".join(task_config.read_fields)
    requirements = [
        "Return JSON only.",
        "Return every edited note keyed by note_key.",
        "Do not change note_key values.",
        "Return the same fields structure per note.",
        f"Edit only these fields: {allowed_fields}.",
        f"You may use these fields for context: {context_fields}.",
    ]
    return {
        "model": model,
        "temperature": task_config.temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": task_config.instructions},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "inline_json_edit",
                        "requirements": requirements,
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


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    retry_after = raw.strip()
    if not retry_after:
        return None

    try:
        parsed_seconds = float(retry_after)
    except ValueError:
        parsed_seconds = None

    if parsed_seconds is not None:
        if parsed_seconds < 0:
            return None
        return min(parsed_seconds, DEFAULT_RETRY_AFTER_MAX_SECONDS)

    try:
        parsed_date = parsedate_to_datetime(retry_after)
    except (TypeError, ValueError, OverflowError):
        return None

    if parsed_date.tzinfo is None:
        parsed_date = parsed_date.replace(tzinfo=timezone.utc)
    delay_seconds = (parsed_date - datetime.now(timezone.utc)).total_seconds()
    if delay_seconds <= 0:
        return 0.0
    return min(delay_seconds, DEFAULT_RETRY_AFTER_MAX_SECONDS)


def _retry_delay(
    attempt: int,
    *,
    retry_after_seconds: float | None = None,
) -> float:
    if retry_after_seconds is not None:
        return retry_after_seconds
    backoff = DEFAULT_RETRY_MIN_SECONDS * float(2 ** (attempt - 1))
    jitter = random.uniform(0.0, backoff * DEFAULT_RETRY_JITTER_FACTOR)
    return float(min(DEFAULT_RETRY_MAX_SECONDS, backoff + jitter))
