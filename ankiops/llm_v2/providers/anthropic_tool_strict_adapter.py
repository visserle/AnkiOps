"""Anthropic Messages API adapter for strict tool-use structured output."""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Mapping
from typing import Protocol

import requests

from ..domain.contracts import ContractValidationError
from ..domain.outcomes import ProviderOutcome, ProviderOutcomeKind, ProviderUsage
from .adapter_base import AdapterRequest

_TRANSIENT_STATUS_CODES = {408, 409, 429}
_MAX_RETRY_BACKOFF_SECONDS = 30.0
_ANTHROPIC_VERSION = "2023-06-01"


class _HttpPost(Protocol):
    def __call__(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: dict[str, object],
        timeout: float,
    ) -> requests.Response: ...


class AnthropicToolStrictAdapter:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: int = 60,
        retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        retry_backoff_jitter: bool = True,
        http_post: _HttpPost | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout_seconds = float(timeout_seconds)
        self._retries = retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._retry_backoff_jitter = retry_backoff_jitter
        self._http_post = http_post or requests.post

    async def close(self) -> None:
        return None

    async def generate(self, request: AdapterRequest) -> ProviderOutcome:
        started_at = time.monotonic()
        user_message_text = (
            request.user_message_text
            if request.user_message_text is not None
            else _build_user_message(request)
        )
        url = _messages_url(self._base_url)
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        payload: dict[str, object] = {
            "model": request.model_id,
            "max_tokens": request.max_output_tokens,
            "system": request.system_prompt.strip(),
            "messages": [{"role": "user", "content": user_message_text}],
            "tools": [
                {
                    "name": request.contract.schema_name,
                    "description": "Return the structured note update payload",
                    "input_schema": request.contract.json_schema,
                }
            ],
            "tool_choice": {
                "type": "tool",
                "name": request.contract.schema_name,
            },
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature

        response: requests.Response | None = None
        retry_count = 0
        while True:
            try:
                response = await asyncio.to_thread(
                    self._http_post,
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout_seconds,
                )
            except requests.RequestException as error:
                if retry_count >= self._retries:
                    return ProviderOutcome(
                        kind=ProviderOutcomeKind.PROVIDER_ERROR,
                        error_message=f"Provider connection error: {error}",
                    )
                retry_count += 1
                await asyncio.sleep(
                    _retry_delay_seconds(
                        base_seconds=self._retry_backoff_seconds,
                        retry_count=retry_count,
                        jitter=self._retry_backoff_jitter,
                    )
                )
                continue

            status_code = response.status_code
            if _is_retryable_status(status_code):
                if retry_count >= self._retries:
                    return ProviderOutcome(
                        kind=ProviderOutcomeKind.PROVIDER_ERROR,
                        error_message=(
                            f"Provider returned HTTP {status_code}: "
                            f"{_response_text_excerpt(response)}"
                        ),
                    )
                retry_count += 1
                await asyncio.sleep(
                    _retry_delay_seconds(
                        base_seconds=self._retry_backoff_seconds,
                        retry_count=retry_count,
                        jitter=self._retry_backoff_jitter,
                    )
                )
                continue

            break

        if response is None:
            return ProviderOutcome(
                kind=ProviderOutcomeKind.FATAL_ERROR,
                error_message="Provider response was not captured",
            )

        status_code = response.status_code
        if status_code in {401, 403}:
            return ProviderOutcome(
                kind=ProviderOutcomeKind.FATAL_ERROR,
                error_message=(
                    "Provider authentication failed with HTTP "
                    f"{status_code}"
                ),
            )

        if status_code >= 400:
            return ProviderOutcome(
                kind=ProviderOutcomeKind.PROVIDER_ERROR,
                error_message=(
                    f"Provider returned HTTP {status_code}: "
                    f"{_response_text_excerpt(response)}"
                ),
            )

        response_data = _safe_json(response)
        if response_data is None:
            return ProviderOutcome(
                kind=ProviderOutcomeKind.PROVIDER_ERROR,
                error_message="Provider response was not valid JSON",
            )

        latency_ms = round((time.monotonic() - started_at) * 1000)
        raw_json = _response_json(response, response_data)
        provider_message_id = _as_non_empty_str(response_data.get("id"))
        response_model_id = (
            _as_non_empty_str(response_data.get("model")) or request.model_id
        )
        request_id = _as_non_empty_str(
            response.headers.get("request-id")
            or response.headers.get("x-request-id")
        )

        usage_raw = response_data.get("usage")
        usage = ProviderUsage(
            input_tokens=_mapping_int(usage_raw, "input_tokens"),
            output_tokens=_mapping_int(usage_raw, "output_tokens"),
        )

        stop_reason = _as_non_empty_str(response_data.get("stop_reason"))
        text_parts, tool_input = _extract_content(
            response_data.get("content"),
            tool_name=request.contract.schema_name,
        )

        if tool_input is not None:
            raw_text = json.dumps(
                tool_input,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            try:
                update = request.contract.parse_data(tool_input)
            except ContractValidationError as error:
                return ProviderOutcome(
                    kind=ProviderOutcomeKind.VALIDATION_ERROR,
                    error_message=str(error),
                    provider_message_id=provider_message_id,
                    response_model_id=response_model_id,
                    request_id=request_id,
                    usage=usage,
                    latency_ms=latency_ms,
                    raw_text=raw_text,
                    raw_json=raw_json,
                )
            return ProviderOutcome(
                kind=ProviderOutcomeKind.SUCCESS,
                update=update,
                provider_message_id=provider_message_id,
                response_model_id=response_model_id,
                request_id=request_id,
                usage=usage,
                latency_ms=latency_ms,
                raw_text=raw_text,
                raw_json=raw_json,
            )

        text_output = "\n".join(text_parts).strip() if text_parts else None
        if stop_reason == "refusal":
            return ProviderOutcome(
                kind=ProviderOutcomeKind.REFUSAL,
                refusal_text=text_output or "Provider refused request",
                provider_message_id=provider_message_id,
                response_model_id=response_model_id,
                request_id=request_id,
                usage=usage,
                latency_ms=latency_ms,
                raw_text=text_output,
                raw_json=raw_json,
            )

        if text_output:
            try:
                update = request.contract.parse_raw_json(text_output)
            except ContractValidationError as error:
                return ProviderOutcome(
                    kind=ProviderOutcomeKind.VALIDATION_ERROR,
                    error_message=str(error),
                    provider_message_id=provider_message_id,
                    response_model_id=response_model_id,
                    request_id=request_id,
                    usage=usage,
                    latency_ms=latency_ms,
                    raw_text=text_output,
                    raw_json=raw_json,
                )
            return ProviderOutcome(
                kind=ProviderOutcomeKind.SUCCESS,
                update=update,
                provider_message_id=provider_message_id,
                response_model_id=response_model_id,
                request_id=request_id,
                usage=usage,
                latency_ms=latency_ms,
                raw_text=text_output,
                raw_json=raw_json,
            )

        return ProviderOutcome(
            kind=ProviderOutcomeKind.PROVIDER_ERROR,
            error_message=(
                "Provider returned no structured tool payload"
                + (f" (stop_reason={stop_reason})" if stop_reason else "")
            ),
            provider_message_id=provider_message_id,
            response_model_id=response_model_id,
            request_id=request_id,
            usage=usage,
            latency_ms=latency_ms,
            raw_json=raw_json,
        )


def _build_user_message(request: AdapterRequest) -> str:
    payload: dict[str, object] = {
        "note_key": request.note_payload.note_key,
        "note_type": request.note_payload.note_type,
        "editable_fields": request.note_payload.editable_fields,
    }
    if request.note_payload.read_only_fields:
        payload["read_only_fields"] = request.note_payload.read_only_fields
    note_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return (
        f"<task>\n{request.task_prompt.strip()}\n</task>\n\n"
        f"<note>\n{note_json}\n</note>"
    )


def _messages_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/messages"):
        return stripped
    return f"{stripped}/messages"


def _safe_json(response: requests.Response) -> dict[str, object] | None:
    try:
        value = response.json()
    except ValueError:
        return None
    if not isinstance(value, dict):
        return None
    return value


def _response_json(
    response: requests.Response,
    response_data: dict[str, object],
) -> str:
    if response.text:
        return response.text
    return json.dumps(response_data, ensure_ascii=False, separators=(",", ":"))


def _response_text_excerpt(response: requests.Response, limit: int = 300) -> str:
    text = response.text.strip()
    if not text:
        return "<empty body>"
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _extract_content(
    content: object,
    *,
    tool_name: str,
) -> tuple[list[str], object | None]:
    if not isinstance(content, list):
        return [], None

    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        if block_type == "tool_use" and block.get("name") == tool_name:
            return text_parts, block.get("input")

        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)

    return text_parts, None


def _mapping_int(value: object, key: str) -> int:
    if not isinstance(value, Mapping):
        return 0
    item = value.get(key)
    return item if isinstance(item, int) else 0


def _as_non_empty_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _is_retryable_status(status_code: int) -> bool:
    return status_code in _TRANSIENT_STATUS_CODES or status_code >= 500


def _retry_delay_seconds(
    *,
    base_seconds: float,
    retry_count: int,
    jitter: bool,
) -> float:
    delay = min(
        base_seconds * (2 ** max(retry_count - 1, 0)),
        _MAX_RETRY_BACKOFF_SECONDS,
    )
    if jitter:
        delay = min(
            delay * random.uniform(0.5, 1.5),
            _MAX_RETRY_BACKOFF_SECONDS,
        )
    return max(delay, 0.0)
