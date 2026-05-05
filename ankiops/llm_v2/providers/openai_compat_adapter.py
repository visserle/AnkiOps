"""OpenAI-compatible chat completions adapter for strict structured output."""

from __future__ import annotations

import asyncio
import json
import random
import time

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, AuthenticationError

from ..domain.contracts import ContractValidationError
from ..domain.outcomes import ProviderOutcome, ProviderOutcomeKind, ProviderUsage
from .adapter_base import AdapterRequest

_TRANSIENT_STATUS_CODES = {408, 409, 429}
_PROVIDER_TRANSIENT_STATUS_CODES = {
    "groq": {498},
}
_MAX_RETRY_BACKOFF_SECONDS = 30.0


class OpenAICompatStructuredAdapter:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        provider: str,
        timeout_seconds: int = 60,
        retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        retry_backoff_jitter: bool = True,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._provider = provider.strip().lower()
        self._retries = retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._retry_backoff_jitter = retry_backoff_jitter
        self._owns_client = client is None
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(timeout_seconds),
            max_retries=0,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()

    async def generate(self, request: AdapterRequest) -> ProviderOutcome:
        request_kwargs: dict[str, object] = {
            "model": request.model_id,
            "max_tokens": request.max_output_tokens,
            "messages": [
                {"role": "system", "content": request.system_prompt.strip()},
                {
                    "role": "user",
                    "content": request.user_message_text
                    if request.user_message_text is not None
                    else _build_user_message(request),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request.contract.schema_name,
                    "strict": True,
                    "schema": request.contract.json_schema,
                },
            },
        }
        if request.temperature is not None:
            request_kwargs["temperature"] = request.temperature
        if self._provider == "ollama":
            request_kwargs["extra_body"] = {"think": False}

        response: object | None = None
        started_at = time.monotonic()
        retry_count = 0
        while True:
            try:
                response = await self._client.chat.completions.create(**request_kwargs)  # type: ignore[call-overload]
                break
            except AuthenticationError as error:
                return ProviderOutcome(
                    kind=ProviderOutcomeKind.FATAL_ERROR,
                    error_message=f"Provider authentication failed: {error}",
                )
            except APIConnectionError as error:
                if retry_count >= self._retries:
                    return ProviderOutcome(
                        kind=ProviderOutcomeKind.FATAL_ERROR,
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
            except APIStatusError as error:
                status_code = _status_code(error)
                status_text = str(status_code) if status_code is not None else "unknown"
                if _is_retryable_status_error(status_code, provider=self._provider):
                    if retry_count >= self._retries:
                        return ProviderOutcome(
                            kind=ProviderOutcomeKind.PROVIDER_ERROR,
                            error_message=(
                                f"Provider returned HTTP {status_text}: {error}"
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
                return ProviderOutcome(
                    kind=ProviderOutcomeKind.PROVIDER_ERROR,
                    error_message=f"Provider returned HTTP {status_text}: {error}",
                )
            except Exception as error:
                return ProviderOutcome(
                    kind=ProviderOutcomeKind.FATAL_ERROR,
                    error_message=f"Unexpected provider error: {error}",
                )

        if response is None:
            return ProviderOutcome(
                kind=ProviderOutcomeKind.FATAL_ERROR,
                error_message="Provider response was not captured",
            )

        latency_ms = round((time.monotonic() - started_at) * 1000)
        usage = getattr(response, "usage", None)
        provider_message_id = _as_non_empty_str(getattr(response, "id", None))
        response_model_id = (
            _as_non_empty_str(getattr(response, "model", None)) or request.model_id
        )

        stop_reason, raw_text = _extract_stop_reason_and_text(response)
        raw_json = _response_to_json(response)

        if stop_reason == "content_filter":
            return ProviderOutcome(
                kind=ProviderOutcomeKind.REFUSAL,
                refusal_text=(
                    raw_text or "Model output was blocked by content filtering"
                ),
                provider_message_id=provider_message_id,
                response_model_id=response_model_id,
                usage=ProviderUsage(
                    input_tokens=_usage_value(usage, "prompt_tokens"),
                    output_tokens=_usage_value(usage, "completion_tokens"),
                ),
                latency_ms=latency_ms,
                raw_text=raw_text,
                raw_json=raw_json,
            )

        if not raw_text:
            return ProviderOutcome(
                kind=ProviderOutcomeKind.PROVIDER_ERROR,
                error_message="Provider response contained no JSON text output",
                provider_message_id=provider_message_id,
                response_model_id=response_model_id,
                usage=ProviderUsage(
                    input_tokens=_usage_value(usage, "prompt_tokens"),
                    output_tokens=_usage_value(usage, "completion_tokens"),
                ),
                latency_ms=latency_ms,
                raw_json=raw_json,
            )

        try:
            update = request.contract.parse_raw_json(raw_text)
        except ContractValidationError as error:
            return ProviderOutcome(
                kind=ProviderOutcomeKind.VALIDATION_ERROR,
                error_message=str(error),
                provider_message_id=provider_message_id,
                response_model_id=response_model_id,
                usage=ProviderUsage(
                    input_tokens=_usage_value(usage, "prompt_tokens"),
                    output_tokens=_usage_value(usage, "completion_tokens"),
                ),
                latency_ms=latency_ms,
                raw_text=raw_text,
                raw_json=raw_json,
            )

        return ProviderOutcome(
            kind=ProviderOutcomeKind.SUCCESS,
            update=update,
            provider_message_id=provider_message_id,
            response_model_id=response_model_id,
            usage=ProviderUsage(
                input_tokens=_usage_value(usage, "prompt_tokens"),
                output_tokens=_usage_value(usage, "completion_tokens"),
            ),
            latency_ms=latency_ms,
            raw_text=raw_text,
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


def _extract_stop_reason_and_text(response: object) -> tuple[str | None, str | None]:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        return None, None

    choice = choices[0]
    stop_reason = getattr(choice, "finish_reason", None)
    stop_reason_text = (
        stop_reason if isinstance(stop_reason, str) and stop_reason else None
    )

    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return stop_reason_text, content

    if not isinstance(content, list):
        return stop_reason_text, None

    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
            continue
        text = getattr(item, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return stop_reason_text, "".join(parts) if parts else None


def _usage_value(usage: object, name: str) -> int:
    value = getattr(usage, name, 0)
    return value if isinstance(value, int) else 0


def _as_non_empty_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _status_code(error: APIStatusError) -> int | None:
    value = getattr(error, "status_code", None)
    return value if isinstance(value, int) else None


def _is_retryable_status_error(status_code: int | None, *, provider: str) -> bool:
    if status_code is None:
        return False
    if status_code in _TRANSIENT_STATUS_CODES:
        return True
    if status_code >= 500:
        return True
    provider_codes = _PROVIDER_TRANSIENT_STATUS_CODES.get(provider, set())
    return status_code in provider_codes


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


def _response_to_json(response: object) -> str | None:
    model_dump_json = getattr(response, "model_dump_json", None)
    if callable(model_dump_json):
        try:
            value = model_dump_json()
            return value if isinstance(value, str) else None
        except Exception:
            return None

    to_json = getattr(response, "to_json", None)
    if callable(to_json):
        try:
            value = to_json()
            return value if isinstance(value, str) else None
        except Exception:
            return None

    return None
