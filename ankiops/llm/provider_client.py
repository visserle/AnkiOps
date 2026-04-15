"""Unified OpenAI-protocol client for note editing tasks."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, AuthenticationError

from .llm_errors import LlmFatalError, LlmNoteError, LlmNoteErrorCategory
from .prompting import build_system_prompt, build_user_message
from .structured_output import (
    NoteUpdateContract,
    StructuredOutputError,
    build_note_update_contract,
    parse_note_update_json,
)
from .task_types import (
    PreparedAttemptRequest,
    ProviderAttemptErrorContext,
    ProviderAttemptOutcome,
    TaskConfig,
    TaskRequestOptions,
)

logger = logging.getLogger(__name__)
_MAX_RETRY_BACKOFF_SECONDS = 30.0
_TRANSIENT_STATUS_CODES = {408, 409, 429}
_PROVIDER_TRANSIENT_STATUS_CODES = {
    # Groq can return 498 when flex capacity is exhausted.
    "groq": {498},
}
_NON_RETRYABLE_429_MARKERS = {
    "default": (
        "insufficient_quota",
        "billing_hard_limit_reached",
        "quota exhausted",
        "insufficient credits",
        "credit balance",
    ),
    # OpenRouter often maps exhausted credit state to payment-required semantics.
    "openrouter": ("payment required",),
}
_RESET_DURATION_PART_PATTERN = re.compile(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>[hms])")
_THROTTLE_RESET_KEYS = (
    ("x-ratelimit-remaining-requests", "x-ratelimit-reset-requests"),
    ("x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens"),
    ("anthropic-ratelimit-requests-remaining", "anthropic-ratelimit-requests-reset"),
    ("anthropic-ratelimit-tokens-remaining", "anthropic-ratelimit-tokens-reset"),
)


class ProviderClient:
    """Async wrapper around OpenAI-compatible chat completions APIs."""

    def __init__(self, task: TaskConfig) -> None:
        api_key = _resolve_api_key(task.model.api_key)

        self._system_prompt = task.system_prompt
        self._provider = task.model.provider.strip().lower()
        self._retries = task.model.retries
        self._retry_backoff_seconds = task.model.retry_backoff_seconds
        self._retry_backoff_jitter = task.model.retry_backoff_jitter
        self._throttle = _InJobThrottle()
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=task.model.base_url,
            timeout=float(task.timeout_seconds),
            max_retries=0,
        )

    async def close(self) -> None:
        await self._client.close()

    def prepare_attempt_request(
        self,
        *,
        note_payload,
        task_prompt: str,
        request_options: TaskRequestOptions,
        model_id: str,
    ) -> PreparedAttemptRequest:
        contract = build_note_update_contract(note_payload)
        max_tokens = request_options.max_output_tokens
        system_prompt_text = build_system_prompt(self._system_prompt)
        user_message_text = build_user_message(task_prompt, note_payload)
        # Anthropic's OpenAI-compatible endpoint requires strict=true for json_schema.
        strict_json_schema = self._provider == "anthropic"
        request_params: dict[str, object] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "note_update",
                    "strict": strict_json_schema,
                    "schema": contract.schema,
                },
            },
        }
        if request_options.temperature is not None:
            request_params["temperature"] = request_options.temperature
        if self._provider == "ollama":
            request_params["extra_body"] = {"think": False}

        return PreparedAttemptRequest(
            note_payload=note_payload,
            system_prompt_text=system_prompt_text,
            user_message_text=user_message_text,
            request_params=request_params,
            output_schema=contract.schema,
            editable_fields=contract.editable_fields,
        )

    async def generate_update(
        self,
        *,
        prepared_request: PreparedAttemptRequest,
    ) -> ProviderAttemptOutcome:
        note_payload = prepared_request.note_payload
        model_id = str(prepared_request.request_params.get("model") or "")
        if not model_id:
            raise LlmFatalError("Prepared request is missing provider model")

        kwargs: dict[str, object] = dict(prepared_request.request_params)
        kwargs["messages"] = [
            {
                "role": "system",
                "content": prepared_request.system_prompt_text,
            },
            {
                "role": "user",
                "content": prepared_request.user_message_text,
            },
        ]

        logger.debug(
            "Requesting update for %s (%s, editable=%d, read_only=%d)",
            note_payload.note_key,
            note_payload.note_type,
            len(note_payload.editable_fields),
            len(note_payload.read_only_fields),
        )
        started_at = time.monotonic()
        retry_count = 0
        response: object | None = None
        request_id: str | None = None
        rate_limit_headers: dict[str, str] = {}
        skip_throttle_once = False
        try:
            while True:
                try:
                    if not skip_throttle_once:
                        await self._throttle.wait()
                    skip_throttle_once = False
                    (
                        response,
                        request_id,
                        rate_limit_headers,
                    ) = await self._create_message_with_headers(kwargs)
                    await self._throttle.observe_headers(rate_limit_headers)
                    break
                except AuthenticationError as error:
                    raise LlmFatalError(
                        f"Provider authentication failed: {_api_error_message(error)}"
                    ) from error
                except APIConnectionError as error:
                    if retry_count >= self._retries:
                        raise LlmFatalError(
                            f"Provider connection error: {_api_error_message(error)}"
                        ) from error
                    retry_count += 1
                    delay = _retry_delay_seconds(
                        base_seconds=self._retry_backoff_seconds,
                        retry_count=retry_count,
                        jitter=self._retry_backoff_jitter,
                    )
                    await self._throttle.bump(delay)
                    skip_throttle_once = True
                    logger.warning(
                        "Retrying %s after connection error (%d/%d) in %.2fs: %s",
                        note_payload.note_key,
                        retry_count,
                        self._retries,
                        delay,
                        _api_error_message(error),
                    )
                    await asyncio.sleep(delay)
                except APIStatusError as error:
                    status_code = _status_code(error)
                    message = _api_error_message(error)
                    non_retryable_quota_429 = _is_non_retryable_429(
                        error,
                        provider=self._provider,
                    )
                    if _is_retryable_status_error(
                        status_code,
                        provider=self._provider,
                        non_retryable_quota_429=non_retryable_quota_429,
                    ):
                        if retry_count >= self._retries:
                            raise LlmFatalError(
                                f"Provider returned HTTP {status_code}: {message}"
                            ) from error
                        retry_count += 1
                        delay = _retry_delay_seconds(
                            base_seconds=self._retry_backoff_seconds,
                            retry_count=retry_count,
                            jitter=self._retry_backoff_jitter,
                        )
                        retry_after = _retry_after_seconds(error)
                        if retry_after is not None:
                            delay = max(delay, retry_after)
                        await self._throttle.bump(delay)
                        skip_throttle_once = True
                        logger.warning(
                            "Retrying %s after HTTP %s (%d/%d) in %.2fs: %s",
                            note_payload.note_key,
                            status_code if status_code is not None else "unknown",
                            retry_count,
                            self._retries,
                            delay,
                            message,
                        )
                        await asyncio.sleep(delay)
                        continue

                    if non_retryable_quota_429:
                        raise LlmFatalError(
                            f"Provider quota or billing limit reached: {message}"
                        ) from error

                    if status_code is None:
                        raise LlmNoteError(
                            f"Provider request failed: {message}",
                            category=LlmNoteErrorCategory.PROVIDER,
                        ) from error

                    raise LlmNoteError(
                        f"Provider returned HTTP {status_code}: {message}",
                        category=LlmNoteErrorCategory.PROVIDER,
                    ) from error
        except LlmFatalError:
            raise

        if response is None:
            raise RuntimeError("Provider response was not captured")

        latency_ms = round((time.monotonic() - started_at) * 1000)
        usage = getattr(response, "usage", None)
        input_tokens = _usage_value(usage, "prompt_tokens")
        output_tokens = _usage_value(usage, "completion_tokens")
        logger.debug(
            "Response for %s: message_id=%s, stop_reason=%s, input_tokens=%d, "
            "output_tokens=%d, latency_ms=%d, retries=%d",
            note_payload.note_key,
            getattr(response, "id", "unknown"),
            _stop_reason(response),
            input_tokens,
            output_tokens,
            latency_ms,
            retry_count,
        )

        return self._outcome_from_response(
            response=response,
            prepared_request=prepared_request,
            retry_count=retry_count,
            latency_ms=latency_ms,
            request_id=request_id,
            rate_limit_headers=rate_limit_headers,
        )

    async def _create_message_with_headers(
        self,
        kwargs: dict[str, object],
    ) -> tuple[object, str | None, dict[str, str]]:
        with_raw_response = getattr(
            self._client.chat.completions,
            "with_raw_response",
            None,
        )
        if with_raw_response is not None and hasattr(with_raw_response, "create"):
            raw_response = await with_raw_response.create(**kwargs)
            headers = _headers_to_dict(getattr(raw_response, "headers", None))
            parse = getattr(raw_response, "parse", None)
            response = parse() if callable(parse) else raw_response
            request_id = _extract_request_id(response=response, headers=headers)
            return response, request_id, _extract_rate_limit_headers(headers)

        response = await self._client.chat.completions.create(**kwargs)  # type: ignore[call-overload]
        request_id = _extract_request_id(response=response, headers={})
        return response, request_id, {}

    def _outcome_from_response(
        self,
        *,
        response: object,
        prepared_request: PreparedAttemptRequest,
        retry_count: int,
        latency_ms: int,
        request_id: str | None,
        rate_limit_headers: dict[str, str],
    ) -> ProviderAttemptOutcome:
        model_id = str(prepared_request.request_params.get("model") or "")
        context = _attempt_context_from_response(
            response,
            fallback_model_id=model_id,
            retry_count=retry_count,
            latency_ms=latency_ms,
            request_id=request_id,
            rate_limit_headers=rate_limit_headers,
        )
        raw_text = context.response_raw_text
        if context.stop_reason == "content_filter":
            message = raw_text or "Model output was blocked by content filtering"
            raise LlmNoteError(
                f"Provider refused request: {message}",
                category=LlmNoteErrorCategory.PROVIDER,
                attempt_context=context,
            )

        if not raw_text:
            raise LlmNoteError(
                "Provider response contained no JSON text output",
                category=LlmNoteErrorCategory.PROVIDER,
                attempt_context=context,
            )

        contract = NoteUpdateContract(
            schema=prepared_request.output_schema,
            editable_fields=prepared_request.editable_fields,
        )

        try:
            update = parse_note_update_json(raw_text, contract=contract)
        except StructuredOutputError as error:
            raise LlmNoteError(
                str(error),
                category=LlmNoteErrorCategory.NOTE,
                attempt_context=context,
            ) from error

        return ProviderAttemptOutcome(
            update=update,
            provider_message_id=context.provider_message_id,
            response_model_id=context.response_model_id,
            stop_reason=context.stop_reason,
            request_id=context.request_id,
            rate_limit_headers=context.rate_limit_headers,
            input_tokens=context.input_tokens,
            output_tokens=context.output_tokens,
            latency_ms=context.latency_ms,
            retry_count=context.retry_count,
            response_raw_text=raw_text,
            response_full_json=context.response_full_json,
        )


def _status_code(error: APIStatusError) -> int | None:
    value = getattr(error, "status_code", None)
    return value if isinstance(value, int) else None


def _resolve_api_key(configured_value: str) -> str:
    if configured_value.startswith("$"):
        env_name = configured_value[1:].strip()
        if not env_name:
            raise LlmFatalError("Model api_key env reference must include a variable")
        resolved = os.environ.get(env_name)
        if not resolved:
            raise LlmFatalError(
                f"Required environment variable '{env_name}' is not set"
            )
        return resolved
    return configured_value


def _api_error_message(error: Exception) -> str:
    value = getattr(error, "message", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    rendered = str(error).strip()
    if rendered:
        return rendered
    return error.__class__.__name__


def _first_choice(response: object) -> object | None:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        return None
    return choices[0]


def _extract_text_content(message: object) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


def _stop_reason(response: object) -> str | None:
    choice = _first_choice(response)
    if choice is None:
        return None
    value = getattr(choice, "finish_reason", None)
    return value if isinstance(value, str) and value else None


def _usage_value(usage: object, name: str) -> int:
    value = getattr(usage, name, 0)
    return value if isinstance(value, int) else 0


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


def _extract_request_id(*, response: object, headers: dict[str, str]) -> str | None:
    request_id = headers.get("request-id") or headers.get("x-request-id")
    if not request_id:
        request_id = getattr(response, "_request_id", None)
    return request_id if isinstance(request_id, str) and request_id else None


def _extract_rate_limit_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.startswith("x-ratelimit-")
        or key.startswith("anthropic-ratelimit-")
        or key in {"retry-after", "retry-after-ms"}
    }


class _InJobThrottle:
    """Best-effort in-process throttling shared across one job's provider calls."""

    def __init__(self) -> None:
        self._next_request_at = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        delay = 0.0
        async with self._lock:
            delay = self._next_request_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

    async def bump(self, delay_seconds: float) -> None:
        if delay_seconds <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            self._next_request_at = max(self._next_request_at, now + delay_seconds)

    async def observe_headers(self, headers: dict[str, str]) -> None:
        delay = _throttle_delay_from_headers(headers)
        if delay is None:
            return
        await self.bump(delay)


def _throttle_delay_from_headers(headers: dict[str, str]) -> float | None:
    delay_candidates: list[float] = []

    retry_after_ms = _parse_non_negative_float(headers.get("retry-after-ms"))
    if retry_after_ms is not None:
        delay_candidates.append(retry_after_ms / 1000.0)
    else:
        retry_after = _parse_retry_after_header_value(headers.get("retry-after"))
        if retry_after is not None:
            delay_candidates.append(retry_after)

    for remaining_key, reset_key in _THROTTLE_RESET_KEYS:
        if not _is_limit_exhausted(headers.get(remaining_key)):
            continue
        reset_delay = _parse_reset_seconds(headers.get(reset_key))
        if reset_delay is not None:
            delay_candidates.append(reset_delay)

    if not delay_candidates:
        return None
    return max(delay_candidates)


def _is_limit_exhausted(raw: str | None) -> bool:
    value = _parse_non_negative_float(raw)
    if value is None:
        return False
    return value <= 0.0


def _parse_reset_seconds(raw: str | None) -> float | None:
    numeric = _parse_non_negative_float(raw)
    if numeric is not None:
        return numeric
    if raw is None:
        return None

    compact_duration = _parse_compact_duration_seconds(raw)
    if compact_duration is not None:
        return compact_duration

    http_date = _parse_retry_after_http_date(raw)
    if http_date is not None:
        return http_date

    return _parse_iso_datetime_delta_seconds(raw)


def _parse_compact_duration_seconds(raw: str) -> float | None:
    normalized = "".join(raw.strip().lower().split())
    if not normalized:
        return None

    consumed = 0
    total = 0.0
    for match in _RESET_DURATION_PART_PATTERN.finditer(normalized):
        if match.start() != consumed:
            return None
        consumed = match.end()
        value = float(match.group("value"))
        unit = match.group("unit")
        if unit == "h":
            total += value * 3600
        elif unit == "m":
            total += value * 60
        elif unit == "s":
            total += value
    if consumed != len(normalized):
        return None
    return max(total, 0.0)


def _parse_iso_datetime_delta_seconds(raw: str) -> float | None:
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max((parsed - datetime.now(timezone.utc)).total_seconds(), 0.0)


def _is_retryable_status_error(
    status_code: int | None,
    *,
    provider: str,
    non_retryable_quota_429: bool,
) -> bool:
    if status_code is None:
        return False
    if status_code == 429:
        return not non_retryable_quota_429
    if status_code in _TRANSIENT_STATUS_CODES:
        return True
    if status_code >= 500:
        return True
    provider_codes = _PROVIDER_TRANSIENT_STATUS_CODES.get(provider, set())
    return status_code in provider_codes


def _is_non_retryable_429(error: APIStatusError, *, provider: str) -> bool:
    if _status_code(error) != 429:
        return False
    haystack = " ".join(_error_text_fragments(error)).lower()
    if not haystack:
        return False
    markers = list(_NON_RETRYABLE_429_MARKERS["default"])
    markers.extend(_NON_RETRYABLE_429_MARKERS.get(provider, ()))
    return any(marker in haystack for marker in markers)


def _error_text_fragments(error: APIStatusError) -> list[str]:
    fragments = [_api_error_message(error)]
    body = getattr(error, "body", None)
    fragments.extend(_flatten_text_values(body))

    response = getattr(error, "response", None)
    if response is not None:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            fragments.append(text)

    return [fragment for fragment in fragments if fragment]


def _flatten_text_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        output: list[str] = []
        for nested in value.values():
            output.extend(_flatten_text_values(nested))
        return output
    if isinstance(value, list):
        output: list[str] = []
        for nested in value:
            output.extend(_flatten_text_values(nested))
        return output
    return []


def _headers_to_dict(value: object) -> dict[str, str]:
    if value is None:
        return {}
    items = getattr(value, "items", None)
    if not callable(items):
        return {}
    out: dict[str, str] = {}
    try:
        header_items = items()
        if not isinstance(header_items, Iterable):
            return {}
        for pair in header_items:
            if not isinstance(pair, tuple) or len(pair) != 2:
                continue
            key, header_value = pair
            if isinstance(key, str) and isinstance(header_value, str):
                out[key.lower()] = header_value
    except Exception:
        return {}
    return out


def _attempt_context_from_response(
    response: object,
    *,
    fallback_model_id: str,
    retry_count: int,
    latency_ms: int,
    request_id: str | None,
    rate_limit_headers: dict[str, str],
) -> ProviderAttemptErrorContext:
    message_id = getattr(response, "id", None)
    if not isinstance(message_id, str):
        message_id = None

    model = getattr(response, "model", None)
    response_model_id = model if isinstance(model, str) else fallback_model_id

    choice = _first_choice(response)
    stop_reason = _stop_reason(response)
    raw_text = None
    if choice is not None:
        raw_text = _extract_text_content(getattr(choice, "message", None)) or None

    usage = getattr(response, "usage", None)
    return ProviderAttemptErrorContext(
        provider_message_id=message_id,
        response_model_id=response_model_id,
        stop_reason=stop_reason,
        request_id=request_id,
        rate_limit_headers=rate_limit_headers,
        input_tokens=_usage_value(usage, "prompt_tokens"),
        output_tokens=_usage_value(usage, "completion_tokens"),
        latency_ms=latency_ms,
        retry_count=retry_count,
        response_raw_text=raw_text,
        response_full_json=_response_to_json(response),
    )


def _retry_after_seconds(error: APIStatusError) -> float | None:
    response = getattr(error, "response", None)
    if response is None:
        return None
    headers = _headers_to_dict(getattr(response, "headers", None))

    retry_after_ms = _parse_non_negative_float(headers.get("retry-after-ms"))
    if retry_after_ms is not None:
        return retry_after_ms / 1000.0

    return _parse_retry_after_header_value(headers.get("retry-after"))


def _parse_retry_after_header_value(raw: str | None) -> float | None:
    if raw is None:
        return None
    parsed = _parse_non_negative_float(raw)
    if parsed is not None:
        return parsed
    return _parse_retry_after_http_date(raw)


def _parse_non_negative_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return None
    if parsed < 0:
        return None
    return parsed


def _parse_retry_after_http_date(raw: str) -> float | None:
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max((parsed - datetime.now(timezone.utc)).total_seconds(), 0.0)


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
