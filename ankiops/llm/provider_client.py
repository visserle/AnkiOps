"""Unified OpenAI-protocol client for note editing tasks."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from collections.abc import Iterable

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, AuthenticationError

from .llm_errors import LlmFatalError, LlmNoteError
from .llm_models import (
    PreparedAttemptRequest,
    ProviderAttemptErrorContext,
    ProviderAttemptOutcome,
    TaskConfig,
    TaskRequestOptions,
)
from .prompting import build_system_prompt, build_user_message
from .structured_output import (
    NoteUpdateContract,
    StructuredOutputError,
    build_note_update_contract,
    parse_note_update_json,
)

logger = logging.getLogger(__name__)
_MAX_RETRY_BACKOFF_SECONDS = 30.0


class ProviderClient:
    """Async wrapper around OpenAI-compatible chat completions APIs."""

    def __init__(self, task: TaskConfig) -> None:
        api_key_env = task.model.api_key_env or task.api_key_env
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise LlmFatalError(
                f"Required environment variable '{api_key_env}' is not set"
            )

        self._system_prompt = task.system_prompt
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
        api_model: str,
    ) -> PreparedAttemptRequest:
        contract = build_note_update_contract(note_payload)
        max_tokens = request_options.max_output_tokens or 2048
        system_prompt_text = build_system_prompt(self._system_prompt)
        user_message_text = build_user_message(task_prompt, note_payload)
        request_params: dict[str, object] = {
            "model": api_model,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "note_update",
                    "strict": True,
                    "schema": contract.schema,
                },
            },
        }
        if request_options.temperature is not None:
            request_params["temperature"] = request_options.temperature

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
        request_options: TaskRequestOptions,
    ) -> ProviderAttemptOutcome:
        note_payload = prepared_request.note_payload
        api_model = str(prepared_request.request_params.get("model") or "")
        if not api_model:
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
        try:
            while True:
                try:
                    (
                        response,
                        request_id,
                        rate_limit_headers,
                    ) = await self._create_message_with_headers(kwargs)
                    break
                except AuthenticationError as error:
                    raise LlmFatalError(
                        f"Provider authentication failed: {_api_error_message(error)}"
                    ) from error
                except APIConnectionError as error:
                    if retry_count >= request_options.retries:
                        raise LlmFatalError(
                            f"Provider connection error: {_api_error_message(error)}"
                        ) from error
                    retry_count += 1
                    delay = _retry_delay_seconds(
                        base_seconds=request_options.retry_backoff_seconds,
                        retry_count=retry_count,
                        jitter=request_options.retry_backoff_jitter,
                    )
                    logger.warning(
                        "Retrying %s after connection error (%d/%d) in %.2fs: %s",
                        note_payload.note_key,
                        retry_count,
                        request_options.retries,
                        delay,
                        _api_error_message(error),
                    )
                    await asyncio.sleep(delay)
                except APIStatusError as error:
                    status_code = _status_code(error)
                    message = _api_error_message(error)
                    if status_code == 429 or (
                        status_code is not None and status_code >= 500
                    ):
                        if retry_count >= request_options.retries:
                            raise LlmFatalError(
                                f"Provider returned HTTP {status_code}: {message}"
                            ) from error
                        retry_count += 1
                        delay = _retry_delay_seconds(
                            base_seconds=request_options.retry_backoff_seconds,
                            retry_count=retry_count,
                            jitter=request_options.retry_backoff_jitter,
                        )
                        retry_after = _retry_after_seconds(error)
                        if retry_after is not None:
                            delay = max(delay, retry_after)
                        logger.warning(
                            "Retrying %s after HTTP %s (%d/%d) in %.2fs: %s",
                            note_payload.note_key,
                            status_code if status_code is not None else "unknown",
                            retry_count,
                            request_options.retries,
                            delay,
                            message,
                        )
                        await asyncio.sleep(delay)
                        continue

                    if status_code is None:
                        raise LlmNoteError(
                            f"Provider request failed: {message}"
                        ) from error

                    raise LlmNoteError(
                        f"Provider returned HTTP {status_code}: {message}"
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
        api_model = str(prepared_request.request_params.get("model") or "")
        context = _attempt_context_from_response(
            response,
            fallback_model=api_model,
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
                attempt_context=context,
            )

        if not raw_text:
            raise LlmNoteError(
                "Provider response contained no JSON text output",
                attempt_context=context,
            )

        contract = NoteUpdateContract(
            schema=prepared_request.output_schema,
            editable_fields=prepared_request.editable_fields,
        )

        try:
            update = parse_note_update_json(raw_text, contract=contract)
        except StructuredOutputError as error:
            raise LlmNoteError(str(error), attempt_context=context) from error

        return ProviderAttemptOutcome(
            update=update,
            provider_message_id=context.provider_message_id,
            provider_model=context.provider_model,
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
        if key.startswith("x-ratelimit-") or key == "retry-after"
    }


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
    fallback_model: str,
    retry_count: int,
    latency_ms: int,
    request_id: str | None,
    rate_limit_headers: dict[str, str],
) -> ProviderAttemptErrorContext:
    message_id = getattr(response, "id", None)
    if not isinstance(message_id, str):
        message_id = None

    model = getattr(response, "model", None)
    provider_model = model if isinstance(model, str) else fallback_model

    choice = _first_choice(response)
    stop_reason = _stop_reason(response)
    raw_text = None
    if choice is not None:
        raw_text = _extract_text_content(getattr(choice, "message", None)) or None

    usage = getattr(response, "usage", None)
    return ProviderAttemptErrorContext(
        provider_message_id=message_id,
        provider_model=provider_model,
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
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return None
    if parsed < 0:
        return None
    return parsed


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
