"""Anthropic Claude client for note editing tasks."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime

from anthropic import (
    APIConnectionError,
    APIStatusError,
    AsyncAnthropic,
    AuthenticationError,
)

from .llm_errors import LlmFatalError, LlmNoteError
from .llm_models import (
    LlmAttemptResultType,
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


@dataclass(frozen=True)
class ProviderBatchState:
    provider_batch_id: str
    processing_status: str
    results_url: str | None
    created_at_remote: str | None
    expires_at_remote: str | None
    ended_at_remote: str | None
    archived_at_remote: str | None
    cancel_initiated_at_remote: str | None
    count_processing: int
    count_succeeded: int
    count_errored: int
    count_canceled: int
    count_expired: int
    request_id: str | None
    rate_limit_headers: dict[str, str]


@dataclass(frozen=True)
class ProviderBatchResult:
    custom_id: str
    result_type: LlmAttemptResultType
    outcome: ProviderAttemptOutcome | None
    error_type: str | None
    error_message: str | None
    response_raw_text: str | None
    response_full_json: str | None
    request_id: str | None
    rate_limit_headers: dict[str, str]


class ClaudeClient:
    """Async wrapper around Anthropic's Messages and Message Batches APIs."""

    def __init__(self, task: TaskConfig) -> None:
        api_key = os.environ.get(task.api_key_env)
        if not api_key:
            raise LlmFatalError(
                f"Required environment variable '{task.api_key_env}' is not set"
            )

        self._system_prompt = task.system_prompt
        self._client = AsyncAnthropic(
            api_key=api_key,
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
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": contract.schema,
                }
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
        kwargs["system"] = prepared_request.system_prompt_text
        kwargs["messages"] = [
            {
                "role": "user",
                "content": prepared_request.user_message_text,
            }
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
                        f"Provider authentication failed: {error.message}"
                    ) from error
                except APIConnectionError as error:
                    if retry_count >= request_options.retries:
                        raise LlmFatalError(
                            f"Provider connection error: {error}"
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
                        error,
                    )
                    await asyncio.sleep(delay)
                except APIStatusError as error:
                    if "does not support output format" in error.message:
                        raise LlmFatalError(
                            "Configured Anthropic model does not support structured "
                            "outputs. Use a current supported Claude model."
                        ) from error

                    if error.status_code == 429 or error.status_code >= 500:
                        if retry_count >= request_options.retries:
                            raise LlmFatalError(
                                f"Provider returned HTTP {error.status_code}: "
                                f"{error.message}"
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
                            "Retrying %s after HTTP %d (%d/%d) in %.2fs: %s",
                            note_payload.note_key,
                            error.status_code,
                            retry_count,
                            request_options.retries,
                            delay,
                            error.message,
                        )
                        await asyncio.sleep(delay)
                        continue

                    raise LlmNoteError(
                        f"Provider returned HTTP {error.status_code}: {error.message}"
                    ) from error
        except LlmFatalError:
            raise

        if response is None:
            raise RuntimeError("Provider response was not captured")

        latency_ms = round((time.monotonic() - started_at) * 1000)
        usage = getattr(response, "usage", None)
        input_tokens = _usage_value(usage, "input_tokens")
        output_tokens = _usage_value(usage, "output_tokens")
        logger.debug(
            "Response for %s: message_id=%s, stop_reason=%s, input_tokens=%d, "
            "output_tokens=%d, latency_ms=%d, retries=%d",
            note_payload.note_key,
            getattr(response, "id", "unknown"),
            getattr(response, "stop_reason", None),
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

    async def create_batch(
        self,
        *,
        requests: list[tuple[str, PreparedAttemptRequest]],
    ) -> ProviderBatchState:
        payload: list[dict[str, object]] = []
        for custom_id, prepared_request in requests:
            request_params: dict[str, object] = dict(prepared_request.request_params)
            request_params["system"] = prepared_request.system_prompt_text
            request_params["messages"] = [
                {
                    "role": "user",
                    "content": prepared_request.user_message_text,
                }
            ]
            payload.append(
                {
                    "custom_id": custom_id,
                    "params": request_params,
                }
            )
        (
            response,
            request_id,
            rate_limit_headers,
        ) = await self._create_batch_with_headers(payload)
        return _batch_state_from_response(
            response,
            request_id=request_id,
            rate_limit_headers=rate_limit_headers,
        )

    async def retrieve_batch(self, provider_batch_id: str) -> ProviderBatchState:
        (
            response,
            request_id,
            rate_limit_headers,
        ) = await self._retrieve_batch_with_headers(provider_batch_id)
        return _batch_state_from_response(
            response,
            request_id=request_id,
            rate_limit_headers=rate_limit_headers,
        )

    async def cancel_batch(self, provider_batch_id: str) -> ProviderBatchState:
        (
            response,
            request_id,
            rate_limit_headers,
        ) = await self._cancel_batch_with_headers(provider_batch_id)
        return _batch_state_from_response(
            response,
            request_id=request_id,
            rate_limit_headers=rate_limit_headers,
        )

    async def get_batch_results(
        self,
        *,
        provider_batch_id: str,
        prepared_by_custom_id: dict[str, PreparedAttemptRequest],
    ) -> list[ProviderBatchResult]:
        entries: list[ProviderBatchResult] = []
        results_stream = None
        try:
            results_stream = await self._client.messages.batches.results(
                provider_batch_id
            )
            async for entry in results_stream:
                custom_id = getattr(entry, "custom_id", None)
                if not isinstance(custom_id, str):
                    continue
                result = getattr(entry, "result", None)
                result_type = getattr(result, "type", None)

                if result_type == "succeeded":
                    prepared_request = prepared_by_custom_id.get(custom_id)
                    if prepared_request is None:
                        entries.append(
                            ProviderBatchResult(
                                custom_id=custom_id,
                                result_type=LlmAttemptResultType.ERRORED,
                                outcome=None,
                                error_type="provider_error",
                                error_message=(
                                    "Batch result referenced unknown custom_id "
                                    f"'{custom_id}'"
                                ),
                                response_raw_text=None,
                                response_full_json=None,
                                request_id=None,
                                rate_limit_headers={},
                            )
                        )
                        continue

                    message = getattr(result, "message", None)
                    try:
                        outcome = self._outcome_from_response(
                            response=message,
                            prepared_request=prepared_request,
                            retry_count=0,
                            latency_ms=0,
                            request_id=getattr(message, "_request_id", None),
                            rate_limit_headers={},
                        )
                    except LlmNoteError as error:
                        context = error.attempt_context
                        entries.append(
                            ProviderBatchResult(
                                custom_id=custom_id,
                                result_type=LlmAttemptResultType.ERRORED,
                                outcome=None,
                                error_type="note_error",
                                error_message=str(error),
                                response_raw_text=(
                                    context.response_raw_text if context else None
                                ),
                                response_full_json=(
                                    context.response_full_json if context else None
                                ),
                                request_id=context.request_id if context else None,
                                rate_limit_headers=(
                                    context.rate_limit_headers if context else {}
                                ),
                            )
                        )
                        continue

                    entries.append(
                        ProviderBatchResult(
                            custom_id=custom_id,
                            result_type=LlmAttemptResultType.SUCCEEDED,
                            outcome=outcome,
                            error_type=None,
                            error_message=None,
                            response_raw_text=outcome.response_raw_text,
                            response_full_json=outcome.response_full_json,
                            request_id=outcome.request_id,
                            rate_limit_headers=outcome.rate_limit_headers,
                        )
                    )
                    continue

                if result_type == "errored":
                    error = getattr(result, "error", None)
                    error_type = getattr(error, "type", None)
                    error_message = getattr(error, "message", None)
                    if not isinstance(error_type, str):
                        error_type = "provider_error"
                    if not isinstance(error_message, str) or not error_message:
                        error_message = "Provider batch request failed"
                    entries.append(
                        ProviderBatchResult(
                            custom_id=custom_id,
                            result_type=LlmAttemptResultType.ERRORED,
                            outcome=None,
                            error_type=error_type,
                            error_message=error_message,
                            response_raw_text=None,
                            response_full_json=None,
                            request_id=None,
                            rate_limit_headers={},
                        )
                    )
                    continue

                if result_type == "canceled":
                    entries.append(
                        ProviderBatchResult(
                            custom_id=custom_id,
                            result_type=LlmAttemptResultType.CANCELED,
                            outcome=None,
                            error_type=None,
                            error_message="Batch request was canceled",
                            response_raw_text=None,
                            response_full_json=None,
                            request_id=None,
                            rate_limit_headers={},
                        )
                    )
                    continue

                if result_type == "expired":
                    entries.append(
                        ProviderBatchResult(
                            custom_id=custom_id,
                            result_type=LlmAttemptResultType.EXPIRED,
                            outcome=None,
                            error_type=None,
                            error_message="Batch request expired",
                            response_raw_text=None,
                            response_full_json=None,
                            request_id=None,
                            rate_limit_headers={},
                        )
                    )
                    continue

                entries.append(
                    ProviderBatchResult(
                        custom_id=custom_id,
                        result_type=LlmAttemptResultType.ERRORED,
                        outcome=None,
                        error_type="provider_error",
                        error_message=f"Unknown batch result type '{result_type}'",
                        response_raw_text=None,
                        response_full_json=None,
                        request_id=None,
                        rate_limit_headers={},
                    )
                )
        except LlmFatalError:
            raise
        except Exception as error:
            raise LlmFatalError(f"Provider batch results failed: {error}") from error
        finally:
            if results_stream is not None:
                close = getattr(results_stream, "close", None)
                if callable(close):
                    maybe_close = close()
                    if hasattr(maybe_close, "__await__"):
                        await maybe_close

        return entries

    async def _create_message_with_headers(
        self,
        kwargs: dict[str, object],
    ) -> tuple[object, str | None, dict[str, str]]:
        with_raw_response = getattr(self._client.messages, "with_raw_response", None)
        if with_raw_response is not None and hasattr(with_raw_response, "create"):
            raw_response = await with_raw_response.create(**kwargs)
            headers = _headers_to_dict(getattr(raw_response, "headers", None))
            parse = getattr(raw_response, "parse", None)
            response = parse() if callable(parse) else raw_response
            request_id = _extract_request_id(response=response, headers=headers)
            return response, request_id, _extract_rate_limit_headers(headers)

        response = await self._client.messages.create(**kwargs)  # type: ignore[call-overload]
        request_id = _extract_request_id(response=response, headers={})
        return response, request_id, {}

    async def _create_batch_with_headers(
        self,
        requests: list[dict[str, object]],
    ) -> tuple[object, str | None, dict[str, str]]:
        with_raw_response = getattr(
            self._client.messages.batches, "with_raw_response", None
        )
        if with_raw_response is not None and hasattr(with_raw_response, "create"):
            raw_response = await with_raw_response.create(requests=requests)
            headers = _headers_to_dict(getattr(raw_response, "headers", None))
            parse = getattr(raw_response, "parse", None)
            response = parse() if callable(parse) else raw_response
            request_id = _extract_request_id(response=response, headers=headers)
            return response, request_id, _extract_rate_limit_headers(headers)

        response = await self._client.messages.batches.create(requests=requests)
        request_id = _extract_request_id(response=response, headers={})
        return response, request_id, {}

    async def _retrieve_batch_with_headers(
        self,
        provider_batch_id: str,
    ) -> tuple[object, str | None, dict[str, str]]:
        with_raw_response = getattr(
            self._client.messages.batches, "with_raw_response", None
        )
        if with_raw_response is not None and hasattr(with_raw_response, "retrieve"):
            raw_response = await with_raw_response.retrieve(provider_batch_id)
            headers = _headers_to_dict(getattr(raw_response, "headers", None))
            parse = getattr(raw_response, "parse", None)
            response = parse() if callable(parse) else raw_response
            request_id = _extract_request_id(response=response, headers=headers)
            return response, request_id, _extract_rate_limit_headers(headers)

        response = await self._client.messages.batches.retrieve(provider_batch_id)
        request_id = _extract_request_id(response=response, headers={})
        return response, request_id, {}

    async def _cancel_batch_with_headers(
        self,
        provider_batch_id: str,
    ) -> tuple[object, str | None, dict[str, str]]:
        with_raw_response = getattr(
            self._client.messages.batches, "with_raw_response", None
        )
        if with_raw_response is not None and hasattr(with_raw_response, "cancel"):
            raw_response = await with_raw_response.cancel(provider_batch_id)
            headers = _headers_to_dict(getattr(raw_response, "headers", None))
            parse = getattr(raw_response, "parse", None)
            response = parse() if callable(parse) else raw_response
            request_id = _extract_request_id(response=response, headers=headers)
            return response, request_id, _extract_rate_limit_headers(headers)

        response = await self._client.messages.batches.cancel(provider_batch_id)
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
        if context.stop_reason == "refusal":
            message = raw_text or "Model refused to produce a structured response"
            raise LlmNoteError(
                f"Provider refused request: {message}",
                attempt_context=context,
            )

        if not raw_text:
            raise LlmNoteError(
                "Anthropic response contained no JSON text output",
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


def _extract_text_content(blocks: object) -> str:
    parts: list[str] = []
    if not isinstance(blocks, list):
        return ""
    for block in blocks:
        if getattr(block, "type", None) != "text":
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


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
        if key.startswith("anthropic-ratelimit-") or key == "retry-after"
    }


def _headers_to_dict(value: object) -> dict[str, str]:
    if value is None:
        return {}
    items = getattr(value, "items", None)
    if not callable(items):
        return {}
    out: dict[str, str] = {}
    try:
        for key, header_value in items():
            if isinstance(key, str) and isinstance(header_value, str):
                out[key.lower()] = header_value
    except Exception:
        return {}
    return out


def _as_rfc3339(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        rendered = isoformat()
        if isinstance(rendered, str):
            return rendered
    return str(value)


def _batch_state_from_response(
    response: object,
    *,
    request_id: str | None,
    rate_limit_headers: dict[str, str],
) -> ProviderBatchState:
    counts = getattr(response, "request_counts", None)
    return ProviderBatchState(
        provider_batch_id=str(getattr(response, "id", "")),
        processing_status=str(getattr(response, "processing_status", "unknown")),
        results_url=(
            str(getattr(response, "results_url"))
            if getattr(response, "results_url", None) is not None
            else None
        ),
        created_at_remote=_as_rfc3339(getattr(response, "created_at", None)),
        expires_at_remote=_as_rfc3339(getattr(response, "expires_at", None)),
        ended_at_remote=_as_rfc3339(getattr(response, "ended_at", None)),
        archived_at_remote=_as_rfc3339(getattr(response, "archived_at", None)),
        cancel_initiated_at_remote=_as_rfc3339(
            getattr(response, "cancel_initiated_at", None)
        ),
        count_processing=_usage_value(counts, "processing"),
        count_succeeded=_usage_value(counts, "succeeded"),
        count_errored=_usage_value(counts, "errored"),
        count_canceled=_usage_value(counts, "canceled"),
        count_expired=_usage_value(counts, "expired"),
        request_id=request_id,
        rate_limit_headers=rate_limit_headers,
    )


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

    raw_text = _extract_text_content(getattr(response, "content", None)) or None

    usage = getattr(response, "usage", None)
    return ProviderAttemptErrorContext(
        provider_message_id=message_id,
        provider_model=provider_model,
        stop_reason=getattr(response, "stop_reason", None),
        request_id=request_id,
        rate_limit_headers=rate_limit_headers,
        input_tokens=_usage_value(usage, "input_tokens"),
        output_tokens=_usage_value(usage, "output_tokens"),
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
