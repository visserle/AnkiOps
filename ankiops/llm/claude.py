"""Anthropic Claude client for note editing tasks."""

from __future__ import annotations

import logging
import os
import random
import time

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
)

from .errors import LlmFatalError, LlmNoteError
from .models import (
    NotePayload,
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


class ClaudeClient:
    """Thin wrapper around Anthropic's Messages API."""

    def __init__(self, task: TaskConfig) -> None:
        api_key = os.environ.get(task.api_key_env)
        if not api_key:
            raise LlmFatalError(
                f"Required environment variable '{task.api_key_env}' is not set"
            )

        self._system_prompt = task.system_prompt
        self._client = Anthropic(
            api_key=api_key,
            timeout=float(task.timeout_seconds),
        )

    def prepare_attempt_request(
        self,
        *,
        note_payload: NotePayload,
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

    def generate_update(
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
        try:
            while True:
                try:
                    response = self._client.messages.create(
                        **kwargs
                    )  # type: ignore[call-overload]
                    break
                except AuthenticationError as error:
                    raise LlmFatalError(
                        f"Provider authentication failed: {error.message}"
                    ) from error
                except APIConnectionError as error:
                    if retry_count >= request_options.retries:
                        raise LlmFatalError(
                            "Failed to connect to provider after "
                            f"{retry_count + 1} attempt(s): {error}"
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
                    time.sleep(delay)
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
                        logger.warning(
                            "Retrying %s after HTTP %d (%d/%d) in %.2fs: %s",
                            note_payload.note_key,
                            error.status_code,
                            retry_count,
                            request_options.retries,
                            delay,
                            error.message,
                        )
                        time.sleep(delay)
                        continue

                    raise LlmNoteError(
                        f"Provider returned HTTP {error.status_code}: {error.message}"
                    ) from error
        except LlmFatalError:
            raise

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

        context = _attempt_context_from_response(
            response,
            fallback_model=api_model,
            retry_count=retry_count,
            latency_ms=latency_ms,
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


def _attempt_context_from_response(
    response: object,
    *,
    fallback_model: str,
    retry_count: int,
    latency_ms: int,
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
        input_tokens=_usage_value(usage, "input_tokens"),
        output_tokens=_usage_value(usage, "output_tokens"),
        latency_ms=latency_ms,
        retry_count=retry_count,
        response_raw_text=raw_text,
        response_full_json=_response_to_json(response),
    )


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
