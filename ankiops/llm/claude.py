"""Anthropic Claude client for note editing tasks."""

from __future__ import annotations

import logging
import os
import time

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
)

from .errors import LlmFatalError, LlmNoteError
from .models import (
    GenerateUpdateResult,
    NotePayload,
    TaskConfig,
    TaskRequestOptions,
)
from .prompting import build_system_prompt, build_user_message
from .structured_output import (
    StructuredOutputError,
    build_note_update_contract,
    parse_note_update_json,
)

logger = logging.getLogger(__name__)


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

    def generate_update(
        self,
        *,
        note_payload: NotePayload,
        task_prompt: str,
        request_options: TaskRequestOptions,
        api_model: str,
    ) -> GenerateUpdateResult:
        contract = build_note_update_contract(note_payload)
        max_tokens = request_options.max_output_tokens or 2048
        kwargs: dict[str, object] = {
            "model": api_model,
            "system": build_system_prompt(self._system_prompt),
            "messages": [
                {
                    "role": "user",
                    "content": build_user_message(task_prompt, note_payload),
                }
            ],
            "max_tokens": max_tokens,
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": contract.schema,
                }
            },
        }
        if request_options.temperature is not None:
            kwargs["temperature"] = request_options.temperature

        logger.debug(
            "Requesting update for %s (%s, editable=%d, read_only=%d)",
            note_payload.note_key,
            note_payload.note_type,
            len(note_payload.editable_fields),
            len(note_payload.read_only_fields),
        )
        started_at = time.monotonic()
        try:
            response = self._client.messages.create(**kwargs)  # type: ignore[arg-type]
        except AuthenticationError as error:
            raise LlmFatalError(
                f"Provider authentication failed: {error.message}"
            ) from error
        except APIConnectionError as error:
            raise LlmFatalError(f"Failed to connect to provider: {error}") from error
        except APIStatusError as error:
            if "does not support output format" in error.message:
                raise LlmFatalError(
                    "Configured Anthropic model does not support structured outputs. "
                    "Use a current supported Claude model."
                ) from error
            if error.status_code == 429 or error.status_code >= 500:
                raise LlmFatalError(
                    f"Provider returned HTTP {error.status_code}: {error.message}"
                ) from error
            raise LlmNoteError(
                f"Provider returned HTTP {error.status_code}: {error.message}"
            ) from error

        latency_ms = round((time.monotonic() - started_at) * 1000)
        usage = getattr(response, "usage", None)
        input_tokens = _usage_value(usage, "input_tokens")
        output_tokens = _usage_value(usage, "output_tokens")
        logger.debug(
            "Response for %s: message_id=%s, stop_reason=%s, input_tokens=%d, "
            "output_tokens=%d, latency_ms=%d",
            note_payload.note_key,
            getattr(response, "id", "unknown"),
            getattr(response, "stop_reason", None),
            input_tokens,
            output_tokens,
            latency_ms,
        )

        raw_text = _extract_text_content(response.content)
        if response.stop_reason == "refusal":
            message = raw_text or "Model refused to produce a structured response"
            raise LlmNoteError(f"Provider refused request: {message}")

        if not raw_text:
            raise LlmNoteError("Anthropic response contained no JSON text output")

        try:
            update = parse_note_update_json(raw_text, contract=contract)
        except StructuredOutputError as error:
            raise LlmNoteError(str(error)) from error

        return GenerateUpdateResult(
            update=update,
            message_id=getattr(response, "id", "unknown"),
            model=getattr(response, "model", api_model),
            stop_reason=getattr(response, "stop_reason", None),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
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
