"""Anthropic Claude client for note editing tasks."""

from __future__ import annotations

import os

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
)

from .errors import LlmFatalError, LlmNoteError
from .models import NotePatch, NotePayload, TaskConfig, TaskRequestOptions
from .prompting import build_system_prompt, build_user_message
from .structured_output import (
    StructuredOutputError,
    build_note_patch_contract,
    parse_note_patch_json,
)


class ClaudeClient:
    """Thin wrapper around Anthropic's Messages API."""

    def __init__(self, task: TaskConfig) -> None:
        api_key = os.environ.get(task.api_key_env)
        if not api_key:
            raise LlmFatalError(
                f"Required environment variable '{task.api_key_env}' is not set"
            )

        self._client = Anthropic(
            api_key=api_key,
            timeout=float(task.timeout_seconds),
        )

    def generate_patch(
        self,
        *,
        note_payload: NotePayload,
        task_prompt: str,
        request_options: TaskRequestOptions,
        model: str,
    ) -> NotePatch:
        contract = build_note_patch_contract(note_payload)
        kwargs: dict[str, object] = {
            "model": model,
            "system": build_system_prompt(),
            "messages": [
                {
                    "role": "user",
                    "content": build_user_message(task_prompt, note_payload),
                }
            ],
            "max_tokens": request_options.max_output_tokens or 2048,
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": contract.schema,
                }
            },
        }
        if request_options.temperature is not None:
            kwargs["temperature"] = request_options.temperature

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

        raw_text = _extract_text_content(response.content)
        if response.stop_reason == "refusal":
            message = raw_text or "Model refused to produce a structured response"
            raise LlmNoteError(f"Provider refused request: {message}")

        if not raw_text:
            raise LlmNoteError("Anthropic response contained no JSON text output")

        try:
            return parse_note_patch_json(raw_text, contract=contract)
        except StructuredOutputError as error:
            raise LlmNoteError(str(error)) from error


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
