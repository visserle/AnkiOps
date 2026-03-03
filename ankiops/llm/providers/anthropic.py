"""Anthropic SDK adapter."""

from __future__ import annotations

import os

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
)

from ankiops.llm.models import (
    NotePatch,
    NotePayload,
    ProviderConfig,
    TaskRequestOptions,
)
from ankiops.llm.prompting import build_user_payload
from ankiops.llm.structured_output import (
    StructuredOutputError,
    build_note_patch_contract,
    parse_note_patch_json,
)

from .errors import ProviderFatalError, ProviderNoteError


class AnthropicProvider:
    """Provider adapter using the ``anthropic`` SDK."""

    def __init__(self, config: ProviderConfig) -> None:
        api_key: str | None = None
        if config.api_key_env:
            api_key = os.environ.get(config.api_key_env)
            if not api_key:
                raise ProviderFatalError(
                    f"Required environment variable '{config.api_key_env}' is not set"
                )
        self._config = config
        self._client = Anthropic(
            api_key=api_key,
            base_url=config.base_url,
            timeout=float(config.timeout_seconds),
        )

    def generate_patch(
        self,
        *,
        note_payload: NotePayload,
        instructions: str,
        request_options: TaskRequestOptions,
        model: str,
    ) -> NotePatch:
        contract = build_note_patch_contract(note_payload)
        kwargs: dict[str, object] = {
            "model": model,
            "system": instructions,
            "messages": [
                {"role": "user", "content": build_user_payload(note_payload)},
            ],
            "max_tokens": request_options.max_output_tokens or 4096,
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
            raise ProviderFatalError(
                f"Provider authentication failed: {error.message}"
            ) from error
        except APIConnectionError as error:
            raise ProviderFatalError(
                f"Failed to connect to provider: {error}"
            ) from error
        except APIStatusError as error:
            if "does not support output format" in error.message:
                raise ProviderFatalError(
                    "Configured Anthropic model does not support structured outputs. "
                    "Use a current supported model such as 'claude-sonnet-4-6'."
                ) from error
            raise ProviderNoteError(
                f"Provider returned HTTP {error.status_code}: {error.message}"
            ) from error

        raw_text = _extract_text_content(response.content)
        if response.stop_reason == "refusal":
            message = raw_text or "Model refused to produce a structured response"
            raise ProviderNoteError(f"Provider refused request: {message}")

        if not raw_text:
            raise ProviderNoteError("Anthropic response contained no JSON text output")

        try:
            return parse_note_patch_json(raw_text, contract=contract)
        except StructuredOutputError as error:
            raise ProviderNoteError(str(error)) from error


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
