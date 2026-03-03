"""OpenAI SDK adapter.

Works with any OpenAI-compatible API (OpenAI, Groq, Together, etc.)
by setting ``base_url`` in the provider config.
"""

from __future__ import annotations

import os

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
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


class OpenAIProvider:
    """Provider adapter using the ``openai`` SDK."""

    def __init__(self, config: ProviderConfig) -> None:
        api_key: str | None = None
        if config.api_key_env:
            api_key = os.environ.get(config.api_key_env)
            if not api_key:
                raise ProviderFatalError(
                    f"Required environment variable '{config.api_key_env}' is not set"
                )
        self._config = config
        self._client = OpenAI(
            api_key=api_key or "unused",
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
            "instructions": instructions,
            "input": build_user_payload(note_payload),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "note_patch",
                    "schema": contract.schema,
                    "strict": True,
                }
            },
        }
        if request_options.temperature is not None:
            kwargs["temperature"] = request_options.temperature
        if request_options.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request_options.max_output_tokens

        try:
            response = self._client.responses.create(**kwargs)  # type: ignore[arg-type]
        except AuthenticationError as error:
            raise ProviderFatalError(
                f"Provider authentication failed: {error.message}"
            ) from error
        except APIConnectionError as error:
            raise ProviderFatalError(
                f"Failed to connect to provider: {error}"
            ) from error
        except APIStatusError as error:
            raise ProviderNoteError(
                f"Provider returned HTTP {error.status_code}: {error.message}"
            ) from error

        refusal_message = _extract_refusal_message(response)
        if refusal_message is not None:
            raise ProviderNoteError(f"Provider refused request: {refusal_message}")

        raw_text = response.output_text
        if not raw_text:
            raise ProviderNoteError(
                "OpenAI response contained no structured output text"
            )
        try:
            return parse_note_patch_json(raw_text, contract=contract)
        except StructuredOutputError as error:
            raise ProviderNoteError(str(error)) from error


def _extract_refusal_message(response: object) -> str | None:
    output = getattr(response, "output", None)
    if not isinstance(output, list):
        return None
    for item in output:
        if getattr(item, "type", None) != "message":
            continue
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            continue
        for block in content:
            if getattr(block, "type", None) != "refusal":
                continue
            refusal = getattr(block, "refusal", None)
            if isinstance(refusal, str) and refusal:
                return refusal
    return None
