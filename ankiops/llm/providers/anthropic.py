"""Anthropic SDK adapter."""

from __future__ import annotations

import logging
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
from ankiops.llm.prompting import build_note_patch_schema, build_user_payload

from .errors import ProviderFatalError, ProviderNoteError

logger = logging.getLogger(__name__)


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
        response_schema = build_note_patch_schema(note_payload)
        tool_def = {
            "name": "note_patch",
            "description": "Return the edited note fields.",
            "input_schema": response_schema,
        }

        kwargs: dict[str, object] = {
            "model": model,
            "system": instructions,
            "messages": [
                {"role": "user", "content": build_user_payload(note_payload)},
            ],
            "tools": [tool_def],
            "tool_choice": {"type": "tool", "name": "note_patch"},
            "max_tokens": request_options.max_output_tokens or 4096,
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
            raise ProviderNoteError(
                f"Provider returned HTTP {error.status_code}: {error.message}"
            ) from error

        # Extract the tool use block from the response
        for block in response.content:
            if block.type == "tool_use" and block.name == "note_patch":
                return _parse_tool_input(block.input)  # type: ignore[arg-type]

        raise ProviderNoteError("Anthropic response contained no tool_use block")


def _parse_tool_input(data: dict[str, object]) -> NotePatch:
    note_key = data.get("note_key")
    edits = data.get("edits")
    if not isinstance(note_key, str) or not isinstance(edits, dict):
        raise ProviderNoteError("Response is missing note_key or edits")

    parsed_fields: dict[str, str] = {}
    for field_name, value in edits.items():
        if not isinstance(field_name, str):
            raise ProviderNoteError("Response edits keys must be strings")
        if value is None:
            continue
        if not isinstance(value, str):
            raise ProviderNoteError("Response edits values must be strings")
        parsed_fields[field_name] = value

    return NotePatch(note_key=note_key, edits=parsed_fields)
