"""Ollama SDK adapter."""

from __future__ import annotations

import json
import logging

from ollama import Client, ResponseError

from ankiops.llm.models import (
    NotePatch,
    NotePayload,
    ProviderConfig,
    TaskRequestOptions,
)
from ankiops.llm.prompting import build_note_patch_schema, build_user_payload

from .errors import ProviderFatalError, ProviderNoteError

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "http://127.0.0.1:11434"


class OllamaProvider:
    """Provider adapter using the ``ollama`` SDK."""

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._client = Client(
            host=config.base_url or _DEFAULT_HOST,
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
        options: dict[str, object] = {}
        if request_options.temperature is not None:
            options["temperature"] = request_options.temperature
        if request_options.max_output_tokens is not None:
            options["num_predict"] = request_options.max_output_tokens

        try:
            response = self._client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": build_user_payload(note_payload)},
                ],
                format=response_schema,  # type: ignore[arg-type]
                options=options,  # type: ignore[arg-type]
            )
        except ResponseError as error:
            if error.status_code and error.status_code >= 500:
                raise ProviderFatalError(f"Ollama server error: {error}") from error
            raise ProviderNoteError(f"Ollama request failed: {error}") from error
        except Exception as error:
            if "connect" in str(error).lower():
                raise ProviderFatalError(
                    f"Failed to connect to Ollama: {error}"
                ) from error
            raise ProviderNoteError(f"Ollama request failed: {error}") from error

        content = response.message.content
        if not content:
            raise ProviderNoteError("Ollama response is missing message content")
        return _parse_note_patch(content)


def _parse_note_patch(raw_text: str) -> NotePatch:
    try:
        data = json.loads(raw_text)
    except ValueError as error:
        raise ProviderNoteError("Response was not valid JSON") from error
    if not isinstance(data, dict):
        raise ProviderNoteError("Response must be a JSON object")

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
