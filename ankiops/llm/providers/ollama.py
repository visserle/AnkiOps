"""Ollama chat API adapter."""

from __future__ import annotations

import json

from ankiops.llm.models import NotePatch, NotePayload, TaskRequestOptions
from ankiops.llm.prompting import NOTE_PATCH_JSON_SCHEMA, build_user_payload

from .base import LlmProvider, ProviderNoteError


class OllamaProvider(LlmProvider):
    """Provider adapter for Ollama's native chat API."""

    def generate_patch(
        self,
        *,
        note_payload: NotePayload,
        instructions: str,
        request_options: TaskRequestOptions,
        model: str,
    ) -> NotePatch:
        options: dict[str, object] = {}
        if request_options.temperature is not None:
            options["temperature"] = request_options.temperature
        if request_options.max_output_tokens is not None:
            options["num_predict"] = request_options.max_output_tokens

        body: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": build_user_payload(note_payload)},
            ],
            "stream": False,
            "format": NOTE_PATCH_JSON_SCHEMA,
        }
        if options:
            body["options"] = options

        data = self._post_json(f"{self.config.base_url}/api/chat", body)
        message = data.get("message")
        if not isinstance(message, dict):
            raise ProviderNoteError("Ollama response is missing message content")
        content = message.get("content")
        if not isinstance(content, str):
            raise ProviderNoteError("Ollama response content must be a string")
        return _parse_note_patch(content)


def _parse_note_patch(raw_text: str) -> NotePatch:
    try:
        data = json.loads(raw_text)
    except ValueError as error:
        raise ProviderNoteError("Ollama response was not valid JSON") from error
    if not isinstance(data, dict):
        raise ProviderNoteError("Ollama response must be a JSON object")

    note_key = data.get("note_key")
    edits = data.get("edits")
    if not isinstance(note_key, str) or not isinstance(edits, dict):
        raise ProviderNoteError("Ollama response is missing note_key or edits")

    parsed_fields: dict[str, str] = {}
    for field_name, value in edits.items():
        if not isinstance(field_name, str) or not isinstance(value, str):
            raise ProviderNoteError("Ollama response edits must be string values")
        parsed_fields[field_name] = value

    return NotePatch(note_key=note_key, edits=parsed_fields)
