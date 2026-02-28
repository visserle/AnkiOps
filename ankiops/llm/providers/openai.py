"""OpenAI Responses API adapter."""

from __future__ import annotations

import json

from ankiops.llm.models import NotePatch, NotePayload, TaskRequestOptions
from ankiops.llm.prompting import NOTE_PATCH_JSON_SCHEMA, build_user_payload

from .base import LlmProvider, ProviderNoteError


class OpenAIProvider(LlmProvider):
    """Provider adapter for OpenAI's Responses API."""

    def generate_patch(
        self,
        *,
        note_payload: NotePayload,
        instructions: str,
        request_options: TaskRequestOptions,
        model: str,
    ) -> NotePatch:
        body: dict[str, object] = {
            "model": model,
            "instructions": instructions,
            "input": build_user_payload(note_payload),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "note_patch",
                    "schema": NOTE_PATCH_JSON_SCHEMA,
                    "strict": True,
                }
            },
        }
        if request_options.temperature is not None:
            body["temperature"] = request_options.temperature
        if request_options.max_output_tokens is not None:
            body["max_output_tokens"] = request_options.max_output_tokens

        data = self._post_json(f"{self.config.base_url}/responses", body)
        if data.get("refusal"):
            raise ProviderNoteError(f"Model refused request: {data['refusal']}")

        output_text = data.get("output_text")
        raw_text: str | None = output_text if isinstance(output_text, str) else None
        if raw_text is None:
            raw_text = self._extract_output_text(data)
        if raw_text is None:
            raise ProviderNoteError(
                "OpenAI response contained no structured output text"
            )

        return _parse_note_patch(raw_text)

    def _extract_output_text(self, data: dict[str, object]) -> str | None:
        output = data.get("output")
        if not isinstance(output, list):
            return None

        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                text = content_item.get("text")
                if isinstance(text, str) and content_item.get("type") in {
                    "output_text",
                    "text",
                }:
                    parts.append(text)
        return "".join(parts) if parts else None


def _parse_note_patch(raw_text: str) -> NotePatch:
    try:
        data = json.loads(raw_text)
    except ValueError as error:
        raise ProviderNoteError("OpenAI response was not valid JSON") from error
    if not isinstance(data, dict):
        raise ProviderNoteError("OpenAI response must be a JSON object")

    note_key = data.get("note_key")
    updated_fields = data.get("updated_fields")
    if not isinstance(note_key, str) or not isinstance(updated_fields, dict):
        raise ProviderNoteError("OpenAI response is missing note_key or updated_fields")

    parsed_fields: dict[str, str] = {}
    for field_name, value in updated_fields.items():
        if not isinstance(field_name, str) or not isinstance(value, str):
            raise ProviderNoteError(
                "OpenAI response updated_fields must be string values"
            )
        parsed_fields[field_name] = value

    return NotePatch(note_key=note_key, updated_fields=parsed_fields)
