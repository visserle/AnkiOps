"""Ollama SDK adapter."""

from __future__ import annotations

from ollama import Client, ResponseError

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
        contract = build_note_patch_contract(note_payload)
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
                format=contract.schema,  # type: ignore[arg-type]
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
        if not isinstance(content, str) or not content:
            raise ProviderNoteError("Ollama response is missing message content")
        try:
            return parse_note_patch_json(content, contract=contract)
        except StructuredOutputError as error:
            raise ProviderNoteError(str(error)) from error
