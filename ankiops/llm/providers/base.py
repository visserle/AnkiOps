"""Provider base classes and shared request helpers."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import requests

from ankiops.llm.models import (
    NotePatch,
    NotePayload,
    ProviderConfig,
    TaskRequestOptions,
)


class ProviderFatalError(RuntimeError):
    """Raised for fatal provider failures that should abort the run."""


class ProviderNoteError(RuntimeError):
    """Raised for note-scoped provider failures."""


class LlmProvider(ABC):
    """Base class for note-at-a-time LLM providers."""

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._session = requests.Session()

    @property
    def config(self) -> ProviderConfig:
        return self._config

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            api_key = os.environ.get(self.config.api_key_env)
            if not api_key:
                raise ProviderFatalError(
                    "Required environment variable "
                    f"'{self.config.api_key_env}' is not set"
                )
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _post_json(self, url: str, body: dict[str, object]) -> dict[str, object]:
        try:
            response = self._session.post(
                url,
                json=body,
                headers=self._build_headers(),
                timeout=self.config.timeout_seconds,
            )
        except requests.Timeout as error:
            raise ProviderNoteError(f"Request timed out: {error}") from error
        except requests.ConnectionError as error:
            raise ProviderFatalError(
                f"Failed to connect to provider: {error}"
            ) from error
        except requests.RequestException as error:
            raise ProviderNoteError(f"Request failed: {error}") from error

        if response.status_code in {401, 403}:
            raise ProviderFatalError(
                f"Provider authentication failed ({response.status_code})"
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise ProviderNoteError(
                f"Provider returned HTTP {response.status_code}: {response.text}"
            )
        if response.status_code >= 400:
            raise ProviderNoteError(
                f"Provider returned HTTP {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except ValueError as error:
            raise ProviderNoteError(
                "Provider returned invalid JSON response"
            ) from error
        if not isinstance(data, dict):
            raise ProviderNoteError("Provider returned a non-object JSON response")
        return data

    @abstractmethod
    def generate_patch(
        self,
        *,
        note_payload: NotePayload,
        instructions: str,
        request_options: TaskRequestOptions,
        model: str,
    ) -> NotePatch:
        """Generate a structured patch for a single note."""
