from __future__ import annotations

import json
from dataclasses import dataclass

from ankiops.llm.models import (
    NotePayload,
    ProviderConfig,
    ProviderType,
    TaskRequestOptions,
)
from ankiops.llm.providers.ollama import OllamaProvider
from ankiops.llm.providers.openai import OpenAIProvider


@dataclass
class _FakeResponse:
    status_code: int
    payload: dict[str, object]
    text: str = ""

    def json(self) -> dict[str, object]:
        return self.payload


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object], dict[str, str], int]] = []

    def post(
        self,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
        timeout: int,
    ) -> _FakeResponse:
        self.calls.append((url, json, headers, timeout))
        return self.response


def test_openai_provider_shapes_responses_request(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenAIProvider(
        ProviderConfig(
            version=1,
            name="openai-default",
            type=ProviderType.OPENAI,
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            model="gpt-5",
        )
    )
    fake_session = _FakeSession(
        _FakeResponse(
            status_code=200,
            payload={
                "output_text": ('{"note_key":"nk-1","edits":{"Question":"Fixed"}}')
            },
        )
    )
    provider._session = fake_session

    patch = provider.generate_patch(
        note_payload=NotePayload(
            note_key="nk-1",
            note_type="AnkiOpsQA",
            editable_fields={"Question": "Broken"},
            read_only_fields={"Source": "Book"},
        ),
        instructions="Fix grammar",
        request_options=TaskRequestOptions(temperature=0, max_output_tokens=200),
        model="gpt-5",
    )

    assert patch.note_key == "nk-1"
    url, body, headers, timeout = fake_session.calls[0]
    assert url == "https://api.openai.com/v1/responses"
    assert body["model"] == "gpt-5"
    assert body["text"] == {
        "format": {
            "type": "json_schema",
            "name": "note_patch",
            "schema": {
                "type": "object",
                "properties": {
                    "note_key": {"type": "string"},
                    "edits": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["note_key", "edits"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    }
    assert json.loads(body["input"]) == {
        "note_key": "nk-1",
        "note_type": "AnkiOpsQA",
        "editable_fields": {"Question": "Broken"},
        "read_only_fields": {"Source": "Book"},
    }
    assert headers["Authorization"] == "Bearer test-key"
    assert timeout == 60


def test_ollama_provider_shapes_chat_request():
    provider = OllamaProvider(
        ProviderConfig(
            version=1,
            name="ollama-local",
            type=ProviderType.OLLAMA,
            base_url="http://127.0.0.1:11434",
            model="gpt-oss",
            timeout_seconds=120,
        )
    )
    fake_session = _FakeSession(
        _FakeResponse(
            status_code=200,
            payload={
                "message": {
                    "content": ('{"note_key":"nk-1","edits":{"Question":"Fixed"}}')
                }
            },
        )
    )
    provider._session = fake_session

    patch = provider.generate_patch(
        note_payload=NotePayload(
            note_key="nk-1",
            note_type="AnkiOpsQA",
            editable_fields={"Question": "Broken"},
            read_only_fields={"Source": "Book"},
        ),
        instructions="Fix grammar",
        request_options=TaskRequestOptions(temperature=0, max_output_tokens=128),
        model="gpt-oss",
    )

    assert patch.edits["Question"] == "Fixed"
    url, body, headers, timeout = fake_session.calls[0]
    assert url == "http://127.0.0.1:11434/api/chat"
    assert body["model"] == "gpt-oss"
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0.0, "num_predict": 128}
    assert json.loads(body["messages"][1]["content"]) == {
        "note_key": "nk-1",
        "note_type": "AnkiOpsQA",
        "editable_fields": {"Question": "Broken"},
        "read_only_fields": {"Source": "Book"},
    }
    assert "Authorization" not in headers
    assert timeout == 120


def test_openai_provider_omits_empty_read_only_fields(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = OpenAIProvider(
        ProviderConfig(
            version=1,
            name="openai-default",
            type=ProviderType.OPENAI,
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            model="gpt-5",
        )
    )
    fake_session = _FakeSession(
        _FakeResponse(
            status_code=200,
            payload={"output_text": '{"note_key":"nk-1","edits":{}}'},
        )
    )
    provider._session = fake_session

    provider.generate_patch(
        note_payload=NotePayload(
            note_key="nk-1",
            note_type="AnkiOpsQA",
            editable_fields={"Question": "Broken"},
        ),
        instructions="Fix grammar",
        request_options=TaskRequestOptions(),
        model="gpt-5",
    )

    _, body, _, _ = fake_session.calls[0]
    assert json.loads(body["input"]) == {
        "note_key": "nk-1",
        "note_type": "AnkiOpsQA",
        "editable_fields": {"Question": "Broken"},
    }
