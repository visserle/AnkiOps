from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ankiops.llm.claude import ClaudeClient
from ankiops.llm.errors import LlmFatalError, LlmNoteError
from ankiops.llm.models import NotePayload, TaskConfig, TaskRequestOptions
from ankiops.llm.prompting import build_system_prompt


def _extract_note_json(message: str) -> dict[str, object]:
    start = message.index("<note>\n") + len("<note>\n")
    end = message.index("\n</note>")
    return json.loads(message[start:end])


def _response(*, blocks: list[object], stop_reason: str = "end_turn") -> MagicMock:
    response = MagicMock()
    response.content = blocks
    response.stop_reason = stop_reason
    return response


@pytest.fixture
def task_config() -> TaskConfig:
    return TaskConfig(
        name="grammar",
        model="claude-sonnet-4-20250514",
        prompt="Fix grammar",
        api_key_env="ANTHROPIC_API_KEY",
    )


@pytest.fixture
def client(monkeypatch, task_config: TaskConfig) -> ClaudeClient:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return ClaudeClient(task_config)


@pytest.fixture
def note_payload() -> NotePayload:
    return NotePayload(
        note_key="nk-1",
        note_type="AnkiOpsQA",
        editable_fields={"Question": "Broken"},
        read_only_fields={"Source": "Book"},
    )


def test_generate_patch_builds_request_and_returns_patch(
    client: ClaudeClient,
    note_payload: NotePayload,
):
    response = _response(
        blocks=[
            SimpleNamespace(
                type="text",
                text='{"note_key":"nk-1","edits":{"Question":"Fixed"}}',
            )
        ]
    )

    with patch.object(
        client._client.messages,
        "create",
        return_value=response,
    ) as create:
        patch_result = client.generate_patch(
            note_payload=note_payload,
            task_prompt="Fix grammar",
            request_options=TaskRequestOptions(temperature=0, max_output_tokens=200),
            model="claude-sonnet-4-20250514",
        )

    assert patch_result.note_key == "nk-1"
    assert patch_result.edits == {"Question": "Fixed"}

    call_kwargs = create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-20250514"
    assert call_kwargs["system"] == build_system_prompt()
    assert "<task>\nFix grammar\n</task>" in call_kwargs["messages"][0]["content"]
    assert _extract_note_json(call_kwargs["messages"][0]["content"]) == {
        "note_key": "nk-1",
        "note_type": "AnkiOpsQA",
        "editable_fields": {"Question": "Broken"},
        "read_only_fields": {"Source": "Book"},
    }
    assert call_kwargs["temperature"] == 0
    assert call_kwargs["max_tokens"] == 200
    assert call_kwargs["output_config"]["format"]["type"] == "json_schema"
    schema = call_kwargs["output_config"]["format"]["schema"]
    assert "Question" in schema["properties"]["edits"]["properties"]
    assert "required" not in schema["properties"]["edits"]


@pytest.mark.parametrize(
    ("blocks", "stop_reason", "expected_error"),
    [
        (
            [SimpleNamespace(type="text", text="Cannot comply")],
            "refusal",
            "Provider refused request",
        ),
        (
            [SimpleNamespace(type="thinking", thinking="...")],
            "end_turn",
            "no JSON text output",
        ),
    ],
    ids=["refusal", "missing-json-text"],
)
def test_generate_patch_rejects_note_level_failures(
    client: ClaudeClient,
    note_payload: NotePayload,
    blocks: list[object],
    stop_reason: str,
    expected_error: str,
):
    response = _response(blocks=blocks, stop_reason=stop_reason)

    with patch.object(client._client.messages, "create", return_value=response):
        with pytest.raises(LlmNoteError, match=expected_error):
            client.generate_patch(
                note_payload=note_payload,
                task_prompt="Fix grammar",
                request_options=TaskRequestOptions(),
                model="claude-sonnet-4-20250514",
            )


def test_generate_patch_surfaces_unsupported_model_as_fatal(
    client: ClaudeClient,
    note_payload: NotePayload,
):
    from anthropic import APIStatusError

    error = APIStatusError(
        message=(
            "Error code: 400 - {'type': 'error', 'error': "
            "{'type': 'invalid_request_error', 'message': "
            "\"'claude-sonnet-4-20250514' does not support output format.\"}}"
        ),
        response=MagicMock(status_code=400),
        body=None,
    )

    with patch.object(client._client.messages, "create", side_effect=error):
        with pytest.raises(
            LlmFatalError,
            match="Configured Anthropic model does not support structured outputs",
        ):
            client.generate_patch(
                note_payload=note_payload,
                task_prompt="Fix grammar",
                request_options=TaskRequestOptions(),
                model="claude-sonnet-4-20250514",
            )


def test_missing_api_key_raises_fatal(task_config: TaskConfig, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(LlmFatalError, match="ANTHROPIC_API_KEY"):
        ClaudeClient(task_config)
