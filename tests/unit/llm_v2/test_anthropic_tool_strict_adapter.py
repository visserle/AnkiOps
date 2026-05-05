from __future__ import annotations

import asyncio
import json

from ankiops.llm_v2.domain.contracts import build_note_update_contract
from ankiops.llm_v2.domain.outcomes import ProviderOutcomeKind
from ankiops.llm_v2.domain.payloads import NotePayload
from ankiops.llm_v2.providers.adapter_base import AdapterRequest
from ankiops.llm_v2.providers.anthropic_tool_strict_adapter import (
    AnthropicToolStrictAdapter,
)


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        payload: object,
        headers: dict[str, str] | None = None,
        text: str | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text if text is not None else json.dumps(payload)

    def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakePost:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
    ) -> _FakeResponse:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "json": dict(json),
                "timeout": timeout,
            }
        )
        return self._responses.pop(0)


def _build_request() -> AdapterRequest:
    payload = NotePayload(
        note_key="nk-1",
        note_type="AnkiOpsQA",
        editable_fields={"Question": "Broken"},
        read_only_fields={"Source": "Book"},
    )
    contract = build_note_update_contract(payload)
    return AdapterRequest(
        note_payload=payload,
        contract=contract,
        system_prompt="System",
        task_prompt="Fix grammar",
        model_id="claude-sonnet-4-6",
        max_output_tokens=128,
    )


def test_generate_success_from_tool_use_input() -> None:
    fake_post = _FakePost(
        [
            _FakeResponse(
                status_code=200,
                payload={
                    "id": "msg_123",
                    "model": "claude-sonnet-4-6",
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 12, "output_tokens": 4},
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "note_update",
                            "input": {
                                "note_key": "nk-1",
                                "edits": {"Question": "Fixed"},
                            },
                        }
                    ],
                },
            )
        ]
    )
    adapter = AnthropicToolStrictAdapter(
        api_key="key",
        base_url="https://api.anthropic.com/v1",
        http_post=fake_post,
    )

    outcome = asyncio.run(adapter.generate(_build_request()))

    assert outcome.kind is ProviderOutcomeKind.SUCCESS
    assert outcome.update is not None
    assert outcome.update.edits == {"Question": "Fixed"}
    assert outcome.usage.input_tokens == 12
    assert outcome.usage.output_tokens == 4

    assert len(fake_post.calls) == 1
    call_json = fake_post.calls[0]["json"]
    assert isinstance(call_json, dict)
    tools = call_json["tools"]
    assert isinstance(tools, list)
    assert tools[0]["name"] == "note_update"


def test_generate_refusal_when_anthropic_refuses() -> None:
    fake_post = _FakePost(
        [
            _FakeResponse(
                status_code=200,
                payload={
                    "id": "msg_123",
                    "model": "claude-sonnet-4-6",
                    "stop_reason": "refusal",
                    "usage": {"input_tokens": 2, "output_tokens": 1},
                    "content": [{"type": "text", "text": "I cannot comply."}],
                },
            )
        ]
    )
    adapter = AnthropicToolStrictAdapter(
        api_key="key",
        base_url="https://api.anthropic.com/v1",
        http_post=fake_post,
    )

    outcome = asyncio.run(adapter.generate(_build_request()))

    assert outcome.kind is ProviderOutcomeKind.REFUSAL
    assert outcome.refusal_text == "I cannot comply."


def test_generate_validation_error_for_invalid_tool_input() -> None:
    fake_post = _FakePost(
        [
            _FakeResponse(
                status_code=200,
                payload={
                    "id": "msg_123",
                    "model": "claude-sonnet-4-6",
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 2, "output_tokens": 1},
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "note_update",
                            "input": {
                                "note_key": "nk-1",
                                "edits": {"Other": "Not editable"},
                            },
                        }
                    ],
                },
            )
        ]
    )
    adapter = AnthropicToolStrictAdapter(
        api_key="key",
        base_url="https://api.anthropic.com/v1",
        http_post=fake_post,
    )

    outcome = asyncio.run(adapter.generate(_build_request()))

    assert outcome.kind is ProviderOutcomeKind.VALIDATION_ERROR
    assert outcome.error_message is not None


def test_generate_retries_transient_http_status() -> None:
    fake_post = _FakePost(
        [
            _FakeResponse(status_code=429, payload={"error": "rate_limited"}),
            _FakeResponse(
                status_code=200,
                payload={
                    "id": "msg_123",
                    "model": "claude-sonnet-4-6",
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "note_update",
                            "input": {
                                "note_key": "nk-1",
                                "edits": {"Question": "Fixed"},
                            },
                        }
                    ],
                },
            ),
        ]
    )
    adapter = AnthropicToolStrictAdapter(
        api_key="key",
        base_url="https://api.anthropic.com/v1",
        retries=1,
        retry_backoff_seconds=0.0,
        retry_backoff_jitter=False,
        http_post=fake_post,
    )

    outcome = asyncio.run(adapter.generate(_build_request()))

    assert outcome.kind is ProviderOutcomeKind.SUCCESS
    assert len(fake_post.calls) == 2
