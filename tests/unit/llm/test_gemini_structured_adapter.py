from __future__ import annotations

import asyncio
import json

from ankiops.llm.domain.contracts import build_note_update_contract
from ankiops.llm.domain.outcomes import ProviderOutcomeKind
from ankiops.llm.domain.payloads import NotePayload
from ankiops.llm.providers.adapter_base import AdapterRequest
from ankiops.llm.providers.gemini_structured_adapter import GeminiStructuredAdapter


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
        model_id="gemini-2.5-pro",
        max_output_tokens=128,
    )


def test_generate_success_from_candidate_json_text() -> None:
    fake_post = _FakePost(
        [
            _FakeResponse(
                status_code=200,
                payload={
                    "responseId": "resp_123",
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {
                                "parts": [
                                    {
                                        "text": (
                                            '{"note_key":"nk-1",'
                                            '"edits":{"Question":"Fixed"}}'
                                        )
                                    }
                                ]
                            },
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 9,
                        "candidatesTokenCount": 4,
                    },
                },
                headers={"x-request-id": "req_123"},
            )
        ]
    )
    adapter = GeminiStructuredAdapter(
        api_key="key",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        http_post=fake_post,
    )

    outcome = asyncio.run(adapter.generate(_build_request()))

    assert outcome.kind is ProviderOutcomeKind.SUCCESS
    assert outcome.update is not None
    assert outcome.update.edits == {"Question": "Fixed"}
    assert outcome.usage.input_tokens == 9
    assert outcome.usage.output_tokens == 4

    assert len(fake_post.calls) == 1
    call_url = fake_post.calls[0]["url"]
    assert isinstance(call_url, str)
    assert call_url.endswith("/models/gemini-2.5-pro:generateContent")


def test_generate_refusal_for_safety_finish_reason() -> None:
    fake_post = _FakePost(
        [
            _FakeResponse(
                status_code=200,
                payload={
                    "candidates": [
                        {
                            "finishReason": "SAFETY",
                            "content": {"parts": [{"text": "Blocked by safety"}]},
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 3,
                        "candidatesTokenCount": 0,
                    },
                },
            )
        ]
    )
    adapter = GeminiStructuredAdapter(
        api_key="key",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        http_post=fake_post,
    )

    outcome = asyncio.run(adapter.generate(_build_request()))

    assert outcome.kind is ProviderOutcomeKind.REFUSAL
    assert outcome.refusal_text == "Blocked by safety"


def test_generate_validation_error_for_invalid_json_output() -> None:
    fake_post = _FakePost(
        [
            _FakeResponse(
                status_code=200,
                payload={
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {"parts": [{"text": "{not valid json}"}]},
                        }
                    ]
                },
            )
        ]
    )
    adapter = GeminiStructuredAdapter(
        api_key="key",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        http_post=fake_post,
    )

    outcome = asyncio.run(adapter.generate(_build_request()))

    assert outcome.kind is ProviderOutcomeKind.VALIDATION_ERROR
    assert outcome.error_message is not None


def test_generate_retries_transient_http_status() -> None:
    fake_post = _FakePost(
        [
            _FakeResponse(status_code=503, payload={"error": "unavailable"}),
            _FakeResponse(
                status_code=200,
                payload={
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {
                                "parts": [
                                    {
                                        "text": (
                                            '{"note_key":"nk-1",'
                                            '"edits":{"Question":"Fixed"}}'
                                        )
                                    }
                                ]
                            },
                        }
                    ]
                },
            ),
        ]
    )
    adapter = GeminiStructuredAdapter(
        api_key="key",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        retries=1,
        retry_backoff_seconds=0.0,
        retry_backoff_jitter=False,
        http_post=fake_post,
    )

    outcome = asyncio.run(adapter.generate(_build_request()))

    assert outcome.kind is ProviderOutcomeKind.SUCCESS
    assert len(fake_post.calls) == 2
