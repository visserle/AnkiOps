"""Tests for LLM task execution behavior."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from ankiops.llm.model_registry import ModelSpec
from ankiops.llm.runner import (
    LlmTaskExecutor,
    MaterializedTaskContext,
    OpenAIResult,
)
from ankiops.llm.types import (
    DiscoveryCounts,
    DiscoveryItem,
    DiscoverySnapshot,
    LlmItemStatus,
    NotePayload,
    TaskConfig,
    TaskRequestOptions,
)


class _FakeAsyncOpenAI:
    def __init__(self, **_kwargs) -> None:
        pass

    async def close(self) -> None:
        pass


def _parsed_response(*updates):
    return SimpleNamespace(
        updates=[
            SimpleNamespace(note_key=note_key, field=field, value=value)
            for note_key, field, value in updates
        ],
        model_dump=lambda mode: {"updates": []},
    )


def _success(parsed_response, *, input_tokens: int, output_tokens: int) -> OpenAIResult:
    return OpenAIResult(
        parsed_response=parsed_response,
        request_json={"model": "gpt-test"},
        parsed_response_json={"updates": []},
        response_json="{}",
        error_message=None,
        outcome="success",
        latency_ms=100,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _fatal_error() -> OpenAIResult:
    return OpenAIResult(
        parsed_response=None,
        request_json={"model": "gpt-test"},
        parsed_response_json=None,
        response_json=None,
        error_message="OpenAI authentication failed: no key",
        outcome="fatal_error",
        latency_ms=1,
        input_tokens=0,
        output_tokens=0,
        fatal=True,
    )


def _context(
    llm_qa_config,
    *,
    notes_per_request: int = 2,
    concurrency: int = 2,
) -> MaterializedTaskContext:
    task = TaskConfig(
        name="grammar",
        model=ModelSpec(
            model="test",
            model_id="gpt-test",
            api_url="https://api.openai.com/v1/responses",
            api_key="$OPENAI_API_KEY",
            concurrency=concurrency,
        ),
        system_prompt="system",
        user_prompt="user",
        request=TaskRequestOptions(notes_per_request=notes_per_request),
    )
    notes = [
        {
            "note_key": "nk-1",
            "note_type": "AnkiOpsQA",
            "fields": {
                "Question": "Broken",
                "Answer": "Existing answer",
                "Source": "Book",
            },
        },
        {
            "note_key": "nk-2",
            "note_type": "AnkiOpsQA",
            "fields": {
                "Question": "Already good",
                "Answer": "Existing answer",
                "Source": "Book",
            },
        },
    ]
    items = [
        DiscoveryItem(
            ordinal=index,
            deck_name="Deck",
            note_key=note["note_key"],
            note_type="AnkiOpsQA",
            item_status=LlmItemStatus.QUEUED,
            skip_reason=None,
            error_message=None,
            payload=NotePayload(
                note_key=note["note_key"],
                note_type="AnkiOpsQA",
                editable_fields={"Question": note["fields"]["Question"]},
                read_only_fields={"Source": note["fields"]["Source"]},
            ),
            note_type_config=llm_qa_config,
            serialized_note=note,
        )
        for index, note in enumerate(notes, start=1)
    ]
    return MaterializedTaskContext(
        task=task,
        note_type_configs={"AnkiOpsQA": llm_qa_config},
        serialized_data={"decks": [{"name": "Deck", "notes": notes}]},
        discovery_snapshot=DiscoverySnapshot(
            counts=DiscoveryCounts(decks_seen=1, decks_matched=1, notes_seen=2),
            items=items,
        ),
    )


def test_executor_persists_successful_updates(
    tmp_path,
    monkeypatch,
    llm_qa_config,
):
    context = _context(llm_qa_config)
    persisted = {}

    async def fake_call_openai(*, batch, **_kwargs):
        assert [candidate.payload.note_key for candidate in batch.candidates] == [
            "nk-1",
            "nk-2",
        ]
        return _success(
            _parsed_response(("nk-1", "Question", "Fixed question")),
            input_tokens=12,
            output_tokens=6,
        )

    def fake_deserialize(data, **_kwargs):
        persisted["data"] = data

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("ankiops.llm.runner.AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr("ankiops.llm.runner._call_openai", fake_call_openai)
    monkeypatch.setattr("ankiops.llm.runner.deserialize", fake_deserialize)

    result = asyncio.run(
        LlmTaskExecutor(
            collection_dir=tmp_path,
            materialized_context=context,
            no_auto_commit=True,
        ).execute()
    )

    assert not result.failed
    assert result.persisted
    assert result.summary.updated == 1
    assert result.summary.unchanged == 1
    assert result.summary.requests == 1
    assert result.summary.input_tokens == 12
    assert result.summary.output_tokens == 6
    assert persisted["data"]["decks"][0]["notes"][0]["fields"]["Question"] == (
        "Fixed question"
    )


def test_executor_cancels_queued_items_after_fatal_error(
    tmp_path,
    monkeypatch,
    llm_qa_config,
):
    context = _context(llm_qa_config, notes_per_request=1, concurrency=1)

    async def fake_call_openai(**_kwargs):
        return _fatal_error()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("ankiops.llm.runner.AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr("ankiops.llm.runner._call_openai", fake_call_openai)

    result = asyncio.run(
        LlmTaskExecutor(
            collection_dir=tmp_path,
            materialized_context=context,
            no_auto_commit=True,
        ).execute()
    )

    assert result.failed
    assert not result.persisted
    assert result.summary.errors == 1
    assert result.summary.canceled == 1
