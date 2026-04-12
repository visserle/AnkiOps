from __future__ import annotations

from pathlib import Path

import pytest

from ankiops.llm.llm_models import (
    DeckScope,
    ExecutionMode,
    FieldExceptionRule,
    TaskConfig,
    TaskExecutionOptions,
    TaskRequestOptions,
)
from ankiops.llm.model_registry import CLAUDE_SONNET_4_6
from ankiops.llm.task_snapshot import task_from_snapshot, task_to_snapshot


def test_task_snapshot_roundtrip_preserves_core_fields():
    task = TaskConfig(
        name="grammar",
        model=CLAUDE_SONNET_4_6,
        system_prompt="system prompt",
        prompt="task prompt",
        system_prompt_path=Path("llm/system_prompt.md"),
        prompt_path=Path("llm/prompts/grammar.md"),
        api_key_env="ANTHROPIC_API_KEY",
        timeout_seconds=45,
        decks=DeckScope(deck_root="DeckA"),
        field_exceptions=[
            FieldExceptionRule(
                note_types=["AnkiOpsQA"],
                read_only=["Source"],
                hidden=["AI Notes"],
            )
        ],
        request=TaskRequestOptions(
            temperature=0.2,
            max_output_tokens=1234,
            retries=3,
            retry_backoff_seconds=0.75,
            retry_backoff_jitter=False,
        ),
        execution=TaskExecutionOptions(
            mode=ExecutionMode.ONLINE,
            concurrency=4,
        ),
    )

    snapshot = task_to_snapshot(task)
    loaded = task_from_snapshot(snapshot)

    assert loaded.name == task.name
    assert loaded.model == task.model
    assert loaded.system_prompt == task.system_prompt
    assert loaded.prompt == task.prompt
    assert loaded.api_key_env == task.api_key_env
    assert loaded.timeout_seconds == task.timeout_seconds
    assert loaded.decks == task.decks
    assert loaded.field_exceptions == task.field_exceptions
    assert loaded.request == task.request
    assert loaded.execution == task.execution


def test_task_from_snapshot_rejects_invalid_model():
    with pytest.raises(ValueError, match="unsupported model"):
        task_from_snapshot(
            {
                "model": "unknown",
                "decks": {"deck_root": None},
                "request": {},
                "execution": {"mode": "online"},
            }
        )


def test_task_from_snapshot_requires_execution_mapping():
    with pytest.raises(ValueError, match="missing execution options"):
        task_from_snapshot(
            {
                "model": "claude-sonnet-4-6",
                "decks": {"deck_root": None},
                "request": {},
            }
        )
