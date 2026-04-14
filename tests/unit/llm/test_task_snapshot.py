from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from ankiops.llm.llm_models import (
    DeckScope,
    ExecutionMode,
    FieldExceptionRule,
    TaskConfig,
    TaskExecutionOptions,
    TaskRequestOptions,
)
from ankiops.llm.model_registry import ProviderModel
from ankiops.llm.task_snapshot import task_from_snapshot, task_to_snapshot

TEST_MODEL = ProviderModel(
    name="claude-sonnet-4-6",
    api_id="claude-sonnet-4-6",
    provider="anthropic",
    base_url="https://api.anthropic.com/v1/",
    api_key_env="ANTHROPIC_API_KEY",
    input_usd_per_mtok=3,
    output_usd_per_mtok=15,
)


def _write_models_file(tmp_path: Path, *, content: str) -> None:
    (tmp_path / "llm").mkdir(parents=True, exist_ok=True)
    (tmp_path / "llm/models.yaml").write_text(
        dedent(content).strip() + "\n",
        encoding="utf-8",
    )


def test_task_snapshot_roundtrip_preserves_core_fields(tmp_path):
    _write_models_file(
        tmp_path,
        content="""
        - name: claude-sonnet-4-6
          api_id: claude-sonnet-4-6
          provider: anthropic
          base_url: https://api.anthropic.com/v1/
          api_key_env: ANTHROPIC_API_KEY
          input_usd_per_mtok: 3
          output_usd_per_mtok: 15
        """,
    )

    task = TaskConfig(
        name="grammar",
        model=TEST_MODEL,
        system_prompt="system prompt",
        prompt="task prompt",
        system_prompt_path=Path("llm/system.md"),
        prompt_path=Path("llm/grammar.md"),
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
    loaded = task_from_snapshot(snapshot, collection_dir=tmp_path)

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


def test_task_snapshot_roundtrip_supports_inline_prompt_without_prompt_path(tmp_path):
    _write_models_file(
        tmp_path,
        content="""
        - name: claude-sonnet-4-6
          api_id: claude-sonnet-4-6
          provider: anthropic
          base_url: https://api.anthropic.com/v1/
          api_key_env: ANTHROPIC_API_KEY
        """,
    )

    task = TaskConfig(
        name="grammar",
        model=TEST_MODEL,
        system_prompt="system prompt",
        prompt="inline task prompt",
        system_prompt_path=None,
        prompt_path=None,
    )

    snapshot = task_to_snapshot(task)
    loaded = task_from_snapshot(snapshot, collection_dir=tmp_path)

    assert snapshot["system_prompt_path"] is None
    assert snapshot["prompt_path"] is None
    assert loaded.system_prompt_path is None
    assert loaded.prompt_path is None


def test_task_from_snapshot_rejects_missing_model_registry(tmp_path):
    missing_collection = tmp_path / "missing-collection"
    with pytest.raises(ValueError, match="model registry file not found"):
        task_from_snapshot(
            {
                "model": "unknown",
                "decks": {"deck_root": None},
                "request": {},
                "execution": {"mode": "online"},
            },
            collection_dir=missing_collection,
        )


def test_task_from_snapshot_rejects_unknown_model(tmp_path):
    _write_models_file(
        tmp_path,
        content="""
        - name: claude-sonnet-4-6
          api_id: claude-sonnet-4-6
          provider: anthropic
          base_url: https://api.anthropic.com/v1/
          api_key_env: ANTHROPIC_API_KEY
        """,
    )

    with pytest.raises(ValueError, match="unsupported model"):
        task_from_snapshot(
            {
                "model": "unknown",
                "decks": {"deck_root": None},
                "request": {},
                "execution": {"mode": "online"},
            },
            collection_dir=tmp_path,
        )


def test_task_from_snapshot_requires_execution_mapping(tmp_path):
    _write_models_file(
        tmp_path,
        content="""
        - name: claude-sonnet-4-6
          api_id: claude-sonnet-4-6
          provider: anthropic
          base_url: https://api.anthropic.com/v1/
          api_key_env: ANTHROPIC_API_KEY
        """,
    )

    with pytest.raises(ValueError, match="missing execution options"):
        task_from_snapshot(
            {
                "model": "claude-sonnet-4-6",
                "decks": {"deck_root": None},
                "request": {},
            },
            collection_dir=tmp_path,
        )


def test_task_from_snapshot_uses_collection_local_model_registry(tmp_path):
    _write_models_file(
        tmp_path,
        content="""
        - name: qwen3-32b
          api_id: qwen3-32b
          provider: openai-compatible
          base_url: https://api.example.com/v1
          api_key_env: EXAMPLE_API_KEY
        """,
    )

    loaded = task_from_snapshot(
        {
            "name": "grammar",
            "model": "qwen3-32b",
            "decks": {"deck_root": None},
            "request": {},
            "execution": {"mode": "online"},
        },
        collection_dir=tmp_path,
    )

    assert loaded.model.name == "qwen3-32b"
    assert loaded.model.base_url == "https://api.example.com/v1"
