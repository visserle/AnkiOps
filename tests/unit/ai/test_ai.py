"""Unit tests for AI config, parser, HTTP client, and task execution."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import respx

from ankiops.ai.client import OpenAICompatibleAsyncEditor
from ankiops.ai.errors import (
    AIConfigError,
    AIRequestError,
    AIResponseError,
    TaskConfigError,
    TaskExecutionError,
)
from ankiops.ai.model_config import (
    load_model_configs,
    provider_choices,
    resolve_runtime_config,
)
from ankiops.ai.orchestration import AIRuntimeOverrides, prepare_ai_run
from ankiops.ai.paths import AIPaths
from ankiops.ai.runner import TaskRunner
from ankiops.ai.task_config import load_task_config
from ankiops.ai.types import (
    InlineEditedNote,
    InlineNotePayload,
    RuntimeAIConfig,
    TaskConfig,
    TaskRunOptions,
)
from ankiops.ai.validators import normalize_patch_response, parse_json_object


@dataclass
class _InlineBatchEditor:
    """Deterministic inline JSON batch editor test double."""

    async def edit_notes(
        self,
        task: TaskConfig,
        notes: list[InlineNotePayload],
    ) -> dict[str, InlineEditedNote]:
        _ = task
        edited: dict[str, InlineEditedNote] = {}
        for note in notes:
            fields = dict(note.editable)
            if "Question" in fields and isinstance(fields["Question"], str):
                fields["Question"] = fields["Question"].replace("I has", "I have")
            edited[note.note_key] = InlineEditedNote.from_parts(
                note_key=note.note_key,
                fields=fields,
            )
        return edited


@dataclass
class _WrongKeyBatchEditor:
    """Editor that returns a mismatched note_key to verify rejection logic."""

    async def edit_notes(
        self,
        task: TaskConfig,
        notes: list[InlineNotePayload],
    ) -> dict[str, InlineEditedNote]:
        _ = task
        _ = notes
        return {
            "n1": InlineEditedNote.from_parts(
                note_key="wrong-key",
                note_type="AnkiOpsQA",
                fields={"Question": "x"},
            )
        }


@dataclass
class _TargetOnlyBatchEditor:
    """Editor that returns only target fields while preserving note_key."""

    async def edit_notes(
        self,
        task: TaskConfig,
        notes: list[InlineNotePayload],
    ) -> dict[str, InlineEditedNote]:
        _ = task
        return {
            note.note_key: InlineEditedNote.from_parts(
                note_key=note.note_key,
                fields={
                    "Question": note.editable["Question"].replace("I has", "I have"),
                },
            )
            for note in notes
        }


@dataclass
class _ClosableBatchEditor:
    """Editor that tracks whether cleanup was called by TaskRunner."""

    closed: bool = False

    async def edit_notes(
        self,
        task: TaskConfig,
        notes: list[InlineNotePayload],
    ) -> dict[str, InlineEditedNote]:
        _ = task
        return {
            note.note_key: InlineEditedNote.from_parts(
                note_key=note.note_key,
                fields=dict(note.editable),
            )
            for note in notes
        }

    async def aclose(self) -> None:
        self.closed = True


@dataclass
class _FailingBatchEditor:
    """Editor that fails all chunk calls to stress warning handling."""

    async def edit_notes(
        self,
        task: TaskConfig,
        notes: list[InlineNotePayload],
    ) -> dict[str, InlineEditedNote]:
        _ = task
        _ = notes
        raise RuntimeError("boom")


def _runtime() -> RuntimeAIConfig:
    return RuntimeAIConfig(
        profile="openai-fast",
        provider="openai",
        transport="openai_chat_completions",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        requires_api_key=True,
        timeout_seconds=10,
        max_in_flight=3,
        api_key="test-key",
    )


def _task(
    *,
    task_id: str = "grammar",
    instructions: str = "Return inline JSON.",
    read_fields: list[str] | None = None,
    write_fields: list[str] | None = None,
    scope_note_types: list[str] | None = None,
    scope_decks: list[str] | None = None,
    scope_subdecks: bool = True,
    batch: str = "single",
    batch_size: int = 1,
    model: str | None = "openai-fast",
    temperature: float = 0.0,
) -> TaskConfig:
    return TaskConfig(
        id=task_id,
        description="",
        model=model,
        instructions=instructions,
        batch=batch,
        batch_size=batch_size,
        scope_decks=scope_decks or ["*"],
        scope_subdecks=scope_subdecks,
        scope_note_types=scope_note_types or ["AnkiOps*"],
        read_fields=read_fields or ["Question"],
        write_fields=write_fields or ["Question"],
        temperature=temperature,
    )


def _notes_payload() -> list[InlineNotePayload]:
    return [
        InlineNotePayload(
            note_key="n1",
            editable={"Question": "I has two lungs."},
            context={},
        )
    ]


async def _run_editor_once(
    *,
    task: TaskConfig | None = None,
    notes: list[InlineNotePayload] | None = None,
) -> dict[str, InlineEditedNote]:
    editor = OpenAICompatibleAsyncEditor(_runtime())
    try:
        return await editor.edit_notes(task or _task(), notes or _notes_payload())
    finally:
        await editor.aclose()


def _mk_ai_paths(tmp_path: Path) -> AIPaths:
    collection_dir = tmp_path / "collection"
    collection_dir.mkdir(parents=True, exist_ok=True)
    ai_paths = AIPaths.from_collection_dir(collection_dir)
    ai_paths.models.mkdir(parents=True, exist_ok=True)
    ai_paths.tasks.mkdir(parents=True, exist_ok=True)
    return ai_paths


def test_load_task_from_yaml(tmp_path):
    ai_paths = _mk_ai_paths(tmp_path)
    (ai_paths.tasks / "grammar.yaml").write_text(
        (
            "schema: ai.task.v1\n"
            "model: openai-fast\n"
            "instructions: |\n"
            "  Return inline JSON.\n"
            "batch: batch\n"
            "batch_size: 2\n"
            "scope_decks:\n"
            "  - Biology\n"
            "scope_subdecks: false\n"
            "scope_note_types:\n"
            "  - AnkiOps*\n"
            "read_fields:\n"
            "  - Question\n"
            "  - Answer\n"
            "write_fields:\n"
            "  - Question\n"
            "temperature: 0.1\n"
        ),
        encoding="utf-8",
    )

    task = load_task_config(ai_paths, "grammar")

    assert task.id == "grammar"
    assert task.model == "openai-fast"
    assert task.read_fields == ["Question", "Answer"]
    assert task.write_fields == ["Question"]
    assert task.batch == "batch"
    assert task.batch_size == 2
    assert task.scope_decks == ["Biology"]
    assert task.scope_subdecks is False
    assert task.temperature == pytest.approx(0.1)
    assert task.matches_note_type("AnkiOpsQA")


def test_load_task_accepts_default_model_alias(tmp_path):
    ai_paths = _mk_ai_paths(tmp_path)
    (ai_paths.tasks / "grammar.yaml").write_text(
        (
            "schema: ai.task.v1\n"
            "model: default\n"
            "instructions: Return inline JSON.\n"
            "read_fields:\n"
            "  - Question\n"
            "write_fields:\n"
            "  - Question\n"
        ),
        encoding="utf-8",
    )

    task = load_task_config(ai_paths, "grammar")
    assert task.model is None


def test_load_task_accepts_scalar_scope_and_field_values(tmp_path):
    ai_paths = _mk_ai_paths(tmp_path)
    (ai_paths.tasks / "grammar.yaml").write_text(
        (
            "schema: ai.task.v1\n"
            "instructions: Return inline JSON.\n"
            "scope_decks: Biology\n"
            "scope_note_types: AnkiOps*\n"
            "read_fields: Question\n"
            "write_fields: Question\n"
        ),
        encoding="utf-8",
    )

    task = load_task_config(ai_paths, "grammar")
    assert task.scope_decks == ["Biology"]
    assert task.scope_note_types == ["AnkiOps*"]
    assert task.read_fields == ["Question"]
    assert task.write_fields == ["Question"]


def test_load_task_rejects_out_of_range_temperature(tmp_path):
    ai_paths = _mk_ai_paths(tmp_path)
    (ai_paths.tasks / "grammar.yaml").write_text(
        "instructions: Return inline JSON.\nread_fields: [Question]\n"
        "write_fields: [Question]\ntemperature: 3\n",
        encoding="utf-8",
    )

    with pytest.raises(TaskConfigError, match="temperature"):
        load_task_config(ai_paths, "grammar")


def test_load_task_rejects_invalid_yaml(tmp_path):
    ai_paths = _mk_ai_paths(tmp_path)
    (ai_paths.tasks / "grammar.yaml").write_text(
        "instructions: [broken\nread_fields: [Question]\nwrite_fields: [Question]\n",
        encoding="utf-8",
    )

    with pytest.raises(TaskConfigError, match="Invalid YAML"):
        load_task_config(ai_paths, "grammar")


def test_load_task_rejects_unknown_fields(tmp_path):
    ai_paths = _mk_ai_paths(tmp_path)
    (ai_paths.tasks / "grammar.yaml").write_text(
        (
            "instructions: Return inline JSON.\n"
            "read_fields:\n"
            "  - Question\n"
            "write_fields:\n"
            "  - Question\n"
            "unexpected: true\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(TaskConfigError, match="unexpected"):
        load_task_config(ai_paths, "grammar")


def test_load_task_rejects_legacy_id_field(tmp_path):
    ai_paths = _mk_ai_paths(tmp_path)
    (ai_paths.tasks / "grammar.yaml").write_text(
        (
            "instructions: Return inline JSON.\n"
            "id: grammar\n"
            "read_fields:\n"
            "  - Question\n"
            "write_fields:\n"
            "  - Question\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(TaskConfigError, match="id"):
        load_task_config(ai_paths, "grammar")


def test_load_task_rejects_write_fields_outside_read_fields(tmp_path):
    ai_paths = _mk_ai_paths(tmp_path)
    (ai_paths.tasks / "grammar.yaml").write_text(
        (
            "instructions: Return inline JSON.\n"
            "read_fields:\n"
            "  - Question\n"
            "write_fields:\n"
            "  - Answer\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(TaskConfigError, match="subset of read_fields"):
        load_task_config(ai_paths, "grammar")


def test_load_task_rejects_absolute_task_path(tmp_path):
    ai_paths = _mk_ai_paths(tmp_path)
    (ai_paths.tasks / "grammar.yaml").write_text(
        (
            "instructions: Return inline JSON.\n"
            "read_fields:\n"
            "  - Question\n"
            "write_fields:\n"
            "  - Question\n"
        ),
        encoding="utf-8",
    )

    task_path = (ai_paths.tasks / "grammar.yaml").resolve()
    with pytest.raises(TaskConfigError, match="relative to ai/tasks/"):
        load_task_config(ai_paths, str(task_path))


def test_load_task_rejects_paths_outside_tasks_directory(tmp_path):
    ai_paths = _mk_ai_paths(tmp_path)
    external_task = ai_paths.root / "outside.yaml"
    external_task.write_text(
        (
            "instructions: Return inline JSON.\n"
            "read_fields:\n"
            "  - Question\n"
            "write_fields:\n"
            "  - Question\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(TaskConfigError, match="stay within ai/tasks/"):
        load_task_config(ai_paths, "../outside")


def test_load_models_config_and_resolve_runtime(tmp_path, monkeypatch):
    ai_paths = _mk_ai_paths(tmp_path)
    (ai_paths.models / "ollama-fast.yaml").write_text(
        (
            "schema: ai.model.v1\n"
            "default: true\n"
            "provider: ollama\n"
            "model: llama3.1:8b\n"
            "base_url: http://localhost:11434/v1\n"
            "api_key_env: TEST_KEY\n"
            "timeout_seconds: 60\n"
            "max_in_flight: 3\n"
        ),
        encoding="utf-8",
    )
    (ai_paths.models / "openai-fast.yaml").write_text(
        (
            "schema: ai.model.v1\n"
            "provider: openai\n"
            "model: gpt-4o-mini\n"
            "base_url: https://api.openai.com/v1\n"
            "api_key_env: TEST_KEY\n"
            "timeout_seconds: 45\n"
            "max_in_flight: 6\n"
        ),
        encoding="utf-8",
    )

    config = load_model_configs(ai_paths)
    assert config.default_profile == "ollama-fast"
    assert config.profiles["openai-fast"].max_in_flight == 6

    monkeypatch.setenv("TEST_KEY", "secret")
    runtime = resolve_runtime_config(config, profile="openai-fast")
    assert runtime.provider == "openai"
    assert runtime.timeout_seconds == 45
    assert runtime.max_in_flight == 6
    assert runtime.api_key == "secret"


def test_load_models_config_defaults_groq_api_key_env(tmp_path):
    ai_paths = _mk_ai_paths(tmp_path)
    (ai_paths.models / "groq-fast.yaml").write_text(
        (
            "schema: ai.model.v1\n"
            "provider: groq\n"
            "model: llama-3.1-8b-instant\n"
        ),
        encoding="utf-8",
    )

    config = load_model_configs(ai_paths)
    assert config.profiles["groq-fast"].api_key_env_override is None

    runtime = resolve_runtime_config(config, profile="groq-fast")
    assert runtime.base_url == "https://api.groq.com/openai/v1"
    assert runtime.api_key_env == "GROQ_API_KEY"


def test_resolve_runtime_provider_override_to_groq_uses_groq_defaults(
    tmp_path,
    monkeypatch,
):
    ai_paths = _mk_ai_paths(tmp_path)
    (ai_paths.models / "openai-fast.yaml").write_text(
        (
            "schema: ai.model.v1\n"
            "default: true\n"
            "provider: openai\n"
            "model: gpt-4o-mini\n"
        ),
        encoding="utf-8",
    )
    config = load_model_configs(ai_paths)

    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    runtime = resolve_runtime_config(config, provider="groq")

    assert runtime.provider == "groq"
    assert runtime.base_url == "https://api.groq.com/openai/v1"
    assert runtime.api_key_env == "GROQ_API_KEY"
    assert runtime.api_key == "groq-secret"


def test_provider_choices_include_supported_registry_ids():
    assert set(provider_choices()) == {"groq", "ollama", "openai"}


def test_load_models_config_rejects_unknown_fields(tmp_path):
    ai_paths = _mk_ai_paths(tmp_path)
    (ai_paths.models / "ollama-fast.yaml").write_text(
        (
            "schema: ai.model.v1\n"
            "provider: ollama\n"
            "model: llama3.1:8b\n"
            "base_url: http://localhost:11434/v1\n"
            "unexpected_field: true\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(AIConfigError, match="unexpected_field"):
        load_model_configs(ai_paths)


def test_load_models_config_rejects_legacy_id_field(tmp_path):
    ai_paths = _mk_ai_paths(tmp_path)
    (ai_paths.models / "ollama-fast.yaml").write_text(
        (
            "schema: ai.model.v1\n"
            "id: ollama-fast\n"
            "provider: ollama\n"
            "model: llama3.1:8b\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(AIConfigError, match="id"):
        load_model_configs(ai_paths)


def test_resolve_runtime_rejects_non_positive_overrides(tmp_path):
    ai_paths = _mk_ai_paths(tmp_path)
    (ai_paths.models / "ollama-fast.yaml").write_text(
        (
            "schema: ai.model.v1\n"
            "provider: ollama\n"
            "model: llama3.1:8b\n"
            "base_url: http://localhost:11434/v1\n"
        ),
        encoding="utf-8",
    )
    config = load_model_configs(ai_paths)

    with pytest.raises(AIConfigError, match="max_in_flight"):
        resolve_runtime_config(config, max_in_flight=0)


def test_prepare_ai_run_uses_task_model_and_overrides(tmp_path):
    ai_paths = _mk_ai_paths(tmp_path)
    (ai_paths.models / "ollama-fast.yaml").write_text(
        (
            "schema: ai.model.v1\n"
            "default: true\n"
            "provider: ollama\n"
            "model: llama3.1:8b\n"
            "base_url: http://localhost:11434/v1\n"
        ),
        encoding="utf-8",
    )
    (ai_paths.models / "openai-fast.yaml").write_text(
        (
            "schema: ai.model.v1\n"
            "provider: openai\n"
            "model: gpt-4o-mini\n"
            "base_url: https://api.openai.com/v1\n"
        ),
        encoding="utf-8",
    )
    (ai_paths.tasks / "grammar.yaml").write_text(
        (
            "schema: ai.task.v1\n"
            "model: openai-fast\n"
            "instructions: Return inline JSON.\n"
            "read_fields:\n"
            "  - Question\n"
            "write_fields:\n"
            "  - Question\n"
        ),
        encoding="utf-8",
    )

    collection_dir = ai_paths.root.parent
    _, runtime_from_task = prepare_ai_run(collection_dir, "grammar")
    assert runtime_from_task.profile == "openai-fast"

    _, runtime_from_override = prepare_ai_run(
        collection_dir,
        "grammar",
        overrides=AIRuntimeOverrides(profile="ollama-fast"),
    )
    assert runtime_from_override.profile == "ollama-fast"


def test_prepare_ai_run_uses_default_profile_when_task_model_omitted(tmp_path):
    ai_paths = _mk_ai_paths(tmp_path)
    (ai_paths.models / "ollama-fast.yaml").write_text(
        (
            "schema: ai.model.v1\n"
            "default: true\n"
            "provider: ollama\n"
            "model: llama3.1:8b\n"
            "base_url: http://localhost:11434/v1\n"
        ),
        encoding="utf-8",
    )
    (ai_paths.models / "openai-fast.yaml").write_text(
        (
            "schema: ai.model.v1\n"
            "provider: openai\n"
            "model: gpt-4o-mini\n"
            "base_url: https://api.openai.com/v1\n"
        ),
        encoding="utf-8",
    )
    (ai_paths.tasks / "grammar.yaml").write_text(
        (
            "schema: ai.task.v1\n"
            "instructions: Return inline JSON.\n"
            "read_fields:\n"
            "  - Question\n"
            "write_fields:\n"
            "  - Question\n"
        ),
        encoding="utf-8",
    )

    collection_dir = ai_paths.root.parent
    task_config, runtime = prepare_ai_run(collection_dir, "grammar")
    assert task_config.model is None
    assert runtime.profile == "ollama-fast"


def test_prepare_ai_run_uses_default_profile_when_task_model_is_default(
    tmp_path,
):
    ai_paths = _mk_ai_paths(tmp_path)
    (ai_paths.models / "ollama-fast.yaml").write_text(
        (
            "schema: ai.model.v1\n"
            "default: true\n"
            "provider: ollama\n"
            "model: llama3.1:8b\n"
            "base_url: http://localhost:11434/v1\n"
        ),
        encoding="utf-8",
    )
    (ai_paths.models / "openai-fast.yaml").write_text(
        (
            "schema: ai.model.v1\n"
            "provider: openai\n"
            "model: gpt-4o-mini\n"
            "base_url: https://api.openai.com/v1\n"
        ),
        encoding="utf-8",
    )
    (ai_paths.tasks / "grammar.yaml").write_text(
        (
            "schema: ai.task.v1\n"
            "model: default\n"
            "instructions: Return inline JSON.\n"
            "read_fields:\n"
            "  - Question\n"
            "write_fields:\n"
            "  - Question\n"
        ),
        encoding="utf-8",
    )

    collection_dir = ai_paths.root.parent
    task_config, runtime = prepare_ai_run(collection_dir, "grammar")
    assert task_config.model is None
    assert runtime.profile == "ollama-fast"


def test_inline_task_updates_only_target_fields_batch_mode():
    task = _task(
        read_fields=["Question", "Answer"],
        write_fields=["Question"],
        batch="batch",
        batch_size=2,
    )
    data = {
        "decks": [
            {
                "name": "Biology",
                "notes": [
                    {
                        "note_key": "n1",
                        "note_type": "AnkiOpsQA",
                        "fields": {
                            "Question": "I has two lungs.",
                            "Answer": "Humans have two lungs.",
                        },
                    },
                    {
                        "note_key": "n2",
                        "note_type": "AnkiOpsQA",
                        "fields": {
                            "Question": "I has one heart.",
                            "Answer": "Humans have one heart.",
                        },
                    },
                ],
            }
        ]
    }

    options = TaskRunOptions(include_decks=["Biology"], batch_size=2, max_in_flight=2)
    result = TaskRunner(_InlineBatchEditor()).run(data, task, options=options)

    assert result.matched_notes == 2
    assert result.changed_fields == 2
    fields = result.changed_decks[0]["notes"][0]["fields"]
    assert fields["Question"] == "I have two lungs."
    assert fields["Answer"] == "Humans have two lungs."


def test_inline_task_accepts_target_only_response_fields():
    task = _task(
        read_fields=["Question", "Answer"],
        write_fields=["Question"],
    )
    data = {
        "decks": [
            {
                "name": "Biology",
                "notes": [
                    {
                        "note_key": "n1",
                        "note_type": "AnkiOpsQA",
                        "fields": {
                            "Question": "I has two lungs.",
                            "Answer": "Humans have two lungs.",
                        },
                    }
                ],
            }
        ]
    }

    result = TaskRunner(_TargetOnlyBatchEditor()).run(data, task)

    assert result.changed_fields == 1
    assert not result.warnings
    fields = result.changed_decks[0]["notes"][0]["fields"]
    assert fields["Question"] == "I have two lungs."
    assert fields["Answer"] == "Humans have two lungs."


def test_inline_task_recursive_deck_selection():
    task = _task()
    data = {
        "decks": [
            {
                "name": "Biology",
                "notes": [
                    {
                        "note_key": "n1",
                        "note_type": "AnkiOpsQA",
                        "fields": {"Question": "I has two lungs."},
                    }
                ],
            },
            {
                "name": "Biology::Cells",
                "notes": [
                    {
                        "note_key": "n2",
                        "note_type": "AnkiOpsQA",
                        "fields": {"Question": "I has one nucleus."},
                    }
                ],
            },
            {
                "name": "History",
                "notes": [
                    {
                        "note_key": "n3",
                        "note_type": "AnkiOpsQA",
                        "fields": {"Question": "I has a timeline."},
                    }
                ],
            },
        ]
    }
    result = TaskRunner(_InlineBatchEditor()).run(
        data,
        task,
        options=TaskRunOptions(include_decks=["Biology"]),
    )

    assert result.processed_decks == 2
    assert result.matched_notes == 2
    assert len(result.changed_decks) == 2


def test_inline_task_rejects_mismatched_note_key():
    task = _task()
    data = {
        "decks": [
            {
                "name": "Biology",
                "notes": [
                    {
                        "note_key": "n1",
                        "note_type": "AnkiOpsQA",
                        "fields": {"Question": "I has two lungs."},
                    }
                ],
            }
        ]
    }

    result = TaskRunner(_WrongKeyBatchEditor()).run(data, task)

    assert result.changed_fields == 0
    assert result.warnings
    assert "note_key mismatch" in result.warnings[0]


def test_inline_task_skips_notes_without_note_key():
    task = _task()
    data = {
        "decks": [
            {
                "name": "Biology",
                "notes": [
                    {
                        "note_key": None,
                        "note_type": "AnkiOpsQA",
                        "fields": {"Question": "I has two lungs."},
                    }
                ],
            }
        ]
    }

    result = TaskRunner(_InlineBatchEditor()).run(data, task)

    assert result.matched_notes == 0
    assert result.changed_fields == 0
    assert "skipped note without note_key" in result.warnings[0]


def test_inline_task_skips_notes_without_note_type():
    task = _task(scope_note_types=["*"])
    data = {
        "decks": [
            {
                "name": "Biology",
                "notes": [
                    {
                        "note_key": "n1",
                        "note_type": None,
                        "fields": {"Question": "I has two lungs."},
                    }
                ],
            }
        ]
    }

    result = TaskRunner(_InlineBatchEditor()).run(data, task)

    assert result.matched_notes == 0
    assert result.changed_fields == 0
    assert "skipped note without note_type" in result.warnings[0]


def test_inline_task_invalid_batch_size_raises():
    task = _task()
    with pytest.raises(TaskExecutionError, match="batch_size"):
        TaskRunner(_InlineBatchEditor()).run(
            {"decks": []},
            task,
            options=TaskRunOptions(batch_size=0),
        )


def test_inline_task_invalid_max_warnings_raises():
    task = _task()
    with pytest.raises(TaskExecutionError, match="max_warnings"):
        TaskRunner(_InlineBatchEditor()).run(
            {"decks": []},
            task,
            options=TaskRunOptions(max_warnings=0),
        )


def test_inline_task_caps_warnings():
    task = _task()
    data = {
        "decks": [
            {
                "name": "Biology",
                "notes": [
                    {
                        "note_key": f"n{i}",
                        "note_type": "AnkiOpsQA",
                        "fields": {"Question": "I has two lungs."},
                    }
                    for i in range(1, 6)
                ],
            }
        ]
    }

    result = TaskRunner(_FailingBatchEditor()).run(
        data,
        task,
        options=TaskRunOptions(batch_size=1, max_in_flight=2, max_warnings=2),
    )

    assert len(result.warnings) == 2
    assert result.dropped_warnings == 3


def test_inline_task_warning_includes_exception_type():
    task = _task()
    data = {
        "decks": [
            {
                "name": "Biology",
                "notes": [
                    {
                        "note_key": "n1",
                        "note_type": "AnkiOpsQA",
                        "fields": {"Question": "I has two lungs."},
                    }
                ],
            }
        ]
    }

    result = TaskRunner(_FailingBatchEditor()).run(data, task)

    assert result.warnings
    assert "RuntimeError: boom" in result.warnings[0]


def test_task_runner_closes_editor_after_run():
    editor = _ClosableBatchEditor()
    result = TaskRunner(editor).run({"decks": []}, _task())

    assert result.changed_fields == 0
    assert editor.closed


def test_normalize_patch_response_accepts_strict_json_object():
    content = '{"patches":[{"ref":"r1","fields":{"Question":"I have two lungs."}}]}'

    normalized = normalize_patch_response(content)
    assert normalized["r1"]["Question"] == "I have two lungs."


def test_parse_json_object_rejects_prefix_suffix_noise():
    content = (
        "Sure, here you go:\n"
        '{"patches":[{"ref":"r1","fields":{"Question":"I have two lungs."}}]}\n'
        "Thanks!"
    )

    with pytest.raises(AIResponseError, match="strict JSON object"):
        parse_json_object(content)


def test_parse_json_object_rejects_malformed_or_embedded_json():
    content = (
        'Noise {"broken": } still noise '
        '{"patches":[{"ref":"r1","fields":{"Question":"I have two lungs."}}]}'
    )

    with pytest.raises(AIResponseError, match="strict JSON object"):
        parse_json_object(content)


def test_normalize_patch_response_rejects_missing_patches_list():
    with pytest.raises(AIResponseError, match="top-level 'patches' list"):
        normalize_patch_response("{}")


@respx.mock
def test_client_reuses_http_client_within_run():
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"patches":[{"ref":"r1","fields":'
                                    '{"Question":"I have two lungs."}}]}'
                                )
                            }
                        }
                    ]
                },
            ),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"patches":[{"ref":"r1","fields":'
                                    '{"Question":"I have two lungs."}}]}'
                                )
                            }
                        }
                    ]
                },
            ),
        ]
    )

    async def _run() -> None:
        editor = OpenAICompatibleAsyncEditor(_runtime())
        try:
            await editor.edit_notes(_task(), _notes_payload())
            first_client = editor._client
            assert first_client is not None
            await editor.edit_notes(_task(), _notes_payload())
            assert editor._client is first_client
        finally:
            await editor.aclose()

    asyncio.run(_run())
    assert route.call_count == 2


@respx.mock
def test_client_retries_transient_429_then_succeeds(monkeypatch):
    monkeypatch.setattr("ankiops.ai.client.DEFAULT_RETRY_MIN_SECONDS", 0.0)
    monkeypatch.setattr("ankiops.ai.client.DEFAULT_RETRY_MAX_SECONDS", 0.0)

    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "rate limited"}}),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"patches":[{"ref":"r1","fields":'
                                    '{"Question":"I have two lungs."}}]}'
                                )
                            }
                        }
                    ]
                },
            ),
        ]
    )

    result = asyncio.run(_run_editor_once())
    assert route.call_count == 2
    assert result["n1"].fields["Question"] == "I have two lungs."


@respx.mock
def test_client_sends_task_temperature_in_payload():
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"patches":[{"ref":"r1","fields":'
                                '{"Question":"I have two lungs."}}]}'
                            )
                        }
                    }
                ]
            },
        )
    )

    asyncio.run(_run_editor_once(task=_task(temperature=0.7)))

    request_payload = json.loads(route.calls[0].request.content.decode("utf-8"))
    assert request_payload["temperature"] == pytest.approx(0.7)
    request_body = json.loads(request_payload["messages"][1]["content"])
    assert request_body["task"] == "inline_json_edit"
    assert request_body["notes"][0]["ref"] == "r1"
    assert request_body["notes"][0]["editable"] == {"Question": "I has two lungs."}
    assert request_body["notes"][0]["context"] == {}
    assert "field_policy" not in request_body
    assert "output_schema" not in request_body
    assert "note_key" not in request_body["notes"][0]
    assert "note_type" not in request_body["notes"][0]


@respx.mock
def test_client_respects_retry_after_header(monkeypatch):
    monkeypatch.setattr("ankiops.ai.client.DEFAULT_RETRY_MIN_SECONDS", 0.0)
    monkeypatch.setattr("ankiops.ai.client.DEFAULT_RETRY_MAX_SECONDS", 0.0)

    delays: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("ankiops.ai.client.asyncio.sleep", _fake_sleep)

    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                429,
                headers={"Retry-After": "1.5"},
                json={"error": {"message": "rate limited"}},
            ),
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"patches":[{"ref":"r1","fields":'
                                    '{"Question":"I have two lungs."}}]}'
                                )
                            }
                        }
                    ]
                },
            ),
        ]
    )

    result = asyncio.run(_run_editor_once())
    assert route.call_count == 2
    assert result["n1"].fields["Question"] == "I have two lungs."
    assert delays == [pytest.approx(1.5)]


@respx.mock
def test_client_fail_fast_non_retryable_400():
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            400,
            json={"error": {"message": "bad request"}},
        )
    )

    with pytest.raises(AIRequestError, match="400"):
        asyncio.run(_run_editor_once())
    assert route.call_count == 1


@respx.mock
def test_client_retries_retryable_response_and_exhausts_attempts(monkeypatch):
    monkeypatch.setattr("ankiops.ai.client.DEFAULT_RETRY_MIN_SECONDS", 0.0)
    monkeypatch.setattr("ankiops.ai.client.DEFAULT_RETRY_MAX_SECONDS", 0.0)

    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            500,
            text="server error",
        )
    )

    with pytest.raises(AIRequestError, match="500"):
        asyncio.run(_run_editor_once())
    assert route.call_count == 3


@respx.mock
def test_client_raises_response_error_on_non_json_response():
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text="not-json",
        )
    )

    with pytest.raises(AIResponseError, match="valid JSON"):
        asyncio.run(_run_editor_once())
    assert route.call_count == 1


@respx.mock
def test_client_raises_response_error_on_missing_assistant_text():
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": []}}]},
        )
    )

    with pytest.raises(AIResponseError, match="assistant text content"):
        asyncio.run(_run_editor_once())
    assert route.call_count == 1


@respx.mock
def test_client_rejects_response_with_missing_ref_patch():
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"patches":[{"ref":"r2","fields":{"Question":"x"}}]}'
                        }
                    }
                ]
            },
        )
    )

    with pytest.raises(AIResponseError, match="missing refs"):
        asyncio.run(_run_editor_once())
    assert route.call_count == 1


@respx.mock
def test_client_rejects_response_with_unknown_ref_patch():
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"patches":['
                                '{"ref":"r1","fields":{"Question":"x"}},'
                                '{"ref":"r9","fields":{"Question":"y"}}]}'
                            )
                        }
                    }
                ]
            },
        )
    )

    with pytest.raises(AIResponseError, match="unknown refs"):
        asyncio.run(_run_editor_once())
    assert route.call_count == 1
