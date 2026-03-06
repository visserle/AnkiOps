from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path
from textwrap import dedent

import pytest

from ankiops.db import SQLiteDbAdapter
from ankiops.fs import FileSystemAdapter
from ankiops.init import initialize_collection
from ankiops.llm.anthropic_models import SONNET
from ankiops.llm.config_loader import load_llm_task_catalog
from ankiops.llm.models import GenerateUpdateResult, NoteUpdate
from ankiops.llm.runner import run_task

TASK_FILE = Path("llm/grammar.yaml")
TEST_DECK = "TestDeck"
TEST_DECK_MARKDOWN = """
<!-- note_key: nk-1 -->
Q: this is a broken question.
A: this is a broken answer.
S: grammar book
AI: hidden content

---

<!-- note_key: nk-2 -->
Q: pick one
C1: yes
C2: no
A: 1
"""
DEFAULT_TASK_EXTRA = """
fields:
  exceptions:
    - read_only: ["Source"]
    - note_types: ["AnkiOpsChoice"]
      read_only: ["Answer"]
    - hidden: ["AI Notes"]
"""


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def _task_config(
    *,
    model: str = "sonnet",
    prompt: str = "fix grammar",
    extra: str = "",
) -> str:
    suffix = f"\n{dedent(extra).strip()}" if extra.strip() else ""
    return f"name: grammar\nmodel: {model}\nprompt: {prompt}{suffix}\n"


def _write_task(collection_dir: Path, *, content: str) -> None:
    _write(collection_dir / TASK_FILE, content)


def _load_note_type_configs(collection_dir: Path):
    fs = FileSystemAdapter()
    note_types_dir = collection_dir / "note_types"
    fs.eject_builtin_note_types(note_types_dir)
    return fs.load_note_type_configs(note_types_dir)


def _patch_collection_paths(monkeypatch, collection_dir: Path) -> None:
    monkeypatch.setattr("ankiops.config.get_collection_dir", lambda: collection_dir)
    monkeypatch.setattr(
        "ankiops.collection_serializer.get_collection_dir",
        lambda: collection_dir,
    )
    monkeypatch.setattr(
        "ankiops.config.get_note_types_dir",
        lambda: collection_dir / "note_types",
    )
    monkeypatch.setattr(
        "ankiops.collection_serializer.get_note_types_dir",
        lambda: collection_dir / "note_types",
    )


def _prepare_runner_collection(
    tmp_path: Path,
    monkeypatch,
    *,
    task_content: str | None = None,
) -> Path:
    _patch_collection_paths(monkeypatch, tmp_path)
    db = SQLiteDbAdapter.load(tmp_path)
    db.close()

    _load_note_type_configs(tmp_path)
    _write(tmp_path / f"{TEST_DECK}.md", TEST_DECK_MARKDOWN)
    _write_task(
        tmp_path,
        content=task_content or _task_config(extra=DEFAULT_TASK_EXTRA),
    )
    return tmp_path


class _StubClient:
    def __init__(self, results: list[GenerateUpdateResult]) -> None:
        self._results = results

    def generate_update(self, **_kwargs) -> GenerateUpdateResult:
        return self._results.pop(0)


def _result(
    note_key: str,
    edits: dict[str, str],
    *,
    message_id: str = "msg_123",
    model: str = "claude-sonnet-4-6",
    stop_reason: str = "end_turn",
    input_tokens: int = 11,
    output_tokens: int = 7,
    latency_ms: int = 900,
) -> GenerateUpdateResult:
    return GenerateUpdateResult(
        update=NoteUpdate(note_key=note_key, edits=edits),
        message_id=message_id,
        model=model,
        stop_reason=stop_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
    )


@pytest.fixture
def note_type_configs(tmp_path: Path):
    return _load_note_type_configs(tmp_path)


def test_initialize_collection_ejects_packaged_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr("ankiops.init.get_collection_dir", lambda: tmp_path)
    monkeypatch.setattr("ankiops.init._setup_git", lambda _collection_dir: None)

    collection_dir = initialize_collection("TestProfile")

    packaged_tasks = sorted(
        resource.name
        for resource in resources.files("ankiops.llm.tasks").iterdir()
        if resource.is_file() and resource.suffix == ".yaml"
    )
    ejected_tasks = sorted(path.name for path in (tmp_path / "llm").glob("*.yaml"))

    assert collection_dir == tmp_path
    assert ejected_tasks == packaged_tasks
    assert "model: sonnet" in (
        tmp_path / "llm/grammar.yaml"
    ).read_text(encoding="utf-8")


def test_initialize_collection_preserves_existing_task(tmp_path, monkeypatch):
    monkeypatch.setattr("ankiops.init.get_collection_dir", lambda: tmp_path)
    monkeypatch.setattr("ankiops.init._setup_git", lambda _collection_dir: None)

    existing_task = tmp_path / TASK_FILE
    existing_task.parent.mkdir(parents=True, exist_ok=True)
    existing_task.write_text("name: grammar\nprompt: keep mine\n", encoding="utf-8")

    initialize_collection("TestProfile")

    assert existing_task.read_text(encoding="utf-8") == (
        "name: grammar\nprompt: keep mine\n"
    )


def test_load_llm_task_catalog_loads_valid_task(note_type_configs, tmp_path: Path):
    _write_task(
        tmp_path,
        content=_task_config(
            extra="""
            decks:
              include: ["Parent"]
              include_subdecks: false
            fields:
              exceptions:
                - read_only: ["Source"]
                - note_types: ["AnkiOpsChoice"]
                  read_only: ["Answer"]
                - hidden: ["AI Notes"]
            """
        ),
    )

    catalog = load_llm_task_catalog(tmp_path, note_type_configs=note_type_configs)

    assert not catalog.errors
    task = catalog.tasks_by_name["grammar"]
    assert task.model == SONNET
    assert task.api_key_env == "ANTHROPIC_API_KEY"
    assert task.decks.include == ["Parent"]
    assert task.decks.include_subdecks is False


@pytest.mark.parametrize(
    ("task_content", "expected_error"),
    [
        (
            _task_config(
                extra="""
                fields:
                  exceptions:
                    - note_types: ["AnkiOpsQA"]
                      read_only: ["DoesNotExist"]
                """
            ),
            "DoesNotExist",
        ),
        (
            _task_config(
                extra="""
                decks:
                  include: ["Parent"]
                  include_subdecks: "yes"
                """
            ),
            "decks.include_subdecks",
        ),
        (
            _task_config(model="gpt-5"),
            "must be one of: opus, sonnet, haiku",
        ),
        (
            _task_config(extra="sdk: anthropic"),
            "unknown task key(s): sdk",
        ),
    ],
    ids=[
        "invalid-exception-field",
        "invalid-include-subdecks",
        "non-claude-model",
        "legacy-provider-key",
    ],
)
def test_load_llm_task_catalog_rejects_invalid_tasks(
    note_type_configs,
    tmp_path: Path,
    task_content: str,
    expected_error: str,
):
    _write_task(tmp_path, content=task_content)

    catalog = load_llm_task_catalog(tmp_path, note_type_configs=note_type_configs)

    assert not catalog.tasks_by_name
    assert expected_error in next(iter(catalog.errors.values()))


def test_load_llm_task_catalog_ignores_non_task_dirs(
    note_type_configs,
    tmp_path: Path,
):
    _write_task(tmp_path, content=_task_config())
    _write(
        tmp_path / "llm/actions/old.yaml",
        """
        name: old
        prompt: should be ignored
        """,
    )
    _write(
        tmp_path / "llm/providers/anthropic.yaml",
        """
        name: anthropic
        model: sonnet
        """,
    )

    catalog = load_llm_task_catalog(tmp_path, note_type_configs=note_type_configs)

    assert not catalog.errors
    assert set(catalog.tasks_by_name) == {"grammar"}


def test_run_task_updates_only_editable_fields(tmp_path, monkeypatch):
    collection = _prepare_runner_collection(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "ankiops.llm.runner.git_snapshot", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        "ankiops.llm.runner.ClaudeClient",
        lambda _task: _StubClient(
            [
                _result(
                    "nk-1",
                    {"Question": "This is a fixed question."},
                    input_tokens=21,
                    output_tokens=9,
                    latency_ms=1200,
                ),
                _result(
                    "nk-2",
                    {},
                    input_tokens=13,
                    output_tokens=4,
                    latency_ms=800,
                ),
            ]
        ),
    )

    summary = run_task(
        collection_dir=collection,
        task_name="grammar",
        no_auto_commit=True,
    )

    content = (collection / f"{TEST_DECK}.md").read_text(encoding="utf-8")

    assert summary.model == SONNET
    assert summary.requests == 2
    assert summary.input_tokens == 34
    assert summary.output_tokens == 13
    assert summary.provider_latency_ms_total == 2000
    assert summary.updated == 1
    assert summary.unchanged == 1
    assert "Q: This is a fixed question." in content
    assert "S: grammar book" in content
    assert "AI: hidden content" in content
    assert "A: 1" in content


def test_run_task_logs_debug_lifecycle(tmp_path, monkeypatch, caplog):
    collection = _prepare_runner_collection(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "ankiops.llm.runner.git_snapshot", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        "ankiops.llm.runner.ClaudeClient",
        lambda _task: _StubClient(
            [
                _result(
                    "nk-1",
                    {"Question": "This is a fixed question."},
                    input_tokens=21,
                    output_tokens=9,
                    latency_ms=1200,
                ),
                _result(
                    "nk-2",
                    {},
                    input_tokens=13,
                    output_tokens=4,
                    latency_ms=800,
                ),
            ]
        ),
    )

    with caplog.at_level(logging.DEBUG):
        summary = run_task(
            collection_dir=collection,
            task_name="grammar",
            no_auto_commit=True,
        )

    assert summary.requests == 2
    assert (
        "Starting LLM task 'grammar' (model=sonnet, "
        "api_model=claude-sonnet-4-6"
    ) in caplog.text
    assert (
        "LLM request defaults: timeout=60s max_tokens=2048 temperature=default"
    ) in caplog.text
    assert "LLM serializer scope: *" in caplog.text
    assert "Auto-commit disabled (--no-auto-commit)" in caplog.text
    assert "Serialized 1 deck(s), 2 note(s) in memory" in caplog.text
    assert "  Updated nk-1 in 'TestDeck' (AnkiOpsQA): fields=Question" in caplog.text
    assert "  Unchanged nk-2 in 'TestDeck' (AnkiOpsChoice)" in caplog.text
    assert (
        "LLM task 'grammar' (sonnet): 2 notes — "
        "1 updated, 1 unchanged"
    ) in caplog.text
    assert (
        "LLM usage: 2 requests, 34 input tokens, 13 output tokens, "
        "2.0s provider time"
    ) in caplog.text
    assert (
        "LLM estimated cost: $0.000297 ($0.000102 input + $0.000195 output)"
    ) in caplog.text
    assert "Broken" not in caplog.text
    assert "<task>" not in caplog.text
    assert '{"note_key"' not in caplog.text


def test_run_task_rejects_read_only_updates(tmp_path, monkeypatch, caplog):
    collection = _prepare_runner_collection(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "ankiops.llm.runner.git_snapshot", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        "ankiops.llm.runner.ClaudeClient",
        lambda _task: _StubClient(
            [
                _result("nk-1", {}),
                _result("nk-2", {"Answer": "2"}),
            ]
        ),
    )

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(SystemExit) as exc:
            run_task(
                collection_dir=collection,
                task_name="grammar",
                no_auto_commit=True,
            )

    assert exc.value.code == 1
    assert "A: 1" in (collection / f"{TEST_DECK}.md").read_text(encoding="utf-8")
    assert (
        "LLM note error for nk-2 in 'TestDeck' (AnkiOpsChoice): "
        "Model attempted to update read-only field 'Answer'"
    ) in caplog.text
    assert (
        "LLM task 'grammar' (sonnet): 2 notes — "
        "1 unchanged, 1 error"
    ) in caplog.text
    assert (
        "LLM usage: 2 requests, 22 input tokens, 14 output tokens, "
        "1.8s provider time"
    ) in caplog.text
    assert (
        "LLM estimated cost: $0.000276 ($0.000066 input + $0.00021 output)"
    ) in caplog.text


def test_run_task_logs_deck_scope_skips(tmp_path, monkeypatch, caplog):
    collection = _prepare_runner_collection(
        tmp_path,
        monkeypatch,
        task_content=_task_config(
            extra="""
            decks:
              include: ["Other*"]
            fields:
              exceptions:
                - read_only: ["Source"]
            """
        ),
    )
    monkeypatch.setattr(
        "ankiops.llm.runner.git_snapshot", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        "ankiops.llm.runner.ClaudeClient",
        lambda _task: _StubClient([]),
    )

    with caplog.at_level(logging.DEBUG):
        summary = run_task(
            collection_dir=collection,
            task_name="grammar",
            no_auto_commit=True,
        )

    assert summary.decks_seen == 1
    assert summary.decks_matched == 0
    assert summary.notes_seen == 2
    assert summary.skipped_deck_scope == 2
    assert "Skipping deck 'TestDeck' (2 notes): outside task scope" in caplog.text
    assert "LLM task 'grammar' (sonnet): 0 notes — 2 skipped" in caplog.text


def test_run_task_logs_no_editable_field_skips(tmp_path, monkeypatch, caplog):
    collection = _prepare_runner_collection(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "ankiops.llm.runner.git_snapshot", lambda *_args, **_kwargs: False
    )

    def _fake_serialize(*_args, **_kwargs):
        return {
            "collection": {"serialized_at": "2026-03-02T00:00:00Z"},
            "decks": [
                {
                    "name": TEST_DECK,
                    "notes": [
                        {
                            "note_key": "nk-ro",
                            "note_type": "AnkiOpsQA",
                            "fields": {
                                "Source": "grammar book",
                                "AI Notes": "hidden content",
                            },
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr("ankiops.llm.runner.serialize_collection", _fake_serialize)
    monkeypatch.setattr(
        "ankiops.llm.runner.ClaudeClient",
        lambda _task: _StubClient([]),
    )

    with caplog.at_level(logging.DEBUG):
        summary = run_task(
            collection_dir=collection,
            task_name="grammar",
            no_auto_commit=True,
        )

    assert summary.notes_seen == 1
    assert summary.skipped_no_editable_fields == 1
    assert (
        "  Skipped nk-ro in 'TestDeck' (AnkiOpsQA): "
        "no editable non-empty fields"
    ) in caplog.text
    assert "LLM task 'grammar' (sonnet): 0 notes — 1 skipped" in caplog.text


@pytest.mark.parametrize(
    ("task_content", "expected_scope"),
    [
        (
            _task_config(
                extra=f"""
                decks:
                  include: ["{TEST_DECK}"]
                fields:
                  exceptions:
                    - read_only: ["Source"]
                """
            ),
            {"deck": TEST_DECK, "no_subdecks": False},
        ),
        (
            _task_config(
                extra=f"""
                decks:
                  include: ["{TEST_DECK}"]
                  include_subdecks: false
                fields:
                  exceptions:
                    - read_only: ["Source"]
                """
            ),
            {"deck": TEST_DECK, "no_subdecks": True},
        ),
        (
            _task_config(
                extra="""
                decks:
                  include: ["Test*"]
                fields:
                  exceptions:
                    - read_only: ["Source"]
                """
            ),
            {"deck": None, "no_subdecks": False},
        ),
    ],
    ids=["exact-deck", "exact-deck-without-subdecks", "wildcard-deck"],
)
def test_run_task_uses_expected_serialize_scope(
    tmp_path: Path,
    monkeypatch,
    task_content: str,
    expected_scope: dict[str, object],
):
    collection = _prepare_runner_collection(
        tmp_path,
        monkeypatch,
        task_content=task_content,
    )
    captured: dict[str, object] = {}

    def _fake_serialize(
        _collection_dir,
        *,
        deck: str | None = None,
        no_subdecks: bool = False,
    ):
        captured["deck"] = deck
        captured["no_subdecks"] = no_subdecks
        return {
            "collection": {"serialized_at": "2026-03-02T00:00:00Z"},
            "decks": [],
        }

    monkeypatch.setattr("ankiops.llm.runner.serialize_collection", _fake_serialize)
    monkeypatch.setattr(
        "ankiops.llm.runner.ClaudeClient",
        lambda _task: _StubClient([]),
    )

    run_task(
        collection_dir=collection,
        task_name="grammar",
        no_auto_commit=True,
    )

    assert captured == expected_scope
