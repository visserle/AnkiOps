"""Tests for LLM task execution behavior."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from types import SimpleNamespace

from ankiops.collection import LLM_DB_FILENAME
from ankiops.llm.execution import OpenAIResult, _validate_cloze_text_fields, run_task
from ankiops.note_types import CardTemplate
from ankiops.notes import Note
from ankiops.sync.state import SyncState
from tests.support.deck_files import DeckFileHarness


class _FakeAsyncOpenAI:
    def __init__(self, **_kwargs) -> None:
        pass

    async def close(self) -> None:
        pass


def _init_collection(collection_root) -> None:
    db = SyncState.open(collection_root)
    try:
        db.set_profile_name("test")
    finally:
        db.close()
    DeckFileHarness().eject_default_note_types(collection_root / "note_types")
    (collection_root / ".gitignore").write_text(
        "/collab/\n.ankiops.db*\n.ankiops/\n",
        encoding="utf-8",
    )


def _init_git_repo(collection_root) -> None:
    subprocess.run(
        ["git", "init"], cwd=collection_root, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=collection_root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=collection_root,
        check=True,
    )


def _commit_all(collection_root, message="root") -> None:
    subprocess.run(["git", "add", "-A", "."], cwd=collection_root, check=True)
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=collection_root,
        check=True,
        capture_output=True,
    )


def _head_name_status(collection_root) -> str:
    return subprocess.run(
        ["git", "show", "--name-status", "--format=", "HEAD"],
        cwd=collection_root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout


def _parsed_response(*updates, tag_updates=()):
    return SimpleNamespace(
        updates=[
            SimpleNamespace(note_key=note_key, field=field, value=value)
            for note_key, field, value in updates
        ],
        tag_updates=[
            SimpleNamespace(note_key=note_key, tags=tags)
            for note_key, tags in tag_updates
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


def test_executor_persists_successful_field_updates(
    llm_collection,
    write_file,
    monkeypatch,
):
    _init_collection(llm_collection)
    write_file(
        llm_collection / "Deck.md",
        """
        <!-- note_key: nk-1 -->
        <!-- tags: keep-me -->
        Q: Broken
        A: Existing answer
        E: Book

        ---

        <!-- note_key: nk-2 -->
        Q: Already good
        A: Existing answer
        E: Book
        """,
    )
    write_file(
        llm_collection / "llm/grammar.yaml",
        """
        model: test
        system_prompt: system
        user_prompt: user
        request:
          max_notes_per_request: 2
        fields:
          default_access: hidden
          editable:
            "AnkiOpsQA": ["Question"]
          read_only:
                        "AnkiOpsQA": ["Extra"]
        """,
    )

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

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("ankiops.llm.execution.AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr("ankiops.llm.execution._call_openai", fake_call_openai)

    result = run_task(
        collection_root=llm_collection,
        task_name="grammar",
        no_auto_commit=True,
    )

    content = (llm_collection / "Deck.md").read_text(encoding="utf-8")
    assert not result.failed
    assert result.persisted
    assert result.summary.updated == 1
    assert result.summary.unchanged == 1
    assert result.summary.requests == 1
    assert result.summary.input_tokens == 12
    assert result.summary.output_tokens == 6
    assert "Q: Fixed question" in content
    assert "<!-- tags: keep-me -->" in content
    assert "E: Book" in content
    assert "Q: Already good" in content


def test_executor_snapshots_only_queued_local_deck_paths(
    llm_collection,
    write_file,
    monkeypatch,
):
    _init_collection(llm_collection)
    write_file(
        llm_collection / "Deck.md",
        """
        <!-- note_key: nk-1 -->
        Q: Original
        A: Existing answer
        """,
    )
    write_file(llm_collection / "Other.txt", "unrelated\n")
    write_file(
        llm_collection / "llm/grammar.yaml",
        """
        model: test
        system_prompt: system
        user_prompt: user
        request:
          max_notes_per_request: 2
        fields:
          default_access: hidden
          editable:
            "AnkiOpsQA": ["Question"]
        """,
    )
    _init_git_repo(llm_collection)
    _commit_all(llm_collection)
    write_file(
        llm_collection / "Deck.md",
        """
        <!-- note_key: nk-1 -->
        Q: Broken before snapshot
        A: Existing answer
        """,
    )
    write_file(llm_collection / "Other.txt", "changed\n")

    async def fake_call_openai(*, batch, **_kwargs):
        assert [candidate.payload.note_key for candidate in batch.candidates] == [
            "nk-1"
        ]
        return _success(
            _parsed_response(("nk-1", "Question", "Fixed question")),
            input_tokens=12,
            output_tokens=6,
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("ankiops.llm.execution.AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr("ankiops.llm.execution._call_openai", fake_call_openai)

    result = run_task(
        collection_root=llm_collection,
        task_name="grammar",
        no_auto_commit=False,
    )

    committed = subprocess.run(
        ["git", "show", "HEAD:Deck.md"],
        cwd=llm_collection,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    working = (llm_collection / "Deck.md").read_text(encoding="utf-8")
    show = _head_name_status(llm_collection)

    assert result.persisted
    assert "M\tDeck.md" in show
    assert "Other.txt" not in show
    assert "Q: Broken before snapshot" in committed
    assert "Q: Fixed question" in working


def test_executor_uses_broad_snapshot_with_collab_without_committing_llm_db(
    llm_collection,
    write_file,
    monkeypatch,
):
    _init_collection(llm_collection)
    collab_root = llm_collection / "collab" / "owner" / "repo"
    DeckFileHarness().eject_default_note_types(collab_root / "note_types")
    _init_git_repo(collab_root)
    _commit_all(collab_root)
    write_file(
        llm_collection / "Deck.md",
        """
        <!-- note_key: nk-1 -->
        Q: Original
        A: Existing answer
        """,
    )
    write_file(llm_collection / "Other.txt", "unrelated\n")
    write_file(
        llm_collection / "llm/grammar.yaml",
        """
        model: test
        system_prompt: system
        user_prompt: user
        request:
          max_notes_per_request: 2
        fields:
          default_access: hidden
          editable:
            "AnkiOpsQA": ["Question"]
        """,
    )
    _init_git_repo(llm_collection)
    _commit_all(llm_collection)
    write_file(
        llm_collection / "Deck.md",
        """
        <!-- note_key: nk-1 -->
        Q: Broken before broad snapshot
        A: Existing answer
        """,
    )
    write_file(llm_collection / "Other.txt", "also changed\n")

    async def fake_call_openai(*, batch, **_kwargs):
        return _success(
            _parsed_response(("nk-1", "Question", "Fixed question")),
            input_tokens=12,
            output_tokens=6,
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("ankiops.llm.execution.AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr("ankiops.llm.execution._call_openai", fake_call_openai)

    result = run_task(
        collection_root=llm_collection,
        task_name="grammar",
        no_auto_commit=False,
    )

    show = _head_name_status(llm_collection)
    assert result.persisted
    assert "M\tDeck.md" in show
    assert "M\tOther.txt" in show
    assert LLM_DB_FILENAME not in show
    assert (llm_collection / "llm" / LLM_DB_FILENAME).exists()


def test_executor_persists_successful_tag_updates(
    llm_collection,
    write_file,
    monkeypatch,
):
    _init_collection(llm_collection)
    write_file(
        llm_collection / "Deck.md",
        """
        <!-- note_key: nk-1 -->
        <!-- tags: old -->
        Q: Question
        A: Answer
        """,
    )
    write_file(
        llm_collection / "llm/autotagger.yaml",
        """
        model: test
        system_prompt: system
        user_prompt: user
        tags: editable
        request:
          max_notes_per_request: 1
        fields:
          default_access: hidden
          read_only:
            "AnkiOpsQA": ["Question", "Answer"]
        """,
    )

    async def fake_call_openai(*, batch, **_kwargs):
        assert batch.candidates[0].payload.editable_tags == ("old",)
        return _success(
            _parsed_response(tag_updates=[("nk-1", ["new", "old", "new"])]),
            input_tokens=7,
            output_tokens=4,
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("ankiops.llm.execution.AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr("ankiops.llm.execution._call_openai", fake_call_openai)

    result = run_task(
        collection_root=llm_collection,
        task_name="autotagger",
        no_auto_commit=True,
    )

    content = (llm_collection / "Deck.md").read_text(encoding="utf-8")
    assert not result.failed
    assert result.persisted
    assert result.summary.updated == 1
    assert "<!-- tags: new old -->" in content


def test_executor_cancels_queued_items_after_fatal_error(
    llm_collection,
    write_file,
    monkeypatch,
):
    _init_collection(llm_collection)
    write_file(
        llm_collection / "Deck.md",
        """
        <!-- note_key: nk-1 -->
        Q: Broken
        A: Existing answer

        ---

        <!-- note_key: nk-2 -->
        Q: Already good
        A: Existing answer
        """,
    )
    write_file(
        llm_collection / "llm/_models.yaml",
        """
        - model: test
          model_id: gpt-test
          base_url: https://api.openai.com/v1
          api_key: $OPENAI_API_KEY
          concurrency: 1
        """,
    )
    write_file(
        llm_collection / "llm/grammar.yaml",
        """
        model: test
        system_prompt: system
        user_prompt: user
        request:
          max_notes_per_request: 1
        fields:
          default_access: hidden
          editable:
            "AnkiOpsQA": ["Question"]
        """,
    )

    async def fake_call_openai(**_kwargs):
        return _fatal_error()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("ankiops.llm.execution.AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr("ankiops.llm.execution._call_openai", fake_call_openai)

    result = run_task(
        collection_root=llm_collection,
        task_name="grammar",
        no_auto_commit=True,
    )

    assert result.failed
    assert not result.persisted
    assert result.summary.errors == 1
    assert result.summary.canceled == 1
    assert "Q: Broken" in (llm_collection / "Deck.md").read_text(encoding="utf-8")


def test_validate_cloze_text_fields_uses_template_cloze_sources(llm_qa_config):
    config = replace(
        llm_qa_config,
        is_cloze=True,
        templates=[CardTemplate("Cloze", "{{edit:cloze:Question}}", "{{Answer}}")],
    )
    valid_note = Note(
        note_key="nk-1",
        note_type="AnkiOpsQA",
        fields={
            "Question": "{{c1::broken question}}",
            "Answer": "Existing answer",
        },
    )
    invalid_note = Note(
        note_key="nk-2",
        note_type="AnkiOpsQA",
        fields={
            "Question": "Broken question",
            "Answer": "{{c1::wrong field}}",
        },
    )

    assert _validate_cloze_text_fields(valid_note, config) == []
    assert _validate_cloze_text_fields(invalid_note, config) == [
        "AnkiOpsQA field 'Question' must contain cloze syntax (e.g. {{c1::answer}})"
    ]
