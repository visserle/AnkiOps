from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from ankiops.db import SQLiteDbAdapter
from ankiops.fs import FileSystemAdapter
from ankiops.llm.models import NotePatch
from ankiops.llm.runner import run_task


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def _set_collection_paths(monkeypatch, collection_dir: Path) -> None:
    monkeypatch.setattr("ankiops.config.get_collection_dir", lambda: collection_dir)
    monkeypatch.setattr(
        "ankiops.collection_serializer.get_collection_dir",
        lambda: collection_dir,
    )
    monkeypatch.setattr(
        "ankiops.config.get_note_types_dir", lambda: collection_dir / "note_types"
    )
    monkeypatch.setattr(
        "ankiops.collection_serializer.get_note_types_dir",
        lambda: collection_dir / "note_types",
    )


def _prepare_collection(tmp_path: Path, monkeypatch) -> Path:
    _set_collection_paths(monkeypatch, tmp_path)
    db = SQLiteDbAdapter.load(tmp_path)
    db.close()

    fs = FileSystemAdapter()
    fs.eject_builtin_note_types(tmp_path / "note_types")

    _write(
        tmp_path / "TestDeck.md",
        """
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
        """,
    )
    _write(
        tmp_path / "llm/providers/ollama-local.yaml",
        """
        name: ollama-local
        type: ollama
        base_url: http://127.0.0.1:11434
        model: gpt-oss
        """,
    )
    _write(
        tmp_path / "llm/config.yaml",
        """
        default_provider: ollama-local
        """,
    )
    _write(
        tmp_path / "llm/tasks/grammar.yaml",
        """
        name: grammar
        prompt: fix grammar
        fields:
          exceptions:
            - read_only: ["Source"]
            - note_types: ["AnkiOpsChoice"]
              read_only: ["Answer"]
            - hidden: ["AI Notes"]
        """,
    )
    return tmp_path


class _StubProvider:
    def __init__(self, patches: list[NotePatch]) -> None:
        self._patches = patches

    def generate_patch(self, **_kwargs) -> NotePatch:
        return self._patches.pop(0)


def test_run_task_updates_markdown_and_respects_read_only(tmp_path, monkeypatch):
    collection = _prepare_collection(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "ankiops.llm.runner.git_snapshot", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        "ankiops.llm.runner._provider_for",
        lambda _config: _StubProvider(
            [
                NotePatch(
                    note_key="nk-1",
                    edits={"Question": "This is a fixed question."},
                ),
                NotePatch(note_key="nk-2", edits={}),
            ]
        ),
    )

    summary = run_task(
        collection_dir=collection,
        task_name="grammar",
        dry_run=False,
        no_auto_commit=True,
    )

    assert summary.updated == 1
    assert summary.unchanged == 1
    content = (collection / "TestDeck.md").read_text(encoding="utf-8")
    assert "Q: This is a fixed question." in content
    assert "S: grammar book" in content
    assert "AI: hidden content" in content
    assert "A: 1" in content


def test_run_task_rejects_read_only_updates(tmp_path, monkeypatch):
    collection = _prepare_collection(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "ankiops.llm.runner.git_snapshot", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        "ankiops.llm.runner._provider_for",
        lambda _config: _StubProvider(
            [
                NotePatch(note_key="nk-1", edits={}),
                NotePatch(note_key="nk-2", edits={"Answer": "2"}),
            ]
        ),
    )

    with pytest.raises(SystemExit) as exc:
        run_task(
            collection_dir=collection,
            task_name="grammar",
            dry_run=True,
            no_auto_commit=True,
        )

    assert exc.value.code == 1
    content = (collection / "TestDeck.md").read_text(encoding="utf-8")
    assert "A: 1" in content


def test_run_task_uses_scoped_serialization_for_single_exact_deck(
    tmp_path, monkeypatch
):
    collection = _prepare_collection(tmp_path, monkeypatch)
    _write(
        collection / "llm/tasks/grammar.yaml",
        """
        name: grammar
        prompt: fix grammar
        decks:
          include: ["TestDeck"]
        fields:
          exceptions:
            - read_only: ["Source"]
        """,
    )

    captured: dict[str, object] = {}

    def _fake_serialize(
        _collection_dir,
        *,
        strict=False,
        deck=None,
        no_subdecks=False,
    ):
        captured["strict"] = strict
        captured["deck"] = deck
        captured["no_subdecks"] = no_subdecks
        return {
            "collection": {"serialized_at": "2026-03-02T00:00:00Z"},
            "decks": [],
        }

    monkeypatch.setattr("ankiops.llm.runner.serialize_collection", _fake_serialize)
    monkeypatch.setattr(
        "ankiops.llm.runner._provider_for",
        lambda _config: _StubProvider([]),
    )

    run_task(
        collection_dir=collection,
        task_name="grammar",
        dry_run=True,
        no_auto_commit=True,
    )

    assert captured == {
        "strict": True,
        "deck": "TestDeck",
        "no_subdecks": False,
    }


def test_run_task_maps_include_subdecks_false_to_exact_scope(tmp_path, monkeypatch):
    collection = _prepare_collection(tmp_path, monkeypatch)
    _write(
        collection / "llm/tasks/grammar.yaml",
        (
            "name: grammar\n"
            "prompt: fix grammar\n"
            "decks:\n"
            '  include: ["TestDeck"]\n'
            "  include_subdecks: false\n"
            "fields:\n"
            "  exceptions:\n"
            '    - read_only: ["Source"]\n'
        ),
    )

    captured: dict[str, object] = {}

    def _fake_serialize(
        _collection_dir,
        *,
        strict=False,
        deck=None,
        no_subdecks=False,
    ):
        captured["strict"] = strict
        captured["deck"] = deck
        captured["no_subdecks"] = no_subdecks
        return {
            "collection": {"serialized_at": "2026-03-02T00:00:00Z"},
            "decks": [],
        }

    monkeypatch.setattr("ankiops.llm.runner.serialize_collection", _fake_serialize)
    monkeypatch.setattr(
        "ankiops.llm.runner._provider_for",
        lambda _config: _StubProvider([]),
    )

    run_task(
        collection_dir=collection,
        task_name="grammar",
        dry_run=True,
        no_auto_commit=True,
    )

    assert captured == {
        "strict": True,
        "deck": "TestDeck",
        "no_subdecks": True,
    }


def test_run_task_keeps_full_serialization_for_wildcard_deck_scope(
    tmp_path, monkeypatch
):
    collection = _prepare_collection(tmp_path, monkeypatch)
    _write(
        collection / "llm/tasks/grammar.yaml",
        """
        name: grammar
        prompt: fix grammar
        decks:
          include: ["Test*"]
        fields:
          exceptions:
            - read_only: ["Source"]
        """,
    )

    captured: dict[str, object] = {}

    def _fake_serialize(
        _collection_dir,
        *,
        strict=False,
        deck=None,
        no_subdecks=False,
    ):
        captured["strict"] = strict
        captured["deck"] = deck
        captured["no_subdecks"] = no_subdecks
        return {
            "collection": {"serialized_at": "2026-03-02T00:00:00Z"},
            "decks": [],
        }

    monkeypatch.setattr("ankiops.llm.runner.serialize_collection", _fake_serialize)
    monkeypatch.setattr(
        "ankiops.llm.runner._provider_for",
        lambda _config: _StubProvider([]),
    )

    run_task(
        collection_dir=collection,
        task_name="grammar",
        dry_run=True,
        no_auto_commit=True,
    )

    assert captured == {
        "strict": True,
        "deck": None,
        "no_subdecks": False,
    }
