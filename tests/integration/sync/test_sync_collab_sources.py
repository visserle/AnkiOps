from __future__ import annotations

import pytest

from ankiops.fs import FileSystemAdapter
from ankiops.markdown_format import NOTE_SEPARATOR


def _setup_collab_root(world):
    collab_root = world.root / "collab" / "owner" / "repo"
    FileSystemAdapter().eject_builtin_note_types(collab_root / "note_types")
    collab_root.mkdir(parents=True, exist_ok=True)
    return collab_root


def _write_collab_qa_deck(
    collab_root,
    deck_name: str,
    question: str,
    answer: str,
    note_key: str,
):
    path = collab_root / f"{deck_name}.md"
    path.write_text(
        "\n".join(
            [
                f"<!-- note_key: {note_key} -->",
                "<!-- note_type: collab/owner/repo/AnkiOpsQA -->",
                f"Q: {question}",
                f"A: {answer}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_import_syncs_root_and_collab_sources_as_one_collection(world):
    collab_root = _setup_collab_root(world)

    root_id = world.add_qa_note(
        deck_name="RootDeck",
        question="Root Q",
        answer="Root A",
        note_key="root-key-1",
    )
    world.mock_anki.add_note(
        "CollabDeck",
        "collab/owner/repo/AnkiOpsQA",
        {
            "Question": "Collab Q",
            "Answer": "Collab A",
            "AnkiOps Key": "collab-key-1",
        },
    )
    collab_id = max(world.mock_anki.notes.keys())

    world.write_qa_deck("RootDeck", [("Root Q", "Root A", "root-key-1")])
    _write_collab_qa_deck(
        collab_root,
        "CollabDeck",
        "Collab Q",
        "Collab A",
        "collab-key-1",
    )

    with world.db_session() as db:
        result = world.sync_import(db)

    assert result.summary.deleted == 0
    assert root_id in world.mock_anki.notes
    assert collab_id in world.mock_anki.notes


def test_import_converts_root_note_to_scoped_collab_type_without_duplicate(world):
    collab_root = _setup_collab_root(world)
    note_key = "published-key-1"
    world.mock_anki.add_note(
        "CollabDeck",
        "AnkiOpsQA",
        {
            "Question": "Collab Q",
            "Answer": "Collab A",
            "AnkiOps Key": note_key,
        },
    )
    note_id = max(world.mock_anki.notes.keys())
    _write_collab_qa_deck(
        collab_root,
        "CollabDeck",
        "Collab Q",
        "Collab A",
        note_key,
    )

    with world.db_session() as db:
        result = world.sync_import(db)

    assert result.summary.converted == 1
    assert len(world.mock_anki.notes) == 1
    assert note_id in world.mock_anki.notes
    assert world.mock_anki.notes[note_id]["modelName"] == (
        "collab/owner/repo/AnkiOpsQA"
    )
    assert (
        "changeNotesNotetype",
        {
            "noteIds": [note_id],
            "oldModel": "AnkiOpsQA",
            "newModel": "collab/owner/repo/AnkiOpsQA",
        },
    ) in world.mock_anki.calls


def test_import_ankiops_connect_failure_blocks_conversion_without_duplicate(
    world,
    monkeypatch,
):
    collab_root = _setup_collab_root(world)
    note_key = "published-key-ankiops-connect-down"
    world.mock_anki.add_note(
        "CollabDeck",
        "AnkiOpsQA",
        {
            "Question": "Collab Q",
            "Answer": "Collab A",
            "AnkiOps Key": note_key,
        },
    )
    note_id = max(world.mock_anki.notes.keys())
    _write_collab_qa_deck(
        collab_root,
        "CollabDeck",
        "Collab Q",
        "Collab A",
        note_key,
    )
    def fail_change_notetype(action: str, **params):
        if action == "changeNotesNotetype":
            raise RuntimeError("AnkiOpsConnect down")
        return world.mock_anki.invoke(action, **params)

    monkeypatch.setattr("ankiops.anki.invoke", fail_change_notetype)

    with world.db_session() as db:
        result = world.sync_import(db)

    assert result.summary.errors == 1
    assert len(world.mock_anki.notes) == 1
    assert world.mock_anki.notes[note_id]["modelName"] == "AnkiOpsQA"


def test_import_blocks_cross_collab_conversion_without_duplicate(world):
    collab_root = _setup_collab_root(world)
    note_key = "published-key-wrong-collab"
    world.mock_anki.add_note(
        "CollabDeck",
        "collab/owner/old/AnkiOpsQA",
        {
            "Question": "Collab Q",
            "Answer": "Collab A",
            "AnkiOps Key": note_key,
        },
    )
    note_id = max(world.mock_anki.notes.keys())
    _write_collab_qa_deck(
        collab_root,
        "CollabDeck",
        "Collab Q",
        "Collab A",
        note_key,
    )

    with world.db_session() as db:
        result = world.sync_import(db)

    assert result.summary.errors == 1
    assert len(world.mock_anki.notes) == 1
    assert world.mock_anki.notes[note_id]["modelName"] == (
        "collab/owner/old/AnkiOpsQA"
    )


def test_import_rejects_duplicate_deck_names_across_sources(world):
    collab_root = _setup_collab_root(world)

    world.write_qa_deck("Deck", [("Root Q", "Root A", "root-key-1")])
    (collab_root / "Deck.md").write_text(
        NOTE_SEPARATOR.join(
            [
                "\n".join(
                    [
                        "<!-- note_key: collab-key-1 -->",
                        "<!-- note_type: collab/owner/repo/AnkiOpsQA -->",
                        "Q: Collab Q",
                        "A: Collab A",
                    ]
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with world.db_session() as db:
        try:
            world.sync_import(db)
        except ValueError as error:
            assert "deck ownership conflicts" in str(error)
        else:  # pragma: no cover - assertion clarity
            raise AssertionError("expected deck ownership conflict")


def test_import_rejects_duplicate_ankiops_key_across_root_and_collab_models(world):
    collab_root = _setup_collab_root(world)
    note_key = "collab-duplicate-key"

    root_id = world.add_qa_note(
        deck_name="RootDeck",
        question="Root Q",
        answer="Root A",
        note_key=note_key,
    )
    world.mock_anki.add_note(
        "CollabDeck",
        "collab/owner/repo/AnkiOpsQA",
        {
            "Question": "Collab Q",
            "Answer": "Collab A",
            "AnkiOps Key": note_key,
        },
    )
    collab_id = max(world.mock_anki.notes.keys())
    _write_collab_qa_deck(
        collab_root,
        "CollabDeck",
        "Collab Q",
        "Collab A",
        note_key,
    )

    with world.db_session() as db:
        db.upsert_note_links([(note_key, collab_id)])
        with pytest.raises(ValueError, match=f"Duplicate AnkiOps Key '{note_key}'"):
            world.sync_import(db)

    assert root_id in world.mock_anki.notes
    assert collab_id in world.mock_anki.notes


def test_export_rejects_duplicate_ankiops_key_across_root_and_collab_models(world):
    collab_root = _setup_collab_root(world)
    note_key = "collab-export-duplicate-key"

    root_id = world.add_qa_note(
        deck_name="RootDeck",
        question="Root Q",
        answer="Root A",
        note_key=note_key,
    )
    world.mock_anki.add_note(
        "CollabDeck",
        "collab/owner/repo/AnkiOpsQA",
        {
            "Question": "Collab Q",
            "Answer": "Collab A",
            "AnkiOps Key": note_key,
        },
    )
    collab_id = max(world.mock_anki.notes.keys())

    with world.db_session() as db:
        with pytest.raises(ValueError, match=f"Duplicate AnkiOps Key '{note_key}'"):
            world.sync_export(db)

    assert not world.deck_path("RootDeck").exists()
    assert not (collab_root / "CollabDeck.md").exists()
    assert root_id in world.mock_anki.notes
    assert collab_id in world.mock_anki.notes


def test_import_deletes_root_orphan_without_deleting_collab_note(world):
    collab_root = _setup_collab_root(world)
    root_id = world.add_qa_note(
        deck_name="RootDeck",
        question="Root Q",
        answer="Root A",
        note_key="root-delete-key",
    )
    world.mock_anki.add_note(
        "CollabDeck",
        "collab/owner/repo/AnkiOpsQA",
        {
            "Question": "Collab Q",
            "Answer": "Collab A",
            "AnkiOps Key": "collab-keep-key",
        },
    )
    collab_id = max(world.mock_anki.notes.keys())

    world.write_qa_deck("RootDeck", [])
    _write_collab_qa_deck(
        collab_root,
        "CollabDeck",
        "Collab Q",
        "Collab A",
        "collab-keep-key",
    )

    with world.db_session() as db:
        db.upsert_note_links(
            [
                ("root-delete-key", root_id),
                ("collab-keep-key", collab_id),
            ]
        )
        result = world.sync_import(db)

    assert result.summary.deleted == 1
    assert root_id not in world.mock_anki.notes
    assert collab_id in world.mock_anki.notes


def test_import_deletes_collab_orphan_without_deleting_root_note(world):
    collab_root = _setup_collab_root(world)
    root_id = world.add_qa_note(
        deck_name="RootDeck",
        question="Root Q",
        answer="Root A",
        note_key="root-keep-key",
    )
    world.mock_anki.add_note(
        "CollabDeck",
        "collab/owner/repo/AnkiOpsQA",
        {
            "Question": "Collab Q",
            "Answer": "Collab A",
            "AnkiOps Key": "collab-delete-key",
        },
    )
    collab_id = max(world.mock_anki.notes.keys())

    world.write_qa_deck("RootDeck", [("Root Q", "Root A", "root-keep-key")])
    (collab_root / "CollabDeck.md").write_text("", encoding="utf-8")

    with world.db_session() as db:
        db.upsert_note_links(
            [
                ("root-keep-key", root_id),
                ("collab-delete-key", collab_id),
            ]
        )
        result = world.sync_import(db)

    assert result.summary.deleted == 1
    assert root_id in world.mock_anki.notes
    assert collab_id not in world.mock_anki.notes


def test_export_writes_anki_edits_back_to_collab_source(world):
    collab_root = _setup_collab_root(world)
    _write_collab_qa_deck(
        collab_root,
        "CollabDeck",
        "Collab Q",
        "Old Collab A",
        "collab-export-key",
    )
    world.mock_anki.add_note(
        "CollabDeck",
        "collab/owner/repo/AnkiOpsQA",
        {
            "Question": "Collab Q",
            "Answer": "New Collab A",
            "AnkiOps Key": "collab-export-key",
        },
    )
    collab_id = max(world.mock_anki.notes.keys())

    with world.db_session() as db:
        db.upsert_note_links([("collab-export-key", collab_id)])
        result = world.sync_export(db)

    assert result.summary.updated == 1
    assert "A: New Collab A" in (collab_root / "CollabDeck.md").read_text(
        encoding="utf-8"
    )
    assert not world.deck_path("CollabDeck").exists()


def test_export_blocks_pending_collab_note_type_conversion(world):
    collab_root = _setup_collab_root(world)
    note_key = "pending-conversion-export-key"
    deck = _write_collab_qa_deck(
        collab_root,
        "CollabDeck",
        "Collab Q",
        "Local Collab A",
        note_key,
    )
    original = deck.read_text(encoding="utf-8")
    world.mock_anki.add_note(
        "CollabDeck",
        "AnkiOpsQA",
        {
            "Question": "Collab Q",
            "Answer": "Anki Root A",
            "AnkiOps Key": note_key,
        },
    )
    note_id = max(world.mock_anki.notes.keys())

    with world.db_session() as db:
        db.upsert_note_links([(note_key, note_id)])
        result = world.sync_export(db)

    assert result.summary.errors == 1
    assert deck.read_text(encoding="utf-8") == original
