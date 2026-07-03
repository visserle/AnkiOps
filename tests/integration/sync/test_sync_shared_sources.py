from __future__ import annotations

import pytest

from ankiops.git import GitRepository
from ankiops.markdown import NOTE_SEPARATOR
from tests.support.deck_files import DeckFileHarness


def _setup_shared_root(world):
    shared_root = world.root / "shared" / "owner" / "repo"
    DeckFileHarness().eject_default_note_types(shared_root / "note_types")
    shared_root.mkdir(parents=True, exist_ok=True)
    GitRepository(shared_root).init_repo()
    return shared_root


def _write_shared_qa_deck(
    shared_root,
    deck_name: str,
    question: str,
    answer: str,
    note_key: str,
):
    path = shared_root / f"{deck_name}.md"
    path.write_text(
        "\n".join(
            [
                f"<!-- note_key: {note_key} -->",
                "<!-- note_type: shared/owner/repo/AnkiOpsQA -->",
                f"Q: {question}",
                f"A: {answer}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_import_syncs_root_and_shared_sources_as_one_collection(world):
    shared_root = _setup_shared_root(world)

    root_id = world.add_qa_note(
        deck_name="RootDeck",
        question="Root Q",
        answer="Root A",
        note_key="root-key-1",
    )
    world.mock_anki.add_note(
        "SharedDeck",
        "shared/owner/repo/AnkiOpsQA",
        {
            "Question": "Shared Q",
            "Answer": "Shared A",
            "AnkiOps Key": "shared-key-1",
        },
    )
    shared_id = max(world.mock_anki.notes.keys())

    world.write_qa_deck("RootDeck", [("Root Q", "Root A", "root-key-1")])
    _write_shared_qa_deck(
        shared_root,
        "SharedDeck",
        "Shared Q",
        "Shared A",
        "shared-key-1",
    )

    with world.db_session() as db:
        result = world.sync_import(db)
        assert db.resolve_note_sources(["root-key-1", "shared-key-1"]) == {
            "root-key-1": ".",
            "shared-key-1": "shared/owner/repo",
        }
        assert (
            db.resolve_deck_source(world.mock_anki.decks["SharedDeck"])
            == "shared/owner/repo"
        )

    assert result.summary.deleted == 0
    assert root_id in world.mock_anki.notes
    assert shared_id in world.mock_anki.notes


def test_import_converts_root_note_to_scoped_shared_type_without_duplicate(world):
    shared_root = _setup_shared_root(world)
    note_key = "created-key-1"
    world.mock_anki.add_note(
        "SharedDeck",
        "AnkiOpsQA",
        {
            "Question": "Shared Q",
            "Answer": "Shared A",
            "AnkiOps Key": note_key,
        },
    )
    note_id = max(world.mock_anki.notes.keys())
    _write_shared_qa_deck(
        shared_root,
        "SharedDeck",
        "Shared Q",
        "Shared A",
        note_key,
    )

    with world.db_session() as db:
        result = world.sync_import(db)

    assert result.summary.converted == 1
    assert len(world.mock_anki.notes) == 1
    assert note_id in world.mock_anki.notes
    assert world.mock_anki.notes[note_id]["modelName"] == (
        "shared/owner/repo/AnkiOpsQA"
    )
    assert (
        "convertNotesToNoteType",
        {
            "noteIds": [note_id],
            "oldNoteType": "AnkiOpsQA",
            "newNoteType": "shared/owner/repo/AnkiOpsQA",
        },
    ) in world.mock_anki.calls


def test_import_rejects_implicit_cross_source_move(world):
    shared_root = _setup_shared_root(world)
    note_key = "cross-source-key"
    _write_shared_qa_deck(
        shared_root,
        "SharedDeck",
        "Shared Q",
        "Shared A",
        note_key,
    )
    with world.db_session() as db:
        world.sync_import(db)

        (shared_root / "SharedDeck.md").write_text("", encoding="utf-8")
        world.write_qa_deck("PrivateDeck", [("Shared Q", "Shared A", note_key)])

        with pytest.raises(ValueError, match="Cross-source note moves"):
            world.sync_import(db)


def test_import_ankiops_connect_failure_blocks_conversion_without_duplicate(world):
    shared_root = _setup_shared_root(world)
    note_key = "created-key-ankiops-connect-down"
    world.mock_anki.add_note(
        "SharedDeck",
        "AnkiOpsQA",
        {
            "Question": "Shared Q",
            "Answer": "Shared A",
            "AnkiOps Key": note_key,
        },
    )
    note_id = max(world.mock_anki.notes.keys())
    _write_shared_qa_deck(
        shared_root,
        "SharedDeck",
        "Shared Q",
        "Shared A",
        note_key,
    )

    world.mock_anki.fail_actions["convertNotesToNoteType"] = RuntimeError(
        "AnkiOpsConnect down"
    )

    with world.db_session() as db:
        result = world.sync_import(db)

    assert result.summary.errors == 1
    assert len(world.mock_anki.notes) == 1
    assert world.mock_anki.notes[note_id]["modelName"] == "AnkiOpsQA"


def test_import_blocks_cross_shared_conversion_without_duplicate(world):
    shared_root = _setup_shared_root(world)
    note_key = "created-key-wrong-shared"
    world.mock_anki.add_note(
        "SharedDeck",
        "shared/owner/old/AnkiOpsQA",
        {
            "Question": "Shared Q",
            "Answer": "Shared A",
            "AnkiOps Key": note_key,
        },
    )
    note_id = max(world.mock_anki.notes.keys())
    _write_shared_qa_deck(
        shared_root,
        "SharedDeck",
        "Shared Q",
        "Shared A",
        note_key,
    )

    with world.db_session() as db:
        result = world.sync_import(db)

    assert result.summary.errors == 1
    assert len(world.mock_anki.notes) == 1
    assert world.mock_anki.notes[note_id]["modelName"] == ("shared/owner/old/AnkiOpsQA")


def test_import_rejects_duplicate_deck_names_across_sources(world):
    shared_root = _setup_shared_root(world)

    world.write_qa_deck("Deck", [("Root Q", "Root A", "root-key-1")])
    (shared_root / "Deck.md").write_text(
        NOTE_SEPARATOR.join(
            [
                "\n".join(
                    [
                        "<!-- note_key: shared-key-1 -->",
                        "<!-- note_type: shared/owner/repo/AnkiOpsQA -->",
                        "Q: Shared Q",
                        "A: Shared A",
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


def test_import_rejects_duplicate_ankiops_key_across_root_and_shared_models(world):
    shared_root = _setup_shared_root(world)
    note_key = "shared-duplicate-key"

    root_id = world.add_qa_note(
        deck_name="RootDeck",
        question="Root Q",
        answer="Root A",
        note_key=note_key,
    )
    world.mock_anki.add_note(
        "SharedDeck",
        "shared/owner/repo/AnkiOpsQA",
        {
            "Question": "Shared Q",
            "Answer": "Shared A",
            "AnkiOps Key": note_key,
        },
    )
    shared_id = max(world.mock_anki.notes.keys())
    _write_shared_qa_deck(
        shared_root,
        "SharedDeck",
        "Shared Q",
        "Shared A",
        note_key,
    )

    with world.db_session() as db:
        db.upsert_note_links([(note_key, shared_id)], source_path="shared/owner/repo")
        with pytest.raises(ValueError, match=f"Duplicate AnkiOps Key '{note_key}'"):
            world.sync_import(db)

    assert root_id in world.mock_anki.notes
    assert shared_id in world.mock_anki.notes


def test_export_rejects_duplicate_ankiops_key_across_root_and_shared_models(world):
    shared_root = _setup_shared_root(world)
    note_key = "shared-export-duplicate-key"

    root_id = world.add_qa_note(
        deck_name="RootDeck",
        question="Root Q",
        answer="Root A",
        note_key=note_key,
    )
    world.mock_anki.add_note(
        "SharedDeck",
        "shared/owner/repo/AnkiOpsQA",
        {
            "Question": "Shared Q",
            "Answer": "Shared A",
            "AnkiOps Key": note_key,
        },
    )
    shared_id = max(world.mock_anki.notes.keys())

    with world.db_session() as db:
        with pytest.raises(ValueError, match=f"Duplicate AnkiOps Key '{note_key}'"):
            world.sync_export(db)

    assert not world.deck_path("RootDeck").exists()
    assert not (shared_root / "SharedDeck.md").exists()
    assert root_id in world.mock_anki.notes
    assert shared_id in world.mock_anki.notes


def test_import_deletes_root_orphan_without_deleting_shared_note(world):
    shared_root = _setup_shared_root(world)
    root_id = world.add_qa_note(
        deck_name="RootDeck",
        question="Root Q",
        answer="Root A",
        note_key="root-delete-key",
    )
    world.mock_anki.add_note(
        "SharedDeck",
        "shared/owner/repo/AnkiOpsQA",
        {
            "Question": "Shared Q",
            "Answer": "Shared A",
            "AnkiOps Key": "shared-keep-key",
        },
    )
    shared_id = max(world.mock_anki.notes.keys())

    world.write_qa_deck("RootDeck", [])
    _write_shared_qa_deck(
        shared_root,
        "SharedDeck",
        "Shared Q",
        "Shared A",
        "shared-keep-key",
    )

    with world.db_session() as db:
        db.upsert_note_links([("root-delete-key", root_id)])
        db.upsert_note_links(
            [("shared-keep-key", shared_id)], source_path="shared/owner/repo"
        )
        result = world.sync_import(db)

    assert result.summary.deleted == 1
    assert root_id not in world.mock_anki.notes
    assert shared_id in world.mock_anki.notes


def test_import_deletes_shared_orphan_without_deleting_root_note(world):
    shared_root = _setup_shared_root(world)
    root_id = world.add_qa_note(
        deck_name="RootDeck",
        question="Root Q",
        answer="Root A",
        note_key="root-keep-key",
    )
    world.mock_anki.add_note(
        "SharedDeck",
        "shared/owner/repo/AnkiOpsQA",
        {
            "Question": "Shared Q",
            "Answer": "Shared A",
            "AnkiOps Key": "shared-delete-key",
        },
    )
    shared_id = max(world.mock_anki.notes.keys())

    world.write_qa_deck("RootDeck", [("Root Q", "Root A", "root-keep-key")])
    (shared_root / "SharedDeck.md").write_text("", encoding="utf-8")

    with world.db_session() as db:
        db.upsert_note_links([("root-keep-key", root_id)])
        db.upsert_note_links(
            [("shared-delete-key", shared_id)], source_path="shared/owner/repo"
        )
        result = world.sync_import(db)

    assert result.summary.deleted == 1
    assert root_id in world.mock_anki.notes
    assert shared_id not in world.mock_anki.notes


def test_export_writes_anki_edits_back_to_shared_source(world):
    shared_root = _setup_shared_root(world)
    _write_shared_qa_deck(
        shared_root,
        "SharedDeck",
        "Shared Q",
        "Old Shared A",
        "shared-export-key",
    )
    world.mock_anki.add_note(
        "SharedDeck",
        "shared/owner/repo/AnkiOpsQA",
        {
            "Question": "Shared Q",
            "Answer": "New Shared A",
            "AnkiOps Key": "shared-export-key",
        },
    )
    shared_id = max(world.mock_anki.notes.keys())

    with world.db_session() as db:
        db.upsert_note_links(
            [("shared-export-key", shared_id)], source_path="shared/owner/repo"
        )
        result = world.sync_export(db)

    assert result.summary.updated == 1
    assert "A: New Shared A" in (shared_root / "SharedDeck.md").read_text(
        encoding="utf-8"
    )
    assert not world.deck_path("SharedDeck").exists()


def test_export_blocks_pending_shared_note_type_conversion(world):
    shared_root = _setup_shared_root(world)
    note_key = "pending-conversion-export-key"
    deck = _write_shared_qa_deck(
        shared_root,
        "SharedDeck",
        "Shared Q",
        "Local Shared A",
        note_key,
    )
    original = deck.read_text(encoding="utf-8")
    world.mock_anki.add_note(
        "SharedDeck",
        "AnkiOpsQA",
        {
            "Question": "Shared Q",
            "Answer": "Anki Root A",
            "AnkiOps Key": note_key,
        },
    )
    note_id = max(world.mock_anki.notes.keys())

    with world.db_session() as db:
        db.upsert_note_links([(note_key, note_id)], source_path="shared/owner/repo")
        result = world.sync_export(db)

    assert result.summary.errors == 1
    assert deck.read_text(encoding="utf-8") == original
