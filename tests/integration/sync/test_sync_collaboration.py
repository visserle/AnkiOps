"""Collaboration scenarios where note_key is the only shared identity."""

from __future__ import annotations

from tests.support.assertions import assert_summary


def test_collab_import_move_without_db_mapping_moves_existing_note(world):
    """With empty DB, import should still move by embedded AnkiOps Key."""
    note_key = "collab-move-001"
    note_id = world.add_qa_note(
        deck_name="SourceDeck",
        question="Move Q",
        answer="Move A",
        note_key=note_key,
    )

    world.write_qa_deck("SourceDeck", [])
    world.write_qa_deck("TargetDeck", [("Move Q", "Move A", note_key)])

    with world.db_session() as db:
        result = world.sync_import(db)

        assert_summary(
            result.summary,
            created=0,
            updated=0,
            moved=1,
            deleted=0,
            errors=0,
        )
        assert db.get_note_id(note_key) == note_id

        card_id = world.mock_anki.notes[note_id]["cards"][0]
        assert world.mock_anki.cards[card_id]["deckName"] == "TargetDeck"


def test_collab_import_recovers_all_keys_without_db(world):
    """With empty DB, import recovers key->id mapping from embedded Anki keys."""
    key_a = "collab-recover-001-a"
    key_b = "collab-recover-001-b"
    id_a = world.add_qa_note(
        deck_name="DeckA",
        question="Qa",
        answer="Aa",
        note_key=key_a,
    )
    id_b = world.add_qa_note(
        deck_name="DeckB",
        question="Qb",
        answer="Ab",
        note_key=key_b,
    )

    world.write_qa_deck("DeckA", [("Qa", "Aa", key_a)])
    world.write_qa_deck("DeckB", [("Qb", "Ab", key_b)])

    with world.db_session() as db:
        result = world.sync_import(db)

        # Mock Anki notes may miss optional fields (Extra/More/...), so updates
        # are allowed here. What must remain stable is key-based identity.
        assert_summary(
            result.summary,
            created=0,
            moved=0,
            deleted=0,
            errors=0,
        )
        assert db.get_note_id(key_a) == id_a
        assert db.get_note_id(key_b) == id_b


def test_collab_export_recovers_mapping_from_embedded_key(world):
    """With empty DB, export should keep stable note_key identity."""
    note_key = "collab-export-001"
    note_id = world.add_qa_note(
        deck_name="ExportDeck",
        question="Q1",
        answer="A1",
        note_key=note_key,
    )
    world.write_qa_deck("ExportDeck", [("Q1", "A1", note_key)])

    with world.db_session() as db:
        result = world.sync_export(db)

        assert_summary(
            result.summary,
            created=0,
            updated=0,
            moved=0,
            deleted=0,
            errors=0,
        )
        assert db.get_note_id(note_key) == note_id
        assert f"<!-- note_key: {note_key} -->" in world.read_deck("ExportDeck")
