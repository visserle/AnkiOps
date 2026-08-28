"""Leech tags must not be synced for shared (collab) decks."""

from __future__ import annotations

from typing import Any

from ankiops.git import GitRepository
from ankiops.markdown import NOTE_SEPARATOR
from tests.support.deck_files import DeckFileHarness

_COLLAB_NOTE_TYPE = "collab/owner/repo/AnkiOpsQA"


def _setup_collab_root(world: Any) -> Any:
    collab_root = world.root / "collab" / "owner" / "repo"
    DeckFileHarness().eject_default_note_types(collab_root / "note_types")
    collab_root.mkdir(parents=True, exist_ok=True)
    GitRepository(collab_root).init_repo()
    return collab_root


def _write_collab_deck(collab_root: Any, deck_name: str, tags: Any, answer: str) -> Any:
    lines = [
        "<!-- note_key: leech-key -->",
        f"<!-- note_type: {_COLLAB_NOTE_TYPE} -->",
        f"<!-- tags: {tags} -->",
        "Q: Collab Q",
        f"A: {answer}",
    ]
    path = collab_root / f"{deck_name}.md"
    path.write_text(NOTE_SEPARATOR.join(["\n".join(lines)]) + "\n", encoding="utf-8")
    return path


def test_export_strips_leech_tag_from_collab_deck(world: Any) -> None:
    collab_root = _setup_collab_root(world)
    _write_collab_deck(collab_root, "CollabDeck", tags="shared", answer="Old Collab A")
    note_id = world.add_anki_note(
        deck_name="CollabDeck",
        note_type=_COLLAB_NOTE_TYPE,
        fields={
            "Question": "Collab Q",
            "Answer": "New Collab A",
            "AnkiOps Key": "leech-key",
        },
        tags=("leech", "shared"),
    )

    with world.db_session() as db:
        db.upsert_note_links([("leech-key", note_id)], source_path="collab/owner/repo")
        result = world.sync_export(db)

    assert result.summary.updated == 1
    content = (collab_root / "CollabDeck.md").read_text(encoding="utf-8")
    assert "A: New Collab A" in content
    assert "<!-- tags: shared -->" in content
    assert "<!-- tags: leech" not in content


def test_export_keeps_leech_tag_for_local_decks(world: Any) -> None:
    world.add_qa_note(
        deck_name="LocalDeck",
        question="Local Q",
        answer="Local A",
        note_key="local-leech-key",
        tags=("leech", "shared"),
    )

    with world.db_session() as db:
        world.sync_export(db)

    content = world.read_deck("LocalDeck")
    assert "<!-- tags: leech shared -->" in content


def test_import_never_applies_leech_tag_from_collab_file(world: Any) -> None:
    collab_root = _setup_collab_root(world)
    _write_collab_deck(
        collab_root, "CollabDeck", tags="leech shared", answer="Collab A"
    )

    with world.db_session() as db:
        result = world.sync_import(db)

    assert result.summary.created == 1
    created_id = world.assert_anki_note(
        deck_name="CollabDeck",
        question="Collab Q",
        note_key="leech-key",
    )
    assert world.mock_anki.notes[created_id]["tags"] == ["shared"]


def test_import_preserves_local_leech_tag_on_update(world: Any) -> None:
    collab_root = _setup_collab_root(world)
    note_id = world.add_anki_note(
        deck_name="CollabDeck",
        note_type=_COLLAB_NOTE_TYPE,
        fields={
            "Question": "Collab Q",
            "Answer": "Old Collab A",
            "AnkiOps Key": "leech-key",
        },
        tags=("leech", "shared"),
    )
    _write_collab_deck(collab_root, "CollabDeck", tags="shared", answer="New Collab A")

    with world.db_session() as db:
        db.upsert_note_links([("leech-key", note_id)], source_path="collab/owner/repo")
        result = world.sync_import(db)

    assert result.summary.updated == 1
    assert world.mock_anki.notes[note_id]["fields"]["Answer"]["value"] == "New Collab A"
    assert world.mock_anki.notes[note_id]["tags"] == ["leech", "shared"]

    # The preserved leech tag must not cause churn on the next import.
    with world.db_session() as db:
        result = world.sync_import(db)
    assert result.summary.updated == 0
    assert result.summary.created == 0
    assert world.mock_anki.notes[note_id]["tags"] == ["leech", "shared"]
