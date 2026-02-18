"""Integration tests for AnkiOps.

These tests simulate the entire application stack:
1.  Mock AnkiConnect (using a stateful MockAnki class).
2.  Use real file I/O (via tmp_path).
3.  Run `export_collection` and `import_collection`.
4.  Verify end-to-end behavior (files created, Anki updated).
"""

from typing import Any
from unittest.mock import patch

import pytest

from ankiops.anki_client import AnkiConnectError
from ankiops.anki_to_markdown import export_collection, export_deck
from ankiops.markdown_to_anki import import_collection, import_file


def test_import_with_stale_deck_key_mapping(
    tmp_path, mock_anki, run_ankiops, monkeypatch
):
    """Test exporting a collection from Anki to an empty directory."""
    # Setup Anki state
    mock_anki.add_note("Deck A", "AnkiOpsQA", {"Question": "Q1", "Answer": "A1"})
    mock_anki.add_note("Deck B", "AnkiOpsQA", {"Question": "Q2", "Answer": "A2"})

    # Run export
    summary = export_collection(output_dir=str(tmp_path))

    # Assertions
    assert len(summary.deck_results) == 2

    file_a = tmp_path / "Deck A.md"
    file_b = tmp_path / "Deck B.md"

    assert file_a.exists()
    assert file_b.exists()

    content_a = file_a.read_text()
    assert "Q: Q1" in content_a
    assert "A: A1" in content_a
    assert "<!-- note_key:" in content_a
    assert "<!-- deck_key:" in content_a

    content_b = file_b.read_text()
    assert "Q: Q2" in content_b


def test_import_creates_new_notes(tmp_path, mock_anki, run_ankiops):
    """Test importing new notes from markdown to Anki."""
    # Setup markdown files
    deck_c = tmp_path / "Deck C.md"
    # We use a preset Key to simulate an existing deck file or forcing a Key
    deck_c.write_text("<!-- deck_key: deck-key-999 -->\nQ: New Question\nA: New Answer")

    # Run import
    summary = import_collection(collection_dir=str(tmp_path))

    # Assertions
    # Should create deck "Deck C"
    assert "Deck C" in mock_anki.decks

    # Should create note
    assert len(mock_anki.notes) == 1
    new_note = list(mock_anki.notes.values())[0]
    assert new_note["fields"]["Question"]["value"] == "New Question"

    # Check file was updated with Key
    content = deck_c.read_text()
    assert "<!-- note_key:" in content
    # Note ID should NOT be in the file
    assert "<!-- note_id:" not in content
    # Deck Key should be present (Deck ID replaced by Key)
    assert "<!-- deck_key:" in content


def test_roundtrip_sync_update(tmp_path, mock_anki, run_ankiops):
    """Test full cycle: Export -> Modify File -> Import -> Verify Anki Update."""
    # 1. Setup Anki
    mock_anki.add_note(
        "Deck X", "AnkiOpsQA", {"Question": "Original Q", "Answer": "Original A"}
    )
    note_id = list(mock_anki.notes.keys())[0]

    # 2. Export
    export_collection(output_dir=str(tmp_path))
    file_path = tmp_path / "Deck X.md"

    # 3. Modify File
    content = file_path.read_text()
    new_content = content.replace("Original A", "Updated A")
    file_path.write_text(new_content)

    # 4. Import
    import_collection(collection_dir=str(tmp_path))

    # 5. Verify Anki updated
    updated_note = mock_anki.notes[note_id]
    assert updated_note["fields"]["Answer"]["value"] == "Updated A"


def test_sync_move_cards(tmp_path, mock_anki, run_ankiops):
    """Test moving a note from one deck to another via file system."""
    # 1. Setup Anki
    mock_anki.add_note("SourceDeck", "AnkiOpsQA", {"Question": "MoveMe", "Answer": "A"})
    note_id = list(mock_anki.notes.keys())[0]

    # 2. Export
    export_collection(output_dir=str(tmp_path))
    source_file = tmp_path / "SourceDeck.md"
    target_file = tmp_path / "TargetDeck.md"

    # 3. Move note block to new file
    source_content = source_file.read_text()
    # Extract note block (everything after deck_id)
    lines = source_content.splitlines()
    deck_line = lines[0]
    note_block = "\n".join(lines[1:])

    # Empty source file (keep deck id)
    source_file.write_text(deck_line)

    # Write target file
    # We use a new Key or rely on filename matching
    target_file.write_text("<!-- deck_key: deck-key-move-target -->\n" + note_block)

    # 4. Import
    import_collection(collection_dir=str(tmp_path))

    # 5. Verify card moved in Anki
    # Need to check card, not note (notes don't have deck, cards do)
    card = list(mock_anki.cards.values())[0]
    assert card["deckName"] == "TargetDeck"


def test_sync_delete_orphan_in_anki(tmp_path, mock_anki, run_ankiops):
    """Test deleting a note in file deletes it in Anki."""
    # 1. Setup Anki
    mock_anki.add_note("Deck D", "AnkiOpsQA", {"Question": "DeleteMe", "Answer": "A"})
    note_id = list(mock_anki.notes.keys())[0]

    # 2. Export
    export_collection(output_dir=str(tmp_path))
    file_path = tmp_path / "Deck D.md"

    # 3. Delete note from file (keep deck ID)
    lines = file_path.read_text().splitlines()
    file_path.write_text(lines[0])  # Only deck ID line

    # 4. Import
    import_collection(collection_dir=str(tmp_path))

    # 5. Verify deleted in Anki
    assert note_id not in mock_anki.notes


def test_all_note_types_integration(tmp_path, mock_anki, run_ankiops):
    """Test full cycle for all supported note types."""
    deck_name = "AllTypes"
    md_file = tmp_path / f"{deck_name}.md"

    # 1. Create markdown with all types
    content = (
        "Q: QA Question\n"
        "A: QA Answer\n\n"
        "---\n\n"
        "F: Reversed Front\n"
        "B: Reversed Back\n\n"
        "---\n\n"
        "T: Cloze with {{c1::hidden}} text\n\n"
        "---\n\n"
        "Q: Input Question\n"
        "I: Input Answer\n\n"
        "---\n\n"
        "Q: Choice Question\n"
        "C1: Option A\n"
        "C2: Option B\n"
        "A: 1"
    )
    md_file.write_text(content)

    # 2. Import
    result = import_file(md_file)
    assert not result.errors
    assert result.created_count == 5

    # 3. Verify MockAnki state
    notes = list(mock_anki.notes.values())
    assert len(notes) == 5

    models = {n["modelName"] for n in notes}
    expected_models = {
        "AnkiOpsQA",
        "AnkiOpsReversed",
        "AnkiOpsCloze",
        "AnkiOpsInput",
        "AnkiOpsChoice",
    }
    assert models == expected_models

    # 4. Verify specific fields
    for n in notes:
        model = n["modelName"]
        fields = n["fields"]
        if model == "AnkiOpsQA":
            assert "QA Question" in fields["Question"]["value"]
        elif model == "AnkiOpsReversed":
            assert "Reversed Front" in fields["Front"]["value"]
        elif model == "AnkiOpsCloze":
            assert "{{c1::hidden}}" in fields["Text"]["value"]
        elif model == "AnkiOpsInput":
            assert "Input Answer" in fields["Input"]["value"]
        elif model == "AnkiOpsChoice":
            assert "Option A" in fields["Choice 1"]["value"]
            assert fields["Answer"]["value"] == "1"

    # 5. Export and verify roundtrip
    md_file.unlink()

    export_result = export_deck(deck_name, output_dir=str(tmp_path))
    assert export_result.file_path.exists()

    new_content = export_result.file_path.read_text()

    assert "Q: QA Question" in new_content
    assert "F: Reversed Front" in new_content
    assert "T: Cloze with {{c1::hidden}} text" in new_content
    assert "I: Input Answer" in new_content
    assert "C1: Option A" in new_content


def test_ankiops_id_populated_on_create(tmp_path, mock_anki, run_ankiops):
    """Test that AnkiOps Key is populated when creating a new note."""
    deck_file = tmp_path / "ReproDeck.md"
    deck_file.write_text("Q: Question\nA: Answer")

    import_collection(collection_dir=str(tmp_path))

    assert len(mock_anki.notes) == 1
    new_note = list(mock_anki.notes.values())[0]

    assert "AnkiOps Key" in new_note["fields"], (
        "AnkiOps Key field missing from new note"
    )
    val = new_note["fields"]["AnkiOps Key"]["value"]
    assert val and len(val) > 0


def test_ankiops_id_populated_on_update(tmp_path, mock_anki, run_ankiops):
    """Test that AnkiOps Key is populated when updating an existing note that misses it."""
    # 1. Create a note in Anki that has content but NO AnkiOps Key
    from ankiops.key_map import KeyMap as KM

    # Create dummy existing note in mock
    field_data = {"Question": "OldQ", "Answer": "OldA", "AnkiOps Key": ""}

    deck_id = mock_anki.invoke("createDeck", deck="ReproDeck")
    note_id = 999

    # Manually properly structure the note in the mock
    mock_anki.notes[note_id] = {
        "noteId": note_id,
        "modelName": "AnkiOpsQA",
        "fields": {k: {"value": v} for k, v in field_data.items()},
        "cards": [1001],
    }
    mock_anki.cards[1001] = {
        "cardId": 1001,
        "note": note_id,
        "deckName": "ReproDeck",
        "modelName": "AnkiOpsQA",
    }

    # Setup KeyMap with fixed keys
    km = KM.load(tmp_path)
    deck_key = "deckkeyexisting"
    note_key = "notekeyexisting"
    km.set_deck(deck_key, deck_id)
    km.set_note(note_key, note_id)
    km.close()

    # Create File
    deck_file = tmp_path / "ReproDeck.md"
    deck_file.write_text(
        f"<!-- deck_key: {deck_key} -->\n<!-- note_key: {note_key} -->\nQ: OldQ\nA: UpdatedA"
    )

    # Run Import
    import_collection(collection_dir=str(tmp_path))

    # Verify Anki note was updated
    updated_note = mock_anki.notes[note_id]
    assert updated_note["fields"]["Answer"]["value"] == "UpdatedA"

    # Verify AnkiOps Key was populated
    current_val = updated_note["fields"]["AnkiOps Key"]["value"]
    assert current_val == note_key, (
        f"AnkiOps Key expected '{note_key}', got '{current_val}'"
    )
