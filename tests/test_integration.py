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

from ankiops.anki_to_markdown import export_collection, export_deck
from ankiops.markdown_to_anki import import_collection, import_file


class MockAnki:
    """Stateful mock of AnkiConnect."""

    def __init__(self):
        # State
        self.decks = {"Default": 1}  # Name -> ID
        self.notes = {}  # ID -> Note dict (Anki format)
        self.cards = {}  # ID -> Card dict
        self.next_note_id = 100
        self.next_card_id = 1000
        self.next_deck_id = 10

        # Operation log for assertions
        self.calls = []

    def invoke(self, action: str, **params) -> Any:
        self.calls.append((action, params))

        if action == "deckNamesAndIds":
            return self.decks

        if action == "findCards":
            query = params.get("query", "")
            # Support "note:Type1 OR note:Type2"
            allowed_types = set()
            if "note:" in query:
                parts = query.split(" OR ")
                for p in parts:
                    if "note:" in p:
                        t = p.split("note:")[1].strip().split()[0]
                        allowed_types.add(t)

            found_cards = []
            for card in self.cards.values():
                note_id = card["note"]
                note = self.notes.get(note_id)
                if not note:
                    continue

                if allowed_types and note["modelName"] not in allowed_types:
                    continue

                found_cards.append(card["cardId"])
            return found_cards

        if action == "cardsInfo":
            card_ids = params.get("cards", [])
            return [self.cards[cid] for cid in card_ids if cid in self.cards]

        if action == "notesInfo":
            note_ids = params.get("notes", [])
            return [self.notes[nid] for nid in note_ids if nid in self.notes]

        if action == "multi":
            actions = params.get("actions", [])
            results = []
            for act in actions:
                res = self.invoke(act["action"], **act.get("params", {}))
                results.append(res)
            return results

        if action == "createDeck":
            name = params["deck"]
            new_id = self.next_deck_id
            self.next_deck_id += 1
            self.decks[name] = new_id
            return new_id

        if action == "addNote":
            note_data = params["note"]
            new_id = self.next_note_id
            self.next_note_id += 1

            # Create cards (mock 1 card per note)
            card_id = self.next_card_id
            self.next_card_id += 1

            self.notes[new_id] = {
                "noteId": new_id,
                "modelName": note_data["modelName"],
                "fields": {k: {"value": v} for k, v in note_data["fields"].items()},
                "cards": [card_id],
            }
            self.cards[card_id] = {
                "cardId": card_id,
                "note": new_id,
                "deckName": note_data["deckName"],
                "modelName": note_data["modelName"],
            }
            return new_id

        if action == "updateNoteFields":
            note_info = params["note"]
            nid = note_info["id"]
            if nid in self.notes:
                for k, v in note_info["fields"].items():
                    self.notes[nid]["fields"][k] = {"value": v}
            return None

        if action == "deleteNotes":
            nids = params["notes"]
            for nid in nids:
                if nid in self.notes:
                    # Remove associated cards
                    card_ids = self.notes[nid]["cards"]
                    for cid in card_ids:
                        if cid in self.cards:
                            del self.cards[cid]
                    del self.notes[nid]
            return None

        if action == "changeDeck":
            # cards, deck
            cards = params["cards"]
            deck = params["deck"]
            for cid in cards:
                if cid in self.cards:
                    self.cards[cid]["deckName"] = deck
            return None

        return None

    # -- Setup helpers --
    def add_note(self, deck_name: str, note_type: str, fields: dict):
        if deck_name not in self.decks:
            self.invoke("createDeck", deck=deck_name)

        self.invoke(
            "addNote",
            note={"deckName": deck_name, "modelName": note_type, "fields": fields},
        )


@pytest.fixture
def mock_anki():
    return MockAnki()


@pytest.fixture(autouse=True)
def mock_input():
    """Always answer 'y' to confirmation prompts."""
    with patch("builtins.input", return_value="y"):
        yield


@pytest.fixture
def run_ankiops(mock_anki):
    """Fixture to run ankiops with mocked invoke."""
    # We must patch where it's imported, because models.py and markdown_to_anki.py
    # do 'from ankiops.anki_client import invoke'
    with (
        patch("ankiops.anki_client.invoke", side_effect=mock_anki.invoke),
        patch("ankiops.models.invoke", side_effect=mock_anki.invoke),
        patch("ankiops.markdown_to_anki.invoke", side_effect=mock_anki.invoke),
    ):
        yield


def test_export_fresh_collection(tmp_path, mock_anki, run_ankiops):
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
    assert "<!-- deck_id:" in content_a

    content_b = file_b.read_text()
    assert "Q: Q2" in content_b


def test_import_creates_new_notes(tmp_path, mock_anki, run_ankiops):
    """Test importing new notes from markdown to Anki."""
    # Setup markdown files
    deck_c = tmp_path / "Deck C.md"
    deck_c.write_text("<!-- deck_id: 999 -->\nQ: New Question\nA: New Answer")

    # Run import
    summary = import_collection(collection_dir=str(tmp_path))

    # Assertions
    # Should create deck "Deck C"
    assert "Deck C" in mock_anki.decks

    # Should create note
    assert len(mock_anki.notes) == 1
    new_note = list(mock_anki.notes.values())[0]
    assert new_note["fields"]["Question"]["value"] == "New Question"

    # Check file was updated with ID
    content = deck_c.read_text()
    assert f"<!-- note_id: {new_note['noteId']} -->" in content
    # Deck ID should be updated to the real one assigned by MockAnki
    real_deck_id = mock_anki.decks["Deck C"]
    assert f"<!-- deck_id: {real_deck_id} -->" in content


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
    target_file.write_text(f"<!-- deck_id: {mock_anki.next_deck_id} -->\n" + note_block)

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
