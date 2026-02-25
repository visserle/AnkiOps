"""Smoke tests for end-to-end import/export behavior."""

from __future__ import annotations

from contextlib import contextmanager

from ankiops.anki import AnkiAdapter
from ankiops.config import deck_name_to_file_stem, get_note_types_dir
from ankiops.db import SQLiteDbAdapter
from ankiops.export_notes import export_collection
from ankiops.fs import FileSystemAdapter
from ankiops.import_notes import import_collection
from tests.support.assertions import assert_summary


@contextmanager
def _ports(collection_dir, *, preload_configs: bool):
    anki = AnkiAdapter()
    fs = FileSystemAdapter()
    if preload_configs:
        fs.set_configs(fs.load_note_type_configs(get_note_types_dir()))
    db = SQLiteDbAdapter.load(collection_dir)
    try:
        yield anki, fs, db
    finally:
        db.close()


def _run_import(collection_dir, *, preload_configs: bool = True):
    with _ports(collection_dir, preload_configs=preload_configs) as (anki, fs, db):
        return import_collection(
            anki_port=anki,
            fs_port=fs,
            db_port=db,
            collection_dir=collection_dir,
            note_types_dir=get_note_types_dir(),
        )


def _run_export(collection_dir, *, preload_configs: bool = False):
    with _ports(collection_dir, preload_configs=preload_configs) as (anki, fs, db):
        return export_collection(
            anki_port=anki,
            fs_port=fs,
            db_port=db,
            collection_dir=collection_dir,
            note_types_dir=get_note_types_dir(),
        )


def _insert_mock_note(
    mock_anki, *, deck_name: str, note_id: int, card_id: int, fields: dict[str, str]
):
    mock_anki.invoke("createDeck", deck=deck_name)
    mock_anki.notes[note_id] = {
        "noteId": note_id,
        "modelName": "AnkiOpsQA",
        "fields": {
            field_name: {"value": field_value}
            for field_name, field_value in fields.items()
        },
        "cards": [card_id],
    }
    mock_anki.cards[card_id] = {
        "cardId": card_id,
        "note": note_id,
        "deckName": deck_name,
        "modelName": "AnkiOpsQA",
    }


def test_export_from_anki_creates_markdown_files(tmp_path, mock_anki, run_ankiops):
    """Export should materialize one markdown file per deck."""
    mock_anki.add_note("Deck A", "AnkiOpsQA", {"Question": "Q1", "Answer": "A1"})
    mock_anki.add_note("Deck B", "AnkiOpsQA", {"Question": "Q2", "Answer": "A2"})

    summary = _run_export(tmp_path)
    assert len(summary.results) == 2

    file_a = tmp_path / f"{deck_name_to_file_stem('Deck A')}.md"
    file_b = tmp_path / f"{deck_name_to_file_stem('Deck B')}.md"
    assert file_a.exists()
    assert file_b.exists()

    content_a = file_a.read_text(encoding="utf-8")
    assert "Q: Q1" in content_a
    assert "A: A1" in content_a
    assert "<!-- note_key:" in content_a

    content_b = file_b.read_text(encoding="utf-8")
    assert "Q: Q2" in content_b


def test_all_note_types_integration(tmp_path, mock_anki, run_ankiops):
    """Import + export should preserve all supported note-type content."""
    deck_name = "AllTypes"
    md_file = tmp_path / f"{deck_name_to_file_stem(deck_name)}.md"
    md_file.write_text(
        (
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
        ),
        encoding="utf-8",
    )

    import_result = _run_import(tmp_path)
    assert import_result.results[0].errors == []
    assert_summary(
        import_result.summary, created=5, updated=0, moved=0, deleted=0, errors=0
    )

    notes = list(mock_anki.notes.values())
    assert len(notes) == 5
    assert {note_data["modelName"] for note_data in notes} == {
        "AnkiOpsQA",
        "AnkiOpsReversed",
        "AnkiOpsCloze",
        "AnkiOpsInput",
        "AnkiOpsChoice",
    }

    for note in notes:
        model = note["modelName"]
        fields = note["fields"]
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

    md_file.unlink()
    export_result = _run_export(tmp_path)
    assert export_result.results[0].file_path.exists()
    new_content = export_result.results[0].file_path.read_text(encoding="utf-8")
    assert "Q: QA Question" in new_content
    assert "F: Reversed Front" in new_content
    assert "T: Cloze with {{c1::hidden}} text" in new_content
    assert "I: Input Answer" in new_content
    assert "C1: Option A" in new_content


def test_ankiops_id_populated_on_create(tmp_path, mock_anki, run_ankiops):
    """Import create should always write a non-empty AnkiOps Key."""
    (tmp_path / f"{deck_name_to_file_stem('ReproDeck')}.md").write_text(
        "Q: Question\nA: Answer", encoding="utf-8"
    )

    _run_import(tmp_path)

    assert len(mock_anki.notes) == 1
    new_note = next(iter(mock_anki.notes.values()))
    assert "AnkiOps Key" in new_note["fields"]
    assert new_note["fields"]["AnkiOps Key"]["value"]


def test_ankiops_id_populated_on_update(tmp_path, mock_anki, run_ankiops):
    """Import update should backfill missing AnkiOps Key on existing note."""
    note_key = "notekeyexisting"
    note_id = 999

    _insert_mock_note(
        mock_anki,
        deck_name="ReproDeck",
        note_id=note_id,
        card_id=1001,
        fields={"Question": "OldQ", "Answer": "OldA", "AnkiOps Key": ""},
    )

    db_setup = SQLiteDbAdapter.load(tmp_path)
    try:
        db_setup.set_note(note_key, note_id)
    finally:
        db_setup.close()

    (tmp_path / f"{deck_name_to_file_stem('ReproDeck')}.md").write_text(
        f"<!-- note_key: {note_key} -->\nQ: OldQ\nA: UpdatedA",
        encoding="utf-8",
    )

    _run_import(tmp_path)

    updated_note = mock_anki.notes[note_id]
    assert updated_note["fields"]["Answer"]["value"] == "UpdatedA"
    assert updated_note["fields"]["AnkiOps Key"]["value"] == note_key


def test_export_reuses_existing_ankiops_key(tmp_path, mock_anki, run_ankiops):
    """Export should reuse existing AnkiOps Key from Anki and persist it in DB."""
    existing_note_key = "a1b2c3d4e5f6"
    deck_name = "ExistingKeyDeck"
    note_id = 12345

    _insert_mock_note(
        mock_anki,
        deck_name=deck_name,
        note_id=note_id,
        card_id=20001,
        fields={"Question": "Q1", "Answer": "A1", "AnkiOps Key": existing_note_key},
    )

    db_setup = SQLiteDbAdapter.load(tmp_path)
    try:
        assert db_setup.get_note_key(note_id) is None
    finally:
        db_setup.close()

    _run_export(tmp_path)

    md_file = tmp_path / f"{deck_name_to_file_stem(deck_name)}.md"
    assert md_file.exists()
    content = md_file.read_text(encoding="utf-8")
    assert f"<!-- note_key: {existing_note_key} -->" in content

    db_check = SQLiteDbAdapter.load(tmp_path)
    try:
        assert db_check.get_note_key(note_id) == existing_note_key
    finally:
        db_check.close()


def test_import_note_key_placement_trailing_space(tmp_path, mock_anki, run_ankiops):
    """Generated note_key comment should sit immediately above the note prefix."""
    deck_file = tmp_path / f"{deck_name_to_file_stem('TrailingSpaceDeck')}.md"
    deck_file.write_text(
        "Q: Trailing Space Question \nA: Normal Answer\n",
        encoding="utf-8",
    )

    _run_import(tmp_path)

    content = deck_file.read_text(encoding="utf-8")
    assert "<!-- note_key:" in content

    lines = content.splitlines()
    key_line_idx = next(
        line_index
        for line_index, line in enumerate(lines)
        if line.startswith("<!-- note_key:")
    )
    assert key_line_idx + 1 < len(lines)
    assert lines[key_line_idx + 1].startswith("Q: Trailing Space Question ")
