import pytest

from ankiops.config import deck_name_to_file_stem
from ankiops.db import SQLiteDbAdapter
from ankiops.fs import FileSystemAdapter
from ankiops.image_widths import fix_image_widths_collection


def _make_collection(tmp_path):
    db = SQLiteDbAdapter.open(tmp_path)
    try:
        db.set_profile_name("test")
    finally:
        db.close()

    fs = FileSystemAdapter()
    note_types_dir = tmp_path / "note_types"
    fs.eject_builtin_note_types(note_types_dir)
    return note_types_dir


def _write_qa_deck(tmp_path, deck_name: str, answer: str):
    path = tmp_path / f"{deck_name_to_file_stem(deck_name)}.md"
    note_key = deck_name.replace("::", "-").replace(" ", "-")
    path.write_text(
        f"<!-- note_key: {note_key}-1 -->\nQ: Question\nA: {answer}",
        encoding="utf-8",
    )
    return path


def test_fix_image_widths_collection_includes_subdecks_by_default(tmp_path):
    note_types_dir = _make_collection(tmp_path)
    parent = _write_qa_deck(
        tmp_path,
        "Parent",
        "![a](a.png){width=400}\n![b](b.png){width=404}",
    )
    child = _write_qa_deck(
        tmp_path,
        "Parent::Child",
        "![a](a.png){width=300}\n![b](b.png){width=304}",
    )
    other = _write_qa_deck(
        tmp_path,
        "Other",
        "![a](a.png){width=200}\n![b](b.png){width=204}",
    )

    result = fix_image_widths_collection(
        tmp_path,
        deck="Parent",
        no_subdecks=False,
        tolerance=5,
        note_types_dir=note_types_dir,
    )

    assert result.decks_checked == 2
    assert "{width=404}" not in parent.read_text(encoding="utf-8")
    assert "{width=304}" not in child.read_text(encoding="utf-8")
    assert "{width=204}" in other.read_text(encoding="utf-8")


def test_fix_image_widths_collection_can_exclude_subdecks(tmp_path):
    note_types_dir = _make_collection(tmp_path)
    parent = _write_qa_deck(
        tmp_path,
        "Parent",
        "![a](a.png){width=400}\n![b](b.png){width=404}",
    )
    child = _write_qa_deck(
        tmp_path,
        "Parent::Child",
        "![a](a.png){width=300}\n![b](b.png){width=304}",
    )

    result = fix_image_widths_collection(
        tmp_path,
        deck="Parent",
        no_subdecks=True,
        tolerance=5,
        note_types_dir=note_types_dir,
    )

    assert result.decks_checked == 1
    assert "{width=404}" not in parent.read_text(encoding="utf-8")
    assert "{width=304}" in child.read_text(encoding="utf-8")


def test_fix_image_widths_collection_rewrites_collab_deck(tmp_path):
    note_types_dir = _make_collection(tmp_path)
    fs = FileSystemAdapter()
    collab_root = tmp_path / "collab" / "owner" / "repo"
    fs.eject_builtin_note_types(collab_root / "note_types")
    collab_file = collab_root / "Shared.md"
    collab_file.write_text(
        "<!-- note_key: collab-1 -->\n"
        "Q: Question\n"
        "A: ![a](media/a.png){width=400}\n![b](media/b.png){width=404}",
        encoding="utf-8",
    )

    result = fix_image_widths_collection(
        tmp_path,
        deck="Shared",
        no_subdecks=True,
        tolerance=5,
        note_types_dir=note_types_dir,
    )

    assert result.decks_checked == 1
    assert "{width=404}" not in collab_file.read_text(encoding="utf-8")


def test_fix_image_widths_collection_fails_on_missing_note_key(tmp_path):
    note_types_dir = _make_collection(tmp_path)
    (tmp_path / "Broken.md").write_text(
        "Q: Question\nA: ![a](a.png){width=400}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing a note_key"):
        fix_image_widths_collection(
            tmp_path,
            tolerance=5,
            note_types_dir=note_types_dir,
        )
