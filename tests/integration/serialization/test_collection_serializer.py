"""Collection serialization round-trip tests."""

from __future__ import annotations

import logging

import pytest

from ankiops.collection_serializer import (
    deserialize_collection_data,
    deserialize_collection_from_json,
    serialize_collection,
    serialize_collection_to_json,
)
from ankiops.config import deck_name_to_file_stem
from ankiops.db import SQLiteDbAdapter
from ankiops.fs import FileSystemAdapter


def _set_collection_paths(monkeypatch, collection_dir):
    monkeypatch.setattr("ankiops.config.get_collection_dir", lambda: collection_dir)
    monkeypatch.setattr(
        "ankiops.collection_serializer.get_collection_dir",
        lambda: collection_dir,
    )


def _set_note_type_paths(monkeypatch, note_types_dir):
    monkeypatch.setattr("ankiops.config.get_note_types_dir", lambda: note_types_dir)
    monkeypatch.setattr(
        "ankiops.collection_serializer.get_note_types_dir",
        lambda: note_types_dir,
    )


@pytest.fixture
def collection(tmp_path, fs, monkeypatch):
    """Create a minimal collection with DB, note types, and one deck file."""
    _set_collection_paths(monkeypatch, tmp_path)

    db = SQLiteDbAdapter.load(tmp_path)
    try:
        db.set_profile_name("test")
        db.save()
    finally:
        db.close()

    note_types_dir = tmp_path / "note_types"
    fs.eject_builtin_note_types(note_types_dir)
    _set_note_type_paths(monkeypatch, note_types_dir)

    (tmp_path / f"{deck_name_to_file_stem('TestDeck')}.md").write_text(
        (
            "<!-- note_key: nk-1 -->\n"
            "Q: What is 2+2?\n"
            "A: 4\n\n"
            "---\n\n"
            "<!-- note_key: nk-2 -->\n"
            "Q: What is 3+3?\n"
            "A: 6"
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_basic_serialize(collection, tmp_path, monkeypatch):
    _set_collection_paths(monkeypatch, collection)
    output = tmp_path / "out.json"
    result = serialize_collection_to_json(collection, output)

    assert len(result["decks"]) == 1
    assert len(result["decks"][0]["notes"]) == 2
    assert result["decks"][0]["notes"][0]["note_key"] == "nk-1"


def test_in_memory_serialize(collection, monkeypatch):
    _set_collection_paths(monkeypatch, collection)
    result = serialize_collection(collection)

    assert len(result["decks"]) == 1
    assert len(result["decks"][0]["notes"]) == 2
    assert result["decks"][0]["notes"][1]["note_key"] == "nk-2"


def test_serialize_logs_parsing_errors_before_summary(collection, monkeypatch, caplog):
    _set_collection_paths(monkeypatch, collection)
    _set_note_type_paths(monkeypatch, collection / "note_types")

    broken_name = f"{deck_name_to_file_stem('BrokenDeck')}.md"
    broken_file = collection / broken_name
    broken_file.write_text("Q: Broken", encoding="utf-8")

    original_read = FileSystemAdapter.read_markdown_file

    def _fake_read_markdown_file(self, md_file):
        if md_file.name == broken_name:
            raise ValueError("synthetic parse failure")
        return original_read(self, md_file)

    monkeypatch.setattr(
        FileSystemAdapter,
        "read_markdown_file",
        _fake_read_markdown_file,
    )

    with caplog.at_level(logging.WARNING):
        serialize_collection(collection)

    assert f"Error parsing {broken_name}: synthetic parse failure" in caplog.text
    assert "Serialization completed with 1 error(s)." in caplog.text


def test_roundtrip(collection, tmp_path, monkeypatch):
    """Serialize then deserialize should preserve deck note content."""
    _set_collection_paths(monkeypatch, collection)
    _set_note_type_paths(monkeypatch, collection / "note_types")

    json_file = tmp_path / "export.json"
    serialize_collection_to_json(collection, json_file)

    fresh_dir = tmp_path / "fresh"
    fresh_dir.mkdir()
    _set_collection_paths(monkeypatch, fresh_dir)

    fs = FileSystemAdapter()
    note_types_dst = fresh_dir / "note_types"
    fs.eject_builtin_note_types(note_types_dst)
    _set_note_type_paths(monkeypatch, note_types_dst)

    deserialize_collection_from_json(json_file, overwrite=True)

    md_files = list(fresh_dir.glob("*.md"))
    assert len(md_files) == 1
    content = md_files[0].read_text(encoding="utf-8")
    assert "What is 2+2?" in content
    assert "What is 3+3?" in content


def test_roundtrip_in_memory(collection, tmp_path, monkeypatch):
    _set_collection_paths(monkeypatch, collection)
    _set_note_type_paths(monkeypatch, collection / "note_types")

    serialized_data = serialize_collection(collection)

    fresh_dir = tmp_path / "fresh-in-memory"
    fresh_dir.mkdir()
    _set_collection_paths(monkeypatch, fresh_dir)

    fs = FileSystemAdapter()
    note_types_dst = fresh_dir / "note_types"
    fs.eject_builtin_note_types(note_types_dst)
    _set_note_type_paths(monkeypatch, note_types_dst)

    deserialize_collection_data(serialized_data, overwrite=True)

    md_files = list(fresh_dir.glob("*.md"))
    assert len(md_files) == 1
    content = md_files[0].read_text(encoding="utf-8")
    assert "What is 2+2?" in content
    assert "What is 3+3?" in content


def test_deserialize_unknown_note_type_skips_note_without_orphan_key(
    tmp_path, fs, monkeypatch
):
    _set_collection_paths(monkeypatch, tmp_path)

    note_types_dir = tmp_path / "note_types"
    fs.eject_builtin_note_types(note_types_dir)
    _set_note_type_paths(monkeypatch, note_types_dir)

    json_file = tmp_path / "in.json"
    json_file.write_text(
        """
{
  "collection": {"serialized_at": "2025-01-01T00:00:00Z"},
  "decks": [
        {
          "name": "UnknownDeck",
          "notes": [
            {
              "note_key": "nk-unknown",
              "note_type": "MissingType",
              "fields": {"UnknownField": "Only"}
            }
          ]
        }
      ]
}
""".strip(),
        encoding="utf-8",
    )

    deserialize_collection_from_json(json_file, overwrite=True)

    content = (tmp_path / f"{deck_name_to_file_stem('UnknownDeck')}.md").read_text(
        encoding="utf-8"
    )
    assert "note_key: nk-unknown" not in content
    assert content == ""


def test_deserialize_unknown_note_type_falls_back_to_inference(
    tmp_path, fs, monkeypatch
):
    _set_collection_paths(monkeypatch, tmp_path)

    note_types_dir = tmp_path / "note_types"
    fs.eject_builtin_note_types(note_types_dir)
    _set_note_type_paths(monkeypatch, note_types_dir)

    json_file = tmp_path / "in.json"
    json_file.write_text(
        """
{
  "collection": {"serialized_at": "2025-01-01T00:00:00Z"},
  "decks": [
    {
      "name": "InferDeck",
      "notes": [
        {
          "note_key": "nk-infer",
          "note_type": "MissingType",
          "fields": {"Question": "Q infer", "Answer": "A infer"}
        }
      ]
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    deserialize_collection_from_json(json_file, overwrite=True)

    content = (tmp_path / f"{deck_name_to_file_stem('InferDeck')}.md").read_text(
        encoding="utf-8"
    )
    assert "<!-- note_key: nk-infer -->" in content
    assert "Q: Q infer" in content
    assert "A: A infer" in content
