"""Collection serialization round-trip tests."""

from __future__ import annotations

import pytest

from ankiops.config import deck_name_to_file_stem
from ankiops.db import SQLiteDbAdapter
from ankiops.fs import FileSystemAdapter
from ankiops.serializer import (
    deserialize,
    deserialize_from_file,
    serialize,
    serialize_to_file,
)


def _set_collection_paths(monkeypatch, collection_dir):
    monkeypatch.setattr("ankiops.config.get_collection_dir", lambda: collection_dir)
    monkeypatch.setattr(
        "ankiops.serializer.get_collection_dir",
        lambda: collection_dir,
    )


def _init_collection(collection_dir, fs):
    db = SQLiteDbAdapter.open(collection_dir)
    try:
        db.set_profile_name("test")
    finally:
        db.close()

    note_types_dir = collection_dir / "note_types"
    fs.eject_builtin_note_types(note_types_dir)
    return note_types_dir


def _serialized_deck(
    *,
    source="local",
    name="TestDeck",
    note_key="nk-1",
    note_type="AnkiOpsQA",
    fields=None,
    tags=None,
):
    return {
        "source": source,
        "name": name,
        "notes": [
            {
                "note_key": note_key,
                "note_type": note_type,
                "fields": fields or {"Question": "Q", "Answer": "A"},
                "tags": [] if tags is None else tags,
            }
        ],
    }


@pytest.fixture
def collection(tmp_path, fs, monkeypatch):
    """Create a minimal collection with DB, note types, and one deck file."""
    _set_collection_paths(monkeypatch, tmp_path)
    _init_collection(tmp_path, fs)

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
    result = serialize_to_file(collection, output)

    assert len(result["decks"]) == 1
    assert result["decks"][0]["source"] == "local"
    assert len(result["decks"][0]["notes"]) == 2
    assert result["decks"][0]["notes"][0]["note_key"] == "nk-1"


def test_in_memory_serialize(collection, monkeypatch):
    _set_collection_paths(monkeypatch, collection)
    result = serialize(collection)

    assert len(result["decks"]) == 1
    assert result["decks"][0]["source"] == "local"
    assert len(result["decks"][0]["notes"]) == 2
    assert result["decks"][0]["notes"][1]["note_key"] == "nk-2"


def test_serialize_includes_tags(collection, monkeypatch):
    _set_collection_paths(monkeypatch, collection)
    deck_file = collection / f"{deck_name_to_file_stem('TestDeck')}.md"
    deck_file.write_text(
        deck_file.read_text(encoding="utf-8").replace(
            "<!-- note_key: nk-1 -->",
            "<!-- note_key: nk-1 -->\n<!-- tags: z high-yield -->",
            1,
        ),
        encoding="utf-8",
    )

    result = serialize(collection)

    assert result["decks"][0]["notes"][0]["tags"] == ["high-yield", "z"]


def test_serialize_includes_shared_sources_and_ignores_reserved_docs(
    collection, fs, monkeypatch
):
    _set_collection_paths(monkeypatch, collection)
    shared_root = collection / "shared" / "owner" / "repo"
    fs.eject_builtin_note_types(shared_root / "note_types")
    (shared_root / "README.md").write_text("# docs", encoding="utf-8")
    (shared_root / "LICENSE.md").write_text("# license", encoding="utf-8")
    (shared_root / "CHANGELOG.md").write_text("# changes", encoding="utf-8")
    (shared_root / "_draft.md").write_text(
        "<!-- note_key: draft -->\nQ: draft\nA: draft",
        encoding="utf-8",
    )
    (shared_root / "Shared.md").write_text(
        "<!-- note_key: shared-1 -->\nQ: shared question\nA: shared answer",
        encoding="utf-8",
    )

    result = serialize(collection)

    decks = {(deck["source"], deck["name"]): deck for deck in result["decks"]}
    assert ("local", "TestDeck") in decks
    assert ("shared/owner/repo", "Shared") in decks
    assert ("shared/owner/repo", "README") not in decks
    shared_note = decks[("shared/owner/repo", "Shared")]["notes"][0]
    assert shared_note["note_type"] == "shared/owner/repo/AnkiOpsQA"


def test_serialize_orders_local_before_sorted_shared_sources(
    collection, fs, monkeypatch
):
    _set_collection_paths(monkeypatch, collection)
    shared_b = collection / "shared" / "z-owner" / "deck"
    shared_a = collection / "shared" / "a-owner" / "deck"
    fs.eject_builtin_note_types(shared_b / "note_types")
    fs.eject_builtin_note_types(shared_a / "note_types")
    (shared_b / "B.md").write_text(
        "<!-- note_key: shared-b -->\nQ: b\nA: b",
        encoding="utf-8",
    )
    (shared_a / "A.md").write_text(
        "<!-- note_key: shared-a -->\nQ: a\nA: a",
        encoding="utf-8",
    )

    result = serialize(collection)

    assert [(deck["source"], deck["name"]) for deck in result["decks"]] == [
        ("local", "TestDeck"),
        ("shared/a-owner/deck", "A"),
        ("shared/z-owner/deck", "B"),
    ]


def test_serialize_global_validation_rejects_duplicate_deck_names(
    collection, fs, monkeypatch
):
    _set_collection_paths(monkeypatch, collection)
    shared_root = collection / "shared" / "owner" / "repo"
    fs.eject_builtin_note_types(shared_root / "note_types")
    (shared_root / "TestDeck.md").write_text(
        "<!-- note_key: shared-1 -->\nQ: shared question\nA: shared answer",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate deck name 'TestDeck'"):
        serialize(collection)


def test_serialize_global_validation_rejects_duplicate_note_keys(
    collection, monkeypatch
):
    _set_collection_paths(monkeypatch, collection)
    (collection / "OtherDeck.md").write_text(
        "<!-- note_key: nk-1 -->\nQ: duplicate\nA: duplicate",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate note_key 'nk-1'"):
        serialize(collection)


def test_serialize_global_validation_allows_missing_note_keys(collection, monkeypatch):
    _set_collection_paths(monkeypatch, collection)
    (collection / "MissingKey.md").write_text(
        "Q: missing key\nA: missing key",
        encoding="utf-8",
    )

    result = serialize(collection)

    missing_deck = next(
        deck for deck in result["decks"] if deck["name"] == "MissingKey"
    )
    assert missing_deck["notes"][0]["note_key"] is None


def test_serialize_single_deck_includes_subdecks_by_default(collection, monkeypatch):
    _set_collection_paths(monkeypatch, collection)

    (collection / f"{deck_name_to_file_stem('TestDeck::Child')}.md").write_text(
        ("<!-- note_key: nk-child-1 -->\nQ: Child question\nA: Child answer"),
        encoding="utf-8",
    )
    (collection / f"{deck_name_to_file_stem('OtherDeck')}.md").write_text(
        ("<!-- note_key: nk-other-1 -->\nQ: Other question\nA: Other answer"),
        encoding="utf-8",
    )

    result = serialize(collection, deck="TestDeck")

    deck_names = {deck["name"] for deck in result["decks"]}
    assert deck_names == {"TestDeck", "TestDeck::Child"}


def test_serialize_single_deck_excludes_subdecks_with_no_subdecks(
    collection, monkeypatch
):
    _set_collection_paths(monkeypatch, collection)

    (collection / f"{deck_name_to_file_stem('TestDeck::Child')}.md").write_text(
        ("<!-- note_key: nk-child-1 -->\nQ: Child question\nA: Child answer"),
        encoding="utf-8",
    )

    result = serialize(
        collection,
        deck="TestDeck",
        no_subdecks=True,
    )

    assert [deck["name"] for deck in result["decks"]] == ["TestDeck"]


def test_serialize_to_file_accepts_deck_scope(collection, tmp_path, monkeypatch):
    _set_collection_paths(monkeypatch, collection)

    (collection / f"{deck_name_to_file_stem('TestDeck::Child')}.md").write_text(
        ("<!-- note_key: nk-child-1 -->\nQ: Child question\nA: Child answer"),
        encoding="utf-8",
    )

    output = tmp_path / "scoped.json"
    result = serialize_to_file(
        collection,
        output,
        deck="TestDeck",
        no_subdecks=True,
    )

    assert [deck["name"] for deck in result["decks"]] == ["TestDeck"]


def test_serialize_validates_whole_collection_before_deck_scope(
    collection, monkeypatch
):
    _set_collection_paths(monkeypatch, collection)
    (collection / "OtherDeck.md").write_text(
        "<!-- note_key: nk-1 -->\nQ: duplicate outside scope\nA: duplicate",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate note_key 'nk-1'"):
        serialize(collection, deck="TestDeck", no_subdecks=True)


def test_serialize_fails_fast_on_parsing_error_by_default(collection, monkeypatch):
    _set_collection_paths(monkeypatch, collection)

    broken_name = f"{deck_name_to_file_stem('BrokenDeck')}.md"
    broken_file = collection / broken_name
    broken_file.write_text("Q: Broken", encoding="utf-8")

    original_read = FileSystemAdapter.read_markdown_file

    def _fake_read_markdown_file(self, md_file, *, context_root=None):
        if md_file.name == broken_name:
            raise ValueError("synthetic parse failure")
        return original_read(self, md_file, context_root=context_root)

    monkeypatch.setattr(
        FileSystemAdapter,
        "read_markdown_file",
        _fake_read_markdown_file,
    )

    with pytest.raises(
        ValueError,
        match=f"Error parsing local:{broken_name}: synthetic parse failure",
    ):
        serialize(collection)


def test_roundtrip(collection, tmp_path, fs, monkeypatch):
    """Serialize then deserialize should preserve deck note content."""
    _set_collection_paths(monkeypatch, collection)

    json_file = tmp_path / "export.json"
    serialize_to_file(collection, json_file)

    fresh_dir = tmp_path / "fresh"
    fresh_dir.mkdir()
    _set_collection_paths(monkeypatch, fresh_dir)
    _init_collection(fresh_dir, fs)

    deserialize_from_file(json_file, overwrite=True)

    md_files = list(fresh_dir.glob("*.md"))
    assert len(md_files) == 1
    content = md_files[0].read_text(encoding="utf-8")
    assert "What is 2+2?" in content
    assert "What is 3+3?" in content


def test_roundtrip_in_memory(collection, tmp_path, fs, monkeypatch):
    _set_collection_paths(monkeypatch, collection)

    serialized_data = serialize(collection)

    fresh_dir = tmp_path / "fresh-in-memory"
    fresh_dir.mkdir()
    _set_collection_paths(monkeypatch, fresh_dir)
    note_types_dst = _init_collection(fresh_dir, fs)

    deserialize(
        serialized_data,
        collection_dir=fresh_dir,
        note_types_dir=note_types_dst,
        overwrite=True,
    )

    md_files = list(fresh_dir.glob("*.md"))
    assert len(md_files) == 1
    content = md_files[0].read_text(encoding="utf-8")
    assert "What is 2+2?" in content
    assert "What is 3+3?" in content


def test_roundtrip_preserves_tags(collection, tmp_path, fs, monkeypatch):
    _set_collection_paths(monkeypatch, collection)
    deck_file = collection / f"{deck_name_to_file_stem('TestDeck')}.md"
    deck_file.write_text(
        deck_file.read_text(encoding="utf-8").replace(
            "<!-- note_key: nk-1 -->",
            "<!-- note_key: nk-1 -->\n<!-- tags: z high-yield -->",
            1,
        ),
        encoding="utf-8",
    )

    serialized_data = serialize(collection)

    fresh_dir = tmp_path / "fresh-tags"
    fresh_dir.mkdir()
    _set_collection_paths(monkeypatch, fresh_dir)
    note_types_dst = _init_collection(fresh_dir, fs)

    deserialize(
        serialized_data,
        collection_dir=fresh_dir,
        note_types_dir=note_types_dst,
        overwrite=True,
    )

    md_files = list(fresh_dir.glob("*.md"))
    assert len(md_files) == 1
    content = md_files[0].read_text(encoding="utf-8")
    assert "<!-- tags: high-yield z -->" in content


def test_deserialize_requires_initialized_collection(tmp_path, fs):
    note_types_dir = tmp_path / "note_types"
    fs.eject_builtin_note_types(note_types_dir)

    with pytest.raises(ValueError, match="Not an initialized AnkiOps collection"):
        deserialize(
            {"decks": [_serialized_deck()]},
            collection_dir=tmp_path,
            note_types_dir=note_types_dir,
            overwrite=True,
        )


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (
            {"decks": [{"name": "Deck", "notes": []}]},
            "missing required source",
        ),
        (
            {"decks": [_serialized_deck(source="shared/missing/repo")]},
            "unknown source 'shared/missing/repo'",
        ),
        (
            {"decks": [_serialized_deck(note_type="MissingType")]},
            "unknown note_type 'MissingType'",
        ),
        (
            {"decks": [_serialized_deck(note_key="")]},
            "note_key must be a non-empty string or null",
        ),
        (
            {
                "decks": [
                    {
                        "source": "local",
                        "name": "Deck",
                        "notes": [
                            {
                                "note_type": "AnkiOpsQA",
                                "fields": {"Question": "Q", "Answer": "A"},
                                "tags": [],
                            }
                        ],
                    }
                ]
            },
            "missing required note_key field",
        ),
        (
            {
                "decks": [
                    _serialized_deck(name="Deck", note_key="nk-1"),
                    _serialized_deck(name="Deck", note_key="nk-2"),
                ]
            },
            "Duplicate deck name 'Deck'",
        ),
        (
            {
                "decks": [
                    _serialized_deck(name="Deck", note_key="nk-1"),
                    _serialized_deck(name="Other", note_key="nk-1"),
                ]
            },
            "Duplicate note_key 'nk-1'",
        ),
        (
            {
                "decks": [
                    _serialized_deck(fields={"Question": 123, "Answer": "A"}),
                ]
            },
            "field 'Question' must be a string",
        ),
        (
            {
                "decks": [
                    _serialized_deck(tags=["ok", 123]),
                ]
            },
            "tags must contain only strings",
        ),
    ],
)
def test_deserialize_rejects_invalid_data_before_writes(tmp_path, fs, data, message):
    note_types_dir = _init_collection(tmp_path, fs)

    with pytest.raises(ValueError, match=message):
        deserialize(
            data,
            collection_dir=tmp_path,
            note_types_dir=note_types_dir,
            overwrite=True,
        )

    assert not list(tmp_path.glob("*.md"))


def test_deserialize_accepts_null_note_key(tmp_path, fs):
    note_types_dir = _init_collection(tmp_path, fs)

    deserialize(
        {"decks": [_serialized_deck(name="DraftDeck", note_key=None)]},
        collection_dir=tmp_path,
        note_types_dir=note_types_dir,
        overwrite=True,
    )

    content = (tmp_path / "DraftDeck.md").read_text(encoding="utf-8")
    assert "<!-- note_key:" not in content
    assert "<!-- note_type: AnkiOpsQA -->" in content
    assert "Q: Q" in content


def test_deserialize_rejects_shared_source_with_missing_note_types(tmp_path, fs):
    note_types_dir = _init_collection(tmp_path, fs)
    (tmp_path / "shared" / "owner" / "repo").mkdir(parents=True)

    with pytest.raises(ValueError, match="note_types/ cannot be loaded"):
        deserialize(
            {
                "decks": [
                    _serialized_deck(
                        source="shared/owner/repo",
                        name="Shared",
                        note_type="shared/owner/repo/AnkiOpsQA",
                    )
                ]
            },
            collection_dir=tmp_path,
            note_types_dir=note_types_dir,
            overwrite=True,
        )

    assert not (tmp_path / "shared" / "owner" / "repo" / "Shared.md").exists()


def test_deserialize_writes_local_and_shared_decks_to_owning_sources(tmp_path, fs):
    note_types_dir = _init_collection(tmp_path, fs)
    shared_root = tmp_path / "shared" / "owner" / "repo"
    fs.eject_builtin_note_types(shared_root / "note_types")

    deserialize(
        {
            "decks": [
                _serialized_deck(name="LocalDeck", note_key="local-1"),
                _serialized_deck(
                    source="shared/owner/repo",
                    name="SharedDeck",
                    note_key="shared-1",
                    note_type="shared/owner/repo/AnkiOpsQA",
                    fields={"Question": "CQ", "Answer": "CA"},
                ),
            ]
        },
        collection_dir=tmp_path,
        note_types_dir=note_types_dir,
        overwrite=True,
    )

    assert "<!-- note_type: AnkiOpsQA -->" in (tmp_path / "LocalDeck.md").read_text(
        encoding="utf-8"
    )
    shared_content = (shared_root / "SharedDeck.md").read_text(encoding="utf-8")
    assert "<!-- note_type: shared/owner/repo/AnkiOpsQA -->" in shared_content
    assert "Q: CQ" in shared_content


def test_deserialize_skips_existing_targets_without_overwrite(tmp_path, fs):
    note_types_dir = _init_collection(tmp_path, fs)
    target = tmp_path / "Existing.md"
    target.write_text("keep me", encoding="utf-8")

    deserialize(
        {"decks": [_serialized_deck(name="Existing", note_key="nk-existing")]},
        collection_dir=tmp_path,
        note_types_dir=note_types_dir,
        overwrite=False,
    )

    assert target.read_text(encoding="utf-8") == "keep me"


def test_deserialize_from_file_uses_source_schema(tmp_path, fs, monkeypatch):
    _set_collection_paths(monkeypatch, tmp_path)
    _init_collection(tmp_path, fs)
    json_file = tmp_path / "in.json"
    json_file.write_text(
        """
{
  "collection": {"serialized_at": "2026-06-05T00:00:00Z"},
  "decks": [
    {
      "source": "local",
      "name": "FromFile",
      "notes": [
        {
          "note_key": "nk-file",
          "note_type": "AnkiOpsQA",
          "fields": {"Question": "Q file", "Answer": "A file"},
          "tags": []
        }
      ]
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    deserialize_from_file(json_file, overwrite=True)

    content = (tmp_path / "FromFile.md").read_text(encoding="utf-8")
    assert "<!-- note_key: nk-file -->" in content
    assert "<!-- note_type: AnkiOpsQA -->" in content
