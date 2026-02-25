"""Stateful-ish property tests for sync invariants across operation sequences."""

from __future__ import annotations

import re
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from ankiops.anki import AnkiAdapter
from ankiops.config import ANKIOPS_DB, get_note_types_dir
from ankiops.db import SQLiteDbAdapter
from ankiops.export_notes import export_collection
from ankiops.fs import FileSystemAdapter
from ankiops.import_notes import import_collection
from tests.support.assertions import assert_unique
from tests.support.fake_anki import MockAnki

_NOTE_KEY_RE = re.compile(r"<!--\s*note_key:\s*([a-zA-Z0-9-]+)\s*-->")

_OPS = st.lists(
    st.sampled_from(
        [
            "md_create",
            "md_update",
            "md_delete",
            "anki_update",
            "sync_import",
            "sync_export",
        ]
    ),
    min_size=8,
    max_size=30,
)


def _replace_answer(content: str, new_answer: str) -> str:
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("A: "):
            lines[i] = f"A: {new_answer}"
            break
    suffix = "\n" if content.endswith("\n") else ""
    return "\n".join(lines) + suffix


def _assert_db_bijection(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT key, note_id FROM notes").fetchall()
    finally:
        conn.close()

    keys = [r[0] for r in rows]
    ids = [r[1] for r in rows]
    assert len(keys) == len(set(keys))
    assert len(ids) == len(set(ids))


def _sync_import(anki, fs, db, collection_dir: Path):
    return import_collection(
        anki_port=anki,
        fs_port=fs,
        db_port=db,
        collection_dir=collection_dir,
        note_types_dir=get_note_types_dir(),
    )


def _sync_export(anki, fs, db, collection_dir: Path):
    return export_collection(
        anki_port=anki,
        fs_port=fs,
        db_port=db,
        collection_dir=collection_dir,
        note_types_dir=get_note_types_dir(),
    )


def _assert_mapping_invariants(deck_file: Path, mock_anki: MockAnki, collection_dir: Path) -> None:
    if deck_file.exists():
        keys = _NOTE_KEY_RE.findall(deck_file.read_text(encoding="utf-8"))
        assert_unique(keys)

    anki_keys = []
    for note in mock_anki.notes.values():
        key_val = note["fields"].get("AnkiOps Key", {}).get("value", "")
        if key_val:
            anki_keys.append(key_val)
    assert_unique(anki_keys)

    _assert_db_bijection(collection_dir / ANKIOPS_DB)


@given(ops=_OPS)
@settings(max_examples=40, deadline=None)
def test_sync_sequences_preserve_key_and_mapping_invariants(ops: list[str]):
    with tempfile.TemporaryDirectory() as tdir:
        collection_dir = Path(tdir)
        deck_file = collection_dir / "StatefulDeck.md"

        mock_anki = MockAnki()
        anki = AnkiAdapter()
        fs = FileSystemAdapter()
        fs.set_configs(fs.load_note_type_configs(get_note_types_dir()))
        db = SQLiteDbAdapter.load(collection_dir)

        question_idx = 0

        try:
            with (
                patch("ankiops.anki_client.invoke", side_effect=mock_anki.invoke),
                patch("ankiops.anki.invoke", side_effect=mock_anki.invoke),
            ):
                for step, op in enumerate(ops):
                    if op == "md_create":
                        question_idx += 1
                        q = f"Q{question_idx}"
                        a = f"A{question_idx}"
                        deck_file.write_text(f"Q: {q}\nA: {a}\n", encoding="utf-8")

                    elif op == "md_update":
                        if deck_file.exists() and deck_file.read_text(encoding="utf-8").strip():
                            content = deck_file.read_text(encoding="utf-8")
                            updated = _replace_answer(content, f"A{step}-md")
                            deck_file.write_text(updated, encoding="utf-8")

                    elif op == "md_delete":
                        if deck_file.exists():
                            deck_file.write_text("", encoding="utf-8")

                    elif op == "anki_update":
                        if mock_anki.notes:
                            nid = next(iter(mock_anki.notes.keys()))
                            mock_anki.notes[nid]["fields"]["Answer"] = {
                                "value": f"A{step}-anki"
                            }

                    elif op == "sync_import":
                        _sync_import(anki, fs, db, collection_dir)

                    elif op == "sync_export":
                        _sync_export(anki, fs, db, collection_dir)

                    if op in {"sync_import", "sync_export"}:
                        _assert_mapping_invariants(deck_file, mock_anki, collection_dir)
        finally:
            db.close()
