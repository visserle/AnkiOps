"""Readable orchestration helper for sync scenario tests."""

from __future__ import annotations

from contextlib import contextmanager
import re
from pathlib import Path

from ankiops.anki import AnkiAdapter
from ankiops.config import ANKIOPS_DB, NOTE_SEPARATOR, get_note_types_dir
from ankiops.db import SQLiteDbAdapter
from ankiops.export_notes import export_collection
from ankiops.fs import FileSystemAdapter
from ankiops.import_notes import import_collection

_NOTE_KEY_RE = re.compile(r"<!--\s*note_key:\s*([a-zA-Z0-9-]+)\s*-->")


class SyncWorld:
    """High-level helper to keep scenario tests concise and explicit."""

    def __init__(self, root: Path, mock_anki):
        self.root = root
        self.mock_anki = mock_anki
        self.anki = AnkiAdapter()

        fs = FileSystemAdapter()
        fs.set_configs(fs.load_note_type_configs(get_note_types_dir()))
        self.fs = fs

    def open_db(self) -> SQLiteDbAdapter:
        return SQLiteDbAdapter.load(self.root)

    def db_for_state(
        self,
        state: str,
        *,
        note_map: dict[str, int] | None = None,
        deck_map: dict[str, int] | None = None,
    ) -> SQLiteDbAdapter:
        db = self.open_db()

        if state in {"RUN", "CORR"}:
            for note_key, note_id in (note_map or {}).items():
                db.set_note(note_key, note_id)
            for deck_name, deck_id in (deck_map or {}).items():
                db.set_deck(deck_name, deck_id)

        if state == "CORR":
            db.close()
            self.corrupt_db()
            db = self.open_db()

        return db

    @contextmanager
    def db_session(
        self,
        state: str = "FRESH",
        *,
        note_map: dict[str, int] | None = None,
        deck_map: dict[str, int] | None = None,
    ):
        """Open and auto-close a DB prepared for the requested collection state."""
        db = self.db_for_state(state, note_map=note_map, deck_map=deck_map)
        try:
            yield db
        finally:
            db.close()

    def sync_import(self, db: SQLiteDbAdapter):
        return import_collection(
            anki_port=self.anki,
            fs_port=self.fs,
            db_port=db,
            collection_dir=self.root,
            note_types_dir=get_note_types_dir(),
        )

    def sync_export(self, db: SQLiteDbAdapter):
        return export_collection(
            anki_port=self.anki,
            fs_port=self.fs,
            db_port=db,
            collection_dir=self.root,
            note_types_dir=get_note_types_dir(),
        )

    def deck_path(self, deck_name: str) -> Path:
        safe = deck_name.replace("::", "__")
        return self.root / f"{safe}.md"

    def write_qa_deck(
        self, deck_name: str, notes: list[tuple[str, str, str | None]]
    ) -> Path:
        path = self.deck_path(deck_name)
        if not notes:
            path.write_text("", encoding="utf-8")
            return path

        blocks = []
        for question, answer, note_key in notes:
            lines = []
            if note_key:
                lines.append(f"<!-- note_key: {note_key} -->")
            lines.append(f"Q: {question}")
            lines.append(f"A: {answer}")
            blocks.append("\n".join(lines))

        path.write_text(NOTE_SEPARATOR.join(blocks) + "\n", encoding="utf-8")
        return path

    def read_deck(self, deck_name: str) -> str:
        return self.deck_path(deck_name).read_text(encoding="utf-8")

    def extract_note_keys(self, deck_name: str) -> list[str]:
        return _NOTE_KEY_RE.findall(self.read_deck(deck_name))

    def add_qa_note(
        self,
        *,
        deck_name: str,
        question: str,
        answer: str,
        note_key: str | None = None,
    ) -> int:
        fields = {"Question": question, "Answer": answer}
        if note_key is not None:
            fields["AnkiOps Key"] = note_key
        self.mock_anki.add_note(deck_name, "AnkiOpsQA", fields)
        return max(self.mock_anki.notes.keys())

    def set_note_answer(self, note_id: int, answer: str) -> None:
        self.mock_anki.notes[note_id]["fields"]["Answer"] = {"value": answer}

    def remove_note(self, note_id: int) -> None:
        self.mock_anki.invoke("deleteNotes", notes=[note_id])

    def rename_deck(self, old_name: str, new_name: str) -> None:
        deck_id = self.mock_anki.decks.pop(old_name)
        self.mock_anki.decks[new_name] = deck_id
        for card in self.mock_anki.cards.values():
            if card["deckName"] == old_name:
                card["deckName"] = new_name

    def corrupt_db(self) -> Path:
        db_path = self.root / ANKIOPS_DB
        db_path.write_text("corrupt data", encoding="utf-8")
        return db_path
