"""Readable orchestration helper for sync scenario tests."""

from __future__ import annotations

import re
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ankiops.anki import Anki
from ankiops.cli_commands import run_af as run_cli_af
from ankiops.cli_commands import run_fa as run_cli_fa
from ankiops.collection import (
    ANKIOPS_DB,
    LOCAL_MEDIA_DIR,
    NOTE_TYPES_DIR,
    deck_name_to_file_stem,
)
from ankiops.deck_sources import discover_deck_sources
from ankiops.markdown import NOTE_SEPARATOR, format_tags_comment
from ankiops.sync.from_anki import sync_collection_from_anki
from ankiops.sync.state import SyncState
from ankiops.sync.to_anki import sync_collection_to_anki
from tests.support.deck_files import DeckFileHarness

_NOTE_KEY_RE = re.compile(r"<!--\s*note_key:\s*([a-zA-Z0-9-]+)\s*-->")


class SyncWorld:
    """High-level helper to keep scenario tests concise and explicit."""

    def __init__(self, root: Path, mock_anki):
        self.root = root
        self.mock_anki = mock_anki
        self.anki = Anki(invoke_func=self.mock_anki.invoke)
        self.note_types_dir = self.root / NOTE_TYPES_DIR
        self.mock_anki.media_dir = self.root / "anki_media"
        self.mock_anki.media_dir.mkdir(parents=True, exist_ok=True)

        fs = DeckFileHarness()
        if not self.note_types_dir.exists():
            fs.eject_default_note_types(self.note_types_dir)
        fs.set_note_types(fs.load_note_types(self.note_types_dir))
        self.fs = fs

    def open_db(self) -> SyncState:
        return SyncState.open(self.root)

    def db_for_state(
        self,
        state: str,
        *,
        note_map: dict[str, int] | None = None,
        deck_map: dict[str, int] | None = None,
    ) -> SyncState:
        db = self.open_db()

        if state in {"RUN", "CORR"}:
            for note_key, note_id in (note_map or {}).items():
                db.upsert_note_links([(note_key, note_id)])
            for deck_name, deck_id in (deck_map or {}).items():
                db.upsert_deck(deck_name, deck_id)

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

    def sync_import(self, db: SyncState):
        return sync_collection_to_anki(
            anki_port=self.anki,
            db_port=db,
            collection_dir=self.root,
            note_types_dir=self.note_types_dir,
        )

    def sync_export(self, db: SyncState):
        return sync_collection_from_anki(
            anki_port=self.anki,
            db_port=db,
            collection_dir=self.root,
            sources=discover_deck_sources(
                self.root, note_types_dir=self.note_types_dir
            ),
        )

    def sync_media_to_anki(self, db: SyncState):
        from ankiops.media import sync_all_media_to_anki

        return sync_all_media_to_anki(self.anki, self.root, db)

    def sync_media_from_anki(self, db: SyncState):
        from ankiops.media import sync_all_media_from_anki

        return sync_all_media_from_anki(self.anki, self.root, db)

    def run_fa(self, *, no_auto_commit: bool = True) -> None:
        args = SimpleNamespace(no_auto_commit=no_auto_commit)
        with (
            patch("ankiops.cli_commands.connect_or_exit", return_value=self.anki),
            patch(
                "ankiops.cli_commands.require_collection_dir",
                return_value=self.root,
            ),
        ):
            run_cli_fa(args)

    def run_af(self, *, no_auto_commit: bool = True) -> None:
        args = SimpleNamespace(no_auto_commit=no_auto_commit)
        with (
            patch("ankiops.cli_commands.connect_or_exit", return_value=self.anki),
            patch(
                "ankiops.cli_commands.require_collection_dir",
                return_value=self.root,
            ),
        ):
            run_cli_af(args)

    def deck_path(self, deck_name: str) -> Path:
        safe = deck_name_to_file_stem(deck_name)
        return self.root / f"{safe}.md"

    def write_raw_deck(self, deck_name: str, content: str) -> Path:
        path = self.deck_path(deck_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def write_deck(self, deck_name: str, content: str) -> Path:
        suffix = "\n" if content and not content.endswith("\n") else ""
        return self.write_raw_deck(deck_name, content + suffix)

    def write_shared_deck(
        self,
        owner: str,
        repo: str,
        deck_name: str,
        content: str,
    ) -> Path:
        source_root = self.root / "shared" / owner / repo
        source_note_types = source_root / NOTE_TYPES_DIR
        if not source_note_types.exists():
            DeckFileHarness().eject_default_note_types(source_note_types)
        path = source_root / f"{deck_name_to_file_stem(deck_name)}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = "\n" if content and not content.endswith("\n") else ""
        path.write_text(content + suffix, encoding="utf-8")
        return path

    def write_qa_deck(self, deck_name: str, notes: list[tuple]) -> Path:
        path = self.deck_path(deck_name)
        if not notes:
            path.write_text("", encoding="utf-8")
            return path

        blocks = []
        for note_data in notes:
            question, answer, note_key = note_data[:3]
            tags = note_data[3] if len(note_data) > 3 else ()
            lines = []
            if note_key:
                lines.append(f"<!-- note_key: {note_key} -->")
            tag_comment = format_tags_comment(tags)
            if tag_comment:
                lines.append(tag_comment)
            lines.append(f"Q: {question}")
            lines.append(f"A: {answer}")
            blocks.append("\n".join(lines))

        path.write_text(NOTE_SEPARATOR.join(blocks) + "\n", encoding="utf-8")
        return path

    def write_media(self, name: str, content: bytes = b"media") -> Path:
        path = self.root / LOCAL_MEDIA_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def seed_db_link(self, note_key: str, note_id: int) -> None:
        with self.db_session() as db:
            db.upsert_note_links([(note_key, note_id)])

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
        tags=(),
    ) -> int:
        fields = {"Question": question, "Answer": answer}
        if note_key is not None:
            fields["AnkiOps Key"] = note_key
        self.mock_anki.add_note(deck_name, "AnkiOpsQA", fields, tags=tags)
        return max(self.mock_anki.notes.keys())

    def add_anki_note(
        self,
        *,
        deck_name: str,
        note_type: str = "AnkiOpsQA",
        fields: dict | None = None,
        tags=(),
        note_key: str | None = None,
    ) -> int:
        resolved_fields = dict(fields or {"Question": "Q", "Answer": "A"})
        if note_key is not None:
            resolved_fields["AnkiOps Key"] = note_key
        self.mock_anki.add_note(deck_name, note_type, resolved_fields, tags=tags)
        return max(self.mock_anki.notes.keys())

    def assert_deck_contains(self, deck_name: str, text: str) -> None:
        assert text in self.read_deck(deck_name)

    def assert_deck_missing(self, deck_name: str) -> None:
        assert not self.deck_path(deck_name).exists()

    def assert_anki_note(
        self,
        *,
        deck_name: str,
        question: str | None = None,
        answer: str | None = None,
        note_key: str | None = None,
    ) -> int:
        for note_id, note in self.mock_anki.notes.items():
            card_decks = {
                self.mock_anki.cards[card_id]["deckName"]
                for card_id in note["cards"]
                if card_id in self.mock_anki.cards
            }
            fields = note["fields"]
            if deck_name not in card_decks:
                continue
            if (
                question is not None
                and fields.get("Question", {}).get("value") != question
            ):
                continue
            if answer is not None and fields.get("Answer", {}).get("value") != answer:
                continue
            if (
                note_key is not None
                and fields.get("AnkiOps Key", {}).get("value") != note_key
            ):
                continue
            return note_id
        raise AssertionError(f"No matching Anki note found in {deck_name}")

    def assert_anki_missing(self, *, note_key: str) -> None:
        for note in self.mock_anki.notes.values():
            assert note["fields"].get("AnkiOps Key", {}).get("value") != note_key

    def assert_db_link(self, note_key: str, note_id: int) -> None:
        with self.db_session() as db:
            assert db.resolve_note_ids([note_key]) == {note_key: note_id}

    def assert_media_file(self, name: str, content: bytes | None = None) -> None:
        path = self.root / LOCAL_MEDIA_DIR / name
        assert path.exists()
        if content is not None:
            assert path.read_bytes() == content

    def init_git(self) -> None:
        subprocess.run(["git", "init"], cwd=self.root, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.invalid"],
            cwd=self.root,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=self.root,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "add", "-A", "."],
            cwd=self.root,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=self.root,
            check=True,
            capture_output=True,
        )

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
