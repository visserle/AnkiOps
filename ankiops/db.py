"""SQLite Database Adapter mapping Keys to Anki IDs."""

import logging
import secrets
import sqlite3
from pathlib import Path

from ankiops.config import ANKIOPS_DB

logger = logging.getLogger(__name__)


class SQLiteDbAdapter:
    """Bidirectional Key ↔ ID mapping using SQLite."""

    def __init__(self, conn: sqlite3.Connection, db_path: Path):
        self._conn = conn
        self._db_path = db_path

    @classmethod
    def load(cls, collection_dir: Path) -> "SQLiteDbAdapter":
        db_path = collection_dir / ANKIOPS_DB
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = None
        try:
            conn = sqlite3.connect(db_path)
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS notes (
                        key TEXT PRIMARY KEY,
                        id INTEGER NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS decks (
                        name TEXT PRIMARY KEY,
                        id INTEGER NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS config (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_id ON notes(id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_decks_id ON decks(id)")

        except (sqlite3.DatabaseError, sqlite3.OperationalError):
            if conn:
                conn.close()
            logger.error(
                f"Database at {db_path} is corrupt. Backing up and recreating."
            )
            if db_path.exists():
                try:
                    db_path.rename(str(db_path) + ".corrupt")
                except OSError as e:
                    logger.error(f"Failed to rename corrupt database: {e}")

            return cls.load(collection_dir)

        return cls(conn, db_path)

    def save(self) -> None:
        """Intentional no-op: each write auto-commits via ``with self._conn:``."""
        pass

    def close(self) -> None:
        self._conn.close()

    def get_config(self, key: str) -> str | None:
        cursor = self._conn.execute("SELECT value FROM config WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None

    def set_config(self, key: str, value: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (key, value)
            )

    def get_note_id(self, key: str) -> int | None:
        cursor = self._conn.execute("SELECT id FROM notes WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_note_key(self, note_id: int) -> str | None:
        cursor = self._conn.execute("SELECT key FROM notes WHERE id = ?", (note_id,))
        row = cursor.fetchone()
        return row[0] if row else None

    def set_note(self, key: str, note_id: int) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            self._conn.execute(
                "INSERT OR REPLACE INTO notes (key, id) VALUES (?, ?)",
                (key, note_id),
            )

    def remove_note_by_key(self, key: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM notes WHERE key = ?", (key,))

    def remove_note_by_id(self, note_id: int) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))

    # -- Deck mapping (deck_name ↔ deck_id) ----------------------------------

    def get_deck_id(self, name: str) -> int | None:
        cursor = self._conn.execute("SELECT id FROM decks WHERE name = ?", (name,))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_deck_name(self, deck_id: int) -> str | None:
        cursor = self._conn.execute("SELECT name FROM decks WHERE id = ?", (deck_id,))
        row = cursor.fetchone()
        return row[0] if row else None

    def set_deck(self, name: str, deck_id: int) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM decks WHERE id = ?", (deck_id,))
            self._conn.execute(
                "INSERT OR REPLACE INTO decks (name, id) VALUES (?, ?)",
                (name, deck_id),
            )

    def remove_deck(self, name: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM decks WHERE name = ?", (name,))

    def generate_key(self) -> str:
        return secrets.token_hex(6)
