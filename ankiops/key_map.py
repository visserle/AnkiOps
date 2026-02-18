"""Key to Anki ID mapping for Notes and Decks (SQLite backed).

Maintains a bidirectional mapping between user-managed Keys (stored in markdown)
and Anki-assigned integer IDs (used for API calls).
Supports separate namespaces for "keys" and "decks".

The mapping file is per-profile and should be gitignored.
Now uses SQLite for atomic writes and crash safety.
"""

import logging
import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ankiops.config import KEY_MAP_DB

logger = logging.getLogger(__name__)


@dataclass
class KeyMap:
    """Bidirectional Key ↔ ID mapping for notes and decks using SQLite."""

    _conn: sqlite3.Connection
    _db_path: Path

    @staticmethod
    def load(collection_dir: Path) -> "KeyMap":
        """Load mapping from the collection's key_map.db (migrating from json if needed)."""
        db_path = collection_dir / KEY_MAP_DB

        try:
            conn = sqlite3.connect(db_path)

            # Create schema
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS notes (
                        key TEXT PRIMARY KEY,
                        id INTEGER NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS decks (
                        key TEXT PRIMARY KEY,
                        id INTEGER NOT NULL
                    )
                """)
                # Index on ID for fast reverse lookup
                conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_id ON notes(id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_decks_id ON decks(id)")

        except sqlite3.DatabaseError:
            conn.close()
            logger.error(
                f"Database at {db_path} is corrupt. Backing up and creating new."
            )
            if db_path.exists():
                try:
                    db_path.rename(str(db_path) + ".corrupt")
                except OSError as e:
                    logger.error(f"Failed to rename corrupt database: {e}")

            # Recursive retry (should match clean state now)
            return KeyMap.load(collection_dir)

        map_obj = KeyMap(conn, db_path)

        return map_obj

    def save(self, collection_dir: Path) -> None:
        """No-op for SQLite implementation (auto-commits)."""
        pass

    def close(self) -> None:
        """Close database connection."""
        self._conn.close()

    # --- Note Methods ---

    def get_note_id(self, key_str: str) -> Optional[int]:
        """Look up Anki note_id by Key."""
        cursor = self._conn.execute("SELECT id FROM notes WHERE key = ?", (key_str,))
        row = cursor.fetchone()
        res = row[0] if row else None
        return res

    def get_key(self, note_id: int) -> Optional[str]:
        """Look up Key by Anki note_id (alias for get_note_key)."""
        return self.get_note_key(note_id)

    def get_note_key(self, note_id: int) -> Optional[str]:
        """Look up Key by Anki note_id."""
        cursor = self._conn.execute("SELECT key FROM notes WHERE id = ?", (note_id,))
        row = cursor.fetchone()
        return row[0] if row else None

    def set_note(self, key_str: str, note_id: int) -> None:
        """Add or update a Note Key ↔ note_id mapping."""
        # SQLite REPLACE handles the primary key conflict on 'key'.

        with self._conn:
            # Enforce 1-to-1: if this ID is already assigned to another Key, remove that mapping first
            self._conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            self._conn.execute(
                "INSERT OR REPLACE INTO notes (key, id) VALUES (?, ?)",
                (key_str, note_id),
            )

    def remove_note_by_key(self, key_str: str) -> None:
        """Remove a note mapping by Key."""
        with self._conn:
            self._conn.execute("DELETE FROM notes WHERE key = ?", (key_str,))

    def remove_note_by_id(self, note_id: int) -> None:
        """Remove a note mapping by note_id."""
        with self._conn:
            self._conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))

    # --- Deck Methods ---

    def get_deck_id(self, key_str: str) -> Optional[int]:
        """Look up Anki deck_id by Key."""
        cursor = self._conn.execute("SELECT id FROM decks WHERE key = ?", (key_str,))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_deck_key(self, deck_id: int) -> Optional[str]:
        """Look up Key by Anki deck_id."""
        cursor = self._conn.execute("SELECT key FROM decks WHERE id = ?", (deck_id,))
        row = cursor.fetchone()
        return row[0] if row else None

    def set_deck(self, key_str: str, deck_id: int) -> None:
        """Add or update a Deck Key ↔ deck_id mapping."""
        with self._conn:
            self._conn.execute("DELETE FROM decks WHERE id = ?", (deck_id,))
            self._conn.execute(
                "INSERT OR REPLACE INTO decks (key, id) VALUES (?, ?)",
                (key_str, deck_id),
            )

    def remove_deck_by_key(self, key_str: str) -> None:
        """Remove a deck mapping by Key."""
        with self._conn:
            self._conn.execute("DELETE FROM decks WHERE key = ?", (key_str,))

    # --- Helpers ---

    @staticmethod
    def generate_key() -> str:
        """Generate a new 12-char hex string (used as Key)."""
        return secrets.token_hex(6)
