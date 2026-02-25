"""SQLite Database Adapter mapping Keys to Anki IDs."""

import logging
import secrets
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from ankiops.config import ANKIOPS_DB

logger = logging.getLogger(__name__)


class SQLiteDbAdapter:
    """Bidirectional Key ↔ ID mapping using SQLite."""

    def __init__(self, conn: sqlite3.Connection, db_path: Path):
        self._conn = conn
        self._db_path = db_path
        self._tx_depth = 0

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
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS note_fingerprints (
                        key TEXT PRIMARY KEY,
                        md_hash TEXT NOT NULL,
                        anki_hash TEXT NOT NULL
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
        """Intentional no-op: transaction boundaries are explicit."""
        pass

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[None]:
        """Run DB writes in a single transaction (supports nesting)."""
        is_outer = self._tx_depth == 0
        if is_outer:
            begin = "BEGIN IMMEDIATE" if immediate else "BEGIN"
            self._conn.execute(begin)
        self._tx_depth += 1
        try:
            yield
        except Exception:
            self._tx_depth -= 1
            if is_outer:
                self._conn.rollback()
            raise
        else:
            self._tx_depth -= 1
            if is_outer:
                self._conn.commit()

    def _write(self, sql: str, params: tuple = ()) -> None:
        if self._tx_depth > 0:
            self._conn.execute(sql, params)
        else:
            with self._conn:
                self._conn.execute(sql, params)

    def _write_many(self, statements: Iterable[tuple[str, tuple]]) -> None:
        if self._tx_depth > 0:
            for sql, params in statements:
                self._conn.execute(sql, params)
        else:
            with self._conn:
                for sql, params in statements:
                    self._conn.execute(sql, params)

    def _executemany(self, sql: str, rows: list[tuple]) -> None:
        if not rows:
            return
        if self._tx_depth > 0:
            self._conn.executemany(sql, rows)
        else:
            with self._conn:
                self._conn.executemany(sql, rows)

    @staticmethod
    def _chunked(items: list, chunk_size: int = 500) -> Iterator[list]:
        for i in range(0, len(items), chunk_size):
            yield items[i : i + chunk_size]

    def get_note_ids_bulk(self, keys: Iterable[str]) -> dict[str, int]:
        key_list = list(keys)
        if not key_list:
            return {}

        out: dict[str, int] = {}
        for chunk in self._chunked(key_list):
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                f"SELECT key, id FROM notes WHERE key IN ({placeholders})",
                tuple(chunk),
            ).fetchall()
            out.update({k: note_id for k, note_id in rows})
        return out

    def get_note_keys_bulk(self, note_ids: Iterable[int]) -> dict[int, str]:
        id_list = list(note_ids)
        if not id_list:
            return {}

        out: dict[int, str] = {}
        for chunk in self._chunked(id_list):
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                f"SELECT id, key FROM notes WHERE id IN ({placeholders})",
                tuple(chunk),
            ).fetchall()
            out.update({note_id: key for note_id, key in rows})
        return out

    def set_notes_bulk(self, mappings: Iterable[tuple[str, int]]) -> None:
        """Set many key->id mappings with the same semantics as set_note()."""
        rows = [(key, note_id) for key, note_id in mappings]
        if not rows:
            return

        # Preserve last-write-wins semantics per id and key.
        seen_keys: set[str] = set()
        seen_ids: set[int] = set()
        ordered_rows: list[tuple[str, int]] = []
        for key, note_id in reversed(rows):
            if key in seen_keys or note_id in seen_ids:
                continue
            seen_keys.add(key)
            seen_ids.add(note_id)
            ordered_rows.append((key, note_id))
        ordered_rows.reverse()

        ids = [note_id for _, note_id in ordered_rows]

        statements: list[tuple[str, tuple]] = []
        for chunk in self._chunked(ids):
            placeholders = ",".join("?" * len(chunk))
            statements.append(
                (f"DELETE FROM notes WHERE id IN ({placeholders})", tuple(chunk))
            )
        self._write_many(statements)
        self._executemany(
            "INSERT OR REPLACE INTO notes (key, id) VALUES (?, ?)", ordered_rows
        )

    def remove_notes_by_keys_bulk(self, keys: Iterable[str]) -> None:
        key_list = list(keys)
        if not key_list:
            return
        statements: list[tuple[str, tuple]] = []
        for chunk in self._chunked(key_list):
            placeholders = ",".join("?" * len(chunk))
            statements.append(
                (f"DELETE FROM notes WHERE key IN ({placeholders})", tuple(chunk))
            )
            statements.append(
                (
                    f"DELETE FROM note_fingerprints WHERE key IN ({placeholders})",
                    tuple(chunk),
                )
            )
        self._write_many(statements)

    def get_note_fingerprints_bulk(
        self, keys: Iterable[str]
    ) -> dict[str, tuple[str, str]]:
        key_list = list(keys)
        if not key_list:
            return {}

        out: dict[str, tuple[str, str]] = {}
        for chunk in self._chunked(key_list):
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                "SELECT key, md_hash, anki_hash "
                f"FROM note_fingerprints WHERE key IN ({placeholders})",
                tuple(chunk),
            ).fetchall()
            out.update({key: (md_hash, anki_hash) for key, md_hash, anki_hash in rows})
        return out

    def set_note_fingerprints_bulk(
        self, rows: Iterable[tuple[str, str, str]]
    ) -> None:
        entries = list(rows)
        if not entries:
            return

        seen_keys: set[str] = set()
        deduped: list[tuple[str, str, str]] = []
        for key, md_hash, anki_hash in reversed(entries):
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append((key, md_hash, anki_hash))
        deduped.reverse()

        self._executemany(
            "INSERT OR REPLACE INTO note_fingerprints "
            "(key, md_hash, anki_hash) VALUES (?, ?, ?)",
            deduped,
        )

    def remove_note_fingerprints_by_keys_bulk(self, keys: Iterable[str]) -> None:
        key_list = list(keys)
        if not key_list:
            return
        statements: list[tuple[str, tuple]] = []
        for chunk in self._chunked(key_list):
            placeholders = ",".join("?" * len(chunk))
            statements.append(
                (
                    f"DELETE FROM note_fingerprints WHERE key IN ({placeholders})",
                    tuple(chunk),
                )
            )
        self._write_many(statements)

    def get_config(self, key: str) -> str | None:
        cursor = self._conn.execute("SELECT value FROM config WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None

    def set_config(self, key: str, value: str) -> None:
        self._write(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
            (key, value),
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
        self.set_notes_bulk([(key, note_id)])

    def remove_note_by_key(self, key: str) -> None:
        self.remove_notes_by_keys_bulk([key])

    def remove_note_by_id(self, note_id: int) -> None:
        self._write("DELETE FROM notes WHERE id = ?", (note_id,))

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
        self._write_many(
            [
                ("DELETE FROM decks WHERE id = ?", (deck_id,)),
                (
                    "INSERT OR REPLACE INTO decks (name, id) VALUES (?, ?)",
                    (name, deck_id),
                ),
            ]
        )

    def remove_deck(self, name: str) -> None:
        self._write("DELETE FROM decks WHERE name = ?", (name,))

    def generate_key(self) -> str:
        return secrets.token_hex(6)
