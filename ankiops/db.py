"""SQLite database adapter for AnkiOps note and media metadata caching."""

import logging
import secrets
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from ankiops.config import ANKIOPS_DB

logger = logging.getLogger(__name__)


class SQLiteDbAdapter:
    """Bidirectional note_key ↔ note_id mapping using SQLite,
    plus additional tables for caching note and media metadata."""

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
                        note_key TEXT PRIMARY KEY,
                        note_id INTEGER NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS decks (
                        name TEXT PRIMARY KEY,
                        deck_id INTEGER NOT NULL
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
                        note_key TEXT PRIMARY KEY,
                        md_hash TEXT NOT NULL,
                        anki_hash TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS markdown_media_state (
                        md_path TEXT PRIMARY KEY,
                        md_mtime_ns INTEGER NOT NULL,
                        md_size INTEGER NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS markdown_media_refs (
                        md_path TEXT NOT NULL,
                        media_name TEXT NOT NULL,
                        PRIMARY KEY (md_path, media_name)
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS media_fingerprints (
                        name TEXT PRIMARY KEY,
                        mtime_ns INTEGER NOT NULL,
                        size INTEGER NOT NULL,
                        digest TEXT NOT NULL,
                        hashed_name TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS media_push_state (
                        name TEXT PRIMARY KEY,
                        digest TEXT NOT NULL
                    )
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_notes_id ON notes(note_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_decks_id ON decks(deck_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_markdown_media_refs_media "
                    "ON markdown_media_refs(media_name)"
                )

                # Schema validation: ensure note mapping columns exist in 'notes' table.
                # If not, it will raise OperationalError and trigger recovery.
                conn.execute("SELECT note_key, note_id FROM notes LIMIT 0")

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

    def get_note_ids_bulk(self, note_keys: Iterable[str]) -> dict[str, int]:
        note_key_list = list(note_keys)
        if not note_key_list:
            return {}

        out: dict[str, int] = {}
        for chunk in self._chunked(note_key_list):
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                f"SELECT note_key, note_id FROM notes WHERE note_key IN ({placeholders})",
                tuple(chunk),
            ).fetchall()
            out.update({note_key: note_id for note_key, note_id in rows})
        return out

    def get_note_keys_bulk(self, note_ids: Iterable[int]) -> dict[int, str]:
        id_list = list(note_ids)
        if not id_list:
            return {}

        out: dict[int, str] = {}
        for chunk in self._chunked(id_list):
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                f"SELECT note_id, note_key FROM notes WHERE note_id IN ({placeholders})",
                tuple(chunk),
            ).fetchall()
            out.update({note_id: note_key for note_id, note_key in rows})
        return out

    def set_notes_bulk(self, mappings: Iterable[tuple[str, int]]) -> None:
        """Set many note_key->note_id mappings with set_note() semantics."""
        rows = [(note_key, note_id) for note_key, note_id in mappings]
        if not rows:
            return

        # Preserve last-write-wins semantics per note_id and note_key.
        seen_note_keys: set[str] = set()
        seen_ids: set[int] = set()
        ordered_rows: list[tuple[str, int]] = []
        for note_key, note_id in reversed(rows):
            if note_key in seen_note_keys or note_id in seen_ids:
                continue
            seen_note_keys.add(note_key)
            seen_ids.add(note_id)
            ordered_rows.append((note_key, note_id))
        ordered_rows.reverse()

        ids = [note_id for _, note_id in ordered_rows]

        statements: list[tuple[str, tuple]] = []
        for chunk in self._chunked(ids):
            placeholders = ",".join("?" * len(chunk))
            statements.append(
                (f"DELETE FROM notes WHERE note_id IN ({placeholders})", tuple(chunk))
            )
        self._write_many(statements)
        self._executemany(
            "INSERT OR REPLACE INTO notes (note_key, note_id) VALUES (?, ?)",
            ordered_rows,
        )

    def remove_notes_by_note_keys_bulk(self, note_keys: Iterable[str]) -> None:
        note_key_list = list(note_keys)
        if not note_key_list:
            return
        statements: list[tuple[str, tuple]] = []
        for chunk in self._chunked(note_key_list):
            placeholders = ",".join("?" * len(chunk))
            statements.append(
                (
                    f"DELETE FROM notes WHERE note_key IN ({placeholders})",
                    tuple(chunk),
                )
            )
            statements.append(
                (
                    f"DELETE FROM note_fingerprints WHERE note_key IN ({placeholders})",
                    tuple(chunk),
                )
            )
        self._write_many(statements)

    def get_note_fingerprints_bulk(
        self, note_keys: Iterable[str]
    ) -> dict[str, tuple[str, str]]:
        note_key_list = list(note_keys)
        if not note_key_list:
            return {}

        out: dict[str, tuple[str, str]] = {}
        for chunk in self._chunked(note_key_list):
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                "SELECT note_key, md_hash, anki_hash "
                f"FROM note_fingerprints WHERE note_key IN ({placeholders})",
                tuple(chunk),
            ).fetchall()
            out.update(
                {
                    note_key: (md_hash, anki_hash)
                    for note_key, md_hash, anki_hash in rows
                }
            )
        return out

    def set_note_fingerprints_bulk(self, rows: Iterable[tuple[str, str, str]]) -> None:
        entries = list(rows)
        if not entries:
            return

        seen_note_keys: set[str] = set()
        deduped: list[tuple[str, str, str]] = []
        for note_key, md_hash, anki_hash in reversed(entries):
            if note_key in seen_note_keys:
                continue
            seen_note_keys.add(note_key)
            deduped.append((note_key, md_hash, anki_hash))
        deduped.reverse()

        self._executemany(
            "INSERT OR REPLACE INTO note_fingerprints "
            "(note_key, md_hash, anki_hash) VALUES (?, ?, ?)",
            deduped,
        )

    def remove_note_fingerprints_by_note_keys_bulk(
        self, note_keys: Iterable[str]
    ) -> None:
        note_key_list = list(note_keys)
        if not note_key_list:
            return
        statements: list[tuple[str, tuple]] = []
        for chunk in self._chunked(note_key_list):
            placeholders = ",".join("?" * len(chunk))
            statements.append(
                (
                    f"DELETE FROM note_fingerprints WHERE note_key IN ({placeholders})",
                    tuple(chunk),
                )
            )
        self._write_many(statements)

    def get_markdown_media_cache_bulk(
        self, md_paths: Iterable[str]
    ) -> dict[str, tuple[int, int, set[str]]]:
        path_list = list(md_paths)
        if not path_list:
            return {}

        state_by_path: dict[str, tuple[int, int]] = {}
        for chunk in self._chunked(path_list):
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                "SELECT md_path, md_mtime_ns, md_size "
                f"FROM markdown_media_state WHERE md_path IN ({placeholders})",
                tuple(chunk),
            ).fetchall()
            state_by_path.update(
                {
                    md_path: (md_mtime_ns, md_size)
                    for md_path, md_mtime_ns, md_size in rows
                }
            )

        refs_by_path: dict[str, set[str]] = defaultdict(set)
        for chunk in self._chunked(path_list):
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                "SELECT md_path, media_name "
                f"FROM markdown_media_refs WHERE md_path IN ({placeholders})",
                tuple(chunk),
            ).fetchall()
            for md_path, media_name in rows:
                refs_by_path[md_path].add(media_name)

        out: dict[str, tuple[int, int, set[str]]] = {}
        for md_path, (md_mtime_ns, md_size) in state_by_path.items():
            out[md_path] = (md_mtime_ns, md_size, refs_by_path.get(md_path, set()))
        return out

    def set_markdown_media_cache_bulk(
        self, rows: Iterable[tuple[str, int, int, set[str]]]
    ) -> None:
        entries = [
            (md_path, md_mtime_ns, md_size, set(media_names))
            for md_path, md_mtime_ns, md_size, media_names in rows
        ]
        if not entries:
            return

        seen_paths: set[str] = set()
        deduped: list[tuple[str, int, int, set[str]]] = []
        for md_path, md_mtime_ns, md_size, media_names in reversed(entries):
            if md_path in seen_paths:
                continue
            seen_paths.add(md_path)
            deduped.append((md_path, md_mtime_ns, md_size, media_names))
        deduped.reverse()

        state_rows = [
            (md_path, md_mtime_ns, md_size)
            for md_path, md_mtime_ns, md_size, _ in deduped
        ]
        md_paths = [md_path for md_path, _, _, _ in deduped]
        ref_rows = [
            (md_path, media_name)
            for md_path, _, _, media_names in deduped
            for media_name in media_names
        ]

        with self.transaction():
            self._executemany(
                "INSERT OR REPLACE INTO markdown_media_state "
                "(md_path, md_mtime_ns, md_size) VALUES (?, ?, ?)",
                state_rows,
            )
            statements: list[tuple[str, tuple]] = []
            for chunk in self._chunked(md_paths):
                placeholders = ",".join("?" * len(chunk))
                statements.append(
                    (
                        f"DELETE FROM markdown_media_refs WHERE md_path IN ({placeholders})",
                        tuple(chunk),
                    )
                )
            self._write_many(statements)
            self._executemany(
                "INSERT OR REPLACE INTO markdown_media_refs "
                "(md_path, media_name) VALUES (?, ?)",
                ref_rows,
            )

    def remove_markdown_media_cache_by_paths(self, md_paths: Iterable[str]) -> None:
        path_list = list(md_paths)
        if not path_list:
            return
        statements: list[tuple[str, tuple]] = []
        for chunk in self._chunked(path_list):
            placeholders = ",".join("?" * len(chunk))
            statements.append(
                (
                    f"DELETE FROM markdown_media_state WHERE md_path IN ({placeholders})",
                    tuple(chunk),
                )
            )
            statements.append(
                (
                    f"DELETE FROM markdown_media_refs WHERE md_path IN ({placeholders})",
                    tuple(chunk),
                )
            )
        self._write_many(statements)

    def get_media_fingerprints_bulk(
        self, names: Iterable[str]
    ) -> dict[str, tuple[int, int, str, str]]:
        name_list = list(names)
        if not name_list:
            return {}

        out: dict[str, tuple[int, int, str, str]] = {}
        for chunk in self._chunked(name_list):
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                "SELECT name, mtime_ns, size, digest, hashed_name "
                f"FROM media_fingerprints WHERE name IN ({placeholders})",
                tuple(chunk),
            ).fetchall()
            out.update(
                {
                    name: (mtime_ns, size, digest, hashed_name)
                    for name, mtime_ns, size, digest, hashed_name in rows
                }
            )
        return out

    def set_media_fingerprints_bulk(
        self, rows: Iterable[tuple[str, int, int, str, str]]
    ) -> None:
        entries = list(rows)
        if not entries:
            return

        seen_names: set[str] = set()
        deduped: list[tuple[str, int, int, str, str]] = []
        for name, mtime_ns, size, digest, hashed_name in reversed(entries):
            if name in seen_names:
                continue
            seen_names.add(name)
            deduped.append((name, mtime_ns, size, digest, hashed_name))
        deduped.reverse()

        self._executemany(
            "INSERT OR REPLACE INTO media_fingerprints "
            "(name, mtime_ns, size, digest, hashed_name) VALUES (?, ?, ?, ?, ?)",
            deduped,
        )

    def remove_media_fingerprints_by_names(self, names: Iterable[str]) -> None:
        name_list = list(names)
        if not name_list:
            return
        statements: list[tuple[str, tuple]] = []
        for chunk in self._chunked(name_list):
            placeholders = ",".join("?" * len(chunk))
            statements.append(
                (
                    f"DELETE FROM media_fingerprints WHERE name IN ({placeholders})",
                    tuple(chunk),
                )
            )
        self._write_many(statements)

    def get_media_push_state_bulk(self, names: Iterable[str]) -> dict[str, str]:
        name_list = list(names)
        if not name_list:
            return {}

        out: dict[str, str] = {}
        for chunk in self._chunked(name_list):
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                "SELECT name, digest "
                f"FROM media_push_state WHERE name IN ({placeholders})",
                tuple(chunk),
            ).fetchall()
            out.update({name: digest for name, digest in rows})
        return out

    def set_media_push_state_bulk(self, rows: Iterable[tuple[str, str]]) -> None:
        entries = list(rows)
        if not entries:
            return

        seen_names: set[str] = set()
        deduped: list[tuple[str, str]] = []
        for name, digest in reversed(entries):
            if name in seen_names:
                continue
            seen_names.add(name)
            deduped.append((name, digest))
        deduped.reverse()

        self._executemany(
            "INSERT OR REPLACE INTO media_push_state (name, digest) VALUES (?, ?)",
            deduped,
        )

    def remove_media_push_state_by_names(self, names: Iterable[str]) -> None:
        name_list = list(names)
        if not name_list:
            return
        statements: list[tuple[str, tuple]] = []
        for chunk in self._chunked(name_list):
            placeholders = ",".join("?" * len(chunk))
            statements.append(
                (
                    f"DELETE FROM media_push_state WHERE name IN ({placeholders})",
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

    def get_note_id(self, note_key: str) -> int | None:
        cursor = self._conn.execute(
            "SELECT note_id FROM notes WHERE note_key = ?",
            (note_key,),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def get_note_key(self, note_id: int) -> str | None:
        cursor = self._conn.execute(
            "SELECT note_key FROM notes WHERE note_id = ?",
            (note_id,),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def set_note(self, note_key: str, note_id: int) -> None:
        self.set_notes_bulk([(note_key, note_id)])

    def remove_note_by_note_key(self, note_key: str) -> None:
        self.remove_notes_by_note_keys_bulk([note_key])

    def remove_note_by_id(self, note_id: int) -> None:
        self._write("DELETE FROM notes WHERE note_id = ?", (note_id,))

    # -- Deck mapping (deck_name ↔ deck_id) ----------------------------------

    def get_deck_id(self, name: str) -> int | None:
        cursor = self._conn.execute("SELECT deck_id FROM decks WHERE name = ?", (name,))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_deck_name(self, deck_id: int) -> str | None:
        cursor = self._conn.execute(
            "SELECT name FROM decks WHERE deck_id = ?", (deck_id,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def set_deck(self, name: str, deck_id: int) -> None:
        self._write_many(
            [
                ("DELETE FROM decks WHERE deck_id = ?", (deck_id,)),
                (
                    "INSERT OR REPLACE INTO decks (name, deck_id) VALUES (?, ?)",
                    (name, deck_id),
                ),
            ]
        )

    def remove_deck(self, name: str) -> None:
        self._write("DELETE FROM decks WHERE name = ?", (name,))

    def generate_note_key(self) -> str:
        return secrets.token_hex(6)
