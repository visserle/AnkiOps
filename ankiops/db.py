"""SQLite DB adapter for AnkiOps state and cache metadata."""

import json
import logging
import secrets
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from ankiops.config import ANKIOPS_DB

logger = logging.getLogger(__name__)

_SCHEMA_TABLE_SQL: dict[str, str] = {
    "note_state": """
CREATE TABLE note_state (
    note_key TEXT PRIMARY KEY,
    note_id INTEGER NOT NULL UNIQUE,
    import_md_hash TEXT,
    import_anki_hash TEXT,
    export_md_hash TEXT,
    export_anki_hash TEXT,
    CHECK (
        (import_md_hash IS NULL) = (import_anki_hash IS NULL)
    ),
    CHECK (
        (export_md_hash IS NULL) = (export_anki_hash IS NULL)
    )
)
""",
    "deck_map": """
CREATE TABLE deck_map (
    deck_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
)
""",
    "app_state": """
CREATE TABLE app_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    profile_name TEXT CHECK (profile_name IS NULL OR profile_name <> ''),
    note_type_sync_hash TEXT CHECK (
        note_type_sync_hash IS NULL OR note_type_sync_hash <> ''
    ),
    note_type_names_signature TEXT CHECK (
        note_type_names_signature IS NULL OR note_type_names_signature <> ''
    ),
    CHECK (
        (note_type_sync_hash IS NULL) = (note_type_names_signature IS NULL)
    )
)
""",
    "markdown_media_cache": """
CREATE TABLE markdown_media_cache (
    md_path TEXT PRIMARY KEY,
    md_mtime_ns INTEGER NOT NULL CHECK (md_mtime_ns >= 0),
    md_size INTEGER NOT NULL CHECK (md_size >= 0),
    media_names_json TEXT NOT NULL
)
""",
    "media_state": """
CREATE TABLE media_state (
    name TEXT PRIMARY KEY,
    mtime_ns INTEGER NOT NULL CHECK (mtime_ns >= 0),
    size INTEGER NOT NULL CHECK (size >= 0),
    digest TEXT NOT NULL CHECK (digest <> ''),
    hashed_name TEXT NOT NULL CHECK (hashed_name <> ''),
    pushed_digest TEXT CHECK (pushed_digest IS NULL OR pushed_digest <> '')
)
""",
}


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.split()).lower()


class SQLiteDbAdapter:
    """SQLite adapter for note/deck identity, sync cache, and media cache state."""

    def __init__(self, conn: sqlite3.Connection, db_path: Path):
        self._conn = conn
        self._db_path = db_path
        self._tx_depth = 0

    @classmethod
    def open(cls, collection_dir: Path) -> "SQLiteDbAdapter":
        db_path = collection_dir / ANKIOPS_DB
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = None
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA busy_timeout = 5000")

            with conn:
                existing_rows = conn.execute(
                    "SELECT name, sql "
                    "FROM sqlite_master "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                    "ORDER BY name"
                ).fetchall()
                existing_schema = {name: sql for name, sql in existing_rows}

                if not existing_schema:
                    for table_sql in _SCHEMA_TABLE_SQL.values():
                        conn.execute(table_sql)
                else:
                    expected_table_names = set(_SCHEMA_TABLE_SQL)
                    existing_table_names = set(existing_schema)
                    if existing_table_names != expected_table_names:
                        raise sqlite3.DatabaseError(
                            "Unexpected schema tables. "
                            f"Expected {sorted(expected_table_names)}, "
                            f"found {sorted(existing_table_names)}"
                        )

                    for table_name, expected_sql in _SCHEMA_TABLE_SQL.items():
                        actual_sql = existing_schema.get(table_name)
                        if actual_sql is None:
                            raise sqlite3.DatabaseError(
                                f"Missing expected table '{table_name}'"
                            )
                        if _normalize_sql(actual_sql) != _normalize_sql(expected_sql):
                            raise sqlite3.DatabaseError(
                                f"Unexpected schema for table '{table_name}'"
                            )

                conn.execute(
                    "INSERT INTO app_state (id) VALUES (1) ON CONFLICT(id) DO NOTHING"
                )

                # Strict schema validation: if any query fails, recover by
                # backing up and recreating the DB.
                conn.execute(
                    "SELECT note_key, note_id, import_md_hash, import_anki_hash, "
                    "export_md_hash, export_anki_hash FROM note_state LIMIT 0"
                )
                conn.execute("SELECT deck_id, name FROM deck_map LIMIT 0")
                conn.execute(
                    "SELECT id, profile_name, note_type_sync_hash, "
                    "note_type_names_signature "
                    "FROM app_state LIMIT 0"
                )
                conn.execute(
                    "SELECT md_path, md_mtime_ns, md_size, media_names_json "
                    "FROM markdown_media_cache LIMIT 0"
                )
                conn.execute(
                    "SELECT name, mtime_ns, size, digest, hashed_name, pushed_digest "
                    "FROM media_state LIMIT 0"
                )

        except (sqlite3.DatabaseError, sqlite3.OperationalError):
            if conn:
                conn.close()
            logger.error(
                f"Database at {db_path} is invalid or corrupt. "
                "Backing up and recreating."
            )
            if db_path.exists():
                try:
                    db_path.rename(str(db_path) + ".corrupt")
                except OSError as error:
                    logger.error(f"Failed to rename corrupt database: {error}")

            return cls.open(collection_dir)

        return cls(conn, db_path)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def write_tx(self, *, immediate: bool = True) -> Iterator[None]:
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

    def _write_singleton(self, sql: str, params: tuple = ()) -> None:
        if self._tx_depth > 0:
            cursor = self._conn.execute(sql, params)
        else:
            with self._conn:
                cursor = self._conn.execute(sql, params)
        if cursor.rowcount != 1:
            raise sqlite3.DatabaseError("Missing singleton app_state row")

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
        for start_index in range(0, len(items), chunk_size):
            yield items[start_index : start_index + chunk_size]

    @staticmethod
    def _encode_media_names(media_names: set[str]) -> str:
        return json.dumps(sorted(media_names), separators=(",", ":"), ensure_ascii=True)

    @staticmethod
    def _decode_media_names(raw: str) -> set[str]:
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return set()
        if not isinstance(decoded, list):
            return set()
        return {value for value in decoded if isinstance(value, str)}

    # -- Note identity ------------------------------------------------------

    def resolve_note_ids(self, note_keys: Iterable[str]) -> dict[str, int]:
        note_key_list = list(note_keys)
        if not note_key_list:
            return {}

        out: dict[str, int] = {}
        for chunk in self._chunked(note_key_list):
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                "SELECT note_key, note_id FROM note_state "
                f"WHERE note_key IN ({placeholders})",
                tuple(chunk),
            ).fetchall()
            out.update({note_key: note_id for note_key, note_id in rows})
        return out

    def resolve_note_keys(self, note_ids: Iterable[int]) -> dict[int, str]:
        id_list = list(note_ids)
        if not id_list:
            return {}

        out: dict[int, str] = {}
        for chunk in self._chunked(id_list):
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                "SELECT note_id, note_key FROM note_state "
                f"WHERE note_id IN ({placeholders})",
                tuple(chunk),
            ).fetchall()
            out.update({note_id: note_key for note_id, note_key in rows})
        return out

    def upsert_note_links(self, mappings: Iterable[tuple[str, int]]) -> None:
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

        with self.write_tx():
            self._executemany(
                "DELETE FROM note_state WHERE note_id = ? AND note_key <> ?",
                [(note_id, note_key) for note_key, note_id in ordered_rows],
            )
            self._executemany(
                "INSERT INTO note_state (note_key, note_id) VALUES (?, ?) "
                "ON CONFLICT(note_key) DO UPDATE SET note_id = excluded.note_id",
                ordered_rows,
            )

    def delete_note_links_by_keys(self, note_keys: Iterable[str]) -> None:
        note_key_list = list(note_keys)
        if not note_key_list:
            return
        statements: list[tuple[str, tuple]] = []
        for chunk in self._chunked(note_key_list):
            placeholders = ",".join("?" * len(chunk))
            statements.append(
                (
                    f"DELETE FROM note_state WHERE note_key IN ({placeholders})",
                    tuple(chunk),
                )
            )
        self._write_many(statements)

    def delete_note_link_by_id(self, note_id: int) -> None:
        self._write("DELETE FROM note_state WHERE note_id = ?", (note_id,))

    # -- Note fingerprints --------------------------------------------------

    def _resolve_directional_hashes(
        self,
        note_keys: Iterable[str],
        *,
        md_column: str,
        anki_column: str,
    ) -> dict[str, tuple[str, str]]:
        note_key_list = list(note_keys)
        if not note_key_list:
            return {}

        out: dict[str, tuple[str, str]] = {}
        for chunk in self._chunked(note_key_list):
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                f"SELECT note_key, {md_column}, {anki_column} FROM note_state "
                f"WHERE note_key IN ({placeholders}) "
                f"AND {md_column} IS NOT NULL AND {anki_column} IS NOT NULL",
                tuple(chunk),
            ).fetchall()
            out.update(
                {
                    note_key: (md_hash, anki_hash)
                    for note_key, md_hash, anki_hash in rows
                }
            )
        return out

    def _upsert_directional_hashes(
        self,
        rows: Iterable[tuple[str, str, str]],
        *,
        md_column: str,
        anki_column: str,
    ) -> None:
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

        note_keys = [note_key for note_key, _, _ in deduped]

        with self.write_tx():
            known_note_keys: set[str] = set()
            for chunk in self._chunked(note_keys):
                placeholders = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    "SELECT note_key FROM note_state "
                    f"WHERE note_key IN ({placeholders})",
                    tuple(chunk),
                ).fetchall()
                known_note_keys.update(note_key for (note_key,) in rows)

            missing = sorted(set(note_keys) - known_note_keys)
            if missing:
                raise sqlite3.IntegrityError(
                    f"Unknown note_key(s) for hash update: {', '.join(missing)}"
                )

            self._executemany(
                f"UPDATE note_state SET {md_column} = ?, {anki_column} = ? "
                "WHERE note_key = ?",
                [
                    (md_hash, anki_hash, note_key)
                    for note_key, md_hash, anki_hash in deduped
                ],
            )

    def _clear_directional_hashes(
        self,
        note_keys: Iterable[str],
        *,
        md_column: str,
        anki_column: str,
    ) -> None:
        note_key_list = list(note_keys)
        if not note_key_list:
            return
        statements: list[tuple[str, tuple]] = []
        for chunk in self._chunked(note_key_list):
            placeholders = ",".join("?" * len(chunk))
            statements.append(
                (
                    f"UPDATE note_state SET {md_column} = NULL, "
                    f"{anki_column} = NULL "
                    f"WHERE note_key IN ({placeholders})",
                    tuple(chunk),
                )
            )
        self._write_many(statements)

    def resolve_import_hashes(
        self, note_keys: Iterable[str]
    ) -> dict[str, tuple[str, str]]:
        return self._resolve_directional_hashes(
            note_keys,
            md_column="import_md_hash",
            anki_column="import_anki_hash",
        )

    def resolve_export_hashes(
        self, note_keys: Iterable[str]
    ) -> dict[str, tuple[str, str]]:
        return self._resolve_directional_hashes(
            note_keys,
            md_column="export_md_hash",
            anki_column="export_anki_hash",
        )

    def upsert_import_hashes(self, rows: Iterable[tuple[str, str, str]]) -> None:
        self._upsert_directional_hashes(
            rows,
            md_column="import_md_hash",
            anki_column="import_anki_hash",
        )

    def upsert_export_hashes(self, rows: Iterable[tuple[str, str, str]]) -> None:
        self._upsert_directional_hashes(
            rows,
            md_column="export_md_hash",
            anki_column="export_anki_hash",
        )

    def clear_import_hashes(self, note_keys: Iterable[str]) -> None:
        self._clear_directional_hashes(
            note_keys,
            md_column="import_md_hash",
            anki_column="import_anki_hash",
        )

    def clear_export_hashes(self, note_keys: Iterable[str]) -> None:
        self._clear_directional_hashes(
            note_keys,
            md_column="export_md_hash",
            anki_column="export_anki_hash",
        )

    # -- Deck mapping -------------------------------------------------------

    def resolve_deck_id(self, name: str) -> int | None:
        cursor = self._conn.execute(
            "SELECT deck_id FROM deck_map WHERE name = ?", (name,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def resolve_deck_name(self, deck_id: int) -> str | None:
        cursor = self._conn.execute(
            "SELECT name FROM deck_map WHERE deck_id = ?", (deck_id,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def upsert_deck(self, name: str, deck_id: int) -> None:
        with self.write_tx():
            self._write(
                "DELETE FROM deck_map WHERE name = ? OR deck_id = ?",
                (name, deck_id),
            )
            self._write(
                "INSERT INTO deck_map (deck_id, name) VALUES (?, ?)",
                (deck_id, name),
            )

    def delete_deck(self, name: str) -> None:
        self._write("DELETE FROM deck_map WHERE name = ?", (name,))

    # -- Markdown media cache ----------------------------------------------

    def resolve_markdown_media_cache(
        self, md_paths: Iterable[str]
    ) -> dict[str, tuple[int, int, set[str]]]:
        path_list = list(md_paths)
        if not path_list:
            return {}

        out: dict[str, tuple[int, int, set[str]]] = {}
        for chunk in self._chunked(path_list):
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                "SELECT md_path, md_mtime_ns, md_size, media_names_json "
                f"FROM markdown_media_cache WHERE md_path IN ({placeholders})",
                tuple(chunk),
            ).fetchall()
            out.update(
                {
                    md_path: (md_mtime_ns, md_size, self._decode_media_names(raw_names))
                    for md_path, md_mtime_ns, md_size, raw_names in rows
                }
            )
        return out

    def upsert_markdown_media_cache(
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

        self._executemany(
            "INSERT INTO markdown_media_cache "
            "(md_path, md_mtime_ns, md_size, media_names_json) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(md_path) DO UPDATE SET "
            "md_mtime_ns = excluded.md_mtime_ns, "
            "md_size = excluded.md_size, "
            "media_names_json = excluded.media_names_json",
            [
                (
                    md_path,
                    md_mtime_ns,
                    md_size,
                    self._encode_media_names(media_names),
                )
                for md_path, md_mtime_ns, md_size, media_names in deduped
            ],
        )

    def delete_markdown_media_cache(self, md_paths: Iterable[str]) -> None:
        path_list = list(md_paths)
        if not path_list:
            return
        statements: list[tuple[str, tuple]] = []
        for chunk in self._chunked(path_list):
            placeholders = ",".join("?" * len(chunk))
            statements.append(
                (
                    f"DELETE FROM markdown_media_cache WHERE md_path IN ({placeholders})",
                    tuple(chunk),
                )
            )
        self._write_many(statements)

    def list_markdown_media_paths(self) -> set[str]:
        rows = self._conn.execute("SELECT md_path FROM markdown_media_cache").fetchall()
        return {md_path for (md_path,) in rows}

    def prune_markdown_media_cache(self, valid_md_paths: Iterable[str]) -> int:
        valid = set(valid_md_paths)
        stale = sorted(self.list_markdown_media_paths() - valid)
        if not stale:
            return 0
        self.delete_markdown_media_cache(stale)
        return len(stale)

    # -- Media state --------------------------------------------------------

    def resolve_media_fingerprints(
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
                f"FROM media_state WHERE name IN ({placeholders})",
                tuple(chunk),
            ).fetchall()
            out.update(
                {
                    name: (mtime_ns, size, digest, hashed_name)
                    for name, mtime_ns, size, digest, hashed_name in rows
                }
            )
        return out

    def upsert_media_fingerprints(
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
            "INSERT INTO media_state "
            "(name, mtime_ns, size, digest, hashed_name, pushed_digest) "
            "VALUES (?, ?, ?, ?, ?, NULL) "
            "ON CONFLICT(name) DO UPDATE SET "
            "mtime_ns = excluded.mtime_ns, "
            "size = excluded.size, "
            "digest = excluded.digest, "
            "hashed_name = excluded.hashed_name",
            deduped,
        )

    def delete_media_state(self, names: Iterable[str]) -> None:
        name_list = list(names)
        if not name_list:
            return
        statements: list[tuple[str, tuple]] = []
        for chunk in self._chunked(name_list):
            placeholders = ",".join("?" * len(chunk))
            statements.append(
                (
                    f"DELETE FROM media_state WHERE name IN ({placeholders})",
                    tuple(chunk),
                )
            )
        self._write_many(statements)

    def resolve_media_push_digests(self, names: Iterable[str]) -> dict[str, str]:
        name_list = list(names)
        if not name_list:
            return {}

        out: dict[str, str] = {}
        for chunk in self._chunked(name_list):
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                "SELECT name, pushed_digest "
                f"FROM media_state WHERE name IN ({placeholders}) "
                "AND pushed_digest IS NOT NULL",
                tuple(chunk),
            ).fetchall()
            out.update({name: pushed_digest for name, pushed_digest in rows})
        return out

    def upsert_media_push_digests(self, rows: Iterable[tuple[str, str]]) -> None:
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

        with self.write_tx():
            for name, digest in deduped:
                cursor = self._conn.execute(
                    "UPDATE media_state SET pushed_digest = ? WHERE name = ?",
                    (digest, name),
                )
                if cursor.rowcount == 0:
                    raise sqlite3.IntegrityError(
                        f"Cannot set push digest for unknown media '{name}'"
                    )

    def clear_media_push_digests(self, names: Iterable[str]) -> None:
        name_list = list(names)
        if not name_list:
            return
        statements: list[tuple[str, tuple]] = []
        for chunk in self._chunked(name_list):
            placeholders = ",".join("?" * len(chunk))
            statements.append(
                (
                    f"UPDATE media_state SET pushed_digest = NULL WHERE name IN ({placeholders})",
                    tuple(chunk),
                )
            )
        self._write_many(statements)

    # -- Singleton metadata -------------------------------------------------

    def get_profile_name(self) -> str | None:
        cursor = self._conn.execute("SELECT profile_name FROM app_state WHERE id = 1")
        row = cursor.fetchone()
        if row is None:
            raise sqlite3.DatabaseError("Missing singleton app_state row")
        return row[0]

    def set_profile_name(self, profile_name: str) -> None:
        self._write_singleton(
            "UPDATE app_state SET profile_name = ? WHERE id = 1",
            (profile_name,),
        )

    def get_note_type_sync_state(self) -> tuple[str, str] | None:
        cursor = self._conn.execute(
            "SELECT note_type_sync_hash, note_type_names_signature "
            "FROM app_state WHERE id = 1"
        )
        row = cursor.fetchone()
        if row is None:
            raise sqlite3.DatabaseError("Missing singleton app_state row")
        if row[0] is None:
            return None
        if row[1] is None:
            raise sqlite3.DatabaseError(
                "Invalid singleton app_state row: note type sync state is partial"
            )
        return row[0], row[1]

    def set_note_type_sync_state(self, sync_hash: str, names_signature: str) -> None:
        self._write_singleton(
            "UPDATE app_state "
            "SET note_type_sync_hash = ?, note_type_names_signature = ? "
            "WHERE id = 1",
            (sync_hash, names_signature),
        )

    def generate_note_key(self) -> str:
        return secrets.token_hex(6)
