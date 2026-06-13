"""SQLite job history for LLM runs."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, TypeVar

from ankiops.collection import LLM_DB_FILENAME, LLM_DIR

from .model_registry import ModelSpec, parse_model
from .types import LlmItemStatus, LlmJobStatus, TaskRunSummary

_DEFAULT_JOB_LIST_LIMIT = 20
_JOB_TABLE = "llm_job"
_ITEM_TABLE = "llm_job_item"
_REQUEST_TABLE = "llm_request"
_REQUEST_ITEM_TABLE = "llm_request_item"
EnumT = TypeVar("EnumT", bound=Enum)


@dataclass(frozen=True)
class LlmJobListItem:
    job_id: int
    task_name: str
    model: str
    status: LlmJobStatus
    persisted: bool
    created_at: str
    finished_at: str | None


@dataclass(frozen=True)
class LlmJobItemDetail:
    ordinal: int
    note_key: str | None
    source: str
    deck_name: str
    note_type: str | None
    item_status: LlmItemStatus
    error_message: str | None
    changed_fields: list[str]
    request_count: int


@dataclass(frozen=True)
class LlmJobRequestNoteRef:
    ordinal: int
    note_key: str | None


@dataclass(frozen=True)
class LlmJobRequestDetail:
    request_id: int
    outcome: str
    error_message: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: int
    created_at: str
    notes: tuple[LlmJobRequestNoteRef, ...]


@dataclass(frozen=True)
class LlmJobDetail:
    job_id: int
    task_name: str
    model: str
    model_id: str
    status: LlmJobStatus
    persisted: bool
    created_at: str
    started_at: str
    finished_at: str | None
    fatal_error: str | None
    summary: TaskRunSummary
    items: list[LlmJobItemDetail]
    requests: list[LlmJobRequestDetail]


@dataclass(frozen=True)
class JobAggregate:
    summary: TaskRunSummary
    persisted: bool
    failed: bool
    status: LlmJobStatus


class LlmDb:
    """Tiny SQLite wrapper for LLM job history."""

    def __init__(self, conn: sqlite3.Connection, db_path: Path) -> None:
        self._conn = conn
        self._db_path = db_path
        self._tx_depth = 0

    @classmethod
    def open(cls, collection_dir: Path) -> "LlmDb":
        llm_dir = collection_dir / LLM_DIR
        llm_dir.mkdir(parents=True, exist_ok=True)
        db_path = llm_dir / LLM_DB_FILENAME
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        db = cls(conn, db_path)
        db._create_schema()
        return db

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def write_tx(self) -> Iterator[None]:
        is_outer = self._tx_depth == 0
        if is_outer:
            self._conn.execute("BEGIN IMMEDIATE")
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

    def start_job(self, *, task_name: str, model: str, model_id: str) -> int:
        now = _utc_now()
        cursor = self._write(
            f"""
            INSERT INTO {_JOB_TABLE} (
                task_name, model, model_id, status, persisted, fatal_error,
                created_at, started_at, finished_at,
                decks_seen, decks_matched, notes_seen
            ) VALUES (?, ?, ?, ?, 0, NULL, ?, ?, NULL, 0, 0, 0)
            """,
            (task_name, model, model_id, LlmJobStatus.RUNNING.value, now, now),
        )
        return _lastrowid(cursor, _JOB_TABLE)

    def set_discovery_counts(
        self,
        *,
        job_id: int,
        decks_seen: int,
        decks_matched: int,
        notes_seen: int,
    ) -> None:
        self._write(
            f"""
            UPDATE {_JOB_TABLE}
            SET decks_seen = ?, decks_matched = ?, notes_seen = ?
            WHERE id = ?
            """,
            (decks_seen, decks_matched, notes_seen, job_id),
        )

    def insert_job_item(
        self,
        *,
        job_id: int,
        ordinal: int,
        source: str,
        deck_name: str,
        note_key: str | None,
        note_type: str | None,
        item_status: LlmItemStatus,
        skip_reason: str | None,
        error_message: str | None = None,
        changed_fields: list[str] | None = None,
        applied_to_markdown: bool = False,
    ) -> int:
        cursor = self._write(
            f"""
            INSERT INTO {_ITEM_TABLE} (
                job_id, ordinal, note_key, source, deck_name, note_type,
                item_status, skip_reason, error_message,
                changed_fields_json, applied_to_markdown
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                ordinal,
                note_key,
                source,
                deck_name,
                note_type,
                item_status.value,
                skip_reason,
                error_message,
                _as_json(changed_fields or []),
                1 if applied_to_markdown else 0,
            ),
        )
        return _lastrowid(cursor, _ITEM_TABLE)

    def update_job_item_status(
        self,
        *,
        item_id: int,
        item_status: LlmItemStatus,
        error_message: str | None = None,
        changed_fields: list[str] | None = None,
    ) -> None:
        self._write(
            f"""
            UPDATE {_ITEM_TABLE}
            SET item_status = ?, error_message = ?, changed_fields_json = ?
            WHERE id = ?
            """,
            (item_status.value, error_message, _as_json(changed_fields or []), item_id),
        )

    def insert_request(
        self,
        *,
        job_id: int,
        item_ids: list[int],
        outcome: str,
        request_json: dict[str, Any],
        parsed_response_json: dict[str, Any] | None,
        response_json: str | None,
        error_message: str | None,
        latency_ms: int,
        input_tokens: int,
        output_tokens: int,
    ) -> int:
        if self._tx_depth == 0:
            with self.write_tx():
                return self.insert_request(
                    job_id=job_id,
                    item_ids=item_ids,
                    outcome=outcome,
                    request_json=request_json,
                    parsed_response_json=parsed_response_json,
                    response_json=response_json,
                    error_message=error_message,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
        if not item_ids:
            raise ValueError("LLM request must be linked to at least one job item")
        now = _utc_now()
        cursor = self._write(
            f"""
            INSERT INTO {_REQUEST_TABLE} (
                job_id, outcome, request_json,
                parsed_response_json, response_json, error_message,
                latency_ms, input_tokens, output_tokens, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                outcome,
                _as_json(request_json),
                _as_json(parsed_response_json) if parsed_response_json else None,
                response_json,
                error_message,
                latency_ms,
                input_tokens,
                output_tokens,
                now,
            ),
        )
        request_id = _lastrowid(cursor, _REQUEST_TABLE)
        self._conn.executemany(
            f"""
            INSERT INTO {_REQUEST_ITEM_TABLE} (request_id, job_item_id)
            VALUES (?, ?)
            """,
            [(request_id, item_id) for item_id in item_ids],
        )
        return request_id

    def mark_unfinished_items_canceled(self, *, job_id: int) -> int:
        cursor = self._write(
            f"""
            UPDATE {_ITEM_TABLE}
            SET item_status = ?
            WHERE job_id = ? AND item_status = ?
            """,
            (LlmItemStatus.CANCELED.value, job_id, LlmItemStatus.QUEUED.value),
        )
        return int(cursor.rowcount)

    def set_applied_for_updated_items(self, *, job_id: int) -> None:
        self._write(
            f"""
            UPDATE {_ITEM_TABLE}
            SET applied_to_markdown = 1
            WHERE job_id = ? AND item_status = ?
            """,
            (job_id, LlmItemStatus.SUCCEEDED_UPDATED.value),
        )

    def finalize_job(
        self,
        *,
        job_id: int,
        status: LlmJobStatus,
        persisted: bool,
        fatal_error: str | None = None,
    ) -> None:
        self._write(
            f"""
            UPDATE {_JOB_TABLE}
            SET status = ?, persisted = ?, fatal_error = ?, finished_at = ?
            WHERE id = ?
            """,
            (status.value, 1 if persisted else 0, fatal_error, _utc_now(), job_id),
        )

    def aggregate_job(self, job_id: int) -> JobAggregate:
        row = self._conn.execute(
            f"""
            SELECT task_name, model, model_id, decks_seen, decks_matched, notes_seen,
                   persisted, status
            FROM {_JOB_TABLE}
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown LLM job id {job_id}")

        model = _resolve_model_for_summary(
            row, collection_dir=self._db_path.parent.parent
        )
        summary = TaskRunSummary(
            task_name=str(row["task_name"]),
            model=model,
            decks_seen=int(row["decks_seen"]),
            decks_matched=int(row["decks_matched"]),
            notes_seen=int(row["notes_seen"]),
        )

        item_counts = self._conn.execute(
            f"""
            SELECT
                SUM(CASE WHEN item_status NOT IN (?, ?) THEN 1 ELSE 0 END) eligible,
                SUM(CASE WHEN item_status = ? THEN 1 ELSE 0 END) updated,
                SUM(CASE WHEN item_status = ? THEN 1 ELSE 0 END) unchanged,
                SUM(CASE WHEN item_status = ? THEN 1 ELSE 0 END) skipped,
                SUM(CASE WHEN item_status IN (?, ?, ?, ?) THEN 1 ELSE 0 END) errors,
                SUM(CASE WHEN item_status = ? THEN 1 ELSE 0 END) canceled
            FROM {_ITEM_TABLE}
            WHERE job_id = ?
            """,
            (
                LlmItemStatus.SKIPPED_NO_EDITABLE_FIELDS.value,
                LlmItemStatus.INVALID_NOTE.value,
                LlmItemStatus.SUCCEEDED_UPDATED.value,
                LlmItemStatus.SUCCEEDED_UNCHANGED.value,
                LlmItemStatus.SKIPPED_NO_EDITABLE_FIELDS.value,
                LlmItemStatus.INVALID_NOTE.value,
                LlmItemStatus.NOTE_ERROR.value,
                LlmItemStatus.PROVIDER_ERROR.value,
                LlmItemStatus.FATAL_ERROR.value,
                LlmItemStatus.CANCELED.value,
                job_id,
            ),
        ).fetchone()
        assert item_counts is not None

        requests = self._conn.execute(
            f"""
            SELECT COUNT(*) requests,
                   COALESCE(SUM(input_tokens), 0) input_tokens,
                   COALESCE(SUM(output_tokens), 0) output_tokens,
                   COALESCE(SUM(latency_ms), 0) latency_ms
            FROM {_REQUEST_TABLE}
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        assert requests is not None

        summary.eligible = int(item_counts["eligible"] or 0)
        summary.updated = int(item_counts["updated"] or 0)
        summary.unchanged = int(item_counts["unchanged"] or 0)
        summary.skipped_no_editable_fields = int(item_counts["skipped"] or 0)
        summary.errors = int(item_counts["errors"] or 0)
        summary.canceled = int(item_counts["canceled"] or 0)
        summary.requests = int(requests["requests"] or 0)
        summary.input_tokens = int(requests["input_tokens"] or 0)
        summary.output_tokens = int(requests["output_tokens"] or 0)
        summary.provider_latency_ms_total = int(requests["latency_ms"] or 0)

        status = _parse_enum(LlmJobStatus, row["status"])
        persisted = bool(row["persisted"])
        return JobAggregate(
            summary=summary,
            persisted=persisted,
            failed=summary.errors > 0 or status is LlmJobStatus.FAILED,
            status=status,
        )

    def resolve_job_id(self, identifier: str) -> int | None:
        token = identifier.strip()
        if not token:
            return None
        if token.lower() == "latest":
            row = self._conn.execute(
                f"SELECT id FROM {_JOB_TABLE} ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            return int(row["id"]) if row is not None else None
        if not token.isdigit():
            raise ValueError("Job ID must be numeric or use 'latest'.")
        row = self._conn.execute(
            f"SELECT id FROM {_JOB_TABLE} WHERE id = ?",
            (int(token),),
        ).fetchone()
        return int(row["id"]) if row is not None else None

    def list_jobs(self) -> list[LlmJobListItem]:
        rows = self._conn.execute(
            f"""
            SELECT id, task_name, model, status, persisted, created_at, finished_at
            FROM {_JOB_TABLE}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (_DEFAULT_JOB_LIST_LIMIT,),
        ).fetchall()
        return [
            LlmJobListItem(
                job_id=int(row["id"]),
                task_name=str(row["task_name"]),
                model=str(row["model"]),
                status=_parse_enum(LlmJobStatus, row["status"]),
                persisted=bool(row["persisted"]),
                created_at=str(row["created_at"]),
                finished_at=(str(row["finished_at"]) if row["finished_at"] else None),
            )
            for row in rows
        ]

    def get_job_detail(self, job_id: int) -> LlmJobDetail | None:
        row = self._conn.execute(
            f"""
            SELECT id, task_name, model, model_id, status, persisted, created_at,
                   started_at, finished_at, fatal_error
            FROM {_JOB_TABLE}
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            return None

        aggregate = self.aggregate_job(job_id)
        item_rows = self._conn.execute(
            f"""
            SELECT i.ordinal, i.note_key, i.source, i.deck_name, i.note_type,
                   i.item_status, i.error_message, i.changed_fields_json,
                   COUNT(ri.request_id) request_count
            FROM {_ITEM_TABLE} i
            LEFT JOIN {_REQUEST_ITEM_TABLE} ri ON ri.job_item_id = i.id
            WHERE i.job_id = ?
            GROUP BY i.id
            ORDER BY i.ordinal ASC
            """,
            (job_id,),
        ).fetchall()

        items = [
            LlmJobItemDetail(
                ordinal=int(item["ordinal"]),
                note_key=(str(item["note_key"]) if item["note_key"] else None),
                source=str(item["source"]),
                deck_name=str(item["deck_name"]),
                note_type=(str(item["note_type"]) if item["note_type"] else None),
                item_status=_parse_enum(LlmItemStatus, item["item_status"]),
                error_message=(
                    str(item["error_message"]) if item["error_message"] else None
                ),
                changed_fields=_parse_json_list(item["changed_fields_json"]),
                request_count=int(item["request_count"] or 0),
            )
            for item in item_rows
        ]

        request_rows = self._conn.execute(
            f"""
            SELECT r.id request_id, r.outcome, r.error_message, r.input_tokens,
                   r.output_tokens, r.latency_ms, r.created_at,
                   i.ordinal, i.note_key
            FROM {_REQUEST_TABLE} r
            LEFT JOIN {_REQUEST_ITEM_TABLE} ri ON ri.request_id = r.id
            LEFT JOIN {_ITEM_TABLE} i ON i.id = ri.job_item_id
            WHERE r.job_id = ?
            ORDER BY r.id ASC, i.ordinal ASC
            """,
            (job_id,),
        ).fetchall()
        request_metadata: dict[int, sqlite3.Row] = {}
        request_notes: dict[int, list[LlmJobRequestNoteRef]] = {}
        for request in request_rows:
            request_id = int(request["request_id"])
            if request_id not in request_metadata:
                request_metadata[request_id] = request
                request_notes[request_id] = []
            if request["ordinal"] is not None:
                request_notes[request_id].append(
                    LlmJobRequestNoteRef(
                        ordinal=int(request["ordinal"]),
                        note_key=(
                            str(request["note_key"]) if request["note_key"] else None
                        ),
                    )
                )

        requests = [
            LlmJobRequestDetail(
                request_id=request_id,
                outcome=str(request["outcome"]),
                error_message=(
                    str(request["error_message"]) if request["error_message"] else None
                ),
                input_tokens=int(request["input_tokens"] or 0),
                output_tokens=int(request["output_tokens"] or 0),
                latency_ms=int(request["latency_ms"] or 0),
                created_at=str(request["created_at"]),
                notes=tuple(request_notes[request_id]),
            )
            for request_id, request in request_metadata.items()
        ]

        return LlmJobDetail(
            job_id=job_id,
            task_name=str(row["task_name"]),
            model=str(row["model"]),
            model_id=str(row["model_id"]),
            status=_parse_enum(LlmJobStatus, row["status"]),
            persisted=bool(row["persisted"]),
            created_at=str(row["created_at"]),
            started_at=str(row["started_at"]),
            finished_at=(str(row["finished_at"]) if row["finished_at"] else None),
            fatal_error=(str(row["fatal_error"]) if row["fatal_error"] else None),
            summary=aggregate.summary,
            items=items,
            requests=requests,
        )

    def _create_schema(self) -> None:
        item_status_check = _enum_check_sql([status.value for status in LlmItemStatus])
        job_status_check = _enum_check_sql([status.value for status in LlmJobStatus])
        with self._conn:
            self._conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS {_JOB_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_name TEXT NOT NULL,
                    model TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN {job_status_check}),
                    persisted INTEGER NOT NULL DEFAULT 0 CHECK (persisted IN (0, 1)),
                    fatal_error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    decks_seen INTEGER NOT NULL DEFAULT 0,
                    decks_matched INTEGER NOT NULL DEFAULT 0,
                    notes_seen INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS {_ITEM_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER NOT NULL,
                    ordinal INTEGER NOT NULL,
                    note_key TEXT,
                    source TEXT NOT NULL,
                    deck_name TEXT NOT NULL,
                    note_type TEXT,
                    item_status TEXT NOT NULL
                        CHECK (item_status IN {item_status_check}),
                    skip_reason TEXT,
                    error_message TEXT,
                    changed_fields_json TEXT NOT NULL,
                    applied_to_markdown INTEGER NOT NULL DEFAULT 0
                        CHECK (applied_to_markdown IN (0, 1)),
                    UNIQUE(job_id, ordinal),
                    FOREIGN KEY(job_id) REFERENCES {_JOB_TABLE}(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS {_REQUEST_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER NOT NULL,
                    outcome TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    parsed_response_json TEXT,
                    response_json TEXT,
                    error_message TEXT,
                    latency_ms INTEGER NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES {_JOB_TABLE}(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS {_REQUEST_ITEM_TABLE} (
                    request_id INTEGER NOT NULL,
                    job_item_id INTEGER NOT NULL,
                    PRIMARY KEY(request_id, job_item_id),
                    FOREIGN KEY(request_id)
                        REFERENCES {_REQUEST_TABLE}(id) ON DELETE CASCADE,
                    FOREIGN KEY(job_item_id)
                        REFERENCES {_ITEM_TABLE}(id) ON DELETE CASCADE
                );
                """
            )
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_llm_job_created "
                f"ON {_JOB_TABLE}(created_at DESC)"
            )
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_llm_job_item_job "
                f"ON {_ITEM_TABLE}(job_id)"
            )
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_llm_request_job "
                f"ON {_REQUEST_TABLE}(job_id)"
            )
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_llm_request_item_item "
                f"ON {_REQUEST_ITEM_TABLE}(job_item_id)"
            )
        self._assert_schema()

    def _assert_schema(self) -> None:
        required_item_columns = {
            "id",
            "job_id",
            "ordinal",
            "note_key",
            "source",
            "deck_name",
            "note_type",
            "item_status",
            "skip_reason",
            "error_message",
            "changed_fields_json",
            "applied_to_markdown",
        }
        required_request_columns = {
            "id",
            "job_id",
            "outcome",
            "request_json",
            "parsed_response_json",
            "response_json",
            "error_message",
            "latency_ms",
            "input_tokens",
            "output_tokens",
            "created_at",
        }
        request_columns = {
            str(row["name"])
            for row in self._conn.execute(f"PRAGMA table_info({_REQUEST_TABLE})")
        }
        item_columns = {
            str(row["name"])
            for row in self._conn.execute(f"PRAGMA table_info({_ITEM_TABLE})")
        }
        required_request_item_columns = {"request_id", "job_item_id"}
        request_item_columns = {
            str(row["name"])
            for row in self._conn.execute(f"PRAGMA table_info({_REQUEST_ITEM_TABLE})")
        }
        if (
            required_item_columns - item_columns
            or required_request_columns - request_columns
            or required_request_item_columns - request_item_columns
        ):
            raise RuntimeError(
                "LLM DB schema is from an older AnkiOps build. "
                f"Delete '{self._db_path}' to reinitialize it."
            )

    def _write(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        if self._tx_depth > 0:
            return self._conn.execute(sql, params)
        with self._conn:
            return self._conn.execute(sql, params)


def _resolve_model_for_summary(row: sqlite3.Row, *, collection_dir: Path) -> ModelSpec:
    try:
        model = parse_model(str(row["model"]), collection_dir=collection_dir)
    except Exception:
        model = None
    if model is not None:
        return model
    return ModelSpec(
        model=str(row["model"]),
        model_id=str(row["model_id"]),
        base_url="",
        api_key="",
    )


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _as_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _parse_json_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(decoded, list):
        return []
    return [item for item in decoded if isinstance(item, str)]


def _parse_enum(enum_type: type[EnumT], raw_value: object) -> EnumT:
    if not isinstance(raw_value, str):
        raise ValueError(f"Expected string value for enum {enum_type.__name__}")
    try:
        return enum_type(raw_value)
    except ValueError as error:
        raise ValueError(
            f"Invalid value '{raw_value}' for enum {enum_type.__name__}"
        ) from error


def _enum_check_sql(values: list[str]) -> str:
    quoted = ", ".join(f"'{value}'" for value in values)
    return f"({quoted})"


def _lastrowid(cursor: sqlite3.Cursor, table_name: str) -> int:
    rowid = cursor.lastrowid
    if rowid is None:
        raise RuntimeError(f"Failed to persist {table_name} row")
    return int(rowid)
