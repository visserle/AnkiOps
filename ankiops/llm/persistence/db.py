"""SQLite persistence for runtime execution history."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, TypeVar

from ankiops.config import LLM_DB_FILENAME, LLM_DIR
from ankiops.llm.domain.outcomes import ProviderOutcomeKind
from ankiops.llm.model_registry import parse_model
from ankiops.llm.task_types import LlmItemStatus, LlmJobStatus, TaskRunSummary

_DEFAULT_JOB_LIST_LIMIT = 20
_RAW_PAYLOAD_ENV_VAR = "ANKIOPS_LLM_PERSIST_RAW_PAYLOADS"

_JOB_TABLE = "llm_job"
_ITEM_TABLE = "llm_job_item"
_ATTEMPT_TABLE = "llm_attempt"
_ATTEMPT_PAYLOAD_TABLE = "llm_attempt_payload"

EnumT = TypeVar("EnumT", bound=Enum)


def _enum_check_sql(values: list[str]) -> str:
    quoted = ", ".join(f"'{value}'" for value in values)
    return f"({quoted})"


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
    deck_name: str
    note_type: str | None
    item_status: LlmItemStatus
    error_message: str | None
    changed_fields: list[str]
    attempts: int
    input_tokens: int
    output_tokens: int
    latency_ms: int


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


@dataclass(frozen=True)
class JobAggregate:
    summary: TaskRunSummary
    persisted: bool
    failed: bool
    status: LlmJobStatus


class LlmDb:
    """SQLite database for v2 LLM job/run persistence."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        db_path: Path,
        *,
        persist_raw_payloads: bool,
    ):
        self._conn = conn
        self._db_path = db_path
        self._tx_depth = 0
        self._persist_raw_payloads = persist_raw_payloads

    @property
    def db_path(self) -> Path:
        return self._db_path

    @classmethod
    def open(
        cls,
        collection_dir: Path,
        *,
        persist_raw_payloads: bool | None = None,
    ) -> "LlmDb":
        llm_dir = collection_dir / LLM_DIR
        llm_dir.mkdir(parents=True, exist_ok=True)
        db_path = llm_dir / LLM_DB_FILENAME

        conn = cls._connect(db_path)
        db = cls(
            conn,
            db_path,
            persist_raw_payloads=_resolve_raw_payload_flag(persist_raw_payloads),
        )
        try:
            db._create_schema()
        except sqlite3.DatabaseError as error:
            conn.close()
            raise RuntimeError(
                "LLM DB schema is incompatible with this build. "
                f"Delete '{db_path}' to reinitialize."
            ) from error
        return db

    @staticmethod
    def _connect(db_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path)
        LlmDb._configure_connection(conn)
        return conn

    @staticmethod
    def _configure_connection(conn: sqlite3.Connection) -> None:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def write_tx(self, *, immediate: bool = True) -> Iterator[None]:
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

    def _write(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        if self._tx_depth > 0:
            return self._conn.execute(sql, params)
        with self._conn:
            return self._conn.execute(sql, params)

    @staticmethod
    def _utc_now() -> str:
        return (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

    @staticmethod
    def _as_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
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

    def _create_schema(self) -> None:
        item_status_check = _enum_check_sql([status.value for status in LlmItemStatus])
        job_status_check = _enum_check_sql([status.value for status in LlmJobStatus])
        outcome_check = _enum_check_sql(
            [outcome.value for outcome in ProviderOutcomeKind]
        )

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

                CREATE TABLE IF NOT EXISTS {_ATTEMPT_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_item_id INTEGER NOT NULL UNIQUE,
                    provider TEXT NOT NULL,
                    outcome_kind TEXT NOT NULL CHECK (outcome_kind IN {outcome_check}),
                    transport_mode TEXT NOT NULL,
                    capability_snapshot_json TEXT NOT NULL,
                    contract_fingerprint TEXT NOT NULL,
                    refusal_reason TEXT,
                    provider_message_id TEXT,
                    response_model_id TEXT,
                    provider_request_id TEXT,
                    stop_reason TEXT,
                    latency_ms INTEGER NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    retry_count INTEGER NOT NULL,
                    error_message TEXT,
                    rate_limit_headers_json TEXT,
                    parsed_update_json TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    FOREIGN KEY(job_item_id)
                        REFERENCES {_ITEM_TABLE}(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS {_ATTEMPT_PAYLOAD_TABLE} (
                    attempt_id INTEGER PRIMARY KEY,
                    system_prompt_text TEXT NOT NULL,
                    user_message_text TEXT NOT NULL,
                    request_params_json TEXT NOT NULL,
                    response_raw_text TEXT,
                    response_full_json TEXT,
                    FOREIGN KEY(attempt_id)
                        REFERENCES {_ATTEMPT_TABLE}(id) ON DELETE CASCADE
                );
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_llm_job_created "
                f"ON {_JOB_TABLE}(created_at DESC)"
            )
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_llm_job_status ON {_JOB_TABLE}(status)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_llm_job_item_job "
                f"ON {_ITEM_TABLE}(job_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_llm_job_item_note_key "
                f"ON {_ITEM_TABLE}(note_key)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_llm_attempt_item "
                f"ON {_ATTEMPT_TABLE}(job_item_id)"
            )

    @staticmethod
    def _parse_enum(enum_type: type[EnumT], raw_value: object) -> EnumT:
        if not isinstance(raw_value, str):
            raise ValueError(f"Expected string value for enum {enum_type.__name__}")
        try:
            return enum_type(raw_value)
        except ValueError as error:
            raise ValueError(
                f"Invalid value '{raw_value}' for enum {enum_type.__name__}"
            ) from error

    def start_job(
        self,
        *,
        task_name: str,
        model: str,
        model_id: str,
    ) -> int:
        now = self._utc_now()
        cursor = self._write(
            f"""
            INSERT INTO {_JOB_TABLE} (
                task_name, model, model_id,
                status, persisted, fatal_error,
                created_at, started_at
            ) VALUES (?, ?, ?, ?, 0, NULL, ?, ?)
            """,
            (
                task_name,
                model,
                model_id,
                LlmJobStatus.RUNNING.value,
                now,
                now,
            ),
        )
        rowid = cursor.lastrowid
        if rowid is None:
            raise RuntimeError(f"Failed to persist {_JOB_TABLE} row")
        return int(rowid)

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
                job_id, ordinal, note_key, deck_name, note_type,
                item_status, skip_reason,
                error_message, changed_fields_json, applied_to_markdown
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                ordinal,
                note_key,
                deck_name,
                note_type,
                item_status.value,
                skip_reason,
                error_message,
                self._as_json(changed_fields or []),
                1 if applied_to_markdown else 0,
            ),
        )
        rowid = cursor.lastrowid
        if rowid is None:
            raise RuntimeError(f"Failed to persist {_ITEM_TABLE} row")
        return int(rowid)

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
            SET item_status = ?,
                error_message = ?,
                changed_fields_json = ?
            WHERE id = ?
            """,
            (
                item_status.value,
                error_message,
                self._as_json(changed_fields or []),
                item_id,
            ),
        )

    def mark_unfinished_items_canceled(self, *, job_id: int) -> int:
        cursor = self._write(
            f"""
            UPDATE {_ITEM_TABLE}
            SET item_status = ?
            WHERE job_id = ? AND item_status = ?
            """,
            (
                LlmItemStatus.CANCELED.value,
                job_id,
                LlmItemStatus.QUEUED.value,
            ),
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

    def insert_attempt(
        self,
        *,
        item_id: int,
        provider: str,
        outcome_kind: str,
        transport_mode: str,
        capability_snapshot_json: dict[str, Any],
        contract_fingerprint: str,
        refusal_reason: str | None,
        provider_message_id: str | None,
        response_model_id: str | None,
        provider_request_id: str | None,
        stop_reason: str | None,
        latency_ms: int,
        input_tokens: int,
        output_tokens: int,
        retry_count: int,
        error_message: str | None,
        parsed_update_json: dict[str, Any] | None,
        rate_limit_headers_json: dict[str, str] | None,
    ) -> int:
        now = self._utc_now()
        cursor = self._write(
            f"""
            INSERT INTO {_ATTEMPT_TABLE} (
                job_item_id, provider,
                outcome_kind, transport_mode,
                capability_snapshot_json, contract_fingerprint,
                refusal_reason,
                provider_message_id, response_model_id, provider_request_id,
                stop_reason, latency_ms,
                input_tokens, output_tokens, retry_count,
                error_message, rate_limit_headers_json, parsed_update_json,
                created_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                provider,
                outcome_kind,
                transport_mode,
                self._as_json(capability_snapshot_json),
                contract_fingerprint,
                refusal_reason,
                provider_message_id,
                response_model_id,
                provider_request_id,
                stop_reason,
                latency_ms,
                input_tokens,
                output_tokens,
                retry_count,
                error_message,
                (
                    self._as_json(rate_limit_headers_json)
                    if rate_limit_headers_json is not None
                    else None
                ),
                (
                    self._as_json(parsed_update_json)
                    if parsed_update_json is not None
                    else None
                ),
                now,
                now,
            ),
        )
        rowid = cursor.lastrowid
        if rowid is None:
            raise RuntimeError(f"Failed to persist {_ATTEMPT_TABLE} row")
        return int(rowid)

    def insert_attempt_payload(
        self,
        *,
        attempt_id: int,
        system_prompt_text: str,
        user_message_text: str,
        request_params_json: dict[str, Any],
        response_raw_text: str | None,
        response_full_json: str | None,
    ) -> None:
        self._write(
            f"""
            INSERT INTO {_ATTEMPT_PAYLOAD_TABLE} (
                attempt_id, system_prompt_text, user_message_text,
                request_params_json, response_raw_text, response_full_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                attempt_id,
                system_prompt_text,
                user_message_text,
                self._as_json(request_params_json),
                response_raw_text if self._persist_raw_payloads else None,
                response_full_json if self._persist_raw_payloads else None,
            ),
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
            SET status = ?,
                persisted = ?,
                fatal_error = ?,
                finished_at = ?
            WHERE id = ?
            """,
            (
                status.value,
                1 if persisted else 0,
                fatal_error,
                self._utc_now(),
                job_id,
            ),
        )

    def aggregate_job(self, job_id: int) -> JobAggregate:
        row = self._conn.execute(
            f"""
            SELECT
                task_name,
                model,
                decks_seen,
                decks_matched,
                notes_seen,
                persisted,
                status
            FROM {_JOB_TABLE}
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown LLM job id {job_id}")

        model = parse_model(
            row["model"],
            collection_dir=self._db_path.parent.parent,
        )
        if model is None:
            raise ValueError(f"Unknown model '{row['model']}' in persisted job")

        summary = TaskRunSummary(
            task_name=row["task_name"],
            model=model,
            decks_seen=int(row["decks_seen"]),
            decks_matched=int(row["decks_matched"]),
            notes_seen=int(row["notes_seen"]),
        )

        item_counts = self._conn.execute(
            f"""
            SELECT
                SUM(
                    CASE
                        WHEN item_status NOT IN (
                            'skipped_no_editable_fields',
                            'invalid_note'
                        )
                        THEN 1
                        ELSE 0
                    END
                ) AS eligible,
                SUM(
                    CASE WHEN item_status = 'succeeded_updated' THEN 1 ELSE 0 END
                ) AS updated,
                SUM(
                    CASE WHEN item_status = 'succeeded_unchanged' THEN 1 ELSE 0 END
                ) AS unchanged,
                SUM(
                    CASE
                        WHEN item_status = 'skipped_no_editable_fields'
                        THEN 1
                        ELSE 0
                    END
                ) AS skipped_no_editable_fields,
                SUM(
                    CASE
                        WHEN item_status IN (
                            'invalid_note',
                            'note_error',
                            'provider_error',
                            'fatal_error'
                        )
                        THEN 1
                        ELSE 0
                    END
                ) AS errors,
                SUM(CASE WHEN item_status = 'canceled' THEN 1 ELSE 0 END) AS canceled
            FROM {_ITEM_TABLE}
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        assert item_counts is not None

        attempts = self._conn.execute(
            f"""
            SELECT
                COUNT(*) AS requests,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(latency_ms), 0) AS provider_latency_ms_total,
                COALESCE(SUM(retry_count), 0) AS provider_retries
            FROM {_ATTEMPT_TABLE} a
            JOIN {_ITEM_TABLE} i ON i.id = a.job_item_id
            WHERE i.job_id = ?
            """,
            (job_id,),
        ).fetchone()
        assert attempts is not None

        summary.eligible = int(item_counts["eligible"] or 0)
        summary.updated = int(item_counts["updated"] or 0)
        summary.unchanged = int(item_counts["unchanged"] or 0)
        summary.skipped_no_editable_fields = int(
            item_counts["skipped_no_editable_fields"] or 0
        )
        summary.errors = int(item_counts["errors"] or 0)
        summary.canceled = int(item_counts["canceled"] or 0)
        summary.requests = int(attempts["requests"] or 0)
        summary.input_tokens = int(attempts["input_tokens"] or 0)
        summary.output_tokens = int(attempts["output_tokens"] or 0)
        summary.provider_latency_ms_total = int(
            attempts["provider_latency_ms_total"] or 0
        )
        summary.provider_retries = int(attempts["provider_retries"] or 0)

        status = self._parse_enum(LlmJobStatus, row["status"])
        persisted = bool(row["persisted"])
        failed = summary.errors > 0 or status is LlmJobStatus.FAILED
        return JobAggregate(
            summary=summary,
            persisted=persisted,
            failed=failed,
            status=status,
        )

    def resolve_job_id(self, identifier: str) -> int | None:
        token = identifier.strip()
        if not token:
            return None
        if token.lower() == "latest":
            latest = self._conn.execute(
                f"""
                SELECT id
                FROM {_JOB_TABLE}
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
            if latest is None:
                return None
            return int(latest["id"])
        if not token.isdigit():
            raise ValueError("Job ID must be numeric or use 'latest'.")
        row = self._conn.execute(
            f"SELECT id FROM {_JOB_TABLE} WHERE id = ?",
            (int(token),),
        ).fetchone()
        if row is None:
            return None
        return int(row["id"])

    def list_jobs(self) -> list[LlmJobListItem]:
        rows = self._conn.execute(
            f"""
            SELECT
                id,
                task_name,
                model,
                status,
                persisted,
                created_at,
                finished_at
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
                status=self._parse_enum(LlmJobStatus, row["status"]),
                persisted=bool(row["persisted"]),
                created_at=str(row["created_at"]),
                finished_at=(str(row["finished_at"]) if row["finished_at"] else None),
            )
            for row in rows
        ]

    def get_job_detail(self, job_id: int) -> LlmJobDetail | None:
        row = self._conn.execute(
            f"""
            SELECT
                id,
                task_name,
                model,
                model_id,
                status,
                persisted,
                created_at,
                started_at,
                finished_at,
                fatal_error
            FROM {_JOB_TABLE}
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            return None

        resolved_job_id = int(row["id"])
        aggregate = self.aggregate_job(resolved_job_id)
        item_rows = self._conn.execute(
            f"""
            SELECT
                i.ordinal,
                i.note_key,
                i.deck_name,
                i.note_type,
                i.item_status,
                i.error_message,
                i.changed_fields_json,
                COUNT(a.id) AS attempts,
                COALESCE(SUM(a.input_tokens), 0) AS input_tokens,
                COALESCE(SUM(a.output_tokens), 0) AS output_tokens,
                COALESCE(SUM(a.latency_ms), 0) AS latency_ms
            FROM {_ITEM_TABLE} i
            LEFT JOIN {_ATTEMPT_TABLE} a ON a.job_item_id = i.id
            WHERE i.job_id = ?
            GROUP BY
                i.id,
                i.ordinal,
                i.note_key,
                i.deck_name,
                i.note_type,
                i.item_status,
                i.error_message,
                i.changed_fields_json
            ORDER BY i.ordinal ASC
            """,
            (resolved_job_id,),
        ).fetchall()

        items = [
            LlmJobItemDetail(
                ordinal=int(item["ordinal"]),
                note_key=(str(item["note_key"]) if item["note_key"] else None),
                deck_name=str(item["deck_name"]),
                note_type=(str(item["note_type"]) if item["note_type"] else None),
                item_status=self._parse_enum(
                    LlmItemStatus,
                    item["item_status"],
                ),
                error_message=(
                    str(item["error_message"]) if item["error_message"] else None
                ),
                changed_fields=self._parse_json_list(item["changed_fields_json"]),
                attempts=int(item["attempts"] or 0),
                input_tokens=int(item["input_tokens"] or 0),
                output_tokens=int(item["output_tokens"] or 0),
                latency_ms=int(item["latency_ms"] or 0),
            )
            for item in item_rows
        ]

        return LlmJobDetail(
            job_id=resolved_job_id,
            task_name=str(row["task_name"]),
            model=str(row["model"]),
            model_id=str(row["model_id"]),
            status=self._parse_enum(LlmJobStatus, row["status"]),
            persisted=bool(row["persisted"]),
            created_at=str(row["created_at"]),
            started_at=str(row["started_at"]),
            finished_at=(str(row["finished_at"]) if row["finished_at"] else None),
            fatal_error=(str(row["fatal_error"]) if row["fatal_error"] else None),
            summary=aggregate.summary,
            items=items,
        )


def _resolve_raw_payload_flag(value: bool | None) -> bool:
    if value is not None:
        return value
    raw = os.environ.get(_RAW_PAYLOAD_ENV_VAR, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}
