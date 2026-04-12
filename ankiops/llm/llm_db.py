"""SQLite database for LLM execution history."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, TypeVar

from ankiops.config import LLM_DB_FILENAME, LLM_DIR

from .llm_models import (
    ExecutionMode,
    LlmAttemptResultType,
    LlmCandidateStatus,
    LlmFinalStatus,
    LlmJobStatus,
    RunFailurePolicy,
    TaskRunSummary,
)
from .model_registry import parse_model

_DEFAULT_JOB_LIST_LIMIT = 20

_RESUMABLE_FINAL_STATUSES = (
    LlmFinalStatus.NOT_ATTEMPTED.value,
    LlmFinalStatus.NOTE_ERROR.value,
    LlmFinalStatus.PROVIDER_ERROR.value,
    LlmFinalStatus.FATAL_ERROR.value,
    LlmFinalStatus.CANCELED.value,
    LlmFinalStatus.EXPIRED.value,
)

EnumT = TypeVar("EnumT", bound=Enum)


def _enum_check_sql(values: list[str]) -> str:
    quoted = ", ".join(f"'{value}'" for value in values)
    return f"({quoted})"


@dataclass(frozen=True)
class LlmJobListItem:
    job_id: int
    task_name: str
    model_name: str
    execution_mode: ExecutionMode
    status: LlmJobStatus
    persisted: bool
    created_at: str
    finished_at: str | None
    resume_from_job_id: int | None


@dataclass(frozen=True)
class LlmJobItemDetail:
    ordinal: int
    note_key: str | None
    deck_name: str
    note_type: str | None
    candidate_status: LlmCandidateStatus
    final_status: LlmFinalStatus
    error_message: str | None
    changed_fields: list[str]
    attempts: int
    input_tokens: int
    output_tokens: int
    latency_ms: int
    resume_source_item_id: int | None


@dataclass(frozen=True)
class LlmJobDetail:
    job_id: int
    task_name: str
    model_name: str
    api_model: str
    execution_mode: ExecutionMode
    status: LlmJobStatus
    persisted: bool
    failure_policy: RunFailurePolicy
    created_at: str
    started_at: str
    finished_at: str | None
    fatal_error: str | None
    summary: TaskRunSummary
    items: list[LlmJobItemDetail]
    resume_from_job_id: int | None


@dataclass(frozen=True)
class JobAggregate:
    summary: TaskRunSummary
    persisted: bool
    failed: bool
    status: LlmJobStatus


class LlmDb:
    """SQLite database for LLM job/run persistence."""

    def __init__(self, conn: sqlite3.Connection, db_path: Path):
        self._conn = conn
        self._db_path = db_path
        self._tx_depth = 0

    @property
    def db_path(self) -> Path:
        return self._db_path

    @classmethod
    def open(cls, collection_dir: Path) -> "LlmDb":
        llm_dir = collection_dir / LLM_DIR
        llm_dir.mkdir(parents=True, exist_ok=True)
        db_path = llm_dir / LLM_DB_FILENAME

        conn = cls._connect(db_path)
        db = cls(conn, db_path)
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

    @staticmethod
    def _parse_json_mapping(raw: str | None) -> dict[str, Any]:
        if raw is None:
            return {}
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}

    def _create_schema(self) -> None:
        candidate_status_check = _enum_check_sql(
            [status.value for status in LlmCandidateStatus]
        )
        final_status_check = _enum_check_sql(
            [status.value for status in LlmFinalStatus]
        )
        attempt_result_check = _enum_check_sql(
            [result.value for result in LlmAttemptResultType]
        )
        job_status_check = _enum_check_sql([status.value for status in LlmJobStatus])
        failure_policy_check = _enum_check_sql(
            [policy.value for policy in RunFailurePolicy]
        )
        execution_mode_check = _enum_check_sql([mode.value for mode in ExecutionMode])

        with self._conn:
            self._conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS llm_job (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    api_model TEXT NOT NULL,
                    execution_mode TEXT NOT NULL CHECK (execution_mode IN {execution_mode_check}),
                    failure_policy TEXT NOT NULL CHECK (failure_policy IN {failure_policy_check}),
                    status TEXT NOT NULL CHECK (status IN {job_status_check}),
                    persisted INTEGER NOT NULL DEFAULT 0 CHECK (persisted IN (0, 1)),
                    fatal_error TEXT,
                    config_snapshot_json TEXT NOT NULL,
                    resume_from_job_id INTEGER,
                    created_at TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    decks_seen INTEGER NOT NULL DEFAULT 0,
                    decks_matched INTEGER NOT NULL DEFAULT 0,
                    notes_seen INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(resume_from_job_id) REFERENCES llm_job(id)
                );

                CREATE TABLE IF NOT EXISTS llm_job_item (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER NOT NULL,
                    ordinal INTEGER NOT NULL,
                    note_key TEXT,
                    deck_name TEXT NOT NULL,
                    note_type TEXT,
                    candidate_status TEXT NOT NULL CHECK (candidate_status IN {candidate_status_check}),
                    skip_reason TEXT,
                    final_status TEXT NOT NULL CHECK (final_status IN {final_status_check}),
                    error_message TEXT,
                    changed_fields_json TEXT NOT NULL,
                    applied_to_markdown INTEGER NOT NULL DEFAULT 0 CHECK (applied_to_markdown IN (0, 1)),
                    resume_source_item_id INTEGER,
                    UNIQUE(job_id, ordinal),
                    FOREIGN KEY(job_id) REFERENCES llm_job(id) ON DELETE CASCADE,
                    FOREIGN KEY(resume_source_item_id) REFERENCES llm_job_item(id)
                );

                CREATE TABLE IF NOT EXISTS llm_item_attempt (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_item_id INTEGER NOT NULL,
                    attempt_no INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    provider_message_id TEXT,
                    provider_model TEXT,
                    provider_request_id TEXT,
                    provider_execution_mode TEXT NOT NULL CHECK (provider_execution_mode IN {execution_mode_check}),
                    stop_reason TEXT,
                    result_type TEXT NOT NULL CHECK (result_type IN {attempt_result_check}),
                    latency_ms INTEGER NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    retry_count INTEGER NOT NULL,
                    error_type TEXT,
                    error_message TEXT,
                    rate_limit_headers_json TEXT,
                    parsed_update_json TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    UNIQUE(job_item_id, attempt_no),
                    FOREIGN KEY(job_item_id) REFERENCES llm_job_item(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS llm_attempt_payload (
                    attempt_id INTEGER PRIMARY KEY,
                    system_prompt_text TEXT NOT NULL,
                    user_message_text TEXT NOT NULL,
                    request_params_json TEXT NOT NULL,
                    response_raw_text TEXT,
                    response_full_json TEXT,
                    FOREIGN KEY(attempt_id) REFERENCES llm_item_attempt(id) ON DELETE CASCADE
                );
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_llm_job_created "
                "ON llm_job(created_at DESC)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_llm_job_status ON llm_job(status)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_llm_job_item_job "
                "ON llm_job_item(job_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_llm_job_item_note_key "
                "ON llm_job_item(note_key)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_llm_attempt_result "
                "ON llm_item_attempt(result_type)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_llm_attempt_item "
                "ON llm_item_attempt(job_item_id)"
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
        model_name: str,
        api_model: str,
        execution_mode: ExecutionMode,
        failure_policy: RunFailurePolicy,
        config_snapshot: dict[str, Any],
        resume_from_job_id: int | None = None,
    ) -> int:
        now = self._utc_now()
        cursor = self._write(
            """
            INSERT INTO llm_job (
                task_name, model_name, api_model, execution_mode,
                failure_policy, status, persisted, fatal_error,
                config_snapshot_json, resume_from_job_id,
                created_at, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, ?, ?)
            """,
            (
                task_name,
                model_name,
                api_model,
                execution_mode.value,
                failure_policy.value,
                LlmJobStatus.RUNNING.value,
                self._as_json(config_snapshot),
                resume_from_job_id,
                now,
                now,
            ),
        )
        return int(cursor.lastrowid)

    def set_discovery_counts(
        self,
        *,
        job_id: int,
        decks_seen: int,
        decks_matched: int,
        notes_seen: int,
    ) -> None:
        self._write(
            """
            UPDATE llm_job
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
        candidate_status: LlmCandidateStatus,
        skip_reason: str | None,
        final_status: LlmFinalStatus,
        error_message: str | None = None,
        changed_fields: list[str] | None = None,
        applied_to_markdown: bool = False,
        resume_source_item_id: int | None = None,
    ) -> int:
        cursor = self._write(
            """
            INSERT INTO llm_job_item (
                job_id, ordinal, note_key, deck_name, note_type,
                candidate_status, skip_reason, final_status,
                error_message, changed_fields_json, applied_to_markdown,
                resume_source_item_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                ordinal,
                note_key,
                deck_name,
                note_type,
                candidate_status.value,
                skip_reason,
                final_status.value,
                error_message,
                self._as_json(changed_fields or []),
                1 if applied_to_markdown else 0,
                resume_source_item_id,
            ),
        )
        return int(cursor.lastrowid)

    def update_job_item_result(
        self,
        *,
        item_id: int,
        final_status: LlmFinalStatus,
        error_message: str | None = None,
        changed_fields: list[str] | None = None,
    ) -> None:
        self._write(
            """
            UPDATE llm_job_item
            SET final_status = ?,
                error_message = ?,
                changed_fields_json = ?
            WHERE id = ?
            """,
            (
                final_status.value,
                error_message,
                self._as_json(changed_fields or []),
                item_id,
            ),
        )

    def mark_unfinished_items_canceled(self, *, job_id: int) -> int:
        cursor = self._write(
            """
            UPDATE llm_job_item
            SET final_status = ?
            WHERE job_id = ? AND candidate_status = ? AND final_status = ?
            """,
            (
                LlmFinalStatus.CANCELED.value,
                job_id,
                LlmCandidateStatus.ELIGIBLE.value,
                LlmFinalStatus.NOT_ATTEMPTED.value,
            ),
        )
        return int(cursor.rowcount)

    def set_applied_for_updated_items(self, *, job_id: int) -> None:
        self._write(
            """
            UPDATE llm_job_item
            SET applied_to_markdown = 1
            WHERE job_id = ? AND final_status = ?
            """,
            (job_id, LlmFinalStatus.SUCCEEDED_UPDATED.value),
        )

    def insert_attempt(
        self,
        *,
        item_id: int,
        attempt_no: int,
        provider: str,
        provider_message_id: str | None,
        provider_model: str | None,
        provider_request_id: str | None,
        provider_execution_mode: ExecutionMode,
        stop_reason: str | None,
        result_type: LlmAttemptResultType,
        latency_ms: int,
        input_tokens: int,
        output_tokens: int,
        retry_count: int,
        error_type: str | None,
        error_message: str | None,
        parsed_update_json: dict[str, Any] | None,
        rate_limit_headers_json: dict[str, str] | None,
    ) -> int:
        now = self._utc_now()
        cursor = self._write(
            """
            INSERT INTO llm_item_attempt (
                job_item_id, attempt_no, provider, provider_message_id,
                provider_model, provider_request_id, provider_execution_mode,
                stop_reason, result_type, latency_ms,
                input_tokens, output_tokens, retry_count, error_type,
                error_message, rate_limit_headers_json, parsed_update_json,
                created_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                attempt_no,
                provider,
                provider_message_id,
                provider_model,
                provider_request_id,
                provider_execution_mode.value,
                stop_reason,
                result_type.value,
                latency_ms,
                input_tokens,
                output_tokens,
                retry_count,
                error_type,
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
        return int(cursor.lastrowid)

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
            """
            INSERT INTO llm_attempt_payload (
                attempt_id, system_prompt_text, user_message_text,
                request_params_json, response_raw_text, response_full_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                attempt_id,
                system_prompt_text,
                user_message_text,
                self._as_json(request_params_json),
                response_raw_text,
                response_full_json,
            ),
        )

    def get_resume_source_items(self, *, job_id: int) -> dict[str, int]:
        placeholders = ",".join("?" for _ in _RESUMABLE_FINAL_STATUSES)
        rows = self._conn.execute(
            f"""
            SELECT id, note_key
            FROM llm_job_item
            WHERE job_id = ?
              AND note_key IS NOT NULL
              AND final_status IN ({placeholders})
            """,
            (job_id, *_RESUMABLE_FINAL_STATUSES),
        ).fetchall()
        mapping: dict[str, int] = {}
        for row in rows:
            note_key = row["note_key"]
            if isinstance(note_key, str):
                mapping[note_key] = int(row["id"])
        return mapping

    def get_job_snapshot(self, *, job_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT config_snapshot_json FROM llm_job WHERE id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            return None
        return self._parse_json_mapping(row["config_snapshot_json"])

    def finalize_job(
        self,
        *,
        job_id: int,
        status: LlmJobStatus,
        persisted: bool,
        fatal_error: str | None = None,
    ) -> None:
        self._write(
            """
            UPDATE llm_job
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
            """
            SELECT
                task_name,
                model_name,
                execution_mode,
                decks_seen,
                decks_matched,
                notes_seen,
                persisted,
                status
            FROM llm_job
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown LLM job id {job_id}")

        model = parse_model(row["model_name"])
        if model is None:
            raise ValueError(f"Unknown model '{row['model_name']}' in persisted job")
        execution_mode = self._parse_enum(ExecutionMode, row["execution_mode"])

        summary = TaskRunSummary(
            task_name=row["task_name"],
            model=model,
            execution_mode=execution_mode,
            decks_seen=int(row["decks_seen"]),
            decks_matched=int(row["decks_matched"]),
            notes_seen=int(row["notes_seen"]),
        )

        item_counts = self._conn.execute(
            """
            SELECT
                SUM(CASE WHEN candidate_status = 'eligible' THEN 1 ELSE 0 END) AS eligible,
                SUM(CASE WHEN final_status = 'succeeded_updated' THEN 1 ELSE 0 END) AS updated,
                SUM(CASE WHEN final_status = 'succeeded_unchanged' THEN 1 ELSE 0 END) AS unchanged,
                SUM(CASE WHEN candidate_status = 'skipped_deck_scope' THEN 1 ELSE 0 END) AS skipped_deck_scope,
                SUM(CASE WHEN candidate_status = 'skipped_no_editable_fields' THEN 1 ELSE 0 END) AS skipped_no_editable_fields,
                SUM(CASE WHEN final_status IN ('note_error', 'provider_error', 'fatal_error') THEN 1 ELSE 0 END) AS errors,
                SUM(CASE WHEN final_status = 'canceled' THEN 1 ELSE 0 END) AS canceled,
                SUM(CASE WHEN final_status = 'expired' THEN 1 ELSE 0 END) AS expired
            FROM llm_job_item
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        assert item_counts is not None

        attempts = self._conn.execute(
            """
            SELECT
                COUNT(*) AS requests,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(latency_ms), 0) AS provider_latency_ms_total,
                COALESCE(SUM(retry_count), 0) AS provider_retries
            FROM llm_item_attempt a
            JOIN llm_job_item i ON i.id = a.job_item_id
            WHERE i.job_id = ?
            """,
            (job_id,),
        ).fetchone()
        assert attempts is not None

        summary.eligible = int(item_counts["eligible"] or 0)
        summary.updated = int(item_counts["updated"] or 0)
        summary.unchanged = int(item_counts["unchanged"] or 0)
        summary.skipped_deck_scope = int(item_counts["skipped_deck_scope"] or 0)
        summary.skipped_no_editable_fields = int(
            item_counts["skipped_no_editable_fields"] or 0
        )
        summary.errors = int(item_counts["errors"] or 0)
        summary.canceled = int(item_counts["canceled"] or 0)
        summary.expired = int(item_counts["expired"] or 0)
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
                """
                SELECT id
                FROM llm_job
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
            "SELECT id FROM llm_job WHERE id = ?",
            (int(token),),
        ).fetchone()
        if row is None:
            return None
        return int(row["id"])

    def list_jobs(self) -> list[LlmJobListItem]:
        rows = self._conn.execute(
            """
            SELECT
                id,
                task_name,
                model_name,
                execution_mode,
                status,
                persisted,
                created_at,
                finished_at,
                resume_from_job_id
            FROM llm_job
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (_DEFAULT_JOB_LIST_LIMIT,),
        ).fetchall()
        return [
            LlmJobListItem(
                job_id=int(row["id"]),
                task_name=str(row["task_name"]),
                model_name=str(row["model_name"]),
                execution_mode=self._parse_enum(ExecutionMode, row["execution_mode"]),
                status=self._parse_enum(LlmJobStatus, row["status"]),
                persisted=bool(row["persisted"]),
                created_at=str(row["created_at"]),
                finished_at=(str(row["finished_at"]) if row["finished_at"] else None),
                resume_from_job_id=(
                    int(row["resume_from_job_id"])
                    if row["resume_from_job_id"] is not None
                    else None
                ),
            )
            for row in rows
        ]

    def get_job_detail(self, job_id: int) -> LlmJobDetail | None:
        row = self._conn.execute(
            """
            SELECT
                id,
                task_name,
                model_name,
                api_model,
                execution_mode,
                status,
                persisted,
                failure_policy,
                created_at,
                started_at,
                finished_at,
                fatal_error,
                resume_from_job_id
            FROM llm_job
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            return None

        resolved_job_id = int(row["id"])
        aggregate = self.aggregate_job(resolved_job_id)
        item_rows = self._conn.execute(
            """
            SELECT
                i.ordinal,
                i.note_key,
                i.deck_name,
                i.note_type,
                i.candidate_status,
                i.final_status,
                i.error_message,
                i.changed_fields_json,
                i.resume_source_item_id,
                COUNT(a.id) AS attempts,
                COALESCE(SUM(a.input_tokens), 0) AS input_tokens,
                COALESCE(SUM(a.output_tokens), 0) AS output_tokens,
                COALESCE(SUM(a.latency_ms), 0) AS latency_ms
            FROM llm_job_item i
            LEFT JOIN llm_item_attempt a ON a.job_item_id = i.id
            WHERE i.job_id = ?
            GROUP BY
                i.id,
                i.ordinal,
                i.note_key,
                i.deck_name,
                i.note_type,
                i.candidate_status,
                i.final_status,
                i.error_message,
                i.changed_fields_json,
                i.resume_source_item_id
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
                candidate_status=self._parse_enum(
                    LlmCandidateStatus,
                    item["candidate_status"],
                ),
                final_status=self._parse_enum(
                    LlmFinalStatus,
                    item["final_status"],
                ),
                error_message=(
                    str(item["error_message"]) if item["error_message"] else None
                ),
                changed_fields=self._parse_json_list(item["changed_fields_json"]),
                attempts=int(item["attempts"] or 0),
                input_tokens=int(item["input_tokens"] or 0),
                output_tokens=int(item["output_tokens"] or 0),
                latency_ms=int(item["latency_ms"] or 0),
                resume_source_item_id=(
                    int(item["resume_source_item_id"])
                    if item["resume_source_item_id"] is not None
                    else None
                ),
            )
            for item in item_rows
        ]

        return LlmJobDetail(
            job_id=resolved_job_id,
            task_name=str(row["task_name"]),
            model_name=str(row["model_name"]),
            api_model=str(row["api_model"]),
            execution_mode=self._parse_enum(ExecutionMode, row["execution_mode"]),
            status=self._parse_enum(LlmJobStatus, row["status"]),
            persisted=bool(row["persisted"]),
            failure_policy=self._parse_enum(RunFailurePolicy, row["failure_policy"]),
            created_at=str(row["created_at"]),
            started_at=str(row["started_at"]),
            finished_at=(str(row["finished_at"]) if row["finished_at"] else None),
            fatal_error=(str(row["fatal_error"]) if row["fatal_error"] else None),
            summary=aggregate.summary,
            items=items,
            resume_from_job_id=(
                int(row["resume_from_job_id"])
                if row["resume_from_job_id"] is not None
                else None
            ),
        )
