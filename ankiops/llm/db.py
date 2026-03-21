"""SQLite adapter for LLM execution history and batch metadata."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ankiops.config import LLM_DB_FILENAME, LLM_DIR

from .anthropic_models import parse_model
from .models import (
    LlmAttemptResultType,
    LlmCandidateStatus,
    LlmFinalStatus,
    LlmJobStatus,
    RunFailurePolicy,
    TaskRunSummary,
)

logger = logging.getLogger(__name__)

_EXPECTED_TABLES = {
    "llm_job",
    "llm_job_item",
    "llm_item_attempt",
    "llm_attempt_payload",
    "llm_provider_batch",
    "llm_batch_item_map",
}

_REQUIRED_INDEXES = {
    "idx_llm_job_created",
    "idx_llm_job_item_job",
    "idx_llm_job_item_note_key",
    "idx_llm_attempt_result",
    "idx_llm_attempt_item",
    "idx_llm_provider_batch_status",
}

_EXPECTED_COLUMNS: dict[str, list[str]] = {
    "llm_job": [
        "id",
        "task_name",
        "model_name",
        "api_model",
        "failure_policy",
        "status",
        "persisted",
        "fatal_error",
        "created_at",
        "started_at",
        "finished_at",
        "decks_seen",
        "decks_matched",
        "notes_seen",
    ],
    "llm_job_item": [
        "id",
        "job_id",
        "ordinal",
        "note_key",
        "deck_name",
        "note_type",
        "candidate_status",
        "skip_reason",
        "final_status",
        "error_message",
        "changed_fields_json",
        "applied_to_markdown",
    ],
    "llm_item_attempt": [
        "id",
        "job_item_id",
        "attempt_no",
        "provider",
        "provider_message_id",
        "provider_model",
        "stop_reason",
        "result_type",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "retry_count",
        "error_type",
        "error_message",
        "parsed_update_json",
        "created_at",
        "completed_at",
    ],
    "llm_attempt_payload": [
        "attempt_id",
        "system_prompt_text",
        "user_message_text",
        "request_params_json",
        "response_raw_text",
        "response_full_json",
    ],
    "llm_provider_batch": [
        "id",
        "job_id",
        "provider_batch_id",
        "processing_status",
        "results_url",
        "created_at_remote",
        "expires_at_remote",
        "ended_at_remote",
        "archived_at_remote",
        "cancel_initiated_at_remote",
        "count_processing",
        "count_succeeded",
        "count_errored",
        "count_canceled",
        "count_expired",
    ],
    "llm_batch_item_map": [
        "provider_batch_id",
        "custom_id",
        "job_item_id",
        "attempt_no",
    ],
}


def _enum_check_sql(values: list[str]) -> str:
    quoted = ", ".join(f"'{value}'" for value in values)
    return f"({quoted})"


def _normalized_sql(sql: str | None) -> str:
    if not sql:
        return ""
    return " ".join(sql.lower().split())


@dataclass(frozen=True)
class LlmJobListItem:
    job_id: int
    task_name: str
    model_name: str
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
    candidate_status: LlmCandidateStatus
    final_status: LlmFinalStatus
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
    model_name: str
    api_model: str
    status: LlmJobStatus
    persisted: bool
    failure_policy: RunFailurePolicy
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


class LlmDbAdapter:
    """SQLite adapter for LLM job/run persistence."""

    def __init__(self, conn: sqlite3.Connection, db_path: Path):
        self._conn = conn
        self._db_path = db_path
        self._tx_depth = 0

    @property
    def db_path(self) -> Path:
        return self._db_path

    @classmethod
    def open(cls, collection_dir: Path) -> "LlmDbAdapter":
        llm_dir = collection_dir / LLM_DIR
        llm_dir.mkdir(parents=True, exist_ok=True)
        db_path = llm_dir / LLM_DB_FILENAME

        conn = cls._connect(db_path)
        adapter = cls(conn, db_path)
        if not adapter._ensure_schema():
            logger.warning(
                "LLM DB schema mismatch at %s; recreating database",
                db_path,
            )
            conn.close()
            cls._delete_db_files(db_path)
            conn = cls._connect(db_path)
            adapter = cls(conn, db_path)
            adapter._create_schema()
        return adapter

    @staticmethod
    def _delete_db_files(db_path: Path) -> None:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{db_path}{suffix}")
            if candidate.exists():
                candidate.unlink()

    @staticmethod
    def _connect(db_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

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

    def _ensure_schema(self) -> bool:
        tables = self._user_tables()
        if not tables:
            self._create_schema()
            return True
        return self._schema_matches_expected()

    def _user_tables(self) -> set[str]:
        rows = self._conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
              AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()
        return {str(row["name"]) for row in rows}

    def _table_columns(self, table_name: str) -> list[str]:
        rows = self._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return [str(row["name"]) for row in rows]

    def _table_sql(self, table_name: str) -> str:
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return _normalized_sql(str(row["sql"]) if row and row["sql"] else "")

    def _schema_matches_expected(self) -> bool:
        tables = self._user_tables()
        if tables != _EXPECTED_TABLES:
            return False

        for table_name, expected_columns in _EXPECTED_COLUMNS.items():
            if self._table_columns(table_name) != expected_columns:
                return False

        index_rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        indexes = {str(row["name"]) for row in index_rows}
        if not _REQUIRED_INDEXES.issubset(indexes):
            return False

        job_sql = self._table_sql("llm_job")
        if (
            "check (status in ('running', 'completed', 'failed'))" not in job_sql
            or "check (failure_policy in ('atomic', 'partial'))" not in job_sql
            or "check (persisted in (0, 1))" not in job_sql
        ):
            return False

        item_sql = self._table_sql("llm_job_item")
        if (
            "check (candidate_status in ('eligible', 'skipped_deck_scope', "
            "'skipped_no_editable_fields', 'invalid_note'))" not in item_sql
            or "check (final_status in ('not_attempted', 'succeeded_updated', "
            "'succeeded_unchanged', 'note_error', 'provider_error', "
            "'fatal_error', 'canceled', 'expired'))" not in item_sql
            or "check (applied_to_markdown in (0, 1))" not in item_sql
        ):
            return False

        attempt_sql = self._table_sql("llm_item_attempt")
        if (
            "check (result_type in ('succeeded', 'errored', "
            "'canceled', 'expired'))" not in attempt_sql
        ):
            return False

        return True

    def _create_schema(self) -> None:
        candidate_values = _enum_check_sql(
            [status.value for status in LlmCandidateStatus]
        )
        final_values = _enum_check_sql([status.value for status in LlmFinalStatus])
        attempt_values = _enum_check_sql(
            [status.value for status in LlmAttemptResultType]
        )
        job_values = _enum_check_sql([status.value for status in LlmJobStatus])
        policy_values = _enum_check_sql([policy.value for policy in RunFailurePolicy])

        with self._conn:
            self._conn.execute(
                f"""
                CREATE TABLE llm_job (
                    id INTEGER PRIMARY KEY,
                    task_name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    api_model TEXT NOT NULL,
                    failure_policy TEXT NOT NULL
                        CHECK (failure_policy IN {policy_values}),
                    status TEXT NOT NULL
                        CHECK (status IN {job_values}),
                    persisted INTEGER NOT NULL DEFAULT 0
                        CHECK (persisted IN (0, 1)),
                    fatal_error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    decks_seen INTEGER NOT NULL DEFAULT 0,
                    decks_matched INTEGER NOT NULL DEFAULT 0,
                    notes_seen INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._conn.execute(
                f"""
                CREATE TABLE llm_job_item (
                    id INTEGER PRIMARY KEY,
                    job_id INTEGER NOT NULL,
                    ordinal INTEGER NOT NULL,
                    note_key TEXT,
                    deck_name TEXT NOT NULL,
                    note_type TEXT,
                    candidate_status TEXT NOT NULL
                        CHECK (candidate_status IN {candidate_values}),
                    skip_reason TEXT,
                    final_status TEXT NOT NULL
                        CHECK (final_status IN {final_values}),
                    error_message TEXT,
                    changed_fields_json TEXT NOT NULL DEFAULT '[]',
                    applied_to_markdown INTEGER NOT NULL DEFAULT 0
                        CHECK (applied_to_markdown IN (0, 1)),
                    UNIQUE(job_id, ordinal),
                    FOREIGN KEY(job_id) REFERENCES llm_job(id) ON DELETE CASCADE
                )
                """
            )
            self._conn.execute(
                f"""
                CREATE TABLE llm_item_attempt (
                    id INTEGER PRIMARY KEY,
                    job_item_id INTEGER NOT NULL,
                    attempt_no INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    provider_message_id TEXT,
                    provider_model TEXT,
                    stop_reason TEXT,
                    result_type TEXT NOT NULL
                        CHECK (result_type IN {attempt_values}),
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    error_type TEXT,
                    error_message TEXT,
                    parsed_update_json TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    UNIQUE(job_item_id, attempt_no),
                    FOREIGN KEY(job_item_id)
                        REFERENCES llm_job_item(id)
                        ON DELETE CASCADE
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE llm_attempt_payload (
                    attempt_id INTEGER PRIMARY KEY,
                    system_prompt_text TEXT NOT NULL,
                    user_message_text TEXT NOT NULL,
                    request_params_json TEXT NOT NULL,
                    response_raw_text TEXT,
                    response_full_json TEXT,
                    FOREIGN KEY(attempt_id)
                        REFERENCES llm_item_attempt(id)
                        ON DELETE CASCADE
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE llm_provider_batch (
                    id INTEGER PRIMARY KEY,
                    job_id INTEGER NOT NULL,
                    provider_batch_id TEXT NOT NULL UNIQUE,
                    processing_status TEXT NOT NULL,
                    results_url TEXT,
                    created_at_remote TEXT,
                    expires_at_remote TEXT,
                    ended_at_remote TEXT,
                    archived_at_remote TEXT,
                    cancel_initiated_at_remote TEXT,
                    count_processing INTEGER NOT NULL DEFAULT 0,
                    count_succeeded INTEGER NOT NULL DEFAULT 0,
                    count_errored INTEGER NOT NULL DEFAULT 0,
                    count_canceled INTEGER NOT NULL DEFAULT 0,
                    count_expired INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(job_id) REFERENCES llm_job(id) ON DELETE CASCADE
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE llm_batch_item_map (
                    provider_batch_id INTEGER NOT NULL,
                    custom_id TEXT NOT NULL,
                    job_item_id INTEGER NOT NULL,
                    attempt_no INTEGER NOT NULL,
                    PRIMARY KEY(provider_batch_id, custom_id),
                    FOREIGN KEY(provider_batch_id)
                        REFERENCES llm_provider_batch(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY(job_item_id)
                        REFERENCES llm_job_item(id)
                        ON DELETE CASCADE
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX idx_llm_job_created ON llm_job(created_at DESC)"
            )
            self._conn.execute(
                "CREATE INDEX idx_llm_job_item_job ON llm_job_item(job_id)"
            )
            self._conn.execute(
                "CREATE INDEX idx_llm_job_item_note_key ON llm_job_item(note_key)"
            )
            self._conn.execute(
                "CREATE INDEX idx_llm_attempt_result ON llm_item_attempt(result_type)"
            )
            self._conn.execute(
                "CREATE INDEX idx_llm_attempt_item ON llm_item_attempt(job_item_id)"
            )
            self._conn.execute(
                "CREATE INDEX idx_llm_provider_batch_status "
                "ON llm_provider_batch(processing_status)"
            )

    @staticmethod
    def _parse_enum(enum_type, raw_value: object):
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
        failure_policy: RunFailurePolicy,
    ) -> int:
        now = self._utc_now()
        cursor = self._write(
            """
            INSERT INTO llm_job (
                task_name, model_name, api_model, failure_policy,
                status, created_at, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_name,
                model_name,
                api_model,
                failure_policy.value,
                LlmJobStatus.RUNNING.value,
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
    ) -> int:
        cursor = self._write(
            """
            INSERT INTO llm_job_item (
                job_id, ordinal, note_key, deck_name, note_type,
                candidate_status, skip_reason, final_status,
                error_message, changed_fields_json, applied_to_markdown
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        stop_reason: str | None,
        result_type: LlmAttemptResultType,
        latency_ms: int,
        input_tokens: int,
        output_tokens: int,
        retry_count: int,
        error_type: str | None,
        error_message: str | None,
        parsed_update_json: dict[str, Any] | None,
    ) -> int:
        now = self._utc_now()
        cursor = self._write(
            """
            INSERT INTO llm_item_attempt (
                job_item_id, attempt_no, provider, provider_message_id,
                provider_model, stop_reason, result_type, latency_ms,
                input_tokens, output_tokens, retry_count, error_type,
                error_message, parsed_update_json, created_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                attempt_no,
                provider,
                provider_message_id,
                provider_model,
                stop_reason,
                result_type.value,
                latency_ms,
                input_tokens,
                output_tokens,
                retry_count,
                error_type,
                error_message,
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

        summary = TaskRunSummary(
            task_name=row["task_name"],
            model=model,
            decks_seen=int(row["decks_seen"]),
            decks_matched=int(row["decks_matched"]),
            notes_seen=int(row["notes_seen"]),
        )

        item_counts = self._conn.execute(
            """
            SELECT
                SUM(
                    CASE WHEN candidate_status = 'eligible' THEN 1 ELSE 0 END
                ) AS eligible,
                SUM(
                    CASE WHEN final_status = 'succeeded_updated' THEN 1 ELSE 0 END
                ) AS updated,
                SUM(
                    CASE WHEN final_status = 'succeeded_unchanged' THEN 1 ELSE 0 END
                ) AS unchanged,
                SUM(
                    CASE
                        WHEN candidate_status = 'skipped_deck_scope' THEN 1
                        ELSE 0
                    END
                ) AS skipped_deck_scope,
                SUM(
                    CASE
                        WHEN candidate_status = 'skipped_no_editable_fields' THEN 1
                        ELSE 0
                    END
                ) AS skipped_no_editable_fields,
                SUM(
                    CASE
                        WHEN final_status IN (
                            'note_error',
                            'provider_error',
                            'fatal_error'
                        ) THEN 1
                        ELSE 0
                    END
                ) AS errors
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
        if token.lower() in {"latest", "last"}:
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
            raise ValueError("Job ID must be numeric, or use 'latest'.")
        row = self._conn.execute(
            "SELECT id FROM llm_job WHERE id = ?",
            (int(token),),
        ).fetchone()
        if row is None:
            return None
        return int(row["id"])

    def list_jobs(self, *, limit: int = 20) -> list[LlmJobListItem]:
        rows = self._conn.execute(
            """
            SELECT
                id,
                task_name,
                model_name,
                status,
                persisted,
                created_at,
                finished_at
            FROM llm_job
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            LlmJobListItem(
                job_id=int(row["id"]),
                task_name=str(row["task_name"]),
                model_name=str(row["model_name"]),
                status=self._parse_enum(LlmJobStatus, row["status"]),
                persisted=bool(row["persisted"]),
                created_at=str(row["created_at"]),
                finished_at=(str(row["finished_at"]) if row["finished_at"] else None),
            )
            for row in rows
        ]

    def get_job_detail(self, job_id: int) -> LlmJobDetail | None:
        row = self._conn.execute(
            """
            SELECT id, task_name, model_name, api_model, status, persisted,
                   failure_policy, created_at, started_at, finished_at, fatal_error
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
            )
            for item in item_rows
        ]

        return LlmJobDetail(
            job_id=resolved_job_id,
            task_name=str(row["task_name"]),
            model_name=str(row["model_name"]),
            api_model=str(row["api_model"]),
            status=self._parse_enum(LlmJobStatus, row["status"]),
            persisted=bool(row["persisted"]),
            failure_policy=self._parse_enum(RunFailurePolicy, row["failure_policy"]),
            created_at=str(row["created_at"]),
            started_at=str(row["started_at"]),
            finished_at=(str(row["finished_at"]) if row["finished_at"] else None),
            fatal_error=(str(row["fatal_error"]) if row["fatal_error"] else None),
            summary=aggregate.summary,
            items=items,
        )
