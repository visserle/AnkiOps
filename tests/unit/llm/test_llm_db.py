from __future__ import annotations

import sqlite3

import pytest

from ankiops.llm.db import LlmDbAdapter
from ankiops.llm.models import (
    LlmAttemptResultType,
    LlmCandidateStatus,
    LlmFinalStatus,
    LlmJobStatus,
    RunFailurePolicy,
)


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {name for (name,) in rows}


def _index_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()
    return {name for (name,) in rows}


def test_open_creates_schema_and_indexes(tmp_path):
    adapter = LlmDbAdapter.open(tmp_path)
    try:
        assert adapter.db_path == tmp_path / "llm" / "llm.db"
        assert adapter.db_path.exists()

        tables = _table_names(adapter._conn)
        assert {
            "llm_job",
            "llm_job_item",
            "llm_item_attempt",
            "llm_attempt_payload",
            "llm_provider_batch",
            "llm_batch_item_map",
        }.issubset(tables)
        assert "llm_schema_meta" not in tables

        indexes = _index_names(adapter._conn)
        assert "idx_llm_job_created" in indexes
        assert "idx_llm_job_item_job" in indexes
        assert "idx_llm_job_item_note_key" in indexes
        assert "idx_llm_attempt_result" in indexes
        assert "idx_llm_provider_batch_status" in indexes

        job_columns = [
            row[1] for row in adapter._conn.execute("PRAGMA table_info(llm_job)")
        ]
        assert "eligible" not in job_columns
        assert "requests" not in job_columns
        assert "notes_seen" in job_columns

        job_sql = adapter._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='llm_job'"
        ).fetchone()[0]
        assert "CHECK (status IN ('running', 'completed', 'failed'))" in job_sql
        assert "CHECK (failure_policy IN ('atomic', 'partial'))" in job_sql
    finally:
        adapter.close()


def test_roundtrip_job_item_attempt_payload(tmp_path):
    adapter = LlmDbAdapter.open(tmp_path)
    try:
        job_id = adapter.start_job(
            task_name="grammar",
            model_name="sonnet",
            api_model="claude-sonnet-4-6",
            failure_policy=RunFailurePolicy.ATOMIC,
        )
        adapter.set_discovery_counts(
            job_id=job_id,
            decks_seen=1,
            decks_matched=1,
            notes_seen=1,
        )
        item_id = adapter.insert_job_item(
            job_id=job_id,
            ordinal=1,
            deck_name="Deck",
            note_key="nk-1",
            note_type="AnkiOpsQA",
            candidate_status=LlmCandidateStatus.ELIGIBLE,
            skip_reason=None,
            final_status=LlmFinalStatus.NOT_ATTEMPTED,
        )
        attempt_id = adapter.insert_attempt(
            item_id=item_id,
            attempt_no=1,
            provider="anthropic",
            provider_message_id="msg_123",
            provider_model="claude-sonnet-4-6",
            stop_reason="end_turn",
            result_type=LlmAttemptResultType.SUCCEEDED,
            latency_ms=901,
            input_tokens=11,
            output_tokens=7,
            retry_count=0,
            error_type=None,
            error_message=None,
            parsed_update_json={"note_key": "nk-1", "edits": {"Question": "fixed"}},
        )
        adapter.insert_attempt_payload(
            attempt_id=attempt_id,
            system_prompt_text="system",
            user_message_text="user",
            request_params_json={"model": "claude-sonnet-4-6"},
            response_raw_text='{"note_key":"nk-1","edits":{"Question":"fixed"}}',
            response_full_json='{"id":"msg_123"}',
        )
        adapter.update_job_item_result(
            item_id=item_id,
            final_status=LlmFinalStatus.SUCCEEDED_UPDATED,
            changed_fields=["Question"],
        )
        adapter.set_applied_for_updated_items(job_id=job_id)
        adapter.finalize_job(
            job_id=job_id,
            status=LlmJobStatus.COMPLETED,
            persisted=True,
        )

        payload_row = adapter._conn.execute(
            """
            SELECT request_params_json, response_raw_text, response_full_json
            FROM llm_attempt_payload
            WHERE attempt_id = ?
            """,
            (attempt_id,),
        ).fetchone()
        assert payload_row is not None
        assert payload_row["request_params_json"] == '{"model":"claude-sonnet-4-6"}'
        assert (
            payload_row["response_raw_text"]
            == '{"note_key":"nk-1","edits":{"Question":"fixed"}}'
        )
        assert payload_row["response_full_json"] == '{"id":"msg_123"}'

        detail = adapter.get_job_detail(job_id)
        assert detail is not None
        assert detail.job_id == job_id
        assert detail.status is LlmJobStatus.COMPLETED
        assert detail.persisted is True
        assert detail.summary.updated == 1
        assert detail.summary.errors == 0
        assert detail.summary.requests == 1
        assert detail.summary.input_tokens == 11
        assert detail.summary.output_tokens == 7
        assert len(detail.items) == 1
        assert detail.items[0].changed_fields == ["Question"]
        assert detail.items[0].candidate_status is LlmCandidateStatus.ELIGIBLE
        assert detail.items[0].final_status is LlmFinalStatus.SUCCEEDED_UPDATED
    finally:
        adapter.close()

    reopened = LlmDbAdapter.open(tmp_path)
    try:
        detail = reopened.get_job_detail(job_id)
        assert detail is not None
        assert detail.items[0].attempts == 1
    finally:
        reopened.close()


def test_open_recreates_invalid_schema(tmp_path):
    llm_dir = tmp_path / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = llm_dir / "llm.db"
    conn = sqlite3.connect(legacy_path)
    try:
        conn.execute("CREATE TABLE llm_job (id INTEGER PRIMARY KEY, status TEXT)")
        conn.execute("INSERT INTO llm_job (id, status) VALUES (1, 'legacy')")
        conn.commit()
    finally:
        conn.close()

    adapter = LlmDbAdapter.open(tmp_path)
    try:
        tables = _table_names(adapter._conn)
        assert "llm_job_item" in tables
        assert "llm_schema_meta" not in tables
        assert adapter.list_jobs() == []
    finally:
        adapter.close()


def test_enforces_uniqueness_constraints(tmp_path):
    adapter = LlmDbAdapter.open(tmp_path)
    try:
        job_id = adapter.start_job(
            task_name="grammar",
            model_name="sonnet",
            api_model="claude-sonnet-4-6",
            failure_policy=RunFailurePolicy.ATOMIC,
        )
        item_id = adapter.insert_job_item(
            job_id=job_id,
            ordinal=1,
            deck_name="Deck",
            note_key="nk-1",
            note_type="AnkiOpsQA",
            candidate_status=LlmCandidateStatus.ELIGIBLE,
            skip_reason=None,
            final_status=LlmFinalStatus.NOT_ATTEMPTED,
        )

        with pytest.raises(sqlite3.IntegrityError):
            adapter.insert_job_item(
                job_id=job_id,
                ordinal=1,
                deck_name="Deck",
                note_key="nk-2",
                note_type="AnkiOpsQA",
                candidate_status=LlmCandidateStatus.ELIGIBLE,
                skip_reason=None,
                final_status=LlmFinalStatus.NOT_ATTEMPTED,
            )

        adapter.insert_attempt(
            item_id=item_id,
            attempt_no=1,
            provider="anthropic",
            provider_message_id="msg",
            provider_model="claude-sonnet-4-6",
            stop_reason="end_turn",
            result_type=LlmAttemptResultType.SUCCEEDED,
            latency_ms=1,
            input_tokens=1,
            output_tokens=1,
            retry_count=0,
            error_type=None,
            error_message=None,
            parsed_update_json=None,
        )
        with pytest.raises(sqlite3.IntegrityError):
            adapter.insert_attempt(
                item_id=item_id,
                attempt_no=1,
                provider="anthropic",
                provider_message_id="msg2",
                provider_model="claude-sonnet-4-6",
                stop_reason="end_turn",
                result_type=LlmAttemptResultType.SUCCEEDED,
                latency_ms=1,
                input_tokens=1,
                output_tokens=1,
                retry_count=0,
                error_type=None,
                error_message=None,
                parsed_update_json=None,
            )
    finally:
        adapter.close()


def test_enforces_status_constraints(tmp_path):
    adapter = LlmDbAdapter.open(tmp_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            adapter._conn.execute(
                """
                INSERT INTO llm_job (
                    task_name, model_name, api_model, failure_policy,
                    status, persisted, created_at, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "grammar",
                    "sonnet",
                    "claude-sonnet-4-6",
                    "atomic",
                    "not_a_status",
                    0,
                    "2026-03-21T10:00:00Z",
                    "2026-03-21T10:00:00Z",
                ),
            )
    finally:
        adapter.close()


def test_resolve_job_id_accepts_numeric_latest_and_minus_one_alias(tmp_path):
    adapter = LlmDbAdapter.open(tmp_path)
    try:
        job_id = adapter.start_job(
            task_name="grammar",
            model_name="sonnet",
            api_model="claude-sonnet-4-6",
            failure_policy=RunFailurePolicy.ATOMIC,
        )
        assert adapter.resolve_job_id(str(job_id)) == job_id
        assert adapter.resolve_job_id("latest") == job_id
        assert adapter.resolve_job_id("-1") == job_id
    finally:
        adapter.close()


def test_resolve_job_id_rejects_non_numeric_lookup(tmp_path):
    adapter = LlmDbAdapter.open(tmp_path)
    try:
        _ = adapter.start_job(
            task_name="grammar",
            model_name="sonnet",
            api_model="claude-sonnet-4-6",
            failure_policy=RunFailurePolicy.ATOMIC,
        )
        with pytest.raises(ValueError, match="Job ID must be numeric"):
            adapter.resolve_job_id("abc123")
    finally:
        adapter.close()


def test_resolve_job_id_returns_none_for_missing_numeric_id(tmp_path):
    adapter = LlmDbAdapter.open(tmp_path)
    try:
        _ = adapter.start_job(
            task_name="grammar",
            model_name="sonnet",
            api_model="claude-sonnet-4-6",
            failure_policy=RunFailurePolicy.ATOMIC,
        )
        assert adapter.resolve_job_id("999999") is None
    finally:
        adapter.close()


def test_write_tx_rolls_back_partial_attempt_persistence(tmp_path):
    adapter = LlmDbAdapter.open(tmp_path)
    try:
        job_id = adapter.start_job(
            task_name="grammar",
            model_name="sonnet",
            api_model="claude-sonnet-4-6",
            failure_policy=RunFailurePolicy.ATOMIC,
        )
        item_id = adapter.insert_job_item(
            job_id=job_id,
            ordinal=1,
            deck_name="Deck",
            note_key="nk-1",
            note_type="AnkiOpsQA",
            candidate_status=LlmCandidateStatus.ELIGIBLE,
            skip_reason=None,
            final_status=LlmFinalStatus.NOT_ATTEMPTED,
        )

        with pytest.raises(RuntimeError, match="boom"):
            with adapter.write_tx():
                attempt_id = adapter.insert_attempt(
                    item_id=item_id,
                    attempt_no=1,
                    provider="anthropic",
                    provider_message_id="msg",
                    provider_model="claude-sonnet-4-6",
                    stop_reason="end_turn",
                    result_type=LlmAttemptResultType.ERRORED,
                    latency_ms=1,
                    input_tokens=1,
                    output_tokens=1,
                    retry_count=0,
                    error_type="note_error",
                    error_message="broken",
                    parsed_update_json=None,
                )
                adapter.insert_attempt_payload(
                    attempt_id=attempt_id,
                    system_prompt_text="system",
                    user_message_text="user",
                    request_params_json={"model": "claude-sonnet-4-6"},
                    response_raw_text=None,
                    response_full_json=None,
                )
                raise RuntimeError("boom")

        attempts = adapter._conn.execute(
            "SELECT COUNT(*) FROM llm_item_attempt"
        ).fetchone()[0]
        payloads = adapter._conn.execute(
            "SELECT COUNT(*) FROM llm_attempt_payload"
        ).fetchone()[0]
        assert attempts == 0
        assert payloads == 0
    finally:
        adapter.close()
