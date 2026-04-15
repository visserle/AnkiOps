from __future__ import annotations

import sqlite3
from importlib import resources

import pytest

from ankiops.llm.llm_db import LlmDb
from ankiops.llm.task_types import (
    LlmAttemptResultType,
    LlmCandidateStatus,
    LlmFinalStatus,
    LlmJobStatus,
)


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {name for (name,) in rows}


def _index_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
    return {name for (name,) in rows}


def _start_job(adapter: LlmDb) -> int:
    return adapter.start_job(
        task_name="grammar",
        model="sonnet",
        model_id="sonnet",
        config_snapshot={"task": "grammar"},
    )


def _insert_attempt(
    adapter: LlmDb,
    *,
    item_id: int,
    provider_message_id: str,
    response_model_id: str,
    stop_reason: str,
    result_type: LlmAttemptResultType,
    latency_ms: int,
    input_tokens: int,
    output_tokens: int,
    retry_count: int,
    error_type: str | None,
    error_message: str | None,
    parsed_update_json: dict[str, object] | None,
) -> int:
    return adapter.insert_attempt(
        item_id=item_id,
        attempt_no=1,
        provider="anthropic",
        provider_message_id=provider_message_id,
        response_model_id=response_model_id,
        provider_request_id=None,
        stop_reason=stop_reason,
        result_type=result_type,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        retry_count=retry_count,
        error_type=error_type,
        error_message=error_message,
        parsed_update_json=parsed_update_json,
        rate_limit_headers_json=None,
    )


def _write_models_registry(tmp_path) -> None:
    models_path = tmp_path / "llm" / "_models.yaml"
    models_path.parent.mkdir(parents=True, exist_ok=True)
    models_path.write_text(
        resources.files("ankiops.llm")
        .joinpath("_models.yaml")
        .read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def test_open_creates_schema_and_indexes(tmp_path):
    adapter = LlmDb.open(tmp_path)
    try:
        assert adapter.db_path == tmp_path / "llm" / ".llm.db"
        assert adapter.db_path.exists()

        tables = _table_names(adapter._conn)
        assert {
            "llm_job",
            "llm_job_item",
            "llm_item_attempt",
            "llm_attempt_payload",
        }.issubset(tables)
        assert "llm_schema_meta" not in tables

        indexes = _index_names(adapter._conn)
        assert "idx_llm_job_created" in indexes
        assert "idx_llm_job_item_job" in indexes
        assert "idx_llm_job_item_note_key" in indexes
        assert "idx_llm_attempt_result" in indexes

        job_columns = [
            row[1] for row in adapter._conn.execute("PRAGMA table_info(llm_job)")
        ]
        assert set(job_columns) == {
            "id",
            "task_name",
            "model",
            "model_id",
            "status",
            "persisted",
            "fatal_error",
            "config_snapshot_json",
            "created_at",
            "started_at",
            "finished_at",
            "decks_seen",
            "decks_matched",
            "notes_seen",
        }

        job_sql = adapter._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='llm_job'"
        ).fetchone()[0]
        assert "CHECK (status IN ('running', 'completed', 'failed'))" in job_sql
    finally:
        adapter.close()


def test_roundtrip_job_item_attempt_payload(tmp_path):
    _write_models_registry(tmp_path)
    adapter = LlmDb.open(tmp_path)
    try:
        job_id = _start_job(adapter)
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
        attempt_id = _insert_attempt(
            adapter,
            item_id=item_id,
            provider_message_id="msg_123",
            response_model_id="sonnet",
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
            request_params_json={"model": "sonnet"},
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
        assert payload_row["request_params_json"] == '{"model":"sonnet"}'
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

    reopened = LlmDb.open(tmp_path)
    try:
        detail = reopened.get_job_detail(job_id)
        assert detail is not None
        assert detail.items[0].attempts == 1
    finally:
        reopened.close()


def test_open_fails_for_incompatible_schema(tmp_path):
    llm_dir = tmp_path / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = llm_dir / ".llm.db"
    conn = sqlite3.connect(legacy_path)
    try:
        conn.execute("CREATE TABLE llm_job (id INTEGER PRIMARY KEY, status TEXT)")
        conn.execute("INSERT INTO llm_job (id, status) VALUES (1, 'legacy')")
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(
        RuntimeError,
        match="LLM DB schema is incompatible with this build",
    ):
        _ = LlmDb.open(tmp_path)


def test_enforces_uniqueness_constraints(tmp_path):
    adapter = LlmDb.open(tmp_path)
    try:
        job_id = _start_job(adapter)
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

        _insert_attempt(
            adapter,
            item_id=item_id,
            provider_message_id="msg",
            response_model_id="sonnet",
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
            _insert_attempt(
                adapter,
                item_id=item_id,
                provider_message_id="msg2",
                response_model_id="sonnet",
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
    adapter = LlmDb.open(tmp_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            adapter._conn.execute(
                """
                INSERT INTO llm_job (
                    task_name, model, model_id,
                    status, persisted, config_snapshot_json, created_at, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "grammar",
                    "sonnet",
                    "sonnet",
                    "not_a_status",
                    0,
                    "{}",
                    "2026-03-21T10:00:00Z",
                    "2026-03-21T10:00:00Z",
                ),
            )
    finally:
        adapter.close()


def test_resolve_job_id_accepts_numeric_and_latest(tmp_path):
    adapter = LlmDb.open(tmp_path)
    try:
        job_id = _start_job(adapter)
        assert adapter.resolve_job_id(str(job_id)) == job_id
        assert adapter.resolve_job_id("latest") == job_id
    finally:
        adapter.close()


def test_resolve_job_id_rejects_non_numeric_lookup(tmp_path):
    adapter = LlmDb.open(tmp_path)
    try:
        _ = _start_job(adapter)
        with pytest.raises(ValueError, match="Job ID must be numeric"):
            adapter.resolve_job_id("abc123")
    finally:
        adapter.close()


def test_resolve_job_id_returns_none_for_missing_numeric_id(tmp_path):
    adapter = LlmDb.open(tmp_path)
    try:
        _ = _start_job(adapter)
        assert adapter.resolve_job_id("999999") is None
    finally:
        adapter.close()


def test_write_tx_rolls_back_partial_attempt_persistence(tmp_path):
    adapter = LlmDb.open(tmp_path)
    try:
        job_id = _start_job(adapter)
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
                attempt_id = _insert_attempt(
                    adapter,
                    item_id=item_id,
                    provider_message_id="msg",
                    response_model_id="sonnet",
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
                    request_params_json={"model": "sonnet"},
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


def test_mark_unfinished_items_canceled_only_updates_pending_eligible(tmp_path):
    adapter = LlmDb.open(tmp_path)
    try:
        job_id = _start_job(adapter)
        pending_id = adapter.insert_job_item(
            job_id=job_id,
            ordinal=1,
            deck_name="Deck",
            note_key="nk-1",
            note_type="AnkiOpsQA",
            candidate_status=LlmCandidateStatus.ELIGIBLE,
            skip_reason=None,
            final_status=LlmFinalStatus.NOT_ATTEMPTED,
        )
        skipped_id = adapter.insert_job_item(
            job_id=job_id,
            ordinal=2,
            deck_name="Deck",
            note_key="nk-skip",
            note_type="AnkiOpsChoice",
            candidate_status=LlmCandidateStatus.SKIPPED_NO_EDITABLE_FIELDS,
            skip_reason="No editable non-empty fields",
            final_status=LlmFinalStatus.NOT_ATTEMPTED,
        )
        settled_id = adapter.insert_job_item(
            job_id=job_id,
            ordinal=3,
            deck_name="Deck",
            note_key="nk-3",
            note_type="AnkiOpsQA",
            candidate_status=LlmCandidateStatus.ELIGIBLE,
            skip_reason=None,
            final_status=LlmFinalStatus.SUCCEEDED_UNCHANGED,
        )

        updated = adapter.mark_unfinished_items_canceled(job_id=job_id)
        assert updated == 1

        rows = adapter._conn.execute(
            """
            SELECT id, candidate_status, final_status
            FROM llm_job_item
            WHERE job_id = ?
            ORDER BY ordinal ASC
            """,
            (job_id,),
        ).fetchall()
    finally:
        adapter.close()

    assert len(rows) == 3
    assert int(rows[0]["id"]) == pending_id
    assert rows[0]["candidate_status"] == LlmCandidateStatus.ELIGIBLE.value
    assert rows[0]["final_status"] == LlmFinalStatus.CANCELED.value
    assert int(rows[1]["id"]) == skipped_id
    skipped_candidate_status = LlmCandidateStatus.SKIPPED_NO_EDITABLE_FIELDS.value
    assert rows[1]["candidate_status"] == skipped_candidate_status
    assert rows[1]["final_status"] == LlmFinalStatus.NOT_ATTEMPTED.value
    assert int(rows[2]["id"]) == settled_id
    assert rows[2]["candidate_status"] == LlmCandidateStatus.ELIGIBLE.value
    assert rows[2]["final_status"] == LlmFinalStatus.SUCCEEDED_UNCHANGED.value
