from __future__ import annotations

import sqlite3

import pytest

from ankiops.llm.task_types import LlmItemStatus, LlmJobStatus
from ankiops.llm_v2.domain.outcomes import ProviderOutcomeKind
from ankiops.llm_v2.persistence.db import LlmDb


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {name for (name,) in rows}


def _insert_attempt_with_payload(
    db: LlmDb,
    *,
    item_id: int,
    response_raw_text: str | None,
    response_full_json: str | None,
) -> int:
    attempt_id = db.insert_attempt(
        item_id=item_id,
        provider="openai",
        outcome_kind=ProviderOutcomeKind.SUCCESS.value,
        transport_mode="openai_responses_structured",
        capability_snapshot_json={
            "provider": "openai",
            "model_id": "gpt-5.4",
            "transport_mode": "openai_responses_structured",
            "supports_strict_json": True,
        },
        contract_fingerprint="contract-fingerprint",
        refusal_reason=None,
        provider_message_id="msg_1",
        response_model_id="gpt-5.4",
        provider_request_id="req_1",
        stop_reason=None,
        latency_ms=15,
        input_tokens=11,
        output_tokens=7,
        retry_count=0,
        error_message=None,
        parsed_update_json={"note_key": "nk-1", "edits": {"Question": "Fixed"}},
        rate_limit_headers_json={"x-ratelimit-remaining-requests": "100"},
    )
    db.insert_attempt_payload(
        attempt_id=attempt_id,
        system_prompt_text="system",
        user_message_text="user",
        request_params_json={"model": "gpt-5.4"},
        response_raw_text=response_raw_text,
        response_full_json=response_full_json,
    )
    return attempt_id


def test_open_creates_v2_schema_and_indexes(tmp_path):
    db = LlmDb.open(tmp_path)
    try:
        tables = _table_names(db._conn)
        assert {
            "llm_job_v2",
            "llm_job_item_v2",
            "llm_attempt_v2",
            "llm_attempt_payload_v2",
        }.issubset(tables)

        indexes = {
            name
            for (name,) in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_llm_job_v2_created" in indexes
        assert "idx_llm_job_item_v2_job" in indexes
        assert "idx_llm_attempt_v2_item" in indexes
    finally:
        db.close()


def test_attempt_payload_raw_fields_default_to_redacted(tmp_path):
    db = LlmDb.open(tmp_path)
    try:
        job_id = db.start_job(task_name="grammar", model="gpt", model_id="gpt-5.4")
        item_id = db.insert_job_item(
            job_id=job_id,
            ordinal=1,
            deck_name="Deck",
            note_key="nk-1",
            note_type="AnkiOpsQA",
            item_status=LlmItemStatus.QUEUED,
            skip_reason=None,
        )
        attempt_id = _insert_attempt_with_payload(
            db,
            item_id=item_id,
            response_raw_text='{"note_key":"nk-1"}',
            response_full_json='{"id":"msg_1"}',
        )

        row = db._conn.execute(
            """
            SELECT response_raw_text, response_full_json
            FROM llm_attempt_payload_v2
            WHERE attempt_id = ?
            """,
            (attempt_id,),
        ).fetchone()
        assert row is not None
        assert row["response_raw_text"] is None
        assert row["response_full_json"] is None
    finally:
        db.close()


def test_attempt_payload_raw_fields_persist_when_enabled(tmp_path):
    db = LlmDb.open(tmp_path, persist_raw_payloads=True)
    try:
        job_id = db.start_job(task_name="grammar", model="gpt", model_id="gpt-5.4")
        item_id = db.insert_job_item(
            job_id=job_id,
            ordinal=1,
            deck_name="Deck",
            note_key="nk-1",
            note_type="AnkiOpsQA",
            item_status=LlmItemStatus.QUEUED,
            skip_reason=None,
        )
        attempt_id = _insert_attempt_with_payload(
            db,
            item_id=item_id,
            response_raw_text='{"note_key":"nk-1"}',
            response_full_json='{"id":"msg_1"}',
        )

        row = db._conn.execute(
            """
            SELECT response_raw_text, response_full_json
            FROM llm_attempt_payload_v2
            WHERE attempt_id = ?
            """,
            (attempt_id,),
        ).fetchone()
        assert row is not None
        assert row["response_raw_text"] == '{"note_key":"nk-1"}'
        assert row["response_full_json"] == '{"id":"msg_1"}'
    finally:
        db.close()


def test_write_tx_rolls_back_attempt_and_payload_v2(tmp_path):
    db = LlmDb.open(tmp_path, persist_raw_payloads=True)
    try:
        job_id = db.start_job(task_name="grammar", model="gpt", model_id="gpt-5.4")
        item_id = db.insert_job_item(
            job_id=job_id,
            ordinal=1,
            deck_name="Deck",
            note_key="nk-1",
            note_type="AnkiOpsQA",
            item_status=LlmItemStatus.QUEUED,
            skip_reason=None,
        )

        with pytest.raises(RuntimeError, match="boom"):
            with db.write_tx():
                _insert_attempt_with_payload(
                    db,
                    item_id=item_id,
                    response_raw_text='{"note_key":"nk-1"}',
                    response_full_json='{"id":"msg_1"}',
                )
                raise RuntimeError("boom")

        attempts = db._conn.execute("SELECT COUNT(*) FROM llm_attempt_v2").fetchone()[0]
        payloads = db._conn.execute(
            "SELECT COUNT(*) FROM llm_attempt_payload_v2"
        ).fetchone()[0]
        assert attempts == 0
        assert payloads == 0
    finally:
        db.close()


def test_attempt_metadata_persists_outcome_and_contract_fields(tmp_path):
    db = LlmDb.open(tmp_path)
    try:
        job_id = db.start_job(task_name="grammar", model="gpt", model_id="gpt-5.4")
        item_id = db.insert_job_item(
            job_id=job_id,
            ordinal=1,
            deck_name="Deck",
            note_key="nk-1",
            note_type="AnkiOpsQA",
            item_status=LlmItemStatus.QUEUED,
            skip_reason=None,
        )
        _ = db.insert_attempt(
            item_id=item_id,
            provider="openai",
            outcome_kind=ProviderOutcomeKind.REFUSAL.value,
            transport_mode="openai_responses_structured",
            capability_snapshot_json={
                "provider": "openai",
                "supports_strict_json": True,
            },
            contract_fingerprint="fingerprint-v1",
            refusal_reason="Safety refusal",
            provider_message_id="msg_1",
            response_model_id="gpt-5.4",
            provider_request_id="req_1",
            stop_reason=None,
            latency_ms=17,
            input_tokens=5,
            output_tokens=2,
            retry_count=0,
            error_message="Provider refused request",
            parsed_update_json=None,
            rate_limit_headers_json=None,
        )
        db.update_job_item_status(
            item_id=item_id,
            item_status=LlmItemStatus.PROVIDER_ERROR,
            error_message="Provider refused request",
        )
        db.finalize_job(
            job_id=job_id,
            status=LlmJobStatus.FAILED,
            persisted=False,
            fatal_error=None,
        )

        row = db._conn.execute(
            """
            SELECT outcome_kind, transport_mode, capability_snapshot_json,
                   contract_fingerprint, refusal_reason, error_message
            FROM llm_attempt_v2
            """
        ).fetchone()
        assert row is not None
        assert row["outcome_kind"] == ProviderOutcomeKind.REFUSAL.value
        assert row["transport_mode"] == "openai_responses_structured"
        assert '"provider":"openai"' in row["capability_snapshot_json"]
        assert row["contract_fingerprint"] == "fingerprint-v1"
        assert row["refusal_reason"] == "Safety refusal"
        assert row["error_message"] == "Provider refused request"
    finally:
        db.close()
