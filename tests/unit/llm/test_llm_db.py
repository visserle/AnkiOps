"""Tests for LLM job history persistence invariants."""

from __future__ import annotations

import pytest

from ankiops.llm.llm_db import LlmDb
from ankiops.llm.types import LlmItemStatus


def _start_job(db: LlmDb) -> int:
    return db.start_job(task_name="grammar", model="test", model_id="gpt-test")


def test_resolve_job_id_accepts_numeric_and_latest(tmp_path):
    db = LlmDb.open(tmp_path)
    try:
        job_id = _start_job(db)

        assert db.resolve_job_id(str(job_id)) == job_id
        assert db.resolve_job_id("latest") == job_id
        assert db.resolve_job_id("999999") is None
        with pytest.raises(ValueError, match="Job ID must be numeric"):
            db.resolve_job_id("abc")
    finally:
        db.close()


def test_write_tx_rolls_back_attempt(tmp_path):
    db = LlmDb.open(tmp_path)
    try:
        job_id = _start_job(db)
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
                db.insert_attempt(
                    item_id=item_id,
                    outcome="provider_error",
                    request_json={"model": "gpt-test"},
                    parsed_response_json=None,
                    response_json=None,
                    error_message="broken",
                    latency_ms=1,
                    input_tokens=2,
                    output_tokens=3,
                )
                raise RuntimeError("boom")

        row = db._conn.execute("SELECT COUNT(*) AS total FROM llm_attempt").fetchone()
        assert row is not None
        assert int(row["total"]) == 0
    finally:
        db.close()


def test_mark_unfinished_items_canceled_only_updates_queued(tmp_path):
    db = LlmDb.open(tmp_path)
    try:
        job_id = _start_job(db)
        queued_id = db.insert_job_item(
            job_id=job_id,
            ordinal=1,
            deck_name="Deck",
            note_key="nk-queued",
            note_type="AnkiOpsQA",
            item_status=LlmItemStatus.QUEUED,
            skip_reason=None,
        )
        skipped_id = db.insert_job_item(
            job_id=job_id,
            ordinal=2,
            deck_name="Deck",
            note_key="nk-skipped",
            note_type="AnkiOpsQA",
            item_status=LlmItemStatus.SKIPPED_NO_EDITABLE_FIELDS,
            skip_reason="no editable fields",
        )
        settled_id = db.insert_job_item(
            job_id=job_id,
            ordinal=3,
            deck_name="Deck",
            note_key="nk-settled",
            note_type="AnkiOpsQA",
            item_status=LlmItemStatus.SUCCEEDED_UNCHANGED,
            skip_reason=None,
        )

        assert db.mark_unfinished_items_canceled(job_id=job_id) == 1
        rows = db._conn.execute(
            """
            SELECT id, item_status
            FROM llm_job_item
            WHERE job_id = ?
            ORDER BY ordinal ASC
            """,
            (job_id,),
        ).fetchall()
    finally:
        db.close()

    assert [(int(row["id"]), row["item_status"]) for row in rows] == [
        (queued_id, LlmItemStatus.CANCELED.value),
        (skipped_id, LlmItemStatus.SKIPPED_NO_EDITABLE_FIELDS.value),
        (settled_id, LlmItemStatus.SUCCEEDED_UNCHANGED.value),
    ]
