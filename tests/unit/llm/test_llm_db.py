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


def test_write_tx_rolls_back_request(tmp_path):
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
                db.insert_request(
                    job_id=job_id,
                    item_ids=[item_id],
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

        row = db._conn.execute("SELECT COUNT(*) AS total FROM llm_request").fetchone()
        link_row = db._conn.execute(
            "SELECT COUNT(*) AS total FROM llm_request_item"
        ).fetchone()
        assert row is not None
        assert link_row is not None
        assert int(row["total"]) == 0
        assert int(link_row["total"]) == 0
    finally:
        db.close()


def test_get_job_detail_reports_usage_at_request_level(tmp_path):
    db = LlmDb.open(tmp_path)
    try:
        job_id = _start_job(db)
        first_id = db.insert_job_item(
            job_id=job_id,
            ordinal=1,
            deck_name="Deck",
            note_key="nk-1",
            note_type="AnkiOpsQA",
            item_status=LlmItemStatus.SUCCEEDED_UPDATED,
            skip_reason=None,
            changed_fields=["Question"],
        )
        second_id = db.insert_job_item(
            job_id=job_id,
            ordinal=2,
            deck_name="Deck",
            note_key="nk-2",
            note_type="AnkiOpsQA",
            item_status=LlmItemStatus.SUCCEEDED_UNCHANGED,
            skip_reason=None,
        )
        request_id = db.insert_request(
            job_id=job_id,
            item_ids=[second_id, first_id],
            outcome="success",
            request_json={"model": "gpt-test"},
            parsed_response_json={"updates": []},
            response_json="{}",
            error_message=None,
            latency_ms=101,
            input_tokens=5,
            output_tokens=2,
        )

        detail = db.get_job_detail(job_id)
    finally:
        db.close()

    assert detail is not None
    assert [(item.ordinal, item.request_count) for item in detail.items] == [
        (1, 1),
        (2, 1),
    ]
    assert not hasattr(detail.items[0], "input_tokens")
    assert not hasattr(detail.items[0], "latency_ms")
    assert detail.summary.requests == 1
    assert detail.summary.input_tokens == 5
    assert detail.summary.output_tokens == 2
    assert detail.summary.provider_latency_ms_total == 101
    assert len(detail.requests) == 1
    request = detail.requests[0]
    assert request.request_id == request_id
    assert request.outcome == "success"
    assert request.input_tokens == 5
    assert request.output_tokens == 2
    assert request.latency_ms == 101
    assert [(note.ordinal, note.note_key) for note in request.notes] == [
        (1, "nk-1"),
        (2, "nk-2"),
    ]


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
