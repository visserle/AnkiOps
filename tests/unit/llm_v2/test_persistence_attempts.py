from __future__ import annotations

import json

from ankiops.llm.task_runtime_types import EligibleCandidate
from ankiops.llm.task_types import (
    LlmItemStatus,
    NotePayload,
)
from ankiops.llm_v2.domain.outcomes import (
    ProviderOutcome,
    ProviderOutcomeKind,
    ProviderUsage,
)
from ankiops.llm_v2.domain.payloads import NoteUpdate
from ankiops.llm_v2.persistence.attempts import AttemptRecorder
from ankiops.llm_v2.persistence.db import LlmDb
from ankiops.llm_v2.runtime.provider import PreparedProviderRequest
from ankiops.models import NoteTypeConfig


def _setup_candidate(db: LlmDb) -> EligibleCandidate:
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
    return EligibleCandidate(
        item_id=item_id,
        deck_name="Deck",
        payload=NotePayload(
            note_key="nk-1",
            note_type="AnkiOpsQA",
            editable_fields={"Question": "Broken"},
            read_only_fields={"Source": "Book"},
        ),
        note_type_config=NoteTypeConfig(name="AnkiOpsQA", fields=[]),
        serialized_note={},
    )


def _prepared_request() -> PreparedProviderRequest:
    return PreparedProviderRequest(
        adapter_request=None,
        system_prompt_text="system",
        user_message_text="user",
        request_params={"model": "gpt-5.4"},
        contract_fingerprint="fingerprint-v1",
        transport_mode="openai_responses",
        capability_snapshot={
            "provider": "openai",
            "model_id": "gpt-5.4",
            "transport_mode": "openai_responses",
            "supports_strict_schema": True,
        },
    )


def _success_outcome() -> ProviderOutcome:
    return ProviderOutcome(
        kind=ProviderOutcomeKind.SUCCESS,
        update=NoteUpdate(note_key="nk-1", edits={"Question": "Fixed"}),
        provider_message_id="msg_1",
        response_model_id="gpt-5.4",
        stop_reason=None,
        request_id="req_1",
        rate_limit_headers={},
        usage=ProviderUsage(input_tokens=7, output_tokens=4),
        latency_ms=20,
        retry_count=0,
        raw_text='{"note_key":"nk-1"}',
        raw_json='{"id":"msg_1"}',
    )


def test_record_success_persists_success_outcome_kind(tmp_path):
    db = LlmDb.open(tmp_path)
    try:
        candidate = _setup_candidate(db)
        recorder = AttemptRecorder(db=db, provider="openai")
        with db.write_tx():
            recorder.record_success(
                candidate=candidate,
                prepared_request=_prepared_request(),
                outcome=_success_outcome(),
            )
            db.update_job_item_status(
                item_id=candidate.item_id,
                item_status=LlmItemStatus.SUCCEEDED_UPDATED,
                changed_fields=["Question"],
            )

        row = db._conn.execute(
            "SELECT outcome_kind, contract_fingerprint FROM llm_attempt_v2"
        ).fetchone()
        assert row is not None
        assert row["outcome_kind"] == ProviderOutcomeKind.SUCCESS.value
        assert row["contract_fingerprint"] == "fingerprint-v1"
    finally:
        db.close()


def test_record_error_persists_refusal_outcome_kind(tmp_path):
    db = LlmDb.open(tmp_path)
    try:
        candidate = _setup_candidate(db)
        recorder = AttemptRecorder(db=db, provider="openai")
        refusal = ProviderOutcome(
            kind=ProviderOutcomeKind.REFUSAL,
            refusal_text="Safety refusal",
            provider_message_id="msg_1",
            response_model_id="gpt-5.4",
            request_id="req_1",
            usage=ProviderUsage(input_tokens=3, output_tokens=1),
            latency_ms=10,
            raw_json='{"id":"msg_1"}',
        )

        with db.write_tx():
            recorder.record_error(
                candidate=candidate,
                prepared_request=_prepared_request(),
                outcome=refusal,
                error_message="Provider refused request",
                item_status=LlmItemStatus.PROVIDER_ERROR,
            )

        row = db._conn.execute(
            "SELECT outcome_kind, refusal_reason FROM llm_attempt_v2"
        ).fetchone()
        assert row is not None
        assert row["outcome_kind"] == ProviderOutcomeKind.REFUSAL.value
        assert row["refusal_reason"] == "Safety refusal"
    finally:
        db.close()


def test_record_error_maps_post_parse_note_error_to_validation_outcome_kind(tmp_path):
    db = LlmDb.open(tmp_path)
    try:
        candidate = _setup_candidate(db)
        recorder = AttemptRecorder(db=db, provider="openai")
        with db.write_tx():
            recorder.record_error(
                candidate=candidate,
                prepared_request=_prepared_request(),
                outcome=_success_outcome(),
                error_message="Model attempted read-only edit",
                item_status=LlmItemStatus.NOTE_ERROR,
                outcome_kind=ProviderOutcomeKind.VALIDATION_ERROR,
            )

        row = db._conn.execute(
            "SELECT outcome_kind, parsed_update_json FROM llm_attempt_v2"
        ).fetchone()
        assert row is not None
        assert row["outcome_kind"] == ProviderOutcomeKind.VALIDATION_ERROR.value
        assert json.loads(row["parsed_update_json"])["note_key"] == "nk-1"
    finally:
        db.close()


def test_record_error_persists_fatal_outcome_kind_from_context(tmp_path):
    db = LlmDb.open(tmp_path)
    try:
        candidate = _setup_candidate(db)
        recorder = AttemptRecorder(db=db, provider="openai")
        with db.write_tx():
            recorder.record_error(
                candidate=candidate,
                prepared_request=_prepared_request(),
                outcome=None,
                error_message="Provider authentication failed",
                item_status=LlmItemStatus.FATAL_ERROR,
                outcome_kind=ProviderOutcomeKind.FATAL_ERROR,
            )

        row = db._conn.execute(
            "SELECT outcome_kind, error_message FROM llm_attempt_v2"
        ).fetchone()
        assert row is not None
        assert row["outcome_kind"] == ProviderOutcomeKind.FATAL_ERROR.value
        assert row["error_message"] == "Provider authentication failed"
    finally:
        db.close()
