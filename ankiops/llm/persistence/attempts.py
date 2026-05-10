"""Attempt recorder for runtime persistence."""

from __future__ import annotations

from typing import Any, Protocol

from ankiops.llm.task_runtime_types import EligibleCandidate
from ankiops.llm.task_types import LlmItemStatus

from ..domain.outcomes import ProviderOutcome, ProviderOutcomeKind
from .db import LlmDb


class AttemptRequestRecord(Protocol):
    system_prompt_text: str
    user_message_text: str
    request_params: dict[str, object]
    contract_fingerprint: str
    transport_mode: str
    capability_snapshot: dict[str, object]


class AttemptRecorder:
    def __init__(
        self,
        *,
        db: LlmDb,
        provider: str,
    ) -> None:
        self._db = db
        self._provider = provider

    def record_success(
        self,
        *,
        candidate: EligibleCandidate,
        prepared_request: AttemptRequestRecord,
        outcome: ProviderOutcome,
    ) -> None:
        if outcome.kind is not ProviderOutcomeKind.SUCCESS or outcome.update is None:
            raise RuntimeError("Successful attempt recording requires success outcome")

        self._write_attempt(
            candidate=candidate,
            prepared_request=prepared_request,
            outcome=outcome,
            outcome_kind=ProviderOutcomeKind.SUCCESS,
            error_message=None,
            parsed_update_json={
                "note_key": outcome.update.note_key,
                "edits": outcome.update.edits,
            },
        )

    def record_error(
        self,
        *,
        candidate: EligibleCandidate,
        prepared_request: AttemptRequestRecord,
        outcome: ProviderOutcome | None,
        error_message: str,
        item_status: LlmItemStatus,
        outcome_kind: ProviderOutcomeKind | None = None,
    ) -> None:
        parsed_update = None
        if outcome is not None and outcome.update is not None:
            parsed_update = {
                "note_key": outcome.update.note_key,
                "edits": outcome.update.edits,
            }

        resolved_kind = (
            outcome_kind
            if outcome_kind is not None
            else outcome.kind
            if outcome is not None
            else _fallback_outcome_kind(item_status)
        )
        self._write_attempt(
            candidate=candidate,
            prepared_request=prepared_request,
            outcome=outcome,
            outcome_kind=resolved_kind,
            error_message=error_message,
            parsed_update_json=parsed_update,
        )
        self._db.update_job_item_status(
            item_id=candidate.item_id,
            item_status=item_status,
            error_message=error_message,
        )

    def _write_attempt(
        self,
        *,
        candidate: EligibleCandidate,
        prepared_request: AttemptRequestRecord,
        outcome: ProviderOutcome | None,
        outcome_kind: ProviderOutcomeKind,
        error_message: str | None,
        parsed_update_json: dict[str, Any] | None,
    ) -> None:
        attempt_id = self._db.insert_attempt(
            item_id=candidate.item_id,
            provider=self._provider,
            outcome_kind=outcome_kind.value,
            transport_mode=prepared_request.transport_mode,
            capability_snapshot_json=prepared_request.capability_snapshot,
            contract_fingerprint=prepared_request.contract_fingerprint,
            refusal_reason=outcome.refusal_text if outcome is not None else None,
            provider_message_id=(
                outcome.provider_message_id if outcome is not None else None
            ),
            response_model_id=(
                outcome.response_model_id if outcome is not None else None
            ),
            provider_request_id=outcome.request_id if outcome is not None else None,
            stop_reason=outcome.stop_reason if outcome is not None else None,
            latency_ms=outcome.latency_ms if outcome is not None else 0,
            input_tokens=outcome.usage.input_tokens if outcome is not None else 0,
            output_tokens=outcome.usage.output_tokens if outcome is not None else 0,
            retry_count=outcome.retry_count if outcome is not None else 0,
            error_message=error_message,
            parsed_update_json=parsed_update_json,
            rate_limit_headers_json=(
                outcome.rate_limit_headers if outcome is not None else None
            ),
        )
        self._db.insert_attempt_payload(
            attempt_id=attempt_id,
            system_prompt_text=prepared_request.system_prompt_text,
            user_message_text=prepared_request.user_message_text,
            request_params_json=prepared_request.request_params,
            response_raw_text=outcome.raw_text if outcome is not None else None,
            response_full_json=outcome.raw_json if outcome is not None else None,
        )


def _fallback_outcome_kind(item_status: LlmItemStatus) -> ProviderOutcomeKind:
    if item_status is LlmItemStatus.FATAL_ERROR:
        return ProviderOutcomeKind.FATAL_ERROR
    if item_status is LlmItemStatus.NOTE_ERROR:
        return ProviderOutcomeKind.VALIDATION_ERROR
    return ProviderOutcomeKind.PROVIDER_ERROR
