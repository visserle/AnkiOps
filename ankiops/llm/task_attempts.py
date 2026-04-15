"""Provider attempt persistence for task execution."""

from __future__ import annotations

from typing import Any

from .llm_db import LlmDb
from .llm_errors import LlmFatalError, LlmNoteError
from .task_runtime_types import EligibleCandidate
from .task_types import (
    LlmItemStatus,
    PreparedAttemptRequest,
    ProviderAttemptErrorContext,
    ProviderAttemptOutcome,
)


class AttemptRecorder:
    def __init__(
        self,
        *,
        db: LlmDb,
        provider: str = "openai",
    ) -> None:
        self._db = db
        self._provider = provider

    def record_success(
        self,
        *,
        candidate: EligibleCandidate,
        prepared_request: PreparedAttemptRequest,
        outcome: ProviderAttemptOutcome | None,
    ) -> None:
        if outcome is None:
            raise RuntimeError("Successful attempt recording requires provider outcome")

        parsed_update = {
            "note_key": outcome.update.note_key,
            "edits": outcome.update.edits,
        }
        self._write_attempt(
            candidate=candidate,
            prepared_request=prepared_request,
            provider_message_id=outcome.provider_message_id,
            response_model_id=outcome.response_model_id,
            provider_request_id=outcome.request_id,
            stop_reason=outcome.stop_reason,
            latency_ms=outcome.latency_ms,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            retry_count=outcome.retry_count,
            error_message=None,
            parsed_update_json=parsed_update,
            rate_limit_headers_json=outcome.rate_limit_headers,
            response_raw_text=outcome.response_raw_text,
            response_full_json=outcome.response_full_json,
        )

    def record_error(
        self,
        *,
        candidate: EligibleCandidate,
        prepared_request: PreparedAttemptRequest,
        outcome: ProviderAttemptOutcome | None,
        error: LlmNoteError | LlmFatalError,
        item_status: LlmItemStatus,
    ) -> None:
        context = _error_context_for_attempt(outcome=outcome, error=error)
        parsed_update = None
        if outcome is not None:
            parsed_update = {
                "note_key": outcome.update.note_key,
                "edits": outcome.update.edits,
            }
        self._write_attempt(
            candidate=candidate,
            prepared_request=prepared_request,
            provider_message_id=(
                context.provider_message_id if context is not None else None
            ),
            response_model_id=(
                context.response_model_id if context is not None else None
            ),
            provider_request_id=context.request_id if context is not None else None,
            stop_reason=context.stop_reason if context is not None else None,
            latency_ms=context.latency_ms if context is not None else 0,
            input_tokens=context.input_tokens if context is not None else 0,
            output_tokens=context.output_tokens if context is not None else 0,
            retry_count=context.retry_count if context is not None else 0,
            error_message=str(error),
            parsed_update_json=parsed_update,
            rate_limit_headers_json=(
                context.rate_limit_headers if context is not None else None
            ),
            response_raw_text=(
                context.response_raw_text if context is not None else None
            ),
            response_full_json=(
                context.response_full_json if context is not None else None
            ),
        )
        self._db.update_job_item_status(
            item_id=candidate.item_id,
            item_status=item_status,
            error_message=str(error),
        )

    def _write_attempt(
        self,
        *,
        candidate: EligibleCandidate,
        prepared_request: PreparedAttemptRequest,
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
        response_raw_text: str | None,
        response_full_json: str | None,
    ) -> None:
        attempt_id = self._db.insert_attempt(
            item_id=candidate.item_id,
            provider=self._provider,
            provider_message_id=provider_message_id,
            response_model_id=response_model_id,
            provider_request_id=provider_request_id,
            stop_reason=stop_reason,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            retry_count=retry_count,
            error_message=error_message,
            parsed_update_json=parsed_update_json,
            rate_limit_headers_json=rate_limit_headers_json,
        )
        self._db.insert_attempt_payload(
            attempt_id=attempt_id,
            system_prompt_text=prepared_request.system_prompt_text,
            user_message_text=prepared_request.user_message_text,
            request_params_json=prepared_request.request_params,
            response_raw_text=response_raw_text,
            response_full_json=response_full_json,
        )


def _error_context_for_attempt(
    *,
    outcome: ProviderAttemptOutcome | None,
    error: LlmNoteError | LlmFatalError,
) -> ProviderAttemptErrorContext | None:
    if outcome is not None:
        return ProviderAttemptErrorContext(
            provider_message_id=outcome.provider_message_id,
            response_model_id=outcome.response_model_id,
            stop_reason=outcome.stop_reason,
            request_id=outcome.request_id,
            rate_limit_headers=outcome.rate_limit_headers,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            latency_ms=outcome.latency_ms,
            retry_count=outcome.retry_count,
            response_raw_text=outcome.response_raw_text,
            response_full_json=outcome.response_full_json,
        )
    if isinstance(error, LlmNoteError):
        return error.attempt_context
    return None
