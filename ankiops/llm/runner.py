"""Execution of collection-local Claude tasks."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ankiops.collection_serializer import (
    deserialize_collection_data,
    serialize_collection,
)
from ankiops.config import NOTE_TYPES_DIR
from ankiops.fs import FileSystemAdapter
from ankiops.git import git_snapshot
from ankiops.models import ANKIOPS_KEY_FIELD, Note, NoteTypeConfig

from .anthropic_models import format_supported_model_names, parse_model
from .claude import ClaudeClient
from .config_loader import load_llm_task_catalog
from .db import LlmDbAdapter, LlmJobDetail, LlmJobListItem
from .errors import LlmFatalError, LlmNoteError
from .models import (
    FieldAccess,
    LlmAttemptResultType,
    LlmCandidateStatus,
    LlmFinalStatus,
    LlmJobResult,
    LlmJobStatus,
    NotePayload,
    PreparedAttemptRequest,
    ProviderAttemptErrorContext,
    ProviderAttemptOutcome,
    RunFailurePolicy,
    TaskConfig,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _EligibleCandidate:
    item_id: int
    deck_name: str
    payload: NotePayload
    note_type_config: NoteTypeConfig
    serialized_note: dict[str, Any]


def _resolve_serializer_scope(task: TaskConfig) -> tuple[str | None, bool]:
    include = task.decks.include
    if task.decks.exclude or len(include) != 1:
        return None, False

    deck_name = include[0].strip()
    if not deck_name:
        return None, False

    if any(char in deck_name for char in ("*", "?", "[")):
        return None, False

    return deck_name, not task.decks.include_subdecks


def _format_deck_scope(task: TaskConfig) -> str:
    include = ",".join(task.decks.include)
    exclude = ",".join(task.decks.exclude) if task.decks.exclude else "-"
    if include == "*" and exclude == "-" and task.decks.include_subdecks:
        return "*"
    return (
        f"include={include} exclude={exclude} "
        f"include_subdecks={str(task.decks.include_subdecks).lower()}"
    )


def _format_serializer_scope(deck: str | None, no_subdecks: bool) -> str:
    if deck is None:
        return "*"
    if no_subdecks:
        return f"exact:{deck}"
    return deck


def _format_request_defaults(task: TaskConfig) -> str:
    max_tokens = task.request.max_output_tokens or 2048
    temperature = (
        task.request.temperature if task.request.temperature is not None else "default"
    )
    return (
        f"timeout={task.timeout_seconds}s "
        f"max_tokens={max_tokens} temperature={temperature} "
        f"retries={task.request.retries} "
        f"retry_backoff={task.request.retry_backoff_seconds}s "
        f"retry_jitter={str(task.request.retry_backoff_jitter).lower()}"
    )


def _resolve_model(task: TaskConfig, model_override: str | None):
    if model_override is None:
        return task.model

    model = parse_model(model_override)
    if model is None:
        supported = format_supported_model_names()
        raise ValueError(f"Model must be one of: {supported}")
    return model


def _resolve_failure_policy(
    value: RunFailurePolicy | str,
) -> RunFailurePolicy:
    if isinstance(value, RunFailurePolicy):
        return value

    normalized = value.strip().lower()
    for policy in RunFailurePolicy:
        if policy.value == normalized:
            return policy
    supported = ", ".join(policy.value for policy in RunFailurePolicy)
    raise ValueError(f"Failure policy must be one of: {supported}")


def _build_note_payload(
    task: TaskConfig,
    *,
    note: dict[str, Any],
    note_type_field_names: set[str],
) -> NotePayload | None:
    note_key = note.get("note_key")
    note_type = note.get("note_type")
    fields = note.get("fields")
    if (
        not isinstance(note_key, str)
        or not isinstance(note_type, str)
        or not isinstance(fields, dict)
    ):
        raise ValueError("Serialized note is missing note_key, note_type, or fields")

    editable_fields: dict[str, str] = {}
    read_only_fields: dict[str, str] = {}

    for field_name, raw_value in fields.items():
        if field_name == ANKIOPS_KEY_FIELD.name:
            continue
        if field_name not in note_type_field_names:
            continue
        if not isinstance(raw_value, str) or not raw_value:
            continue

        access = task.field_access(note_type, field_name)
        if access is FieldAccess.HIDDEN:
            continue
        if access is FieldAccess.READ_ONLY:
            read_only_fields[field_name] = raw_value
        else:
            editable_fields[field_name] = raw_value

    if not editable_fields:
        return None

    return NotePayload(
        note_key=note_key,
        note_type=note_type,
        editable_fields=editable_fields,
        read_only_fields=read_only_fields,
    )


def _apply_note_update(
    *,
    serialized_note: dict[str, Any],
    payload: NotePayload,
    edits: dict[str, str],
    note_type_config: NoteTypeConfig,
) -> list[str]:
    if payload.note_key != serialized_note.get("note_key"):
        raise LlmNoteError("Update note_key did not match serialized note")

    raw_fields = serialized_note.get("fields")
    if not isinstance(raw_fields, dict):
        raise LlmNoteError("Serialized note fields must be a mapping")

    next_fields = dict(raw_fields)
    changed_fields: list[str] = []
    for field_name, value in edits.items():
        if field_name not in payload.editable_fields:
            if field_name in payload.read_only_fields:
                raise LlmNoteError(
                    f"Model attempted to update read-only field '{field_name}'"
                )
            raise LlmNoteError(
                f"Model attempted to update hidden or unknown field '{field_name}'"
            )
        if next_fields.get(field_name) != value:
            next_fields[field_name] = value
            changed_fields.append(field_name)

    if not changed_fields:
        return []

    note = Note(
        note_key=payload.note_key,
        note_type=payload.note_type,
        fields=next_fields,
    )
    errors = note.validate(note_type_config)
    if errors:
        raise LlmNoteError("; ".join(errors))

    serialized_note["fields"] = next_fields
    return changed_fields


def _load_task(
    *,
    collection_dir: Path,
    task_name: str,
) -> tuple[TaskConfig, dict[str, NoteTypeConfig]]:
    fs = FileSystemAdapter()
    note_type_configs = fs.load_note_type_configs(collection_dir / NOTE_TYPES_DIR)
    catalog = load_llm_task_catalog(
        collection_dir,
        note_type_configs=note_type_configs,
    )
    if catalog.errors:
        joined_errors = "\n".join(catalog.errors.values())
        raise ValueError(f"Invalid LLM task configuration:\n{joined_errors}")
    task = catalog.tasks_by_name.get(task_name)
    if task is None:
        raise ValueError(f"Unknown or invalid task '{task_name}'")

    config_by_name = {config.name: config for config in note_type_configs}
    return task, config_by_name


def _iter_decks(data: dict[str, Any]) -> Iterator[tuple[str, list[Any]]]:
    decks = data.get("decks")
    if not isinstance(decks, list):
        raise ValueError("Serialized collection is missing a decks list")

    for deck in decks:
        if not isinstance(deck, dict):
            continue
        deck_name = deck.get("name")
        notes = deck.get("notes")
        if not isinstance(deck_name, str) or not isinstance(notes, list):
            continue

        yield deck_name, notes


def _error_context_for_attempt(
    *,
    outcome: ProviderAttemptOutcome | None,
    error: LlmNoteError | LlmFatalError,
) -> ProviderAttemptErrorContext | None:
    if outcome is not None:
        return ProviderAttemptErrorContext(
            provider_message_id=outcome.provider_message_id,
            provider_model=outcome.provider_model,
            stop_reason=outcome.stop_reason,
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


class LlmTaskExecutor:
    """Standalone LLM runtime pipeline with DB-backed state."""

    def __init__(
        self,
        *,
        collection_dir: Path,
        task_name: str,
        model_override: str | None,
        no_auto_commit: bool,
        failure_policy: RunFailurePolicy | str,
    ) -> None:
        self.collection_dir = collection_dir
        self.task_name = task_name
        self.model_override = model_override
        self.no_auto_commit = no_auto_commit
        self.failure_policy = _resolve_failure_policy(failure_policy)

    def execute(self) -> LlmJobResult:
        task, note_type_configs = _load_task(
            collection_dir=self.collection_dir,
            task_name=self.task_name,
        )

        model = _resolve_model(task, self.model_override)
        db = LlmDbAdapter.open(self.collection_dir)

        job_id = db.start_job(
            task_name=task.name,
            model_name=model.name,
            api_model=model.api_id,
            failure_policy=self.failure_policy,
        )

        deck, no_subdecks = _resolve_serializer_scope(task)
        logger.debug(
            "Starting LLM task '%s' (model=%s, api_model=%s, collection=%s, "
            "deck_scope=%s, failure_policy=%s)",
            task.name,
            model,
            model.api_id,
            self.collection_dir,
            _format_deck_scope(task),
            self.failure_policy.value,
        )
        logger.debug("LLM request defaults: %s", _format_request_defaults(task))
        logger.debug(
            "LLM serializer scope: %s",
            _format_serializer_scope(deck, no_subdecks),
        )
        aggregate = None

        try:
            provider_client = ClaudeClient(task)
            if not self.no_auto_commit:
                logger.debug("Creating pre-LLM git snapshot")
                git_snapshot(self.collection_dir, f"llm:{task.name}")
            else:
                logger.debug("Auto-commit disabled (--no-auto-commit)")

            data = serialize_collection(
                self.collection_dir,
                deck=deck,
                no_subdecks=no_subdecks,
                note_types_dir=self.collection_dir / NOTE_TYPES_DIR,
            )

            candidates = self._discover_candidates(
                db=db,
                job_id=job_id,
                data=data,
                task=task,
                note_type_configs=note_type_configs,
            )
            self._execute_candidates(
                db=db,
                task=task,
                api_model=model.api_id,
                provider_client=provider_client,
                candidates=candidates,
            )

            persisted = self._apply_updates(
                db=db,
                job_id=job_id,
                data=data,
            )

            aggregate = db.aggregate_job(job_id)
            status = (
                LlmJobStatus.FAILED
                if aggregate.summary.errors > 0
                else LlmJobStatus.COMPLETED
            )
            db.finalize_job(
                job_id=job_id,
                status=status,
                persisted=persisted,
            )
            aggregate = db.aggregate_job(job_id)
        except LlmFatalError as error:
            db.finalize_job(
                job_id=job_id,
                status=LlmJobStatus.FAILED,
                persisted=False,
                fatal_error=str(error),
            )
            aggregate = db.aggregate_job(job_id)
            summary = aggregate.summary
            logger.error(str(error))
            logger.info(summary.format())
            logger.debug(summary.format_usage())
            logger.debug(summary.format_cost())
            return LlmJobResult(
                job_id=job_id,
                status=LlmJobStatus.FAILED.value,
                summary=summary,
                failed=True,
                persisted=False,
            )
        finally:
            db.close()
        if aggregate is None:
            raise RuntimeError("LLM job aggregation failed")

        summary = aggregate.summary
        logger.info(summary.format())
        logger.debug(summary.format_usage())
        logger.debug(summary.format_cost())

        return LlmJobResult(
            job_id=job_id,
            status=aggregate.status.value,
            summary=summary,
            failed=aggregate.failed,
            persisted=aggregate.persisted,
        )

    def _discover_candidates(
        self,
        *,
        db: LlmDbAdapter,
        job_id: int,
        data: dict[str, Any],
        task: TaskConfig,
        note_type_configs: dict[str, NoteTypeConfig],
    ) -> list[_EligibleCandidate]:
        candidates: list[_EligibleCandidate] = []
        decks_seen = 0
        decks_matched = 0
        notes_seen = 0
        ordinal = 0

        for deck_name, notes in _iter_decks(data):
            decks_seen += 1
            notes_seen += len(notes)
            in_scope = task.decks.matches(deck_name)
            if in_scope:
                decks_matched += 1
            else:
                logger.debug(
                    "Skipping deck '%s' (%d notes): outside task scope",
                    deck_name,
                    len(notes),
                )

            for note in notes:
                if not isinstance(note, dict):
                    continue
                ordinal += 1

                note_key = note.get("note_key")
                note_type_name = note.get("note_type")
                note_label = note_key if isinstance(note_key, str) else "unknown"
                note_type_label = (
                    note_type_name if isinstance(note_type_name, str) else "unknown"
                )
                note_type_value = (
                    note_type_name if isinstance(note_type_name, str) else None
                )

                if not in_scope:
                    db.insert_job_item(
                        job_id=job_id,
                        ordinal=ordinal,
                        deck_name=deck_name,
                        note_key=note_key if isinstance(note_key, str) else None,
                        note_type=note_type_value,
                        candidate_status=LlmCandidateStatus.SKIPPED_DECK_SCOPE,
                        skip_reason="outside task scope",
                        final_status=LlmFinalStatus.NOT_ATTEMPTED,
                    )
                    continue

                note_type_config = (
                    note_type_configs.get(note_type_name)
                    if isinstance(note_type_name, str)
                    else None
                )
                if note_type_config is None:
                    message = f"Unknown note type '{note_type_name}' in serialized note"
                    db.insert_job_item(
                        job_id=job_id,
                        ordinal=ordinal,
                        deck_name=deck_name,
                        note_key=note_key if isinstance(note_key, str) else None,
                        note_type=note_type_value,
                        candidate_status=LlmCandidateStatus.INVALID_NOTE,
                        skip_reason=None,
                        final_status=LlmFinalStatus.NOTE_ERROR,
                        error_message=message,
                    )
                    logger.error(
                        "LLM note error for %s in '%s' (%s): %s",
                        note_label,
                        deck_name,
                        note_type_label,
                        message,
                    )
                    continue

                note_field_names = {
                    field.name
                    for field in note_type_config.fields
                    if field.name != ANKIOPS_KEY_FIELD.name
                }
                try:
                    payload = _build_note_payload(
                        task,
                        note=note,
                        note_type_field_names=note_field_names,
                    )
                except ValueError as error:
                    db.insert_job_item(
                        job_id=job_id,
                        ordinal=ordinal,
                        deck_name=deck_name,
                        note_key=note_key if isinstance(note_key, str) else None,
                        note_type=note_type_value,
                        candidate_status=LlmCandidateStatus.INVALID_NOTE,
                        skip_reason=None,
                        final_status=LlmFinalStatus.NOTE_ERROR,
                        error_message=str(error),
                    )
                    logger.error(
                        "LLM note error for %s in '%s' (%s): %s",
                        note_label,
                        deck_name,
                        note_type_label,
                        error,
                    )
                    continue

                if payload is None:
                    db.insert_job_item(
                        job_id=job_id,
                        ordinal=ordinal,
                        deck_name=deck_name,
                        note_key=note_key if isinstance(note_key, str) else None,
                        note_type=note_type_value,
                        candidate_status=LlmCandidateStatus.SKIPPED_NO_EDITABLE_FIELDS,
                        skip_reason="no editable non-empty fields",
                        final_status=LlmFinalStatus.NOT_ATTEMPTED,
                    )
                    logger.debug(
                        "  Skipped %s in '%s' (%s): no editable non-empty fields",
                        note_label,
                        deck_name,
                        note_type_label,
                    )
                    continue

                item_id = db.insert_job_item(
                    job_id=job_id,
                    ordinal=ordinal,
                    deck_name=deck_name,
                    note_key=payload.note_key,
                    note_type=payload.note_type,
                    candidate_status=LlmCandidateStatus.ELIGIBLE,
                    skip_reason=None,
                    final_status=LlmFinalStatus.NOT_ATTEMPTED,
                )
                candidates.append(
                    _EligibleCandidate(
                        item_id=item_id,
                        deck_name=deck_name,
                        payload=payload,
                        note_type_config=note_type_config,
                        serialized_note=note,
                    )
                )

        db.set_discovery_counts(
            job_id=job_id,
            decks_seen=decks_seen,
            decks_matched=decks_matched,
            notes_seen=notes_seen,
        )
        return candidates

    def _execute_candidates(
        self,
        *,
        db: LlmDbAdapter,
        task: TaskConfig,
        api_model: str,
        provider_client: ClaudeClient,
        candidates: list[_EligibleCandidate],
    ) -> None:
        for candidate in candidates:
            prepared_request = provider_client.prepare_attempt_request(
                note_payload=candidate.payload,
                task_prompt=task.prompt,
                request_options=task.request,
                api_model=api_model,
            )
            outcome: ProviderAttemptOutcome | None = None

            try:
                outcome = provider_client.generate_update(
                    prepared_request=prepared_request,
                    request_options=task.request,
                )
                update = outcome.update
                if update.note_key != candidate.payload.note_key:
                    raise LlmNoteError("Model returned mismatched note_key")

                changed_fields = _apply_note_update(
                    serialized_note=candidate.serialized_note,
                    payload=candidate.payload,
                    edits=update.edits,
                    note_type_config=candidate.note_type_config,
                )
            except LlmFatalError as error:
                with db.write_tx():
                    self._record_error_attempt(
                        db=db,
                        candidate=candidate,
                        prepared_request=prepared_request,
                        outcome=outcome,
                        error=error,
                        error_type="fatal_error",
                        final_status=LlmFinalStatus.FATAL_ERROR,
                    )
                raise
            except LlmNoteError as error:
                provider_error = str(error).startswith("Provider returned HTTP")
                with db.write_tx():
                    self._record_error_attempt(
                        db=db,
                        candidate=candidate,
                        prepared_request=prepared_request,
                        outcome=outcome,
                        error=error,
                        error_type="provider_error" if provider_error else "note_error",
                        final_status=(
                            LlmFinalStatus.PROVIDER_ERROR
                            if provider_error
                            else LlmFinalStatus.NOTE_ERROR
                        ),
                    )
                logger.error(
                    "LLM note error for %s in '%s' (%s): %s",
                    candidate.payload.note_key,
                    candidate.deck_name,
                    candidate.payload.note_type,
                    error,
                )
                continue

            if changed_fields:
                with db.write_tx():
                    self._record_success_attempt(
                        db=db,
                        candidate=candidate,
                        prepared_request=prepared_request,
                        outcome=outcome,
                    )
                    db.update_job_item_result(
                        item_id=candidate.item_id,
                        final_status=LlmFinalStatus.SUCCEEDED_UPDATED,
                        changed_fields=changed_fields,
                    )
                logger.debug(
                    "  Updated %s in '%s' (%s): fields=%s",
                    candidate.payload.note_key,
                    candidate.deck_name,
                    candidate.payload.note_type,
                    ",".join(changed_fields),
                )
                continue

            with db.write_tx():
                self._record_success_attempt(
                    db=db,
                    candidate=candidate,
                    prepared_request=prepared_request,
                    outcome=outcome,
                )
                db.update_job_item_result(
                    item_id=candidate.item_id,
                    final_status=LlmFinalStatus.SUCCEEDED_UNCHANGED,
                    changed_fields=[],
                )
            logger.debug(
                "  Unchanged %s in '%s' (%s)",
                candidate.payload.note_key,
                candidate.deck_name,
                candidate.payload.note_type,
            )

    def _record_success_attempt(
        self,
        *,
        db: LlmDbAdapter,
        candidate: _EligibleCandidate,
        prepared_request: PreparedAttemptRequest,
        outcome: ProviderAttemptOutcome | None,
    ) -> None:
        if outcome is None:
            raise RuntimeError("Successful attempt recording requires provider outcome")

        parsed_update = {
            "note_key": outcome.update.note_key,
            "edits": outcome.update.edits,
        }
        attempt_id = db.insert_attempt(
            item_id=candidate.item_id,
            attempt_no=1,
            provider="anthropic",
            provider_message_id=outcome.provider_message_id,
            provider_model=outcome.provider_model,
            stop_reason=outcome.stop_reason,
            result_type=LlmAttemptResultType.SUCCEEDED,
            latency_ms=outcome.latency_ms,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            retry_count=outcome.retry_count,
            error_type=None,
            error_message=None,
            parsed_update_json=parsed_update,
        )
        db.insert_attempt_payload(
            attempt_id=attempt_id,
            system_prompt_text=prepared_request.system_prompt_text,
            user_message_text=prepared_request.user_message_text,
            request_params_json=prepared_request.request_params,
            response_raw_text=outcome.response_raw_text,
            response_full_json=outcome.response_full_json,
        )

    def _record_error_attempt(
        self,
        *,
        db: LlmDbAdapter,
        candidate: _EligibleCandidate,
        prepared_request: PreparedAttemptRequest,
        outcome: ProviderAttemptOutcome | None,
        error: LlmNoteError | LlmFatalError,
        error_type: str,
        final_status: LlmFinalStatus,
    ) -> None:
        context = _error_context_for_attempt(outcome=outcome, error=error)
        parsed_update = None
        if outcome is not None:
            parsed_update = {
                "note_key": outcome.update.note_key,
                "edits": outcome.update.edits,
            }
        attempt_id = db.insert_attempt(
            item_id=candidate.item_id,
            attempt_no=1,
            provider="anthropic",
            provider_message_id=(
                context.provider_message_id if context is not None else None
            ),
            provider_model=context.provider_model if context is not None else None,
            stop_reason=context.stop_reason if context is not None else None,
            result_type=LlmAttemptResultType.ERRORED,
            latency_ms=context.latency_ms if context is not None else 0,
            input_tokens=context.input_tokens if context is not None else 0,
            output_tokens=context.output_tokens if context is not None else 0,
            retry_count=context.retry_count if context is not None else 0,
            error_type=error_type,
            error_message=str(error),
            parsed_update_json=parsed_update,
        )
        db.insert_attempt_payload(
            attempt_id=attempt_id,
            system_prompt_text=prepared_request.system_prompt_text,
            user_message_text=prepared_request.user_message_text,
            request_params_json=prepared_request.request_params,
            response_raw_text=(
                context.response_raw_text if context is not None else None
            ),
            response_full_json=(
                context.response_full_json if context is not None else None
            ),
        )
        db.update_job_item_result(
            item_id=candidate.item_id,
            final_status=final_status,
            error_message=str(error),
        )

    def _apply_updates(
        self,
        *,
        db: LlmDbAdapter,
        job_id: int,
        data: dict[str, Any],
    ) -> bool:
        aggregate = db.aggregate_job(job_id)
        summary = aggregate.summary
        persisted = False

        if summary.updated > 0:
            if self.failure_policy is RunFailurePolicy.ATOMIC and summary.errors:
                logger.error(
                    "Atomic failure policy prevented persistence: %d update(s) staged, "
                    "%d error(s) observed",
                    summary.updated,
                    summary.errors,
                )
            else:
                deserialize_collection_data(
                    data,
                    root_dir=self.collection_dir,
                    note_types_dir=self.collection_dir / NOTE_TYPES_DIR,
                    overwrite=True,
                )
                persisted = True
                db.set_applied_for_updated_items(job_id=job_id)

        return persisted


def run_task(
    *,
    collection_dir: Path,
    task_name: str,
    model_override: str | None = None,
    no_auto_commit: bool = False,
    failure_policy: RunFailurePolicy | str = RunFailurePolicy.ATOMIC,
) -> LlmJobResult:
    executor = LlmTaskExecutor(
        collection_dir=collection_dir,
        task_name=task_name,
        model_override=model_override,
        no_auto_commit=no_auto_commit,
        failure_policy=failure_policy,
    )
    return executor.execute()


def list_jobs(
    *,
    collection_dir: Path,
    limit: int = 20,
) -> list[LlmJobListItem]:
    db = LlmDbAdapter.open(collection_dir)
    try:
        return db.list_jobs(limit=limit)
    finally:
        db.close()


def show_job(
    *,
    collection_dir: Path,
    job_id: str,
) -> LlmJobDetail | None:
    db = LlmDbAdapter.open(collection_dir)
    try:
        resolved = db.resolve_job_id(job_id)
        if resolved is None:
            return None
        return db.get_job_detail(int(resolved))
    finally:
        db.close()
