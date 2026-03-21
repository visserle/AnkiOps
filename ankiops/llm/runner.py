"""Execution of collection-local Claude tasks."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
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
from .claude import ClaudeClient, ProviderBatchResult
from .config_loader import load_llm_task_catalog
from .db import LlmDbAdapter, LlmJobDetail, LlmJobListItem
from .discovery import DiscoverySnapshot, discover_candidates
from .errors import LlmFatalError, LlmNoteError
from .models import (
    DeckScope,
    ExecutionMode,
    FieldExceptionRule,
    LlmAttemptResultType,
    LlmCandidateStatus,
    LlmFinalStatus,
    LlmJobResult,
    LlmJobStatus,
    NotePayload,
    PlanFieldSurface,
    PreparedAttemptRequest,
    ProviderAttemptErrorContext,
    ProviderAttemptOutcome,
    RunFailurePolicy,
    TaskConfig,
    TaskExecutionOptions,
    TaskPlanResult,
    TaskRequestOptions,
    TaskRunSummary,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _EligibleCandidate:
    item_id: int
    deck_name: str
    payload: NotePayload
    note_type_config: NoteTypeConfig
    serialized_note: dict[str, Any]
    ordinal: int
    resume_source_item_id: int | None = None


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


def _apply_deck_override(task: TaskConfig, deck_override: str | None) -> TaskConfig:
    if deck_override is None:
        return task

    deck_name = deck_override.strip()
    if not deck_name:
        raise ValueError("Deck override must be a non-empty deck name")
    if any(char in deck_name for char in ("*", "?", "[")):
        raise ValueError(
            "Deck override must be an exact deck name "
            "(wildcards are not supported)"
        )

    return replace(
        task,
        decks=DeckScope(
            include=[deck_name],
            exclude=[],
            include_subdecks=False,
        ),
    )


def _apply_mode_override(task: TaskConfig, mode_override: str | None) -> TaskConfig:
    if mode_override is None:
        return task

    normalized = mode_override.strip().lower()
    try:
        execution_mode = ExecutionMode(normalized)
    except ValueError as error:
        supported = ", ".join(mode.value for mode in ExecutionMode)
        raise ValueError(f"Execution mode must be one of: {supported}") from error

    return replace(
        task,
        execution=replace(task.execution, mode=execution_mode),
    )


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
    execution = task.execution
    if execution.mode is ExecutionMode.ONLINE:
        execution_text = (
            f"mode=online concurrency={execution.concurrency} "
            f"fail_fast={str(execution.fail_fast).lower()}"
        )
    else:
        execution_text = (
            f"mode=batch poll={execution.batch_poll_seconds}s "
            f"fail_fast={str(execution.fail_fast).lower()}"
        )

    return (
        f"timeout={task.timeout_seconds}s "
        f"max_tokens={max_tokens} temperature={temperature} "
        f"retries={task.request.retries} "
        f"retry_backoff={task.request.retry_backoff_seconds}s "
        f"retry_jitter={str(task.request.retry_backoff_jitter).lower()} "
        f"{execution_text}"
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


def _task_to_snapshot(task: TaskConfig) -> dict[str, Any]:
    return {
        "name": task.name,
        "model": task.model.name,
        "system_prompt": task.system_prompt,
        "prompt": task.prompt,
        "system_prompt_path": str(task.system_prompt_path),
        "prompt_path": str(task.prompt_path),
        "api_key_env": task.api_key_env,
        "timeout_seconds": task.timeout_seconds,
        "decks": {
            "include": list(task.decks.include),
            "exclude": list(task.decks.exclude),
            "include_subdecks": task.decks.include_subdecks,
        },
        "field_exceptions": [
            {
                "note_types": list(rule.note_types),
                "read_only": list(rule.read_only),
                "hidden": list(rule.hidden),
            }
            for rule in task.field_exceptions
        ],
        "request": {
            "temperature": task.request.temperature,
            "max_output_tokens": task.request.max_output_tokens,
            "retries": task.request.retries,
            "retry_backoff_seconds": task.request.retry_backoff_seconds,
            "retry_backoff_jitter": task.request.retry_backoff_jitter,
        },
        "execution": {
            "mode": task.execution.mode.value,
            "concurrency": task.execution.concurrency,
            "fail_fast": task.execution.fail_fast,
            "batch_poll_seconds": task.execution.batch_poll_seconds,
        },
    }


def _task_from_snapshot(snapshot: dict[str, Any]) -> TaskConfig:
    model_name = snapshot.get("model")
    if not isinstance(model_name, str):
        raise ValueError("Job snapshot is missing model")
    model = parse_model(model_name)
    if model is None:
        raise ValueError(f"Job snapshot references unsupported model '{model_name}'")

    decks = snapshot.get("decks")
    if not isinstance(decks, dict):
        raise ValueError("Job snapshot is missing deck scope")

    field_exceptions_raw = snapshot.get("field_exceptions")
    if not isinstance(field_exceptions_raw, list):
        field_exceptions_raw = []
    field_exceptions: list[FieldExceptionRule] = []
    for entry in field_exceptions_raw:
        if not isinstance(entry, dict):
            continue
        note_types = entry.get("note_types")
        read_only = entry.get("read_only")
        hidden = entry.get("hidden")
        field_exceptions.append(
            FieldExceptionRule(
                note_types=[item for item in note_types if isinstance(item, str)]
                if isinstance(note_types, list)
                else ["*"],
                read_only=[item for item in read_only if isinstance(item, str)]
                if isinstance(read_only, list)
                else [],
                hidden=[item for item in hidden if isinstance(item, str)]
                if isinstance(hidden, list)
                else [],
            )
        )

    request = snapshot.get("request")
    if not isinstance(request, dict):
        raise ValueError("Job snapshot is missing request options")

    execution = snapshot.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("Job snapshot is missing execution options")

    mode_raw = execution.get("mode")
    if not isinstance(mode_raw, str):
        raise ValueError("Job snapshot execution mode is invalid")

    return TaskConfig(
        name=str(snapshot.get("name") or "unknown"),
        model=model,
        system_prompt=str(snapshot.get("system_prompt") or ""),
        prompt=str(snapshot.get("prompt") or ""),
        system_prompt_path=Path(str(snapshot.get("system_prompt_path") or "")),
        prompt_path=Path(str(snapshot.get("prompt_path") or "")),
        api_key_env=str(snapshot.get("api_key_env") or "ANTHROPIC_API_KEY"),
        timeout_seconds=int(snapshot.get("timeout_seconds") or 60),
        decks=DeckScope(
            include=[item for item in decks.get("include", []) if isinstance(item, str)]
            or ["*"],
            exclude=[item for item in decks.get("exclude", []) if isinstance(item, str)]
            if isinstance(decks.get("exclude"), list)
            else [],
            include_subdecks=bool(decks.get("include_subdecks", True)),
        ),
        field_exceptions=field_exceptions,
        request=TaskRequestOptions(
            temperature=(
                float(request["temperature"])
                if request.get("temperature") is not None
                else None
            ),
            max_output_tokens=(
                int(request["max_output_tokens"])
                if request.get("max_output_tokens") is not None
                else None
            ),
            retries=int(request.get("retries", 2)),
            retry_backoff_seconds=float(request.get("retry_backoff_seconds", 0.5)),
            retry_backoff_jitter=bool(request.get("retry_backoff_jitter", True)),
        ),
        execution=TaskExecutionOptions(
            mode=ExecutionMode(mode_raw),
            concurrency=int(execution.get("concurrency", 8)),
            fail_fast=bool(execution.get("fail_fast", True)),
            batch_poll_seconds=int(execution.get("batch_poll_seconds", 15)),
        ),
    )


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
    task = catalog.tasks_by_name.get(task_name)
    if task is None:
        task_errors = [
            message
            for path, message in catalog.errors.items()
            if _is_task_file_for_name(path, task_name)
        ]
        if task_errors:
            joined_errors = "\n".join(task_errors)
            raise ValueError(f"Invalid LLM task configuration:\n{joined_errors}")

        shared_errors = [
            message
            for path, message in catalog.errors.items()
            if not _is_task_file(path)
        ]
        if shared_errors:
            joined_errors = "\n".join(shared_errors)
            raise ValueError(f"Invalid LLM task configuration:\n{joined_errors}")

        raise ValueError(f"Unknown task '{task_name}'")

    config_by_name = {config.name: config for config in note_type_configs}
    return task, config_by_name


def _load_note_type_configs(
    collection_dir: Path,
) -> dict[str, NoteTypeConfig]:
    fs = FileSystemAdapter()
    configs = fs.load_note_type_configs(collection_dir / NOTE_TYPES_DIR)
    return {config.name: config for config in configs}


def _is_task_file(path: str) -> bool:
    path_obj = Path(path)
    return path_obj.suffix in {".yaml", ".yml"} and path_obj.parent.name == "tasks"


def _is_task_file_for_name(path: str, task_name: str) -> bool:
    path_obj = Path(path)
    return _is_task_file(path) and path_obj.stem == task_name


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


class LlmTaskExecutor:
    """Standalone async LLM runtime pipeline with DB-backed state."""

    def __init__(
        self,
        *,
        collection_dir: Path,
        task: TaskConfig,
        note_type_configs: dict[str, NoteTypeConfig],
        model_override: str | None,
        mode_override: str | None,
        no_auto_commit: bool,
        failure_policy: RunFailurePolicy | str,
        resume_from_job_id: int | None = None,
        resume_source_items: dict[str, int] | None = None,
    ) -> None:
        self.collection_dir = collection_dir
        self.task = task
        self.note_type_configs = note_type_configs
        self.model_override = model_override
        self.mode_override = mode_override
        self.no_auto_commit = no_auto_commit
        self.failure_policy = _resolve_failure_policy(failure_policy)
        self.resume_from_job_id = resume_from_job_id
        self.resume_source_items = resume_source_items

    async def execute(self) -> LlmJobResult:
        task = _apply_mode_override(self.task, self.mode_override)
        model = _resolve_model(task, self.model_override)
        task = replace(task, model=model)

        db = LlmDbAdapter.open(self.collection_dir)

        job_id = db.start_job(
            task_name=task.name,
            model_name=model.name,
            api_model=model.api_id,
            execution_mode=task.execution.mode,
            failure_policy=self.failure_policy,
            config_snapshot=_task_to_snapshot(task),
            resume_from_job_id=self.resume_from_job_id,
        )

        deck, no_subdecks = _resolve_serializer_scope(task)
        logger.debug(
            "Starting LLM task '%s' (model=%s, api_model=%s, mode=%s, collection=%s, "
            "deck_scope=%s, failure_policy=%s)",
            task.name,
            model,
            model.api_id,
            task.execution.mode.value,
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
        provider_client: ClaudeClient | None = None

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
                note_type_configs=self.note_type_configs,
                resume_source_items=self.resume_source_items,
            )
            if task.execution.mode is ExecutionMode.ONLINE:
                await self._execute_candidates_online(
                    db=db,
                    job_id=job_id,
                    task=task,
                    api_model=model.api_id,
                    provider_client=provider_client,
                    candidates=candidates,
                )
            else:
                await self._execute_candidates_batch(
                    db=db,
                    job_id=job_id,
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
            if task.execution.mode is ExecutionMode.BATCH:
                fatal_marked = db.mark_unfinished_items_fatal(
                    job_id=job_id,
                    error_message=str(error),
                )
                if fatal_marked:
                    logger.warning(
                        "Marked %d pending batch item(s) as fatal_error due to fatal "
                        "error",
                        fatal_marked,
                    )
            elif task.execution.fail_fast:
                canceled = db.mark_unfinished_items_canceled(job_id=job_id)
                if canceled:
                    logger.warning(
                        "Marked %d pending item(s) as canceled due to fatal error",
                        canceled,
                    )
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
            logger.debug("Usage: %s", summary.format_usage())
            logger.debug("Cost: %s", summary.format_cost())
            return LlmJobResult(
                job_id=job_id,
                status=LlmJobStatus.FAILED.value,
                summary=summary,
                failed=True,
                persisted=False,
            )
        finally:
            if provider_client is not None:
                await provider_client.close()
            db.close()
        if aggregate is None:
            raise RuntimeError("LLM job aggregation failed")

        summary = aggregate.summary
        logger.info(summary.format())
        logger.debug("Usage: %s", summary.format_usage())
        logger.debug("Cost: %s", summary.format_cost())

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
        resume_source_items: dict[str, int] | None,
    ) -> list[_EligibleCandidate]:
        snapshot = discover_candidates(
            data=data,
            task=task,
            note_type_configs=note_type_configs,
        )
        candidates: list[_EligibleCandidate] = []

        selected_items = self._select_snapshot_items(
            snapshot,
            resume_source_items=resume_source_items,
        )

        skipped_deck_counts: dict[str, int] = {}
        for item in selected_items:
            if item.candidate_status is not LlmCandidateStatus.SKIPPED_DECK_SCOPE:
                continue
            skipped_deck_counts[item.deck_name] = (
                skipped_deck_counts.get(item.deck_name, 0) + 1
            )
        for deck_name, skipped_count in skipped_deck_counts.items():
            logger.debug(
                "Skipping deck '%s' (%d notes): outside task scope",
                deck_name,
                skipped_count,
            )

        for item in selected_items:
            resume_source_item_id = (
                resume_source_items.get(item.note_key) if resume_source_items else None
            )
            if item.candidate_status is LlmCandidateStatus.SKIPPED_DECK_SCOPE:
                db.insert_job_item(
                    job_id=job_id,
                    ordinal=item.ordinal,
                    deck_name=item.deck_name,
                    note_key=item.note_key,
                    note_type=item.note_type,
                    candidate_status=item.candidate_status,
                    skip_reason=item.skip_reason,
                    final_status=LlmFinalStatus.NOT_ATTEMPTED,
                    resume_source_item_id=resume_source_item_id,
                )
                continue

            note_label = item.note_key or "unknown"
            note_type_label = item.note_type or "unknown"

            if item.candidate_status is LlmCandidateStatus.INVALID_NOTE:
                message = item.error_message or "Invalid note"
                db.insert_job_item(
                    job_id=job_id,
                    ordinal=item.ordinal,
                    deck_name=item.deck_name,
                    note_key=item.note_key,
                    note_type=item.note_type,
                    candidate_status=item.candidate_status,
                    skip_reason=None,
                    final_status=LlmFinalStatus.NOTE_ERROR,
                    error_message=message,
                    resume_source_item_id=resume_source_item_id,
                )
                logger.error(
                    "LLM note error for %s in '%s' (%s): %s",
                    note_label,
                    item.deck_name,
                    note_type_label,
                    message,
                )
                continue

            if item.candidate_status is LlmCandidateStatus.SKIPPED_NO_EDITABLE_FIELDS:
                db.insert_job_item(
                    job_id=job_id,
                    ordinal=item.ordinal,
                    deck_name=item.deck_name,
                    note_key=item.note_key,
                    note_type=item.note_type,
                    candidate_status=item.candidate_status,
                    skip_reason=item.skip_reason,
                    final_status=LlmFinalStatus.NOT_ATTEMPTED,
                    resume_source_item_id=resume_source_item_id,
                )
                logger.debug(
                    "  Skipped %s in '%s' (%s): no editable non-empty fields",
                    note_label,
                    item.deck_name,
                    note_type_label,
                )
                continue

            if (
                item.candidate_status is LlmCandidateStatus.ELIGIBLE
                and item.payload is not None
                and item.note_type_config is not None
                and item.serialized_note is not None
            ):
                item_id = db.insert_job_item(
                    job_id=job_id,
                    ordinal=item.ordinal,
                    deck_name=item.deck_name,
                    note_key=item.payload.note_key,
                    note_type=item.payload.note_type,
                    candidate_status=item.candidate_status,
                    skip_reason=None,
                    final_status=LlmFinalStatus.NOT_ATTEMPTED,
                    resume_source_item_id=resume_source_item_id,
                )
                candidates.append(
                    _EligibleCandidate(
                        item_id=item_id,
                        deck_name=item.deck_name,
                        payload=item.payload,
                        note_type_config=item.note_type_config,
                        serialized_note=item.serialized_note,
                        ordinal=item.ordinal,
                        resume_source_item_id=resume_source_item_id,
                    )
                )

        if resume_source_items is None:
            db.set_discovery_counts(
                job_id=job_id,
                decks_seen=snapshot.counts.decks_seen,
                decks_matched=snapshot.counts.decks_matched,
                notes_seen=snapshot.counts.notes_seen,
            )
        else:
            deck_names = {item.deck_name for item in selected_items}
            matched_decks = {
                item.deck_name
                for item in selected_items
                if item.candidate_status is not LlmCandidateStatus.SKIPPED_DECK_SCOPE
            }
            db.set_discovery_counts(
                job_id=job_id,
                decks_seen=len(deck_names),
                decks_matched=len(matched_decks),
                notes_seen=len(selected_items),
            )
        return candidates

    @staticmethod
    def _select_snapshot_items(
        snapshot: DiscoverySnapshot,
        *,
        resume_source_items: dict[str, int] | None,
    ) -> list[Any]:
        if resume_source_items is None:
            return list(snapshot.items)
        selected: list[Any] = []
        resume_note_keys = set(resume_source_items)
        for item in snapshot.items:
            note_key = item.note_key
            if note_key is None:
                continue
            if note_key in resume_note_keys:
                selected.append(item)
        return selected

    async def _execute_candidates_online(
        self,
        *,
        db: LlmDbAdapter,
        job_id: int,
        task: TaskConfig,
        api_model: str,
        provider_client: ClaudeClient,
        candidates: list[_EligibleCandidate],
    ) -> None:
        semaphore = asyncio.Semaphore(task.execution.concurrency)
        first_fatal_error: LlmFatalError | None = None

        async def _process_candidate(candidate: _EligibleCandidate) -> None:
            nonlocal first_fatal_error
            async with semaphore:
                prepared_request = provider_client.prepare_attempt_request(
                    note_payload=candidate.payload,
                    task_prompt=task.prompt,
                    request_options=task.request,
                    api_model=api_model,
                )
                outcome: ProviderAttemptOutcome | None = None

                try:
                    outcome = await provider_client.generate_update(
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
                            result_type=LlmAttemptResultType.ERRORED,
                            execution_mode=ExecutionMode.ONLINE,
                        )
                    if first_fatal_error is None:
                        first_fatal_error = error
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
                            error_type=(
                                "provider_error" if provider_error else "note_error"
                            ),
                            final_status=(
                                LlmFinalStatus.PROVIDER_ERROR
                                if provider_error
                                else LlmFinalStatus.NOTE_ERROR
                            ),
                            result_type=LlmAttemptResultType.ERRORED,
                            execution_mode=ExecutionMode.ONLINE,
                        )
                    logger.error(
                        "LLM note error for %s in '%s' (%s): %s",
                        candidate.payload.note_key,
                        candidate.deck_name,
                        candidate.payload.note_type,
                        error,
                    )
                    return

                if changed_fields:
                    with db.write_tx():
                        self._record_success_attempt(
                            db=db,
                            candidate=candidate,
                            prepared_request=prepared_request,
                            outcome=outcome,
                            execution_mode=ExecutionMode.ONLINE,
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
                    return

                with db.write_tx():
                    self._record_success_attempt(
                        db=db,
                        candidate=candidate,
                        prepared_request=prepared_request,
                        outcome=outcome,
                        execution_mode=ExecutionMode.ONLINE,
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

        tasks = [asyncio.create_task(_process_candidate(candidate)) for candidate in candidates]

        if not tasks:
            return

        if task.execution.fail_fast:
            for completed in asyncio.as_completed(tasks):
                try:
                    await completed
                except LlmFatalError:
                    for pending in tasks:
                        if not pending.done():
                            pending.cancel()
                    break
            await asyncio.gather(*tasks, return_exceptions=True)
            if first_fatal_error is not None:
                canceled = db.mark_unfinished_items_canceled(job_id=job_id)
                if canceled:
                    logger.warning(
                        "Fail-fast canceled %d pending online item(s)",
                        canceled,
                    )
                raise first_fatal_error
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, LlmFatalError):
                if first_fatal_error is None:
                    first_fatal_error = result
        if first_fatal_error is not None:
            raise first_fatal_error

    async def _execute_candidates_batch(
        self,
        *,
        db: LlmDbAdapter,
        job_id: int,
        task: TaskConfig,
        api_model: str,
        provider_client: ClaudeClient,
        candidates: list[_EligibleCandidate],
    ) -> None:
        if not candidates:
            return

        prepared_by_custom_id: dict[str, PreparedAttemptRequest] = {}
        candidate_by_custom_id: dict[str, _EligibleCandidate] = {}
        batch_requests: list[tuple[str, PreparedAttemptRequest]] = []
        for candidate in candidates:
            prepared_request = provider_client.prepare_attempt_request(
                note_payload=candidate.payload,
                task_prompt=task.prompt,
                request_options=task.request,
                api_model=api_model,
            )
            custom_id = f"item-{candidate.item_id}"
            prepared_by_custom_id[custom_id] = prepared_request
            candidate_by_custom_id[custom_id] = candidate
            batch_requests.append((custom_id, prepared_request))

        try:
            batch_state = await provider_client.create_batch(requests=batch_requests)
        except LlmFatalError:
            raise
        except Exception as error:
            raise LlmFatalError(f"Provider batch submission failed: {error}") from error
        batch_local_id = db.upsert_provider_batch(job_id=job_id, batch=batch_state)

        for custom_id, candidate in candidate_by_custom_id.items():
            db.insert_batch_item_map(
                provider_batch_local_id=batch_local_id,
                custom_id=custom_id,
                job_item_id=candidate.item_id,
                attempt_no=1,
            )

        terminal_states = {"ended"}
        polling_states = {"in_progress", "canceling"}
        try:
            while batch_state.processing_status not in terminal_states:
                if batch_state.processing_status not in polling_states:
                    raise LlmFatalError(
                        "Provider batch entered unexpected processing status "
                        f"'{batch_state.processing_status}'"
                    )
                await asyncio.sleep(task.execution.batch_poll_seconds)
                batch_state = await provider_client.retrieve_batch(
                    batch_state.provider_batch_id
                )
                db.upsert_provider_batch(job_id=job_id, batch=batch_state)
        except LlmFatalError:
            raise
        except Exception as error:
            raise LlmFatalError(f"Provider batch polling failed: {error}") from error

        try:
            batch_results = await provider_client.get_batch_results(
                provider_batch_id=batch_state.provider_batch_id,
                prepared_by_custom_id=prepared_by_custom_id,
            )
        except LlmFatalError:
            raise
        except Exception as error:
            raise LlmFatalError(f"Provider batch results failed: {error}") from error

        seen_custom_ids: set[str] = set()
        for batch_result in batch_results:
            seen_custom_ids.add(batch_result.custom_id)
            candidate = candidate_by_custom_id.get(batch_result.custom_id)
            prepared_request = prepared_by_custom_id.get(batch_result.custom_id)
            if candidate is None or prepared_request is None:
                continue

            await self._apply_batch_result(
                db=db,
                candidate=candidate,
                prepared_request=prepared_request,
                batch_result=batch_result,
            )

        missing = set(candidate_by_custom_id) - seen_custom_ids
        for custom_id in sorted(missing):
            candidate = candidate_by_custom_id[custom_id]
            db.update_job_item_result(
                item_id=candidate.item_id,
                final_status=LlmFinalStatus.CANCELED,
                error_message="Batch result missing from provider response",
            )

    async def _apply_batch_result(
        self,
        *,
        db: LlmDbAdapter,
        candidate: _EligibleCandidate,
        prepared_request: PreparedAttemptRequest,
        batch_result: ProviderBatchResult,
    ) -> None:
        if (
            batch_result.result_type is LlmAttemptResultType.SUCCEEDED
            and batch_result.outcome is not None
        ):
            outcome = batch_result.outcome
            try:
                if outcome.update.note_key != candidate.payload.note_key:
                    raise LlmNoteError("Model returned mismatched note_key")
                changed_fields = _apply_note_update(
                    serialized_note=candidate.serialized_note,
                    payload=candidate.payload,
                    edits=outcome.update.edits,
                    note_type_config=candidate.note_type_config,
                )
            except LlmNoteError as error:
                with db.write_tx():
                    self._record_error_attempt(
                        db=db,
                        candidate=candidate,
                        prepared_request=prepared_request,
                        outcome=outcome,
                        error=error,
                        error_type="note_error",
                        final_status=LlmFinalStatus.NOTE_ERROR,
                        result_type=LlmAttemptResultType.ERRORED,
                        execution_mode=ExecutionMode.BATCH,
                    )
                return

            with db.write_tx():
                self._record_success_attempt(
                    db=db,
                    candidate=candidate,
                    prepared_request=prepared_request,
                    outcome=outcome,
                    execution_mode=ExecutionMode.BATCH,
                )
                db.update_job_item_result(
                    item_id=candidate.item_id,
                    final_status=(
                        LlmFinalStatus.SUCCEEDED_UPDATED
                        if changed_fields
                        else LlmFinalStatus.SUCCEEDED_UNCHANGED
                    ),
                    changed_fields=changed_fields,
                )
            return

        if batch_result.result_type is LlmAttemptResultType.ERRORED:
            error = LlmNoteError(batch_result.error_message or "Provider batch request failed")
            with db.write_tx():
                self._record_error_attempt(
                    db=db,
                    candidate=candidate,
                    prepared_request=prepared_request,
                    outcome=batch_result.outcome,
                    error=error,
                    error_type=batch_result.error_type or "provider_error",
                    final_status=LlmFinalStatus.PROVIDER_ERROR,
                    result_type=LlmAttemptResultType.ERRORED,
                    execution_mode=ExecutionMode.BATCH,
                )
            return

        if batch_result.result_type is LlmAttemptResultType.CANCELED:
            with db.write_tx():
                self._record_terminal_attempt(
                    db=db,
                    candidate=candidate,
                    prepared_request=prepared_request,
                    execution_mode=ExecutionMode.BATCH,
                    result_type=LlmAttemptResultType.CANCELED,
                    final_status=LlmFinalStatus.CANCELED,
                    error_message=batch_result.error_message,
                )
            return

        if batch_result.result_type is LlmAttemptResultType.EXPIRED:
            with db.write_tx():
                self._record_terminal_attempt(
                    db=db,
                    candidate=candidate,
                    prepared_request=prepared_request,
                    execution_mode=ExecutionMode.BATCH,
                    result_type=LlmAttemptResultType.EXPIRED,
                    final_status=LlmFinalStatus.EXPIRED,
                    error_message=batch_result.error_message,
                )

    def _record_success_attempt(
        self,
        *,
        db: LlmDbAdapter,
        candidate: _EligibleCandidate,
        prepared_request: PreparedAttemptRequest,
        outcome: ProviderAttemptOutcome | None,
        execution_mode: ExecutionMode,
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
            provider_request_id=outcome.request_id,
            provider_execution_mode=execution_mode,
            stop_reason=outcome.stop_reason,
            result_type=LlmAttemptResultType.SUCCEEDED,
            latency_ms=outcome.latency_ms,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            retry_count=outcome.retry_count,
            error_type=None,
            error_message=None,
            parsed_update_json=parsed_update,
            rate_limit_headers_json=outcome.rate_limit_headers,
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
        result_type: LlmAttemptResultType,
        execution_mode: ExecutionMode,
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
            provider_request_id=context.request_id if context is not None else None,
            provider_execution_mode=execution_mode,
            stop_reason=context.stop_reason if context is not None else None,
            result_type=result_type,
            latency_ms=context.latency_ms if context is not None else 0,
            input_tokens=context.input_tokens if context is not None else 0,
            output_tokens=context.output_tokens if context is not None else 0,
            retry_count=context.retry_count if context is not None else 0,
            error_type=error_type,
            error_message=str(error),
            parsed_update_json=parsed_update,
            rate_limit_headers_json=(
                context.rate_limit_headers if context is not None else None
            ),
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

    def _record_terminal_attempt(
        self,
        *,
        db: LlmDbAdapter,
        candidate: _EligibleCandidate,
        prepared_request: PreparedAttemptRequest,
        execution_mode: ExecutionMode,
        result_type: LlmAttemptResultType,
        final_status: LlmFinalStatus,
        error_message: str | None,
    ) -> None:
        attempt_id = db.insert_attempt(
            item_id=candidate.item_id,
            attempt_no=1,
            provider="anthropic",
            provider_message_id=None,
            provider_model=None,
            provider_request_id=None,
            provider_execution_mode=execution_mode,
            stop_reason=None,
            result_type=result_type,
            latency_ms=0,
            input_tokens=0,
            output_tokens=0,
            retry_count=0,
            error_type=result_type.value,
            error_message=error_message,
            parsed_update_json=None,
            rate_limit_headers_json=None,
        )
        db.insert_attempt_payload(
            attempt_id=attempt_id,
            system_prompt_text=prepared_request.system_prompt_text,
            user_message_text=prepared_request.user_message_text,
            request_params_json=prepared_request.request_params,
            response_raw_text=None,
            response_full_json=None,
        )
        db.update_job_item_result(
            item_id=candidate.item_id,
            final_status=final_status,
            error_message=error_message,
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
                    quiet=True,
                )
                persisted = True
                db.set_applied_for_updated_items(job_id=job_id)
                logger.info(
                    "Persisted %d updated note(s) across %d deck file(s)",
                    summary.updated,
                    len(
                        [
                            deck
                            for deck in data.get("decks", [])
                            if isinstance(deck, dict)
                        ]
                    ),
                )

        return persisted


def _estimate_tokens(text: str) -> int:
    value = text.strip()
    if not value:
        return 0
    return max(1, (len(value) + 3) // 4)


def _estimate_note_input_tokens(task: TaskConfig, payload: NotePayload) -> int:
    parts = [
        task.system_prompt,
        task.prompt,
        payload.note_key,
        payload.note_type,
    ]
    for name, value in sorted(payload.editable_fields.items()):
        parts.append(name)
        parts.append(value)
    for name, value in sorted(payload.read_only_fields.items()):
        parts.append(name)
        parts.append(value)
    return _estimate_tokens("\n".join(parts))


def _build_plan_field_surface(
    *,
    task: TaskConfig,
    note_type_configs: dict[str, NoteTypeConfig],
    snapshot_items: list[Any],
) -> list[PlanFieldSurface]:
    observed_note_types = {
        item.note_type
        for item in snapshot_items
        if item.note_type is not None
        and item.note_type_config is not None
        and item.candidate_status is not LlmCandidateStatus.SKIPPED_DECK_SCOPE
    }
    field_surface: list[PlanFieldSurface] = []
    for note_type in sorted(observed_note_types):
        config = note_type_configs.get(note_type)
        if config is None:
            continue
        editable_fields: list[str] = []
        read_only_fields: list[str] = []
        hidden_fields: list[str] = []
        for field in config.fields:
            if field.name == ANKIOPS_KEY_FIELD.name:
                continue
            access = task.field_access(note_type, field.name)
            if access.value == "edit":
                editable_fields.append(field.name)
            elif access.value == "read_only":
                read_only_fields.append(field.name)
            else:
                hidden_fields.append(field.name)
        candidate_notes = sum(
            1
            for item in snapshot_items
            if item.candidate_status is LlmCandidateStatus.ELIGIBLE
            and item.note_type == note_type
        )
        field_surface.append(
            PlanFieldSurface(
                note_type=note_type,
                candidate_notes=candidate_notes,
                editable_fields=editable_fields,
                read_only_fields=read_only_fields,
                hidden_fields=hidden_fields,
            )
        )
    return field_surface


def plan_task(
    *,
    collection_dir: Path,
    task_name: str,
    model_override: str | None = None,
    deck_override: str | None = None,
    mode_override: str | None = None,
) -> TaskPlanResult:
    task, note_type_configs = _load_task(
        collection_dir=collection_dir,
        task_name=task_name,
    )
    task = _apply_deck_override(task, deck_override)
    task = _apply_mode_override(task, mode_override)
    model = _resolve_model(task, model_override)
    task = replace(task, model=model)
    deck, no_subdecks = _resolve_serializer_scope(task)
    data = serialize_collection(
        collection_dir,
        deck=deck,
        no_subdecks=no_subdecks,
        note_types_dir=collection_dir / NOTE_TYPES_DIR,
    )
    snapshot = discover_candidates(
        data=data,
        task=task,
        note_type_configs=note_type_configs,
    )
    eligible_items = [
        item
        for item in snapshot.items
        if item.candidate_status is LlmCandidateStatus.ELIGIBLE
        and item.payload is not None
    ]
    eligible = len(eligible_items)
    skipped_deck_scope = sum(
        1
        for item in snapshot.items
        if item.candidate_status is LlmCandidateStatus.SKIPPED_DECK_SCOPE
    )
    skipped_no_editable_fields = sum(
        1
        for item in snapshot.items
        if item.candidate_status is LlmCandidateStatus.SKIPPED_NO_EDITABLE_FIELDS
    )
    errors = sum(
        1
        for item in snapshot.items
        if item.candidate_status is LlmCandidateStatus.INVALID_NOTE
    )
    input_tokens_estimate = sum(
        _estimate_note_input_tokens(task, item.payload) for item in eligible_items
    )
    max_output_tokens = task.request.max_output_tokens or 2048
    output_tokens_cap = eligible * max_output_tokens
    summary = TaskRunSummary(
        task_name=task.name,
        model=model,
        decks_seen=snapshot.counts.decks_seen,
        decks_matched=snapshot.counts.decks_matched,
        notes_seen=snapshot.counts.notes_seen,
        eligible=eligible,
        skipped_deck_scope=skipped_deck_scope,
        skipped_no_editable_fields=skipped_no_editable_fields,
        errors=errors,
        requests=eligible,
    )
    return TaskPlanResult(
        task_name=task.name,
        model=model,
        deck_scope=_format_deck_scope(task),
        serializer_scope=_format_serializer_scope(deck, no_subdecks),
        system_prompt_path=str(task.system_prompt_path),
        prompt_path=str(task.prompt_path),
        system_prompt=task.system_prompt,
        task_prompt=task.prompt,
        request_defaults=_format_request_defaults(task),
        summary=summary,
        field_surface=_build_plan_field_surface(
            task=task,
            note_type_configs=note_type_configs,
            snapshot_items=snapshot.items,
        ),
        requests_estimate=eligible,
        input_tokens_estimate=input_tokens_estimate,
        output_tokens_cap=output_tokens_cap,
    )


async def run_task_async(
    *,
    collection_dir: Path,
    task_name: str,
    model_override: str | None = None,
    mode_override: str | None = None,
    deck_override: str | None = None,
    no_auto_commit: bool = False,
    failure_policy: RunFailurePolicy | str = RunFailurePolicy.ATOMIC,
) -> LlmJobResult:
    task, note_type_configs = _load_task(
        collection_dir=collection_dir,
        task_name=task_name,
    )
    task = _apply_deck_override(task, deck_override)
    executor = LlmTaskExecutor(
        collection_dir=collection_dir,
        task=task,
        note_type_configs=note_type_configs,
        model_override=model_override,
        mode_override=mode_override,
        no_auto_commit=no_auto_commit,
        failure_policy=failure_policy,
    )
    return await executor.execute()


async def resume_task_async(
    *,
    collection_dir: Path,
    resume_job_id: str,
    mode_override: str | None = None,
    no_auto_commit: bool = False,
    failure_policy: RunFailurePolicy | str = RunFailurePolicy.ATOMIC,
) -> LlmJobResult:
    db = LlmDbAdapter.open(collection_dir)
    try:
        resolved = db.resolve_job_id(resume_job_id)
        if resolved is None:
            raise ValueError(f"Unknown LLM job '{resume_job_id}'")
        snapshot = db.get_job_snapshot(job_id=resolved)
        if snapshot is None:
            raise ValueError(f"Missing config snapshot for LLM job '{resolved}'")
        resume_source_items = db.get_resume_source_items(job_id=resolved)
    finally:
        db.close()

    task = _task_from_snapshot(snapshot)
    note_type_configs = _load_note_type_configs(collection_dir)

    executor = LlmTaskExecutor(
        collection_dir=collection_dir,
        task=task,
        note_type_configs=note_type_configs,
        model_override=None,
        mode_override=mode_override,
        no_auto_commit=no_auto_commit,
        failure_policy=failure_policy,
        resume_from_job_id=resolved,
        resume_source_items=resume_source_items,
    )
    return await executor.execute()


def run_task(
    *,
    collection_dir: Path,
    task_name: str,
    model_override: str | None = None,
    mode_override: str | None = None,
    deck_override: str | None = None,
    no_auto_commit: bool = False,
    failure_policy: RunFailurePolicy | str = RunFailurePolicy.ATOMIC,
) -> LlmJobResult:
    return asyncio.run(
        run_task_async(
            collection_dir=collection_dir,
            task_name=task_name,
            model_override=model_override,
            mode_override=mode_override,
            deck_override=deck_override,
            no_auto_commit=no_auto_commit,
            failure_policy=failure_policy,
        )
    )


def resume_task(
    *,
    collection_dir: Path,
    resume_job_id: str,
    mode_override: str | None = None,
    no_auto_commit: bool = False,
    failure_policy: RunFailurePolicy | str = RunFailurePolicy.ATOMIC,
) -> LlmJobResult:
    return asyncio.run(
        resume_task_async(
            collection_dir=collection_dir,
            resume_job_id=resume_job_id,
            mode_override=mode_override,
            no_auto_commit=no_auto_commit,
            failure_policy=failure_policy,
        )
    )


def list_jobs(
    *,
    collection_dir: Path,
) -> list[LlmJobListItem]:
    db = LlmDbAdapter.open(collection_dir)
    try:
        return db.list_jobs()
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
