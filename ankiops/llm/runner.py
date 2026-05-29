"""OpenAI-only LLM task planning and execution."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, AuthenticationError
from openai.types.responses import (
    ParsedResponse,
    ResponseOutputMessage,
    ResponseOutputRefusal,
)

from ankiops.config import NOTE_TYPES_DIR
from ankiops.fs import FileSystemAdapter
from ankiops.git import git_snapshot
from ankiops.models import ANKIOPS_KEY_FIELD, Note, NoteTypeConfig
from ankiops.serializer import deserialize, serialize
from ankiops.tags import normalize_tags

from .config_loader import is_task_config_file, load_llm_task_catalog
from .llm_db import LlmDb, LlmJobDetail, LlmJobListItem
from .model_registry import parse_model
from .schemas import (
    build_response_model,
    parsed_response_json,
    parsed_tag_updates,
    parsed_updates,
)
from .types import (
    DeckScope,
    DiscoveryCounts,
    DiscoveryItem,
    DiscoverySnapshot,
    EligibleCandidate,
    FieldAccess,
    LlmItemStatus,
    LlmJobResult,
    LlmJobStatus,
    NotePayload,
    PlanFieldSurface,
    TaskConfig,
    TaskExecutionProgress,
    TaskPlanResult,
    TaskRequestOptions,
    TaskRunSummary,
)

logger = logging.getLogger(__name__)


class ProgressReporter(Protocol):
    def __call__(self, progress: TaskExecutionProgress) -> object: ...


@dataclass(frozen=True)
class MaterializedTaskContext:
    task: TaskConfig
    note_type_configs: dict[str, NoteTypeConfig]
    serialized_data: dict[str, Any]
    discovery_snapshot: DiscoverySnapshot


@dataclass(frozen=True)
class OpenAIResult:
    parsed_response: object | None
    request_json: dict[str, Any]
    parsed_response_json: dict[str, Any] | None
    response_json: str | None
    error_message: str | None
    outcome: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    fatal: bool = False


@dataclass(frozen=True)
class EligibleBatch:
    note_type: str
    note_type_config: NoteTypeConfig
    candidates: tuple[EligibleCandidate, ...]

    @property
    def item_ids(self) -> list[int]:
        return [candidate.item_id for candidate in self.candidates]

    @property
    def note_count(self) -> int:
        return len(self.candidates)

    @property
    def payloads(self) -> list[NotePayload]:
        return [candidate.payload for candidate in self.candidates]


@dataclass(frozen=True)
class _BatchProcessResult:
    statuses: tuple[LlmItemStatus, ...]


@dataclass(frozen=True)
class _CandidateApplyResult:
    candidate: EligibleCandidate
    status: LlmItemStatus
    changed_fields: list[str]
    error_message: str | None = None


@dataclass
class _ExecutionProgressState:
    job_id: int
    task_name: str
    total: int
    completed: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0
    errors: int = 0
    canceled: int = 0

    def record_status(self, status: LlmItemStatus, *, count: int = 1) -> None:
        if count <= 0:
            return
        if status is LlmItemStatus.SUCCEEDED_UPDATED:
            self.updated += count
        elif status is LlmItemStatus.SUCCEEDED_UNCHANGED:
            self.unchanged += count
        elif status is LlmItemStatus.SKIPPED_NO_EDITABLE_FIELDS:
            self.skipped += count
        elif status is LlmItemStatus.CANCELED:
            self.canceled += count
        elif status in {
            LlmItemStatus.INVALID_NOTE,
            LlmItemStatus.NOTE_ERROR,
            LlmItemStatus.PROVIDER_ERROR,
            LlmItemStatus.FATAL_ERROR,
        }:
            self.errors += count
        else:
            return
        self.completed += count

    def snapshot(self, *, in_flight: int) -> TaskExecutionProgress:
        queued = max(self.total - self.completed - in_flight, 0)
        return TaskExecutionProgress(
            job_id=self.job_id,
            task_name=self.task_name,
            total=self.total,
            completed=self.completed,
            in_flight=in_flight,
            queued=queued,
            updated=self.updated,
            unchanged=self.unchanged,
            skipped=self.skipped,
            errors=self.errors,
            canceled=self.canceled,
        )


class LlmTaskExecutor:
    """Run one configured LLM task against serialized notes."""

    def __init__(
        self,
        *,
        collection_dir: Path,
        materialized_context: MaterializedTaskContext,
        no_auto_commit: bool,
        progress_callback: ProgressReporter | None = None,
    ) -> None:
        self.collection_dir = collection_dir
        self.materialized_context = materialized_context
        self.no_auto_commit = no_auto_commit
        self.progress_callback = progress_callback

    async def execute(self) -> LlmJobResult:
        task_context = self.materialized_context
        task = task_context.task
        db = LlmDb.open(self.collection_dir)
        try:
            job_id = db.start_job(
                task_name=task.name,
                model=task.model.model,
                model_id=task.model.model_id,
            )
            logger.debug(
                "Starting LLM task '%s' (model=%s, collection=%s, deck_scope=%s)",
                task.name,
                task.model,
                self.collection_dir,
                _format_deck_scope(task),
            )
            if not self.no_auto_commit:
                logger.debug("Creating pre-LLM git snapshot")
                git_snapshot(self.collection_dir, f"llm:{task.name}")
            else:
                logger.debug("Auto-commit disabled (--no-auto-commit)")

            progress_state = _build_progress_state(
                job_id=job_id,
                task_name=task.name,
                snapshot=task_context.discovery_snapshot,
            )
            candidates = _record_discovery_snapshot(
                db=db,
                job_id=job_id,
                snapshot=task_context.discovery_snapshot,
            )
            self._emit_progress(progress_state, in_flight=0)

            fatal_error: str | None = None
            if candidates:
                try:
                    await self._execute_candidates(
                        db=db,
                        job_id=job_id,
                        task=task,
                        candidates=candidates,
                        progress_state=progress_state,
                    )
                except RuntimeError as error:
                    fatal_error = str(error)
                    logger.error(fatal_error)

            if fatal_error is not None:
                canceled = db.mark_unfinished_items_canceled(job_id=job_id)
                progress_state.record_status(LlmItemStatus.CANCELED, count=canceled)

            self._emit_progress(progress_state, in_flight=0)
            aggregate = db.aggregate_job(job_id)
            failed = fatal_error is not None or aggregate.summary.errors > 0
            persisted = False
            if not failed:
                persisted = _persist_updates(
                    db=db,
                    job_id=job_id,
                    data=task_context.serialized_data,
                    collection_dir=self.collection_dir,
                )

            db.finalize_job(
                job_id=job_id,
                status=LlmJobStatus.FAILED if failed else LlmJobStatus.COMPLETED,
                persisted=persisted,
                fatal_error=fatal_error if failed else None,
            )
            aggregate = db.aggregate_job(job_id)
            logger.info(aggregate.summary.format())
            logger.debug("Usage: %s", aggregate.summary.format_usage())
            logger.debug("Cost: %s", aggregate.summary.format_cost())
            return LlmJobResult(
                job_id=job_id,
                status=aggregate.status.value,
                summary=aggregate.summary,
                failed=aggregate.failed,
                persisted=aggregate.persisted,
            )
        finally:
            db.close()

    async def _execute_candidates(
        self,
        *,
        db: LlmDb,
        job_id: int,
        task: TaskConfig,
        candidates: list[EligibleCandidate],
        progress_state: _ExecutionProgressState,
    ) -> None:
        api_key = _resolve_api_key(task.model.api_key)
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=task.model.base_url,
            timeout=60.0,
            max_retries=0,
        )
        try:
            batches = _build_candidate_batches(
                candidates,
                max_notes_per_request=task.request.max_notes_per_request,
            )
            in_flight_limit = min(max(task.model.concurrency, 1), len(batches))
            next_index = 0
            in_flight: dict[asyncio.Task[_BatchProcessResult], EligibleBatch] = {}

            def in_flight_notes() -> int:
                return sum(batch.note_count for batch in in_flight.values())

            def start(batch: EligibleBatch) -> None:
                task_handle = asyncio.create_task(
                    self._process_batch(
                        db=db,
                        job_id=job_id,
                        task=task,
                        client=client,
                        batch=batch,
                    )
                )
                in_flight[task_handle] = batch

            while next_index < in_flight_limit:
                start(batches[next_index])
                next_index += 1

            first_fatal: RuntimeError | None = None
            while in_flight:
                done, _pending = await asyncio.wait(
                    set(in_flight),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for completed in done:
                    batch = in_flight.pop(completed)
                    try:
                        result = await completed
                        for status in result.statuses:
                            progress_state.record_status(status)
                    except RuntimeError as error:
                        progress_state.record_status(
                            LlmItemStatus.FATAL_ERROR,
                            count=batch.note_count,
                        )
                        if first_fatal is None:
                            first_fatal = error
                    self._emit_progress(
                        progress_state,
                        in_flight=in_flight_notes(),
                    )

                if first_fatal is not None:
                    for pending_task in in_flight:
                        pending_task.cancel()
                    await asyncio.gather(*in_flight, return_exceptions=True)
                    raise first_fatal

                while next_index < len(batches) and len(in_flight) < in_flight_limit:
                    start(batches[next_index])
                    next_index += 1
        finally:
            await client.close()

    async def _process_batch(
        self,
        *,
        db: LlmDb,
        job_id: int,
        task: TaskConfig,
        client: AsyncOpenAI,
        batch: EligibleBatch,
    ) -> _BatchProcessResult:
        result = await _call_openai(
            client=client,
            task=task,
            batch=batch,
        )
        if result.parsed_response is None:
            status = (
                LlmItemStatus.FATAL_ERROR
                if result.fatal
                else LlmItemStatus.PROVIDER_ERROR
            )
            with db.write_tx():
                db.insert_request(
                    job_id=job_id,
                    item_ids=batch.item_ids,
                    outcome=result.outcome,
                    request_json=result.request_json,
                    parsed_response_json=result.parsed_response_json,
                    response_json=result.response_json,
                    error_message=result.error_message,
                    latency_ms=result.latency_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                )
                for candidate in batch.candidates:
                    db.update_job_item_status(
                        item_id=candidate.item_id,
                        item_status=status,
                        error_message=result.error_message,
                    )
            if result.fatal:
                raise RuntimeError(result.error_message or "Fatal OpenAI error")
            for candidate in batch.candidates:
                _log_note_error(
                    candidate,
                    result.error_message or "OpenAI request failed",
                )
            return _BatchProcessResult(statuses=tuple(status for _ in batch.candidates))

        try:
            apply_results = _apply_batch_parsed_response(
                parsed_response=result.parsed_response,
                batch=batch,
            )
        except ValueError as error:
            message = str(error)
            with db.write_tx():
                db.insert_request(
                    job_id=job_id,
                    item_ids=batch.item_ids,
                    outcome="validation_error",
                    request_json=result.request_json,
                    parsed_response_json=result.parsed_response_json,
                    response_json=result.response_json,
                    error_message=message,
                    latency_ms=result.latency_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                )
                for candidate in batch.candidates:
                    db.update_job_item_status(
                        item_id=candidate.item_id,
                        item_status=LlmItemStatus.NOTE_ERROR,
                        error_message=message,
                    )
            for candidate in batch.candidates:
                _log_note_error(candidate, message)
            return _BatchProcessResult(
                statuses=tuple(LlmItemStatus.NOTE_ERROR for _ in batch.candidates)
            )

        errors = [
            f"{item.candidate.payload.note_key}: {item.error_message}"
            for item in apply_results
            if item.error_message
        ]
        outcome = "validation_error" if errors else "success"
        request_error = "; ".join(errors) if errors else None
        with db.write_tx():
            db.insert_request(
                job_id=job_id,
                item_ids=batch.item_ids,
                outcome=outcome,
                request_json=result.request_json,
                parsed_response_json=result.parsed_response_json,
                response_json=result.response_json,
                error_message=request_error,
                latency_ms=result.latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
            for item in apply_results:
                db.update_job_item_status(
                    item_id=item.candidate.item_id,
                    item_status=item.status,
                    error_message=item.error_message,
                    changed_fields=item.changed_fields,
                )
        for item in apply_results:
            if item.error_message is not None:
                _log_note_error(item.candidate, item.error_message)
            elif item.changed_fields:
                logger.debug(
                    "  Updated %s in '%s' (%s): changed=%s",
                    item.candidate.payload.note_key,
                    item.candidate.deck_name,
                    item.candidate.payload.note_type,
                    ",".join(item.changed_fields),
                )
        return _BatchProcessResult(
            statuses=tuple(item.status for item in apply_results)
        )

    def _emit_progress(
        self,
        progress_state: _ExecutionProgressState,
        *,
        in_flight: int,
    ) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(progress_state.snapshot(in_flight=in_flight))
        except Exception:
            logger.debug("Progress callback raised an exception", exc_info=True)


def plan_task(
    *,
    collection_dir: Path,
    task_name: str,
    model_override: str | None = None,
    deck_override: str | None = None,
) -> TaskPlanResult:
    context = _materialize_task_context(
        collection_dir=collection_dir,
        task_name=task_name,
        model_override=model_override,
        deck_override=deck_override,
    )
    return _build_task_plan_result(
        task=context.task,
        note_type_configs=context.note_type_configs,
        snapshot=context.discovery_snapshot,
    )


async def run_task_async(
    *,
    collection_dir: Path,
    task_name: str,
    model_override: str | None = None,
    deck_override: str | None = None,
    no_auto_commit: bool = False,
    progress_callback: ProgressReporter | None = None,
) -> LlmJobResult:
    context = _materialize_task_context(
        collection_dir=collection_dir,
        task_name=task_name,
        model_override=model_override,
        deck_override=deck_override,
    )
    return await LlmTaskExecutor(
        collection_dir=collection_dir,
        materialized_context=context,
        no_auto_commit=no_auto_commit,
        progress_callback=progress_callback,
    ).execute()


def run_task(
    *,
    collection_dir: Path,
    task_name: str,
    model_override: str | None = None,
    deck_override: str | None = None,
    no_auto_commit: bool = False,
    progress_callback: ProgressReporter | None = None,
) -> LlmJobResult:
    return asyncio.run(
        run_task_async(
            collection_dir=collection_dir,
            task_name=task_name,
            model_override=model_override,
            deck_override=deck_override,
            no_auto_commit=no_auto_commit,
            progress_callback=progress_callback,
        )
    )


def list_jobs(*, collection_dir: Path) -> list[LlmJobListItem]:
    db = LlmDb.open(collection_dir)
    try:
        return db.list_jobs()
    finally:
        db.close()


def show_job(*, collection_dir: Path, job_id: str | int) -> LlmJobDetail | None:
    db = LlmDb.open(collection_dir)
    try:
        resolved_job_id = (
            db.resolve_job_id(job_id) if isinstance(job_id, str) else int(job_id)
        )
        if resolved_job_id is None:
            return None
        return db.get_job_detail(resolved_job_id)
    finally:
        db.close()


def _materialize_task_context(
    *,
    collection_dir: Path,
    task_name: str,
    model_override: str | None,
    deck_override: str | None,
) -> MaterializedTaskContext:
    task, note_type_configs = _load_task(
        collection_dir=collection_dir,
        task_name=task_name,
    )
    if deck_override is not None:
        task = replace(task, decks=DeckScope(deck_root=deck_override))
    if model_override is not None:
        model = parse_model(model_override, collection_dir=collection_dir)
        if model is None:
            raise ValueError(f"Unknown model '{model_override}'")
        task = replace(task, model=model)

    deck, no_subdecks = _resolve_serializer_scope(task)
    serialized_data = serialize(
        collection_dir,
        deck=deck,
        no_subdecks=no_subdecks,
        note_types_dir=collection_dir / NOTE_TYPES_DIR,
    )
    discovery_snapshot = _discover_candidates(
        data=serialized_data,
        task=task,
        note_type_configs=note_type_configs,
    )
    return MaterializedTaskContext(
        task=task,
        note_type_configs=note_type_configs,
        serialized_data=serialized_data,
        discovery_snapshot=discovery_snapshot,
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
            raise ValueError(
                "Invalid LLM task configuration:\n" + "\n".join(task_errors)
            )
        shared_errors = [
            message
            for path, message in catalog.errors.items()
            if not _is_task_file(path)
        ]
        if shared_errors:
            raise ValueError(
                "Invalid LLM task configuration:\n" + "\n".join(shared_errors)
            )
        raise ValueError(f"Unknown task '{task_name}'")
    return task, {config.name: config for config in note_type_configs}


def _discover_candidates(
    *,
    data: dict[str, Any],
    task: TaskConfig,
    note_type_configs: dict[str, NoteTypeConfig],
) -> DiscoverySnapshot:
    decks = data.get("decks")
    if not isinstance(decks, list):
        raise ValueError("Serialized collection is missing a decks list")

    items: list[DiscoveryItem] = []
    decks_seen = 0
    decks_matched = 0
    notes_seen = 0
    ordinal = 0

    for deck in decks:
        if not isinstance(deck, dict):
            continue
        deck_name = deck.get("name")
        notes = deck.get("notes")
        if not isinstance(deck_name, str) or not isinstance(notes, list):
            continue
        decks_seen += 1
        notes_seen += len(notes)
        if not task.decks.matches(deck_name):
            continue
        decks_matched += 1

        for note in notes:
            if not isinstance(note, dict):
                continue
            ordinal += 1
            items.append(
                _discover_note(
                    task=task,
                    deck_name=deck_name,
                    ordinal=ordinal,
                    note=note,
                    note_type_configs=note_type_configs,
                )
            )

    return DiscoverySnapshot(
        counts=DiscoveryCounts(
            decks_seen=decks_seen,
            decks_matched=decks_matched,
            notes_seen=notes_seen,
        ),
        items=items,
    )


def _discover_note(
    *,
    task: TaskConfig,
    deck_name: str,
    ordinal: int,
    note: dict[str, Any],
    note_type_configs: dict[str, NoteTypeConfig],
) -> DiscoveryItem:
    note_key = note.get("note_key")
    note_type_name = note.get("note_type")
    fields = note.get("fields")
    note_key_value = note_key if isinstance(note_key, str) else None
    note_type_value = note_type_name if isinstance(note_type_name, str) else None

    if not isinstance(note_key, str) or not isinstance(note_type_name, str):
        return _invalid_discovery_item(
            deck_name=deck_name,
            ordinal=ordinal,
            note_key=note_key_value,
            note_type=note_type_value,
            error_message="Serialized note is missing note_key or note_type",
            serialized_note=note,
        )
    if not isinstance(fields, dict):
        return _invalid_discovery_item(
            deck_name=deck_name,
            ordinal=ordinal,
            note_key=note_key,
            note_type=note_type_name,
            error_message="Serialized note fields must be a mapping",
            serialized_note=note,
        )

    note_type_config = note_type_configs.get(note_type_name)
    if note_type_config is None:
        return _invalid_discovery_item(
            deck_name=deck_name,
            ordinal=ordinal,
            note_key=note_key,
            note_type=note_type_name,
            error_message=f"Unknown note type '{note_type_name}' in serialized note",
            serialized_note=note,
        )

    editable_fields: dict[str, str] = {}
    read_only_fields: dict[str, str] = {}
    for field in note_type_config.fields:
        if field.name == ANKIOPS_KEY_FIELD.name:
            continue
        access = task.field_access(note_type_name, field.name)
        if access is FieldAccess.HIDDEN:
            continue
        raw_value = fields.get(field.name, "")
        if raw_value is None:
            raw_value = ""
        if not isinstance(raw_value, str):
            return _invalid_discovery_item(
                deck_name=deck_name,
                ordinal=ordinal,
                note_key=note_key,
                note_type=note_type_name,
                error_message=f"Serialized field '{field.name}' must be a string",
                serialized_note=note,
            )
        if access is FieldAccess.READ_ONLY:
            read_only_fields[field.name] = raw_value
        else:
            editable_fields[field.name] = raw_value

    tags = normalize_tags(note.get("tags", ()))
    editable_tags = tags if task.tag_access is FieldAccess.EDITABLE else None
    read_only_tags = tags if task.tag_access is FieldAccess.READ_ONLY else None
    has_editable_surface = bool(editable_fields) or editable_tags is not None
    has_visible_field_surface = bool(editable_fields) or bool(read_only_fields)
    if not has_editable_surface or (
        editable_tags is not None
        and not editable_fields
        and not has_visible_field_surface
    ):
        skip_reason = (
            "no readable fields" if has_editable_surface else "no editable fields"
        )
        return DiscoveryItem(
            ordinal=ordinal,
            deck_name=deck_name,
            note_key=note_key,
            note_type=note_type_name,
            item_status=LlmItemStatus.SKIPPED_NO_EDITABLE_FIELDS,
            skip_reason=skip_reason,
            error_message=None,
            payload=None,
            note_type_config=note_type_config,
            serialized_note=note,
        )

    return DiscoveryItem(
        ordinal=ordinal,
        deck_name=deck_name,
        note_key=note_key,
        note_type=note_type_name,
        item_status=LlmItemStatus.QUEUED,
        skip_reason=None,
        error_message=None,
        payload=NotePayload(
            note_key=note_key,
            note_type=note_type_name,
            editable_fields=editable_fields,
            read_only_fields=read_only_fields,
            editable_tags=editable_tags,
            read_only_tags=read_only_tags,
        ),
        note_type_config=note_type_config,
        serialized_note=note,
    )


def _invalid_discovery_item(
    *,
    deck_name: str,
    ordinal: int,
    note_key: str | None,
    note_type: str | None,
    error_message: str,
    serialized_note: dict[str, Any] | None,
) -> DiscoveryItem:
    return DiscoveryItem(
        ordinal=ordinal,
        deck_name=deck_name,
        note_key=note_key,
        note_type=note_type,
        item_status=LlmItemStatus.INVALID_NOTE,
        skip_reason=None,
        error_message=error_message,
        payload=None,
        note_type_config=None,
        serialized_note=serialized_note,
    )


def _record_discovery_snapshot(
    *,
    db: LlmDb,
    job_id: int,
    snapshot: DiscoverySnapshot,
) -> list[EligibleCandidate]:
    candidates: list[EligibleCandidate] = []
    for item in snapshot.items:
        candidate = _record_discovery_item(db=db, job_id=job_id, item=item)
        if candidate is not None:
            candidates.append(candidate)

    db.set_discovery_counts(
        job_id=job_id,
        decks_seen=snapshot.counts.decks_seen,
        decks_matched=snapshot.counts.decks_matched,
        notes_seen=snapshot.counts.notes_seen,
    )
    return candidates


def _record_discovery_item(
    *,
    db: LlmDb,
    job_id: int,
    item: DiscoveryItem,
) -> EligibleCandidate | None:
    if item.item_status is LlmItemStatus.INVALID_NOTE:
        db.insert_job_item(
            job_id=job_id,
            ordinal=item.ordinal,
            deck_name=item.deck_name,
            note_key=item.note_key,
            note_type=item.note_type,
            item_status=item.item_status,
            skip_reason=None,
            error_message=item.error_message or "Invalid note",
        )
        logger.error(
            "LLM note error for %s in '%s' (%s): %s",
            item.note_key or "unknown",
            item.deck_name,
            item.note_type or "unknown",
            item.error_message or "Invalid note",
        )
        return None

    if item.item_status is LlmItemStatus.SKIPPED_NO_EDITABLE_FIELDS:
        db.insert_job_item(
            job_id=job_id,
            ordinal=item.ordinal,
            deck_name=item.deck_name,
            note_key=item.note_key,
            note_type=item.note_type,
            item_status=item.item_status,
            skip_reason=item.skip_reason,
        )
        return None

    if (
        item.payload is None
        or item.note_type_config is None
        or item.serialized_note is None
    ):
        return None

    item_id = db.insert_job_item(
        job_id=job_id,
        ordinal=item.ordinal,
        deck_name=item.deck_name,
        note_key=item.payload.note_key,
        note_type=item.payload.note_type,
        item_status=LlmItemStatus.QUEUED,
        skip_reason=None,
    )
    return EligibleCandidate(
        item_id=item_id,
        deck_name=item.deck_name,
        payload=item.payload,
        note_type_config=item.note_type_config,
        serialized_note=item.serialized_note,
    )


def _build_candidate_batches(
    candidates: list[EligibleCandidate],
    *,
    max_notes_per_request: int,
) -> list[EligibleBatch]:
    by_note_type: dict[str, list[EligibleCandidate]] = {}
    for candidate in candidates:
        by_note_type.setdefault(candidate.payload.note_type, []).append(candidate)

    batches: list[EligibleBatch] = []
    for note_type, grouped_candidates in by_note_type.items():
        for start in range(0, len(grouped_candidates), max_notes_per_request):
            chunk = grouped_candidates[start : start + max_notes_per_request]
            batches.append(
                EligibleBatch(
                    note_type=note_type,
                    note_type_config=chunk[0].note_type_config,
                    candidates=tuple(chunk),
                )
            )
    return batches


def _build_payload_batches(
    payloads: list[NotePayload],
    *,
    max_notes_per_request: int,
) -> list[list[NotePayload]]:
    by_note_type: dict[str, list[NotePayload]] = {}
    for payload in payloads:
        by_note_type.setdefault(payload.note_type, []).append(payload)

    batches: list[list[NotePayload]] = []
    for grouped_payloads in by_note_type.values():
        for start in range(0, len(grouped_payloads), max_notes_per_request):
            batches.append(grouped_payloads[start : start + max_notes_per_request])
    return batches


async def _call_openai(
    *,
    client: AsyncOpenAI,
    task: TaskConfig,
    batch: EligibleBatch,
) -> OpenAIResult:
    response_model = build_response_model(
        note_type=batch.note_type,
        payloads=batch.payloads,
    )
    instructions, user_input = _build_request_content(task=task, batch=batch)
    request_kwargs: dict[str, Any] = {
        "model": task.model.model_id,
        "instructions": instructions,
        "input": user_input,
        "text_format": response_model,
    }
    if task.request.temperature is not None:
        request_kwargs["temperature"] = task.request.temperature
    if task.request.reasoning is not None:
        request_kwargs["reasoning"] = {"effort": task.request.reasoning}
    request_json = dict(request_kwargs)
    request_json["text_format"] = response_model.__name__
    started_at = time.monotonic()
    try:
        response: ParsedResponse[object] = await client.responses.parse(
            **request_kwargs
        )
    except AuthenticationError as error:
        return _openai_error_result(
            request_json=request_json,
            error_message=f"OpenAI authentication failed: {error}",
            outcome="fatal_error",
            started_at=started_at,
            fatal=True,
        )
    except APIConnectionError as error:
        return _openai_error_result(
            request_json=request_json,
            error_message=f"OpenAI connection error: {error}",
            outcome="provider_error",
            started_at=started_at,
        )
    except APIStatusError as error:
        status_code = getattr(error, "status_code", "unknown")
        return _openai_error_result(
            request_json=request_json,
            error_message=f"OpenAI returned HTTP {status_code}: {error}",
            outcome="provider_error",
            started_at=started_at,
        )
    except Exception as error:
        return _openai_error_result(
            request_json=request_json,
            error_message=f"Unexpected OpenAI error: {error}",
            outcome="fatal_error",
            started_at=started_at,
            fatal=True,
        )

    latency_ms = round((time.monotonic() - started_at) * 1000)
    refusal = _extract_refusal_text(response)
    response_json = _response_to_json(response)
    if refusal is not None:
        return OpenAIResult(
            parsed_response=None,
            request_json=request_json,
            parsed_response_json=None,
            response_json=response_json,
            error_message=refusal,
            outcome="refusal",
            latency_ms=latency_ms,
            input_tokens=_usage_value(response, "input_tokens"),
            output_tokens=_usage_value(response, "output_tokens"),
        )

    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        return OpenAIResult(
            parsed_response=None,
            request_json=request_json,
            parsed_response_json=None,
            response_json=response_json,
            error_message="OpenAI response did not include parsed structured output",
            outcome="provider_error",
            latency_ms=latency_ms,
            input_tokens=_usage_value(response, "input_tokens"),
            output_tokens=_usage_value(response, "output_tokens"),
        )

    return OpenAIResult(
        parsed_response=parsed,
        request_json=request_json,
        parsed_response_json=parsed_response_json(parsed),
        response_json=response_json,
        error_message=None,
        outcome="success",
        latency_ms=latency_ms,
        input_tokens=_usage_value(response, "input_tokens"),
        output_tokens=_usage_value(response, "output_tokens"),
    )


def _build_request_content(
    *,
    task: TaskConfig,
    batch: EligibleBatch,
) -> tuple[str, str]:
    payload = {
        "user_prompt": task.user_prompt,
        "note_type": batch.note_type,
        "notes": [
            _build_note_request_payload(candidate.payload)
            for candidate in batch.candidates
        ],
    }
    return (
        task.system_prompt.strip(),
        json.dumps(payload, ensure_ascii=False),
    )


def _build_note_request_payload(payload: NotePayload) -> dict[str, object]:
    note_payload: dict[str, object] = {
        "note_key": payload.note_key,
        "editable_fields": payload.editable_fields,
    }
    if payload.read_only_fields:
        note_payload["read_only_fields"] = payload.read_only_fields
    if payload.editable_tags is not None:
        note_payload["editable_tags"] = list(payload.editable_tags)
    if payload.read_only_tags is not None:
        note_payload["read_only_tags"] = list(payload.read_only_tags)
    return note_payload


def _openai_error_result(
    *,
    request_json: dict[str, Any],
    error_message: str,
    outcome: str,
    started_at: float,
    fatal: bool = False,
) -> OpenAIResult:
    return OpenAIResult(
        parsed_response=None,
        request_json=request_json,
        parsed_response_json=None,
        response_json=None,
        error_message=error_message,
        outcome=outcome,
        latency_ms=round((time.monotonic() - started_at) * 1000),
        input_tokens=0,
        output_tokens=0,
        fatal=fatal,
    )


def _apply_batch_parsed_response(
    *,
    parsed_response: object,
    batch: EligibleBatch,
) -> list[_CandidateApplyResult]:
    updates = parsed_updates(parsed_response)
    tag_updates = parsed_tag_updates(parsed_response)
    candidates_by_note_key = {
        candidate.payload.note_key: candidate for candidate in batch.candidates
    }
    unexpected_note_keys = sorted(
        {
            note_key
            for note_key, _field_name, _value in updates
            if note_key not in candidates_by_note_key
        }
        | {
            note_key
            for note_key, _tags in tag_updates
            if note_key not in candidates_by_note_key
        }
    )
    if unexpected_note_keys:
        raise ValueError(
            f"Model returned update for unexpected note_key '{unexpected_note_keys[0]}'"
        )

    updates_by_note_key: dict[str, list[tuple[str, str, str]]] = {
        note_key: [] for note_key in candidates_by_note_key
    }
    tag_updates_by_note_key: dict[str, list[tuple[str, list[str]]]] = {
        note_key: [] for note_key in candidates_by_note_key
    }
    for update in updates:
        updates_by_note_key[update[0]].append(update)
    for tag_update in tag_updates:
        tag_updates_by_note_key[tag_update[0]].append(tag_update)

    results: list[_CandidateApplyResult] = []
    for candidate in batch.candidates:
        try:
            changed_fields = _apply_candidate_updates(
                candidate=candidate,
                updates=updates_by_note_key[candidate.payload.note_key],
                tag_updates=tag_updates_by_note_key[candidate.payload.note_key],
            )
        except ValueError as error:
            results.append(
                _CandidateApplyResult(
                    candidate=candidate,
                    status=LlmItemStatus.NOTE_ERROR,
                    changed_fields=[],
                    error_message=str(error),
                )
            )
        else:
            results.append(
                _CandidateApplyResult(
                    candidate=candidate,
                    status=(
                        LlmItemStatus.SUCCEEDED_UPDATED
                        if changed_fields
                        else LlmItemStatus.SUCCEEDED_UNCHANGED
                    ),
                    changed_fields=changed_fields,
                )
            )
    return results


def _apply_candidate_updates(
    *,
    candidate: EligibleCandidate,
    updates: list[tuple[str, str, str]],
    tag_updates: list[tuple[str, list[str]]] | None = None,
) -> list[str]:
    seen_fields: set[str] = set()
    edits: dict[str, str] = {}
    for _note_key, field_name, value in updates:
        if field_name in seen_fields:
            raise ValueError(f"Model returned duplicate update for '{field_name}'")
        seen_fields.add(field_name)
        if field_name not in candidate.payload.editable_fields:
            if field_name in candidate.payload.read_only_fields:
                raise ValueError(
                    f"Model attempted to update read-only field '{field_name}'"
                )
            raise ValueError(
                f"Model attempted to update hidden or unknown field '{field_name}'"
            )
        edits[field_name] = value

    tag_edit: tuple[str, ...] | None = None
    if tag_updates:
        if len(tag_updates) > 1:
            raise ValueError("Model returned duplicate update for 'tags'")
        if candidate.payload.editable_tags is None:
            if candidate.payload.read_only_tags is not None:
                raise ValueError("Model attempted to update read-only tags")
            raise ValueError("Model attempted to update hidden tags")
        _note_key, raw_tags = tag_updates[0]
        tag_edit = normalize_tags(raw_tags)

    raw_fields = candidate.serialized_note.get("fields")
    if not isinstance(raw_fields, dict):
        raise ValueError("Serialized note fields must be a mapping")

    next_fields = dict(raw_fields)
    next_tags = normalize_tags(candidate.serialized_note.get("tags", ()))
    changed_fields: list[str] = []
    for field_name, value in edits.items():
        if next_fields.get(field_name, "") != value:
            next_fields[field_name] = value
            changed_fields.append(field_name)
    if tag_edit is not None and next_tags != tag_edit:
        next_tags = tag_edit
        changed_fields.append("tags")

    if not changed_fields:
        return []

    note = Note(
        note_key=candidate.payload.note_key,
        note_type=candidate.payload.note_type,
        fields=next_fields,
        tags=next_tags,
    )
    errors = [*_validate_cloze_text_fields(note, candidate.note_type_config)]
    errors.extend(note.validate(candidate.note_type_config))
    if errors:
        raise ValueError("; ".join(errors))

    candidate.serialized_note["fields"] = next_fields
    candidate.serialized_note["tags"] = list(next_tags)
    return changed_fields


def _validate_cloze_text_fields(
    note: Note,
    config: NoteTypeConfig,
) -> list[str]:
    if not config.is_cloze:
        return []
    errors: list[str] = []
    field_names = _cloze_template_field_names(config)
    for field_name in sorted(field_names):
        value = note.fields.get(field_name, "")
        if "{{c" not in value:
            errors.append(
                f"{note.note_type} field '{field_name}' must contain cloze syntax "
                "(e.g. {{c1::answer}})"
            )
    return errors


def _cloze_template_field_names(config: NoteTypeConfig) -> set[str]:
    """Find the actualname of the cloze field used in the cloze template."""
    cloze_template_field_pattern = re.compile(
        r"\{\{\s*(?:[A-Za-z]+:)*cloze:([^}]+?)\s*\}\}",
        re.IGNORECASE,
    )
    field_names: set[str] = set()
    for template in config.templates:
        for template_value in template.values():
            if not isinstance(template_value, str):
                continue
            for match in cloze_template_field_pattern.finditer(template_value):
                field_name = match.group(1).strip()
                if field_name:
                    field_names.add(field_name)
    return field_names


def _persist_updates(
    *,
    db: LlmDb,
    job_id: int,
    data: dict[str, Any],
    collection_dir: Path,
) -> bool:
    aggregate = db.aggregate_job(job_id)
    if aggregate.summary.updated <= 0:
        return False

    deserialize(
        data,
        collection_dir=collection_dir,
        note_types_dir=collection_dir / NOTE_TYPES_DIR,
        overwrite=True,
        quiet=True,
    )
    db.set_applied_for_updated_items(job_id=job_id)
    deck_count = len([deck for deck in data.get("decks", []) if isinstance(deck, dict)])
    logger.info(
        "Persisted %d updated note(s) across %d deck file(s)",
        aggregate.summary.updated,
        deck_count,
    )
    return True


def _build_progress_state(
    *,
    job_id: int,
    task_name: str,
    snapshot: DiscoverySnapshot,
) -> _ExecutionProgressState:
    state = _ExecutionProgressState(
        job_id=job_id,
        task_name=task_name,
        total=len(snapshot.items),
    )
    for item in snapshot.items:
        state.record_status(item.item_status)
    return state


def _build_task_plan_result(
    *,
    task: TaskConfig,
    note_type_configs: dict[str, NoteTypeConfig],
    snapshot: DiscoverySnapshot,
) -> TaskPlanResult:
    eligible_items = [
        item
        for item in snapshot.items
        if item.item_status is LlmItemStatus.QUEUED and item.payload is not None
    ]
    eligible_payloads = [
        item.payload for item in eligible_items if item.payload is not None
    ]
    payload_batches = _build_payload_batches(
        eligible_payloads,
        max_notes_per_request=task.request.max_notes_per_request,
    )
    skipped = sum(
        1
        for item in snapshot.items
        if item.item_status is LlmItemStatus.SKIPPED_NO_EDITABLE_FIELDS
    )
    errors = sum(
        1 for item in snapshot.items if item.item_status is LlmItemStatus.INVALID_NOTE
    )
    input_tokens_estimate = sum(
        _estimate_batch_input_tokens(task, payload_batch)
        for payload_batch in payload_batches
    )
    summary = TaskRunSummary(
        task_name=task.name,
        model=task.model,
        decks_seen=snapshot.counts.decks_seen,
        decks_matched=snapshot.counts.decks_matched,
        notes_seen=snapshot.counts.notes_seen,
        eligible=len(eligible_items),
        skipped_no_editable_fields=skipped,
        errors=errors,
        requests=len(payload_batches),
    )
    return TaskPlanResult(
        task_name=task.name,
        model=task.model,
        deck_scope=_format_deck_scope(task),
        serializer_scope=_format_serializer_scope(task),
        system_prompt_path=(
            str(task.system_prompt_path)
            if task.system_prompt_path is not None
            else None
        ),
        user_prompt_path=(
            str(task.user_prompt_path) if task.user_prompt_path is not None else None
        ),
        system_prompt=task.system_prompt,
        user_prompt=task.user_prompt,
        request_defaults=_format_request_defaults(task.request),
        summary=summary,
        field_surface=_build_plan_field_surface(
            task=task,
            note_type_configs=note_type_configs,
            snapshot_items=snapshot.items,
        ),
        requests_estimate=len(payload_batches),
        input_tokens_estimate=input_tokens_estimate,
    )


def _build_plan_field_surface(
    *,
    task: TaskConfig,
    note_type_configs: dict[str, NoteTypeConfig],
    snapshot_items: list[DiscoveryItem],
) -> list[PlanFieldSurface]:
    observed_note_types = {
        item.note_type
        for item in snapshot_items
        if item.note_type is not None and item.note_type_config is not None
    }
    surface: list[PlanFieldSurface] = []
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
            if access is FieldAccess.EDITABLE:
                editable_fields.append(field.name)
            elif access is FieldAccess.READ_ONLY:
                read_only_fields.append(field.name)
            else:
                hidden_fields.append(field.name)
        candidate_notes = sum(
            1
            for item in snapshot_items
            if item.item_status is LlmItemStatus.QUEUED and item.note_type == note_type
        )
        surface.append(
            PlanFieldSurface(
                note_type=note_type,
                candidate_notes=candidate_notes,
                editable_fields=editable_fields,
                read_only_fields=read_only_fields,
                hidden_fields=hidden_fields,
                tag_access=task.tag_access,
            )
        )
    return surface


def _estimate_batch_input_tokens(task: TaskConfig, payloads: list[NotePayload]) -> int:
    if not payloads:
        return 0
    request_payload = {
        "user_prompt": task.user_prompt,
        "note_type": payloads[0].note_type,
        "notes": [_build_note_request_payload(payload) for payload in payloads],
    }
    return _estimate_tokens(
        "\n".join(
            [
                task.system_prompt,
                json.dumps(request_payload, ensure_ascii=False),
            ]
        )
    )


def _estimate_tokens(text: str) -> int:
    value = text.strip()
    if not value:
        return 0
    return max(1, (len(value) + 3) // 4)


def _resolve_serializer_scope(task: TaskConfig) -> tuple[str | None, bool]:
    return task.decks.deck_root, False


def _format_deck_scope(task: TaskConfig) -> str:
    return task.decks.deck_root or "all decks"


def _format_serializer_scope(task: TaskConfig) -> str:
    deck, no_subdecks = _resolve_serializer_scope(task)
    if deck is None:
        return "all markdown decks"
    suffix = "exact deck only" if no_subdecks else "including subdecks"
    return f"{deck} ({suffix})"


def _format_request_defaults(request: TaskRequestOptions) -> str:
    parts = [f"max_notes_per_request={request.max_notes_per_request}"]
    if request.temperature is not None:
        parts.append(f"temperature={request.temperature:g}")
    if request.reasoning is not None:
        parts.append(f"reasoning={request.reasoning}")
    return ", ".join(parts)


def _is_task_file(path: str) -> bool:
    path_obj = Path(path)
    return path_obj.parent.name == "llm" and is_task_config_file(path_obj)


def _is_task_file_for_name(path: str, task_name: str) -> bool:
    path_obj = Path(path)
    return _is_task_file(path) and path_obj.stem == task_name


def _resolve_api_key(configured_value: str) -> str:
    if not configured_value.startswith("$"):
        return configured_value
    env_name = configured_value[1:].strip()
    if not env_name:
        raise RuntimeError("Model api_key env reference must include a variable")
    resolved = os.environ.get(env_name)
    if not resolved:
        raise RuntimeError(f"Required environment variable '{env_name}' is not set")
    return resolved


def _usage_value(response: ParsedResponse[object], name: str) -> int:
    usage = getattr(response, "usage", None)
    value = getattr(usage, name, 0)
    return value if isinstance(value, int) else 0


def _response_to_json(response: ParsedResponse[object]) -> str | None:
    try:
        return response.model_dump_json(warnings=False)
    except Exception:
        return None


def _extract_refusal_text(response: ParsedResponse[object]) -> str | None:
    refusal_parts = [
        part.refusal
        for item in response.output
        if isinstance(item, ResponseOutputMessage)
        for part in item.content
        if isinstance(part, ResponseOutputRefusal) and part.refusal
    ]
    return "\n".join(refusal_parts) if refusal_parts else None


def _log_note_error(candidate: EligibleCandidate, message: str) -> None:
    logger.error(
        "LLM note error for %s in '%s' (%s): %s",
        candidate.payload.note_key,
        candidate.deck_name,
        candidate.payload.note_type,
        message,
    )
