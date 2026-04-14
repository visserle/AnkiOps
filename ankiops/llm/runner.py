"""Execution of collection-local LLM tasks."""

from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator

from ankiops.collection_serializer import (
    deserialize_collection_data,
    serialize_collection,
)
from ankiops.config import NOTE_TYPES_DIR
from ankiops.fs import FileSystemAdapter
from ankiops.git import git_snapshot
from ankiops.models import Note, NoteTypeConfig

from .config_loader import load_llm_task_catalog
from .llm_db import LlmDb, LlmJobDetail, LlmJobListItem
from .llm_errors import LlmFatalError, LlmNoteError
from .llm_models import (
    LlmAttemptResultType,
    LlmFinalStatus,
    LlmJobResult,
    LlmJobStatus,
    RunFailurePolicy,
    TaskConfig,
    TaskPlanResult,
    TaskRunSummary,
)
from .provider_client import ProviderClient
from .task_attempts import AttemptRecorder
from .task_discovery import discover_and_record_candidates
from .task_options import (
    apply_deck_override,
    format_deck_scope,
    format_request_defaults,
    resolve_failure_policy,
    resolve_model,
    resolve_serializer_scope,
)
from .task_planner import build_task_plan_result
from .task_runtime_types import EligibleCandidate
from .task_snapshot import task_from_snapshot, task_to_snapshot

logger = logging.getLogger(__name__)

ProviderClientFactory = Callable[[TaskConfig], Any]
CollectionSerializer = Callable[..., dict[str, Any]]
CollectionDeserializer = Callable[..., None]
Snapshotter = Callable[[Path, str], bool]


def _default_provider_client_factory(task: TaskConfig) -> ProviderClient:
    return ProviderClient(task)


@contextmanager
def _open_llm_db(collection_dir: Path) -> Iterator[LlmDb]:
    db = LlmDb.open(collection_dir)
    try:
        yield db
    finally:
        db.close()


def _apply_note_update(
    *,
    serialized_note: dict[str, Any],
    payload,
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


def _unexpected_online_execution_error(error: Exception) -> LlmFatalError:
    message = str(error).strip() or error.__class__.__name__
    return LlmFatalError(f"Unexpected online execution error: {message}")


class LlmTaskExecutor:
    """Standalone async LLM runtime pipeline with DB-backed state."""

    def __init__(
        self,
        *,
        collection_dir: Path,
        task: TaskConfig,
        note_type_configs: dict[str, NoteTypeConfig],
        model_override: str | None,
        no_auto_commit: bool,
        failure_policy: RunFailurePolicy | str,
        resume_from_job_id: int | None = None,
        resume_source_items: dict[str, int] | None = None,
        provider_client_factory: ProviderClientFactory | None = None,
        serialize_collection_fn: CollectionSerializer | None = None,
        deserialize_collection_fn: CollectionDeserializer | None = None,
        snapshot_fn: Snapshotter | None = None,
    ) -> None:
        self.collection_dir = collection_dir
        self.task = task
        self.note_type_configs = note_type_configs
        self.model_override = model_override
        self.no_auto_commit = no_auto_commit
        self.failure_policy = resolve_failure_policy(failure_policy)
        self.resume_from_job_id = resume_from_job_id
        self.resume_source_items = resume_source_items
        self.provider_client_factory = (
            provider_client_factory or _default_provider_client_factory
        )
        self.serialize_collection_fn = serialize_collection_fn or serialize_collection
        self.deserialize_collection_fn = (
            deserialize_collection_fn or deserialize_collection_data
        )
        self.snapshot_fn = snapshot_fn or git_snapshot

    async def execute(self) -> LlmJobResult:
        task = self.task
        model = resolve_model(
            task,
            self.model_override,
            collection_dir=self.collection_dir,
        )
        task = replace(task, model=model)

        db = LlmDb.open(self.collection_dir)

        job_id = db.start_job(
            task_name=task.name,
            model_name=model.name,
            api_model=model.api_id,
            failure_policy=self.failure_policy,
            config_snapshot=task_to_snapshot(task),
            resume_from_job_id=self.resume_from_job_id,
        )

        deck, no_subdecks = resolve_serializer_scope(task)
        logger.debug(
            "Starting LLM task '%s' (model=%s, api_model=%s, collection=%s, "
            "deck_scope=%s, failure_policy=%s)",
            task.name,
            model,
            model.api_id,
            self.collection_dir,
            format_deck_scope(task),
            self.failure_policy.value,
        )
        logger.debug("LLM request defaults: %s", format_request_defaults(task))
        logger.debug(
            "LLM serializer scope: %s",
            format_deck_scope(task),
        )
        aggregate = None
        provider_client = None
        attempt_recorder = AttemptRecorder(db=db, provider=model.provider)

        try:
            provider_client = self.provider_client_factory(task)
            if not self.no_auto_commit:
                logger.debug("Creating pre-LLM git snapshot")
                self.snapshot_fn(self.collection_dir, f"llm:{task.name}")
            else:
                logger.debug("Auto-commit disabled (--no-auto-commit)")

            data = self.serialize_collection_fn(
                self.collection_dir,
                deck=deck,
                no_subdecks=no_subdecks,
                note_types_dir=self.collection_dir / NOTE_TYPES_DIR,
            )

            candidates = discover_and_record_candidates(
                db=db,
                job_id=job_id,
                data=data,
                task=task,
                note_type_configs=self.note_type_configs,
                resume_source_items=self.resume_source_items,
                logger=logger,
            )
            await self._execute_candidates_online(
                db=db,
                job_id=job_id,
                task=task,
                api_model=model.api_id,
                provider_client=provider_client,
                candidates=candidates,
                attempt_recorder=attempt_recorder,
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
            self._log_summary(summary)
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

        self._log_summary(aggregate.summary)
        return LlmJobResult(
            job_id=job_id,
            status=aggregate.status.value,
            summary=aggregate.summary,
            failed=aggregate.failed,
            persisted=aggregate.persisted,
        )

    @staticmethod
    def _log_summary(summary: TaskRunSummary) -> None:
        logger.info(summary.format())
        logger.debug("Usage: %s", summary.format_usage())
        logger.debug("Cost: %s", summary.format_cost())

    async def _execute_candidates_online(
        self,
        *,
        db: LlmDb,
        job_id: int,
        task: TaskConfig,
        api_model: str,
        provider_client,
        candidates: list[EligibleCandidate],
        attempt_recorder: AttemptRecorder,
    ) -> None:
        semaphore = asyncio.Semaphore(task.concurrency)
        first_fatal_error: LlmFatalError | None = None

        async def _run_with_semaphore(candidate: EligibleCandidate) -> None:
            async with semaphore:
                await self._process_online_candidate(
                    db=db,
                    task=task,
                    api_model=api_model,
                    provider_client=provider_client,
                    candidate=candidate,
                    attempt_recorder=attempt_recorder,
                )

        tasks = [
            asyncio.create_task(_run_with_semaphore(candidate))
            for candidate in candidates
        ]

        if not tasks:
            return

        for completed in asyncio.as_completed(tasks):
            try:
                await completed
            except LlmFatalError as error:
                if first_fatal_error is None:
                    first_fatal_error = error
                for pending in tasks:
                    if not pending.done():
                        pending.cancel()
                break
            except Exception as error:
                if first_fatal_error is None:
                    first_fatal_error = _unexpected_online_execution_error(error)
                for pending in tasks:
                    if not pending.done():
                        pending.cancel()
                break
        await asyncio.gather(*tasks, return_exceptions=True)
        if first_fatal_error is not None:
            canceled = db.mark_unfinished_items_canceled(job_id=job_id)
            if canceled:
                logger.warning(
                    "Canceled %d pending online item(s) due to fatal error",
                    canceled,
                )
            raise first_fatal_error

    async def _process_online_candidate(
        self,
        *,
        db: LlmDb,
        task: TaskConfig,
        api_model: str,
        provider_client,
        candidate: EligibleCandidate,
        attempt_recorder: AttemptRecorder,
    ) -> None:
        prepared_request = provider_client.prepare_attempt_request(
            note_payload=candidate.payload,
            task_prompt=task.prompt,
            request_options=task.request,
            api_model=api_model,
        )
        outcome = None

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
                attempt_recorder.record_error(
                    candidate=candidate,
                    prepared_request=prepared_request,
                    outcome=outcome,
                    error=error,
                    error_type="fatal_error",
                    final_status=LlmFinalStatus.FATAL_ERROR,
                    result_type=LlmAttemptResultType.ERRORED,
                )
            raise
        except LlmNoteError as error:
            provider_error = str(error).startswith("Provider returned HTTP")
            with db.write_tx():
                attempt_recorder.record_error(
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
                    result_type=LlmAttemptResultType.ERRORED,
                )
            logger.error(
                "LLM note error for %s in '%s' (%s): %s",
                candidate.payload.note_key,
                candidate.deck_name,
                candidate.payload.note_type,
                error,
            )
            return
        except Exception as error:
            logger.exception(
                "Unexpected online execution error for %s in '%s' (%s)",
                candidate.payload.note_key,
                candidate.deck_name,
                candidate.payload.note_type,
            )
            fatal_error = _unexpected_online_execution_error(error)
            with db.write_tx():
                attempt_recorder.record_error(
                    candidate=candidate,
                    prepared_request=prepared_request,
                    outcome=outcome,
                    error=fatal_error,
                    error_type="fatal_error",
                    final_status=LlmFinalStatus.FATAL_ERROR,
                    result_type=LlmAttemptResultType.ERRORED,
                )
            raise fatal_error from error

        if changed_fields:
            with db.write_tx():
                attempt_recorder.record_success(
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
            return

        with db.write_tx():
            attempt_recorder.record_success(
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

    def _apply_updates(
        self,
        *,
        db: LlmDb,
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
                self.deserialize_collection_fn(
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


def plan_task(
    *,
    collection_dir: Path,
    task_name: str,
    model_override: str | None = None,
    deck_override: str | None = None,
) -> TaskPlanResult:
    task, note_type_configs = _load_task(
        collection_dir=collection_dir,
        task_name=task_name,
    )
    task = apply_deck_override(task, deck_override)
    model = resolve_model(task, model_override, collection_dir=collection_dir)
    task = replace(task, model=model)
    return build_task_plan_result(
        collection_dir=collection_dir,
        task=task,
        note_type_configs=note_type_configs,
        serialize_collection_fn=serialize_collection,
    )


async def run_task_async(
    *,
    collection_dir: Path,
    task_name: str,
    model_override: str | None = None,
    deck_override: str | None = None,
    no_auto_commit: bool = False,
    failure_policy: RunFailurePolicy | str = RunFailurePolicy.ATOMIC,
) -> LlmJobResult:
    task, note_type_configs = _load_task(
        collection_dir=collection_dir,
        task_name=task_name,
    )
    task = apply_deck_override(task, deck_override)
    executor = LlmTaskExecutor(
        collection_dir=collection_dir,
        task=task,
        note_type_configs=note_type_configs,
        model_override=model_override,
        no_auto_commit=no_auto_commit,
        failure_policy=failure_policy,
    )
    return await executor.execute()


async def resume_task_async(
    *,
    collection_dir: Path,
    resume_job_id: str,
    no_auto_commit: bool = False,
    failure_policy: RunFailurePolicy | str = RunFailurePolicy.ATOMIC,
) -> LlmJobResult:
    with _open_llm_db(collection_dir) as db:
        resolved = db.resolve_job_id(resume_job_id)
        if resolved is None:
            raise ValueError(f"Unknown LLM job '{resume_job_id}'")
        snapshot = db.get_job_snapshot(job_id=resolved)
        if snapshot is None:
            raise ValueError(f"Missing config snapshot for LLM job '{resolved}'")
        resume_source_items = db.get_resume_source_items(job_id=resolved)

    task = task_from_snapshot(snapshot, collection_dir=collection_dir)
    note_type_configs = _load_note_type_configs(collection_dir)

    executor = LlmTaskExecutor(
        collection_dir=collection_dir,
        task=task,
        note_type_configs=note_type_configs,
        model_override=None,
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
    deck_override: str | None = None,
    no_auto_commit: bool = False,
    failure_policy: RunFailurePolicy | str = RunFailurePolicy.ATOMIC,
) -> LlmJobResult:
    return asyncio.run(
        run_task_async(
            collection_dir=collection_dir,
            task_name=task_name,
            model_override=model_override,
            deck_override=deck_override,
            no_auto_commit=no_auto_commit,
            failure_policy=failure_policy,
        )
    )


def resume_task(
    *,
    collection_dir: Path,
    resume_job_id: str,
    no_auto_commit: bool = False,
    failure_policy: RunFailurePolicy | str = RunFailurePolicy.ATOMIC,
) -> LlmJobResult:
    return asyncio.run(
        resume_task_async(
            collection_dir=collection_dir,
            resume_job_id=resume_job_id,
            no_auto_commit=no_auto_commit,
            failure_policy=failure_policy,
        )
    )


def list_jobs(
    *,
    collection_dir: Path,
) -> list[LlmJobListItem]:
    with _open_llm_db(collection_dir) as db:
        return db.list_jobs()


def show_job(
    *,
    collection_dir: Path,
    job_id: str,
) -> LlmJobDetail | None:
    with _open_llm_db(collection_dir) as db:
        resolved = db.resolve_job_id(job_id)
        if resolved is None:
            return None
        return db.get_job_detail(int(resolved))
