"""Plan LLM tasks against serialized Anki notes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ankiops.collection import deck_name_to_file_stem
from ankiops.deck_sources import discover_deck_sources, load_note_types_for_collection
from ankiops.interchange import serialize
from ankiops.note_types import ANKIOPS_KEY_FIELD, NoteType
from ankiops.notes import normalize_tags

from .jobs import LlmItemStatus, TaskRunSummary
from .models import ModelSpec, parse_model
from .tasks import (
    DeckScope,
    FieldAccess,
    TaskConfig,
    TaskRequestOptions,
    is_task_config_file,
    load_llm_task_catalog,
)


@dataclass(frozen=True)
class NotePayload:
    note_key: str
    note_type: str
    editable_fields: dict[str, str]
    read_only_fields: dict[str, str] = field(default_factory=dict)
    editable_tags: tuple[str, ...] | None = None
    read_only_tags: tuple[str, ...] | None = None


@dataclass(frozen=True)
class DiscoveryCounts:
    decks_seen: int
    decks_matched: int
    notes_seen: int


@dataclass(frozen=True)
class DiscoveryItem:
    ordinal: int
    source: str
    deck_name: str
    note_key: str | None
    note_type: str | None
    item_status: LlmItemStatus
    skip_reason: str | None
    error_message: str | None
    payload: NotePayload | None
    note_type_config: NoteType | None
    serialized_note: dict[str, Any] | None


@dataclass(frozen=True)
class DiscoverySnapshot:
    counts: DiscoveryCounts
    items: list[DiscoveryItem]


@dataclass(frozen=True)
class EligibleCandidate:
    item_id: int
    source: str
    deck_name: str
    payload: NotePayload
    note_type_config: NoteType
    serialized_note: dict[str, Any]


@dataclass(frozen=True)
class PlanFieldSurface:
    source: str
    note_type: str
    candidate_notes: int
    editable_fields: list[str]
    read_only_fields: list[str]
    hidden_fields: list[str]
    tag_access: FieldAccess


@dataclass(frozen=True)
class TaskPlanResult:
    task_name: str
    model: ModelSpec
    deck_scope: str
    serializer_scope: str
    system_prompt_path: str | None
    user_prompt_path: str | None
    system_prompt: str
    user_prompt: str
    request_defaults: str
    summary: TaskRunSummary
    field_surface: list[PlanFieldSurface]
    requests_estimate: int
    input_tokens_estimate: int

    def format_full_prompt(self) -> str:
        return (
            f"<system>\n{self.system_prompt.strip()}\n</system>\n\n"
            f"<user>\n{self.user_prompt.strip()}\n</user>"
        )

    def format_cost_estimate(self) -> str:
        estimate = self.model.estimate_cost(
            input_tokens=self.input_tokens_estimate,
            # we assume input and output tokens are the same for estimation purposes
            output_tokens=self.input_tokens_estimate,
        )
        if estimate is None:
            return "n/a"
        return estimate.format()


@dataclass(frozen=True)
class MaterializedTaskContext:
    task: TaskConfig
    note_type_configs: dict[str, NoteType]
    serialized_data: dict[str, Any]
    discovery_snapshot: DiscoverySnapshot


@dataclass(frozen=True)
class EligibleBatch:
    note_type: str
    note_type_config: NoteType
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


def plan_task(
    *,
    collection_root: Path,
    task_name: str,
    model_override: str | None = None,
    deck_override: str | None = None,
) -> TaskPlanResult:
    context = materialize_task_context(
        collection_root=collection_root,
        task_name=task_name,
        model_override=model_override,
        deck_override=deck_override,
    )
    return _build_task_plan_result(
        task=context.task,
        note_type_configs=context.note_type_configs,
        snapshot=context.discovery_snapshot,
    )


def materialize_task_context(
    *,
    collection_root: Path,
    task_name: str,
    model_override: str | None,
    deck_override: str | None,
) -> MaterializedTaskContext:
    task, note_type_configs = _load_task(
        collection_root=collection_root,
        task_name=task_name,
    )
    if deck_override is not None:
        task = replace(task, decks=DeckScope(deck_root=deck_override))
    if model_override is not None:
        model = parse_model(model_override, collection_root=collection_root)
        if model is None:
            raise ValueError(f"Unknown model '{model_override}'")
        task = replace(task, model=model)

    deck, no_subdecks = _resolve_serializer_scope(task)
    serialized_data = serialize(
        collection_root,
        deck=deck,
        no_subdecks=no_subdecks,
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
    collection_root: Path,
    task_name: str,
) -> tuple[TaskConfig, dict[str, NoteType]]:
    note_type_configs = load_note_types_for_collection(collection_root)
    catalog = load_llm_task_catalog(
        collection_root,
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


def snapshot_paths_for_task(
    collection_root: Path,
    task_context: MaterializedTaskContext,
) -> list[Path]:
    sources = discover_deck_sources(collection_root)
    if any(source.is_collab for source in sources):
        return [collection_root]

    if any(
        item.item_status is LlmItemStatus.QUEUED and item.source != "local"
        for item in task_context.discovery_snapshot.items
    ):
        return [collection_root]

    deck_names = {
        item.deck_name
        for item in task_context.discovery_snapshot.items
        if item.item_status is LlmItemStatus.QUEUED and item.source == "local"
    }
    return [
        collection_root / f"{deck_name_to_file_stem(deck_name)}.md"
        for deck_name in sorted(deck_names)
    ]


def _discover_candidates(
    *,
    data: dict[str, Any],
    task: TaskConfig,
    note_type_configs: dict[str, NoteType],
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
        source = deck.get("source")
        deck_name = deck.get("name")
        notes = deck.get("notes")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Serialized deck is missing source")
        if not isinstance(deck_name, str) or not isinstance(notes, list):
            raise ValueError(f"Serialized deck in source '{source}' is malformed")
        decks_seen += 1
        notes_seen += len(notes)
        if not task.decks.matches(deck_name):
            continue
        decks_matched += 1

        for note_index, note in enumerate(notes, start=1):
            if not isinstance(note, dict):
                continue
            note_key = note.get("note_key")
            if not isinstance(note_key, str) or not note_key.strip():
                raise ValueError(
                    "LLM tasks require note_key metadata: "
                    f"{source}:{deck_name} note {note_index}"
                )
            ordinal += 1
            items.append(
                _discover_note(
                    task=task,
                    source=source,
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
    source: str,
    deck_name: str,
    ordinal: int,
    note: dict[str, Any],
    note_type_configs: dict[str, NoteType],
) -> DiscoveryItem:
    note_key = note.get("note_key")
    note_type_name = note.get("note_type")
    fields = note.get("fields")
    note_key_value = note_key if isinstance(note_key, str) else None
    note_type_value = note_type_name if isinstance(note_type_name, str) else None

    if not isinstance(note_key, str) or not isinstance(note_type_name, str):
        return _invalid_discovery_item(
            source=source,
            deck_name=deck_name,
            ordinal=ordinal,
            note_key=note_key_value,
            note_type=note_type_value,
            error_message="Serialized note is missing note_key or note_type",
            serialized_note=note,
        )
    if not isinstance(fields, dict):
        return _invalid_discovery_item(
            source=source,
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
            source=source,
            deck_name=deck_name,
            ordinal=ordinal,
            note_key=note_key,
            note_type=note_type_name,
            error_message=f"Unknown note type '{note_type_name}' in serialized note",
            serialized_note=note,
        )

    editable_fields: dict[str, str] = {}
    read_only_fields: dict[str, str] = {}
    for note_field in note_type_config.fields:
        if note_field.name == ANKIOPS_KEY_FIELD.name:
            continue
        access = task.field_access(note_type_name, note_field.name)
        if access is FieldAccess.HIDDEN:
            continue
        raw_value = fields.get(note_field.name, "")
        if raw_value is None:
            raw_value = ""
        if not isinstance(raw_value, str):
            return _invalid_discovery_item(
                source=source,
                deck_name=deck_name,
                ordinal=ordinal,
                note_key=note_key,
                note_type=note_type_name,
                error_message=f"Serialized field '{note_field.name}' must be a string",
                serialized_note=note,
            )
        if access is FieldAccess.READ_ONLY:
            read_only_fields[note_field.name] = raw_value
        else:
            editable_fields[note_field.name] = raw_value

    tags = normalize_tags(note.get("tags", ()))
    editable_tags = tags if task.tag_access is FieldAccess.EDITABLE else None
    read_only_tags = tags if task.tag_access is FieldAccess.READ_ONLY else None
    has_editable_surface = bool(editable_fields) or editable_tags is not None
    has_visible_field_surface = bool(editable_fields) or bool(read_only_fields)
    tag_only_without_fields = (
        editable_tags is not None and not has_visible_field_surface
    )
    if not has_editable_surface or tag_only_without_fields:
        skip_reason = (
            "no readable fields" if has_editable_surface else "no editable fields"
        )
        return DiscoveryItem(
            ordinal=ordinal,
            source=source,
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
        source=source,
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
    source: str,
    deck_name: str,
    ordinal: int,
    note_key: str | None,
    note_type: str | None,
    error_message: str,
    serialized_note: dict[str, Any] | None,
) -> DiscoveryItem:
    return DiscoveryItem(
        ordinal=ordinal,
        source=source,
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


def build_candidate_batches(
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


def build_note_request_payload(payload: NotePayload) -> dict[str, object]:
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


def _build_task_plan_result(
    *,
    task: TaskConfig,
    note_type_configs: dict[str, NoteType],
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
        deck_scope=format_deck_scope(task),
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
    note_type_configs: dict[str, NoteType],
    snapshot_items: list[DiscoveryItem],
) -> list[PlanFieldSurface]:
    observed = {
        (item.source, item.note_type)
        for item in snapshot_items
        if item.note_type is not None and item.note_type_config is not None
    }
    surface: list[PlanFieldSurface] = []
    for source, note_type in sorted(
        observed,
        key=lambda item: (0 if item[0] == "local" else 1, item[0], item[1]),
    ):
        config = note_type_configs.get(note_type)
        if config is None:
            continue
        editable_fields: list[str] = []
        read_only_fields: list[str] = []
        hidden_fields: list[str] = []
        for note_field in config.fields:
            if note_field.name == ANKIOPS_KEY_FIELD.name:
                continue
            access = task.field_access(note_type, note_field.name)
            if access is FieldAccess.EDITABLE:
                editable_fields.append(note_field.name)
            elif access is FieldAccess.READ_ONLY:
                read_only_fields.append(note_field.name)
            else:
                hidden_fields.append(note_field.name)
        candidate_notes = sum(
            1
            for item in snapshot_items
            if item.item_status is LlmItemStatus.QUEUED
            and item.source == source
            and item.note_type == note_type
        )
        surface.append(
            PlanFieldSurface(
                source=source,
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
        "notes": [build_note_request_payload(payload) for payload in payloads],
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


def format_deck_scope(task: TaskConfig) -> str:
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
