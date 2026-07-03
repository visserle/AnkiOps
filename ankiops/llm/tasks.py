"""Load OpenAI LLM task YAML files."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import ScalarNode

from ankiops.collection import LLM_DIR
from ankiops.note_types import NoteType

from .models import (
    MODEL_REGISTRY_FILE_NAME,
    ModelRegistry,
    ModelRegistryError,
    ModelSpec,
    load_model_registry,
    model_registry_path,
)

_SUPPORTED_TASK_KEYS = (
    "model",
    "system_prompt",
    "user_prompt",
    "request",
    "fields",
    "tags",
)
_SUPPORTED_REQUEST_KEYS = (
    "max_notes_per_request",
    "temperature",
    "reasoning",
)
_SUPPORTED_REASONING_EFFORTS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)


class LlmConfigError(ValueError):
    """Raised when a task config is invalid."""


class FieldAccess(Enum):
    EDITABLE = "editable"
    READ_ONLY = "read_only"
    HIDDEN = "hidden"


@dataclass(frozen=True)
class DeckScope:
    deck_root: str | None = None

    def matches(self, deck_name: str) -> bool:
        if self.deck_root is None:
            return True
        return deck_name == self.deck_root or deck_name.startswith(
            f"{self.deck_root}::"
        )


@dataclass(frozen=True)
class FieldAccessRule:
    note_types: list[str] = field(default_factory=lambda: ["*"])
    editable: list[str] = field(default_factory=list)
    read_only: list[str] = field(default_factory=list)
    hidden: list[str] = field(default_factory=list)

    def matches_note_type(self, note_type: str) -> bool:
        return any(fnmatchcase(note_type, pattern) for pattern in self.note_types)

    def marks_editable(self, field_name: str) -> bool:
        return _matches_any(self.editable, field_name)

    def marks_read_only(self, field_name: str) -> bool:
        return _matches_any(self.read_only, field_name)

    def marks_hidden(self, field_name: str) -> bool:
        return _matches_any(self.hidden, field_name)


@dataclass(frozen=True)
class TaskRequestOptions:
    max_notes_per_request: int = 1
    temperature: float | None = None
    reasoning: str | None = None


@dataclass(frozen=True)
class TaskConfig:
    name: str
    model: ModelSpec
    system_prompt: str
    user_prompt: str
    system_prompt_path: Path | None = None
    user_prompt_path: Path | None = None
    decks: DeckScope = field(default_factory=DeckScope)
    default_field_access: FieldAccess = FieldAccess.EDITABLE
    field_rules: list[FieldAccessRule] = field(default_factory=list)
    tag_access: FieldAccess = FieldAccess.HIDDEN
    request: TaskRequestOptions = field(default_factory=TaskRequestOptions)

    def field_access(self, note_type: str, field_name: str) -> FieldAccess:
        editable = False
        read_only = False
        hidden = False
        for rule in self.field_rules:
            if not rule.matches_note_type(note_type):
                continue
            editable = editable or rule.marks_editable(field_name)
            read_only = read_only or rule.marks_read_only(field_name)
            hidden = hidden or rule.marks_hidden(field_name)

        if hidden:
            return FieldAccess.HIDDEN
        if editable:
            return FieldAccess.EDITABLE
        if read_only:
            return FieldAccess.READ_ONLY
        return self.default_field_access


@dataclass(frozen=True)
class TaskCatalog:
    tasks_by_name: dict[str, TaskConfig]
    errors: dict[str, str]


class _FileSource:
    def __init__(self, text: str, path: Path) -> None:
        self.text = text
        self.path = path


class _TaskConfigLoader(yaml.SafeLoader):
    task_path: Path
    llm_dir: Path


def _construct_file_source(
    loader: _TaskConfigLoader,
    node: Any,
) -> _FileSource:
    if not isinstance(node, ScalarNode):
        raise LlmConfigError(
            f"{loader.task_path}: !file tag must be used with a scalar path"
        )

    raw_reference = loader.construct_scalar(node).strip()
    if not raw_reference:
        raise LlmConfigError(
            f"{loader.task_path}: !file path must be a non-empty string"
        )

    resolved_path = _resolve_relative_file(
        loader.task_path,
        raw_reference,
        llm_dir=loader.llm_dir,
    )
    return _FileSource(text=_read_text_file(resolved_path), path=resolved_path)


_TaskConfigLoader.add_constructor("!file", _construct_file_source)


def is_task_config_file(path: Path) -> bool:
    return path.suffix in {".yaml", ".yml"} and path.name != MODEL_REGISTRY_FILE_NAME


def load_llm_task_catalog(
    collection_root: Path,
    *,
    note_type_configs: list[NoteType],
) -> TaskCatalog:
    llm_dir = collection_root / LLM_DIR
    tasks_by_name: dict[str, TaskConfig] = {}
    errors: dict[str, str] = {}

    if not llm_dir.exists() or not llm_dir.is_dir():
        errors[str(llm_dir)] = (
            f"{llm_dir}: LLM config directory not found. "
            "Run 'ankiops init' to create LLM configs."
        )
        return TaskCatalog(tasks_by_name=tasks_by_name, errors=errors)

    task_files = [
        path for path in _iter_yaml_files(llm_dir) if is_task_config_file(path)
    ]
    if not task_files:
        errors[str(llm_dir)] = f"{llm_dir}: no task YAML files found in llm/"
        return TaskCatalog(tasks_by_name=tasks_by_name, errors=errors)

    try:
        model_registry = load_model_registry(collection_root=collection_root)
    except ModelRegistryError as error:
        errors[str(model_registry_path(collection_root=collection_root))] = str(error)
        return TaskCatalog(tasks_by_name=tasks_by_name, errors=errors)

    for path in task_files:
        try:
            task = _parse_task(
                path,
                note_type_configs=note_type_configs,
                llm_dir=llm_dir,
                model_registry=model_registry,
            )
            if task.name in tasks_by_name:
                raise LlmConfigError(f"{path}: duplicate task name '{task.name}'")
            tasks_by_name[task.name] = task
        except LlmConfigError as error:
            errors[str(path)] = str(error)

    return TaskCatalog(tasks_by_name=tasks_by_name, errors=errors)


def _parse_task(
    path: Path,
    *,
    note_type_configs: list[NoteType],
    llm_dir: Path,
    model_registry: ModelRegistry,
) -> TaskConfig:
    mapping = _read_yaml_mapping(path, llm_dir=llm_dir)
    _validate_task_keys(mapping, path)

    model_name = _require_str(mapping, "model", path)
    model = model_registry.parse(model_name)
    if model is None:
        raise LlmConfigError(
            f"{path}: 'model' must be one of: {model_registry.format_models()}"
        )

    system_prompt, system_prompt_path = _parse_text_source(
        mapping,
        key="system_prompt",
        path=path,
    )
    user_prompt, user_prompt_path = _parse_text_source(
        mapping,
        key="user_prompt",
        path=path,
    )
    default_field_access, field_rules = _parse_field_rules(
        mapping.get("fields"),
        note_type_configs=note_type_configs,
        path=path,
    )
    tag_access = _parse_tag_access(mapping.get("tags"), path=path)
    request = _parse_request_options(mapping.get("request"), path=path)

    return TaskConfig(
        name=path.stem,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        system_prompt_path=system_prompt_path,
        user_prompt_path=user_prompt_path,
        default_field_access=default_field_access,
        field_rules=field_rules,
        tag_access=tag_access,
        request=request,
    )


def _iter_yaml_files(directory: Path) -> list[Path]:
    return sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")])


def _read_yaml_mapping(path: Path, *, llm_dir: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loader = _TaskConfigLoader(handle)
        loader.task_path = path
        loader.llm_dir = llm_dir
        try:
            raw = loader.get_single_data() or {}
        except yaml.YAMLError as error:
            raise LlmConfigError(f"{path}: invalid YAML ({error})") from error
        finally:
            loader.dispose()
    if not isinstance(raw, dict):
        raise LlmConfigError(f"{path}: config must be a YAML mapping")
    return raw


def _validate_task_keys(mapping: dict[str, Any], path: Path) -> None:
    unknown = sorted(set(mapping.keys()) - set(_SUPPORTED_TASK_KEYS))
    if not unknown:
        return
    allowed = ", ".join(_SUPPORTED_TASK_KEYS)
    raise LlmConfigError(
        f"{path}: unknown task key(s): {', '.join(unknown)}. Allowed keys: {allowed}"
    )


def _require_str(mapping: dict[str, Any], key: str, path: Path) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LlmConfigError(f"{path}: '{key}' must be a non-empty string")
    return value.strip()


def _parse_text_source(
    mapping: dict[str, Any],
    *,
    key: str,
    path: Path,
) -> tuple[str, Path | None]:
    raw_value = mapping.get(key)
    if isinstance(raw_value, _FileSource):
        return raw_value.text, raw_value.path
    if isinstance(raw_value, str) and raw_value.strip():
        return raw_value.strip(), None
    raise LlmConfigError(f"{path}: '{key}' must be a non-empty string")


def _read_text_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise LlmConfigError(f"{path}: file not found")
    value = path.read_text(encoding="utf-8")
    if not value.strip():
        raise LlmConfigError(f"{path}: file must be non-empty")
    return value.strip()


def _resolve_relative_file(
    task_path: Path,
    raw_reference: str,
    *,
    llm_dir: Path,
) -> Path:
    candidate = (task_path.parent / raw_reference).resolve()
    try:
        candidate.relative_to(llm_dir.resolve())
    except ValueError as error:
        raise LlmConfigError(
            f"{task_path}: !file path must stay within {llm_dir}"
        ) from error
    return candidate


def _parse_field_rules(
    value: Any,
    *,
    note_type_configs: list[NoteType],
    path: Path,
) -> tuple[FieldAccess, list[FieldAccessRule]]:
    if value is None:
        return FieldAccess.EDITABLE, []
    if not isinstance(value, dict):
        raise LlmConfigError(f"{path}: 'fields' must be a mapping")

    unknown_fields = sorted(
        set(value.keys()) - {"default_access", "editable", "read_only", "hidden"}
    )
    if unknown_fields:
        raise LlmConfigError(
            f"{path}: unknown fields config key(s): {', '.join(unknown_fields)}"
        )

    default_access = _parse_access_value(
        value.get("default_access"),
        path=path,
        key="fields.default_access",
        default=FieldAccess.EDITABLE,
    )
    note_type_names = [config.name for config in note_type_configs]
    field_names = {
        config.name: {field.name for field in config.fields}
        for config in note_type_configs
    }

    rules: list[FieldAccessRule] = []
    for key in ("editable", "read_only", "hidden"):
        rules.extend(
            _parse_access_rules_by_note_type(
                value.get(key),
                key=key,
                note_type_names=note_type_names,
                field_names=field_names,
                path=path,
            )
        )

    return default_access, rules


def _parse_tag_access(value: Any, *, path: Path) -> FieldAccess:
    return _parse_access_value(
        value,
        path=path,
        key="tags",
        default=FieldAccess.HIDDEN,
    )


def _parse_access_value(
    value: Any,
    *,
    path: Path,
    key: str,
    default: FieldAccess,
) -> FieldAccess:
    if value is None:
        return default
    if not isinstance(value, str):
        return _raise_invalid_access(path, key=key)
    normalized = value.strip()
    for access in FieldAccess:
        if normalized == access.value:
            return access
    return _raise_invalid_access(path, key=key)


def _raise_invalid_access(path: Path, *, key: str) -> FieldAccess:
    supported = ", ".join(access.value for access in FieldAccess)
    raise LlmConfigError(f"{path}: '{key}' must be one of: {supported}")


def _parse_access_rules_by_note_type(
    value: Any,
    *,
    key: str,
    note_type_names: list[str],
    field_names: dict[str, set[str]],
    path: Path,
) -> list[FieldAccessRule]:
    if value is None:
        return []
    if not isinstance(value, dict):
        raise LlmConfigError(
            f"{path}: 'fields.{key}' must be a mapping of note-type patterns to "
            "non-empty field pattern lists"
        )

    rules: list[FieldAccessRule] = []
    for raw_note_type_pattern, raw_field_patterns in value.items():
        if (
            not isinstance(raw_note_type_pattern, str)
            or not raw_note_type_pattern.strip()
        ):
            raise LlmConfigError(
                f"{path}: 'fields.{key}' note type pattern keys must be non-empty "
                "strings"
            )
        note_type_pattern = raw_note_type_pattern.strip()
        entry_path = Path(f"{path}#fields.{key}.{note_type_pattern}")
        field_patterns = _parse_str_list(
            raw_field_patterns,
            f"fields.{key}.{note_type_pattern}",
            entry_path,
            required=True,
        )
        matched_types = _match_note_types(
            [note_type_pattern],
            note_type_names=note_type_names,
            entry_path=entry_path,
        )
        available_fields = set().union(*(field_names[name] for name in matched_types))
        _validate_field_patterns(
            field_patterns,
            available_fields=available_fields,
            entry_path=entry_path,
        )
        rules.append(
            FieldAccessRule(
                note_types=[note_type_pattern],
                editable=field_patterns if key == "editable" else [],
                read_only=field_patterns if key == "read_only" else [],
                hidden=field_patterns if key == "hidden" else [],
            )
        )
    return rules


def _parse_request_options(value: Any, *, path: Path) -> TaskRequestOptions:
    defaults = TaskRequestOptions()
    if value is None:
        raise LlmConfigError(f"{path}: 'request.max_notes_per_request' is required")
    if not isinstance(value, dict):
        raise LlmConfigError(f"{path}: 'request' must be a mapping")

    unknown = sorted(set(value.keys()) - set(_SUPPORTED_REQUEST_KEYS))
    if unknown:
        allowed = ", ".join(_SUPPORTED_REQUEST_KEYS)
        raise LlmConfigError(
            f"{path}: unknown request key(s): {', '.join(unknown)}. "
            f"Allowed request keys: {allowed}"
        )

    raw_max_notes_per_request = value.get("max_notes_per_request")
    if raw_max_notes_per_request is None:
        raise LlmConfigError(f"{path}: 'request.max_notes_per_request' is required")
    if isinstance(raw_max_notes_per_request, bool) or not isinstance(
        raw_max_notes_per_request,
        int,
    ):
        raise LlmConfigError(
            f"{path}: 'request.max_notes_per_request' must be an integer"
        )
    if raw_max_notes_per_request < 1:
        raise LlmConfigError(f"{path}: 'request.max_notes_per_request' must be >= 1")

    temperature = defaults.temperature
    if "temperature" in value:
        raw_temperature = value.get("temperature")
        if raw_temperature is None:
            temperature = None
        elif isinstance(raw_temperature, bool) or not isinstance(
            raw_temperature,
            (int, float),
        ):
            raise LlmConfigError(f"{path}: 'request.temperature' must be numeric")
        else:
            temperature = float(raw_temperature)

    reasoning = defaults.reasoning
    if "reasoning" in value:
        raw_reasoning = value.get("reasoning")
        if raw_reasoning is None:
            reasoning = None
        elif not isinstance(raw_reasoning, str) or not raw_reasoning.strip():
            raise LlmConfigError(
                f"{path}: 'request.reasoning' must be a non-empty string"
            )
        else:
            normalized = raw_reasoning.strip()
            if normalized not in _SUPPORTED_REASONING_EFFORTS:
                allowed = ", ".join(_SUPPORTED_REASONING_EFFORTS)
                raise LlmConfigError(
                    f"{path}: 'request.reasoning' must be one of: {allowed}"
                )
            reasoning = normalized

    return TaskRequestOptions(
        max_notes_per_request=raw_max_notes_per_request,
        temperature=temperature,
        reasoning=reasoning,
    )


def _parse_str_list(
    value: Any,
    key: str,
    path: Path,
    *,
    required: bool,
) -> list[str]:
    if value is None:
        if required:
            raise LlmConfigError(f"{path}: '{key}' must be a non-empty list")
        return []
    if not isinstance(value, list) or (required and not value):
        expected = "non-empty list" if required else "list"
        raise LlmConfigError(f"{path}: '{key}' must be a {expected}")

    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise LlmConfigError(f"{path}: '{key}' must contain only strings")
        parsed.append(item.strip())
    return parsed


def _match_note_types(
    note_types: list[str],
    *,
    note_type_names: list[str],
    entry_path: Path,
) -> set[str]:
    matched = {
        name
        for name in note_type_names
        if any(fnmatchcase(name, pattern) for pattern in note_types)
    }
    if not matched:
        raise LlmConfigError(
            f"{entry_path}: note type pattern(s) match no note types: "
            f"{', '.join(note_types)}"
        )
    return matched


def _validate_field_patterns(
    patterns: list[str],
    *,
    available_fields: set[str],
    entry_path: Path,
) -> None:
    for pattern in patterns:
        if not any(fnmatchcase(field_name, pattern) for field_name in available_fields):
            raise LlmConfigError(
                f"{entry_path}: field pattern '{pattern}' does not exist on any "
                "matched note type"
            )


def _matches_any(patterns: list[str], value: str) -> bool:
    return any(fnmatchcase(value, pattern) for pattern in patterns)
