"""YAML config loading for task specs."""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import yaml

from ankiops.config import LLM_DIR
from ankiops.models import NoteTypeConfig

from .llm_models import FieldExceptionRule, TaskCatalog, TaskConfig
from .model_registry import ModelRegistry, ModelRegistryError, load_model_registry


class LlmConfigError(ValueError):
    """Raised when an LLM task config is invalid."""


_SYSTEM_PROMPT_FILE_NAME = "system_prompt.md"
_TASKS_DIR_NAME = "tasks"
_SUPPORTED_TASK_KEYS_ORDERED = ("model", "prompt_file", "fields")
_SUPPORTED_TASK_KEYS = set(_SUPPORTED_TASK_KEYS_ORDERED)


def _iter_yaml_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.yaml"))
    files.extend(sorted(directory.glob("*.yml")))
    return files


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise LlmConfigError(f"{path}: config must be a YAML mapping")
    return raw


def _require_str(mapping: dict[str, Any], key: str, path: Path) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LlmConfigError(f"{path}: '{key}' must be a non-empty string")
    return value.strip()


def _read_text_file(path: Path, *, label: str) -> str:
    if not path.exists() or not path.is_file():
        raise LlmConfigError(f"{path}: {label} file not found")
    value = path.read_text(encoding="utf-8")
    if not value.strip():
        raise LlmConfigError(f"{path}: {label} file must be non-empty")
    return value.strip()


def _resolve_relative_file(
    task_path: Path,
    raw_reference: str,
    *,
    llm_dir: Path,
    key: str,
) -> Path:
    candidate = (task_path.parent / raw_reference).resolve()
    try:
        candidate.relative_to(llm_dir.resolve())
    except ValueError as error:
        raise LlmConfigError(
            f"{task_path}: '{key}' must stay within {llm_dir}"
        ) from error
    return candidate


def _parse_str_list(
    value: Any,
    key: str,
    path: Path,
    *,
    required: bool,
) -> list[str]:
    if value is None:
        if required:
            raise LlmConfigError(f"{path}: '{key}' must be a non-empty list of strings")
        return []
    if not isinstance(value, list) or (required and not value):
        expected = "non-empty list of strings" if required else "list of strings"
        raise LlmConfigError(f"{path}: '{key}' must be a {expected}")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise LlmConfigError(f"{path}: '{key}' must contain only strings")
        items.append(item.strip())
    return items


def _parse_field_exceptions(
    value: Any,
    *,
    note_type_configs: list[NoteTypeConfig],
    path: Path,
) -> list[FieldExceptionRule]:
    if value is None:
        return []
    if not isinstance(value, dict):
        raise LlmConfigError(f"{path}: 'fields' must be a mapping")

    unknown_fields = sorted(set(value.keys()) - {"exceptions"})
    if unknown_fields:
        raise LlmConfigError(
            f"{path}: unknown fields config key(s): {', '.join(unknown_fields)}"
        )

    exceptions_value = value.get("exceptions", [])
    if not isinstance(exceptions_value, list):
        raise LlmConfigError(f"{path}: 'fields.exceptions' must be a list")

    note_type_names = [config.name for config in note_type_configs]
    field_names = {
        config.name: {field.name for field in config.fields}
        for config in note_type_configs
    }
    parsed: list[FieldExceptionRule] = []

    for index, entry in enumerate(exceptions_value):
        entry_path = Path(f"{path}#fields.exceptions[{index}]")
        if not isinstance(entry, dict):
            raise LlmConfigError(f"{entry_path}: exception must be a mapping")

        unknown = sorted(set(entry.keys()) - {"note_types", "read_only", "hidden"})
        if unknown:
            raise LlmConfigError(
                f"{entry_path}: unknown exception key(s): {', '.join(unknown)}"
            )

        note_types = (
            _parse_str_list(
                entry.get("note_types"),
                "note_types",
                entry_path,
                required=True,
            )
            if "note_types" in entry
            else ["*"]
        )
        read_only = _parse_str_list(
            entry.get("read_only"),
            "read_only",
            entry_path,
            required=False,
        )
        hidden = _parse_str_list(
            entry.get("hidden"),
            "hidden",
            entry_path,
            required=False,
        )
        if not read_only and not hidden:
            raise LlmConfigError(
                f"{entry_path}: exception must declare 'read_only' or 'hidden'"
            )

        matched_types = {
            note_type_name
            for note_type_name in note_type_names
            if any(fnmatchcase(note_type_name, pattern) for pattern in note_types)
        }
        if not matched_types:
            patterns = ", ".join(note_types)
            raise LlmConfigError(
                f"{entry_path}: note type pattern(s) match no note types: {patterns}"
            )

        available_fields = set().union(*(field_names[name] for name in matched_types))
        for field_name in read_only + hidden:
            if field_name not in available_fields:
                raise LlmConfigError(
                    f"{entry_path}: field '{field_name}' does not exist on any matched "
                    "note type"
                )

        parsed.append(
            FieldExceptionRule(
                note_types=note_types,
                read_only=read_only,
                hidden=hidden,
            )
        )

    return parsed


def _validate_task_keys(mapping: dict[str, Any], path: Path) -> None:
    unknown = sorted(set(mapping.keys()) - _SUPPORTED_TASK_KEYS)
    if unknown:
        allowed = ", ".join(_SUPPORTED_TASK_KEYS_ORDERED)
        raise LlmConfigError(
            f"{path}: unknown task key(s): {', '.join(unknown)}. "
            f"Allowed keys: {allowed}"
        )


def _parse_task(
    path: Path,
    *,
    note_type_configs: list[NoteTypeConfig],
    llm_dir: Path,
    model_registry: ModelRegistry,
) -> TaskConfig:
    mapping = _read_yaml_mapping(path)
    _validate_task_keys(mapping, path)
    name = path.stem
    model_name = _require_str(mapping, "model", path)
    model = model_registry.parse(model_name)
    if model is None:
        supported = model_registry.format_names()
        raise LlmConfigError(f"{path}: 'model' must be one of: {supported}")
    prompt_file_ref = _require_str(mapping, "prompt_file", path)
    prompt_file_path = _resolve_relative_file(
        path,
        prompt_file_ref,
        llm_dir=llm_dir,
        key="prompt_file",
    )
    prompt = _read_text_file(prompt_file_path, label="prompt")
    system_prompt_path = (llm_dir / _SYSTEM_PROMPT_FILE_NAME).resolve()
    system_prompt = _read_text_file(system_prompt_path, label="system prompt")
    field_exceptions = _parse_field_exceptions(
        mapping.get("fields"),
        note_type_configs=note_type_configs,
        path=path,
    )

    return TaskConfig(
        name=name,
        model=model,
        system_prompt=system_prompt,
        prompt=prompt,
        system_prompt_path=system_prompt_path,
        prompt_path=prompt_file_path,
        api_key_env=model.api_key_env,
        field_exceptions=field_exceptions,
    )


def load_llm_task_catalog(
    collection_dir: Path,
    *,
    note_type_configs: list[NoteTypeConfig],
) -> TaskCatalog:
    llm_dir = collection_dir / LLM_DIR
    tasks_by_name: dict[str, TaskConfig] = {}
    errors: dict[str, str] = {}

    if not llm_dir.exists() or not llm_dir.is_dir():
        errors[str(llm_dir)] = (
            f"{llm_dir}: LLM config directory not found. "
            "Run 'ankiops init' to create IaC LLM configs."
        )
        return TaskCatalog(tasks_by_name=tasks_by_name, errors=errors)

    tasks_dir = llm_dir / _TASKS_DIR_NAME
    if not tasks_dir.exists() or not tasks_dir.is_dir():
        errors[str(tasks_dir)] = (
            f"{tasks_dir}: task directory not found. "
            "Expected task YAML files in llm/tasks/."
        )
        return TaskCatalog(tasks_by_name=tasks_by_name, errors=errors)

    task_files = _iter_yaml_files(tasks_dir)
    if not task_files:
        errors[str(tasks_dir)] = f"{tasks_dir}: no task YAML files found"
        return TaskCatalog(tasks_by_name=tasks_by_name, errors=errors)

    try:
        model_registry = load_model_registry(collection_dir=collection_dir)
    except ModelRegistryError as error:
        errors[str(llm_dir / "models.yaml")] = str(error)
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

    return TaskCatalog(
        tasks_by_name=tasks_by_name,
        errors=errors,
    )
