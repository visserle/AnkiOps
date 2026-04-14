"""YAML config loading for task specs."""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import ScalarNode

from ankiops.config import LLM_DIR
from ankiops.models import NoteTypeConfig

from .llm_models import FieldExceptionRule, TaskCatalog, TaskConfig
from .model_registry import ModelRegistry, ModelRegistryError, load_model_registry


class LlmConfigError(ValueError):
    """Raised when an LLM task config is invalid."""


_MODEL_REGISTRY_FILE_NAMES = {"models.yaml"}
_SUPPORTED_TASK_KEYS_ORDERED = ("model", "system_prompt", "task_prompt", "fields")
_SUPPORTED_TASK_KEYS = set(_SUPPORTED_TASK_KEYS_ORDERED)


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
    content = _read_text_file(resolved_path)
    return _FileSource(text=content, path=resolved_path)


_TaskConfigLoader.add_constructor("!file", _construct_file_source)


def _iter_yaml_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.yaml"))
    files.extend(sorted(directory.glob("*.yml")))
    return files


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


def _require_str(mapping: dict[str, Any], key: str, path: Path) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LlmConfigError(f"{path}: '{key}' must be a non-empty string")
    return value.strip()


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


def _parse_task(
    path: Path,
    *,
    note_type_configs: list[NoteTypeConfig],
    llm_dir: Path,
    model_registry: ModelRegistry,
) -> TaskConfig:
    mapping = _read_yaml_mapping(path, llm_dir=llm_dir)
    _validate_task_keys(mapping, path)
    name = path.stem
    model_name = _require_str(mapping, "model", path)
    model = model_registry.parse(model_name)
    if model is None:
        supported = model_registry.format_names()
        raise LlmConfigError(f"{path}: 'model' must be one of: {supported}")

    system_prompt, system_prompt_path = _parse_text_source(
        mapping,
        key="system_prompt",
        path=path,
    )
    task_prompt, task_prompt_path = _parse_text_source(
        mapping,
        key="task_prompt",
        path=path,
    )
    field_exceptions = _parse_field_exceptions(
        mapping.get("fields"),
        note_type_configs=note_type_configs,
        path=path,
    )

    return TaskConfig(
        name=name,
        model=model,
        system_prompt=system_prompt,
        prompt=task_prompt,
        system_prompt_path=system_prompt_path,
        prompt_path=task_prompt_path,
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

    task_files = [
        path
        for path in _iter_yaml_files(llm_dir)
        if path.name not in _MODEL_REGISTRY_FILE_NAMES
    ]
    if not task_files:
        errors[str(llm_dir)] = f"{llm_dir}: no task YAML files found in llm/"
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
