"""Task YAML loading and normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ankiops.ai.config_utils import load_yaml_mapping, validate_config_model
from ankiops.ai.errors import TaskConfigError
from ankiops.ai.paths import AIPaths
from ankiops.ai.types import TaskConfig

_VALID_TASK_SUFFIXES = frozenset({".yaml", ".yml"})
_VALID_BATCH_MODES = frozenset({"single", "batch", "collection"})
_DEFAULT_BATCH_SIZE_BY_MODE = {
    "single": 1,
    "batch": 8,
    "collection": 16,
}
_DEFAULT_MODEL_PROFILE_ALIAS = "default"


class _RawTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_name: str = Field(default="ai.task.v1", alias="schema")
    id: str | None = None
    description: str | None = None
    model: str | None = None
    instructions: str
    batch: Literal["single", "batch", "collection"] = "single"
    batch_size: int | None = Field(default=None, gt=0)
    scope_decks: list[str] | None = None
    scope_subdecks: bool = True
    scope_note_types: list[str] | None = None
    read_fields: list[str]
    write_fields: list[str]
    temperature: float = Field(default=0.0, ge=0, le=2)

    @field_validator("schema_name")
    @classmethod
    def _validate_schema(cls, value: str) -> str:
        normalized = value.strip()
        if normalized != "ai.task.v1":
            raise ValueError("must equal 'ai.task.v1'")
        return normalized

    @field_validator("id", "description")
    @classmethod
    def _normalize_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be a non-empty string")
        return normalized

    @field_validator("model")
    @classmethod
    def _normalize_model_profile_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be a non-empty string")
        if normalized.casefold() == _DEFAULT_MODEL_PROFILE_ALIAS:
            return None
        return normalized

    @field_validator("instructions")
    @classmethod
    def _normalize_instructions(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be a non-empty string")
        return normalized

    @field_validator(
        "scope_decks",
        "scope_note_types",
        "read_fields",
        "write_fields",
        mode="before",
    )
    @classmethod
    def _normalize_pattern_list(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            raw_values = [value]
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            raw_values = list(value)
        else:
            raise ValueError("must be a string or list of strings")

        normalized = [item.strip() for item in raw_values if item.strip()]
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @model_validator(mode="after")
    def _validate_batch_and_fields(self) -> _RawTaskConfig:
        if self.batch not in _VALID_BATCH_MODES:
            allowed = ", ".join(sorted(_VALID_BATCH_MODES))
            raise ValueError(f"batch must be one of: {allowed}")

        resolved_batch_size = self.batch_size
        if resolved_batch_size is None:
            resolved_batch_size = _DEFAULT_BATCH_SIZE_BY_MODE[self.batch]
        if self.batch == "single" and resolved_batch_size != 1:
            raise ValueError("batch_size must be 1 when batch is 'single'")
        object.__setattr__(self, "batch_size", resolved_batch_size)

        read_fields = self.read_fields or []
        write_fields = self.write_fields or []
        missing = [field for field in write_fields if field not in read_fields]
        if missing:
            missing_fields = ", ".join(missing)
            raise ValueError(
                "write_fields must be a subset of read_fields; missing: "
                f"{missing_fields}"
            )
        return self


def resolve_task_path(ai_paths: AIPaths, task_ref: str) -> Path:
    """Resolve a task name/path to a YAML file path."""
    raw = _require_task_ref(task_ref)
    candidates = _candidate_task_paths(ai_paths.tasks, raw)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return _assert_is_task_file(candidate)

    tried = ", ".join(str(candidate_path) for candidate_path in candidates)
    raise TaskConfigError(f"Task not found: '{task_ref}'. Tried: {tried}")


def load_task_config(ai_paths: AIPaths, task_ref: str) -> TaskConfig:
    """Load a task YAML file into a validated TaskConfig."""
    path = resolve_task_path(ai_paths, task_ref)
    raw = load_yaml_mapping(
        path,
        error_type=TaskConfigError,
        mapping_label="Task file",
    )
    parsed = validate_config_model(
        raw,
        model_type=_RawTaskConfig,
        path=path,
        error_type=TaskConfigError,
        config_label="task",
    )

    task_id = parsed.id or path.stem
    if parsed.id is not None and parsed.id != path.stem:
        raise TaskConfigError(
            f"Task id must match file name stem in '{path}' "
            f"(expected '{path.stem}', got '{parsed.id}')"
        )

    batch_size = parsed.batch_size
    if batch_size is None:
        raise TaskConfigError("Task batch_size could not be resolved")

    return TaskConfig(
        id=task_id,
        description=parsed.description or "",
        model=parsed.model,
        instructions=parsed.instructions,
        batch=parsed.batch,
        batch_size=batch_size,
        scope_decks=parsed.scope_decks or ["*"],
        scope_subdecks=parsed.scope_subdecks,
        scope_note_types=parsed.scope_note_types or ["*"],
        read_fields=parsed.read_fields,
        write_fields=parsed.write_fields,
        temperature=parsed.temperature,
        source_path=path,
    )


def _require_task_ref(task_ref: str) -> str:
    if not isinstance(task_ref, str) or not task_ref.strip():
        raise TaskConfigError("Task name/path cannot be empty.")
    return task_ref.strip()


def _assert_is_task_file(path: Path) -> Path:
    if path.suffix.lower() not in _VALID_TASK_SUFFIXES:
        raise TaskConfigError(f"Task file must use .yaml or .yml extension: {path}")
    return path


def _candidate_task_paths(tasks_dir: Path, task_ref: str) -> list[Path]:
    ref_path = Path(task_ref)
    if ref_path.is_absolute():
        if ref_path.suffix:
            return [ref_path]
        return [ref_path.with_suffix(".yaml"), ref_path.with_suffix(".yml")]

    candidate_in_dir = tasks_dir / ref_path
    if ref_path.suffix:
        return [candidate_in_dir]
    return [
        candidate_in_dir.with_suffix(".yaml"),
        candidate_in_dir.with_suffix(".yml"),
    ]
