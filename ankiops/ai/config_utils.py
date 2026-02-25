"""Shared YAML loading and schema validation helpers for AI config files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

SchemaModel = TypeVar("SchemaModel", bound=BaseModel)


def load_yaml_mapping(
    path: Path,
    *,
    error_type: type[Exception],
    mapping_label: str,
    missing_message: str | None = None,
) -> dict[str, Any]:
    """Load a YAML file and require a top-level mapping."""
    if missing_message is not None and (not path.exists() or not path.is_file()):
        raise error_type(missing_message)

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise error_type(f"{mapping_label} must be a YAML mapping: {path}")
    return raw


def validate_config_model(
    raw: dict[str, Any],
    *,
    model_type: type[SchemaModel],
    path: Path,
    error_type: type[Exception],
    config_label: str,
) -> SchemaModel:
    """Validate a raw mapping with consistent first-error formatting."""
    try:
        return model_type.model_validate(raw)
    except ValidationError as error:
        first = error.errors()[0]
        field_path = ".".join(str(part) for part in first.get("loc", ()))
        detail = first.get("msg", "invalid value")
        if field_path:
            raise error_type(
                f"Invalid {config_label} '{path}' field '{field_path}': {detail}"
            ) from None
        raise error_type(f"Invalid {config_label} '{path}': {detail}") from None
