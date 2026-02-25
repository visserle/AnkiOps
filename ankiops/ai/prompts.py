"""Prompt YAML loading and normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .model_profiles import MODELS_FILE_NAME
from .types import PromptConfig


def resolve_prompt_path(prompts_dir: Path, prompt_ref: str) -> Path:
    """Resolve prompt name/path to a YAML file path."""
    raw = prompt_ref.strip()
    if not raw:
        raise ValueError("Prompt name/path cannot be empty.")

    direct = Path(raw)
    if direct.exists() and direct.is_file():
        if direct.name == MODELS_FILE_NAME:
            raise ValueError(f"'{MODELS_FILE_NAME}' is model config, not a prompt.")
        return direct

    candidates: list[Path] = []
    candidate_in_dir = prompts_dir / raw
    if candidate_in_dir.suffix:
        candidates.append(candidate_in_dir)
    else:
        candidates.append(prompts_dir / f"{raw}.yaml")
        candidates.append(prompts_dir / f"{raw}.yml")
        candidates.append(prompts_dir / raw)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            if candidate.name == MODELS_FILE_NAME:
                raise ValueError(f"'{MODELS_FILE_NAME}' is model config, not a prompt.")
            return candidate

    tried = ", ".join(str(candidate_path) for candidate_path in candidates)
    raise ValueError(f"Prompt not found: '{prompt_ref}'. Tried: {tried}")


def load_prompt_config(prompts_dir: Path, prompt_ref: str) -> PromptConfig:
    """Load prompt YAML and normalize into PromptConfig."""
    path = resolve_prompt_path(prompts_dir, prompt_ref)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Prompt file must contain a YAML mapping: {path}")

    name = str(raw.get("name") or path.stem).strip()
    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Prompt file '{path}' requires non-empty key: prompt")

    target_fields = _coerce_pattern_list(
        raw.get("target_fields", raw.get("fields_to_edit")),
        key="target_fields",
        fallback=["*"],
    )
    send_fields = _coerce_pattern_list(
        raw.get("send_fields", raw.get("fields_to_send")),
        key="send_fields",
        fallback=target_fields,
    )
    note_types = _coerce_pattern_list(
        raw.get("note_types"),
        key="note_types",
        fallback=["*"],
    )

    model_profile = raw.get("model_profile")
    if model_profile is not None:
        if not isinstance(model_profile, str) or not model_profile.strip():
            raise ValueError(
                f"Prompt key 'model_profile' must be a non-empty string in '{path}'"
            )
        model_profile = model_profile.strip()

    raw_temperature = raw.get("temperature", 0)
    try:
        temperature = float(raw_temperature)
    except (TypeError, ValueError):
        raise ValueError(
            f"Prompt key 'temperature' must be numeric in '{path}'"
        ) from None

    return PromptConfig(
        name=name,
        prompt=prompt.strip(),
        target_fields=target_fields,
        send_fields=send_fields,
        note_types=note_types,
        model_profile=model_profile,
        temperature=temperature,
        source_path=path,
    )


def _coerce_pattern_list(raw: Any, *, key: str, fallback: list[str]) -> list[str]:
    if raw is None:
        return fallback
    if isinstance(raw, str):
        value = [raw]
    elif isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        value = list(raw)
    else:
        raise ValueError(f"Prompt key '{key}' must be a string or list of strings.")

    cleaned = [item.strip() for item in value if item.strip()]
    if not cleaned:
        raise ValueError(f"Prompt key '{key}' must not be empty.")
    return cleaned
