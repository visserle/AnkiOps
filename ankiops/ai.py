"""AI infrastructure for prompt-driven inline JSON editing."""

from __future__ import annotations

import copy
import fnmatch
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import requests
import yaml

from ankiops.db import SQLiteDbAdapter

logger = logging.getLogger(__name__)

_VALID_PROVIDERS = {"local", "remote"}
_DEFAULT_TIMEOUT_SECONDS = 60
_DEFAULT_API_KEY_ENV = "ANKIOPS_AI_API_KEY"

_PROVIDER_DEFAULTS = {
    "local": {
        "model": "llama3.1:8b",
        "base_url": "http://localhost:11434/v1",
        "api_key_env": _DEFAULT_API_KEY_ENV,
    },
    "remote": {
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": _DEFAULT_API_KEY_ENV,
    },
}

_CFG_PROVIDER = "ai.provider"
_CFG_MODEL = "ai.model"
_CFG_BASE_URL = "ai.base_url"
_CFG_API_KEY_ENV = "ai.api_key_env"
_CFG_TIMEOUT_SECONDS = "ai.timeout_seconds"


@dataclass(frozen=True)
class AIConfig:
    provider: str
    model: str
    base_url: str
    api_key_env: str
    timeout_seconds: int
    api_key: str | None = None


@dataclass(frozen=True)
class PromptConfig:
    name: str
    prompt: str
    target_fields: list[str]
    send_fields: list[str]
    note_types: list[str]
    model: str | None = None
    temperature: float = 0.0
    source_path: Path | None = None

    def matches_note_type(self, note_type: str) -> bool:
        return any(
            fnmatch.fnmatchcase(note_type, pattern) for pattern in self.note_types
        )


@dataclass(frozen=True)
class PromptChange:
    prompt_name: str
    deck_name: str
    note_key: str
    note_type: str
    field_name: str
    original_text: str
    edited_text: str


@dataclass
class PromptRunResult:
    processed_decks: int = 0
    processed_notes: int = 0
    prompted_notes: int = 0
    changed_fields: int = 0
    changes: list[PromptChange] = field(default_factory=list)
    changed_decks: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class InlineJSONEditor(Protocol):
    def edit_note(
        self,
        prompt_config: PromptConfig,
        note_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Return edited note JSON with the same structure as input."""


def _provider_defaults(provider: str) -> dict[str, str]:
    if provider not in _VALID_PROVIDERS:
        provider = "local"
    return _PROVIDER_DEFAULTS[provider]


def _parse_positive_int(raw: str | None, fallback: int) -> int:
    if raw is None:
        return fallback
    try:
        value = int(raw)
    except ValueError:
        return fallback
    if value <= 0:
        return fallback
    return value


def load_ai_config(db: SQLiteDbAdapter) -> AIConfig:
    """Load collection-scoped AI defaults from the existing config table."""
    provider = db.get_config(_CFG_PROVIDER) or "local"
    if provider not in _VALID_PROVIDERS:
        provider = "local"
    defaults = _provider_defaults(provider)

    model = db.get_config(_CFG_MODEL) or defaults["model"]
    base_url = db.get_config(_CFG_BASE_URL) or defaults["base_url"]
    api_key_env = db.get_config(_CFG_API_KEY_ENV) or defaults["api_key_env"]
    timeout_seconds = _parse_positive_int(
        db.get_config(_CFG_TIMEOUT_SECONDS),
        _DEFAULT_TIMEOUT_SECONDS,
    )
    return AIConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        timeout_seconds=timeout_seconds,
    )


def save_ai_config(
    db: SQLiteDbAdapter,
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    timeout_seconds: int | None = None,
) -> AIConfig:
    """Persist collection-scoped AI defaults into the existing config table."""
    if provider is not None:
        if provider not in _VALID_PROVIDERS:
            raise ValueError("provider must be one of: local, remote")
        db.set_config(_CFG_PROVIDER, provider)
    if model is not None:
        db.set_config(_CFG_MODEL, model)
    if base_url is not None:
        db.set_config(_CFG_BASE_URL, base_url)
    if api_key_env is not None:
        db.set_config(_CFG_API_KEY_ENV, api_key_env)
    if timeout_seconds is not None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        db.set_config(_CFG_TIMEOUT_SECONDS, str(timeout_seconds))
    return load_ai_config(db)


def resolve_runtime_ai_config(
    stored: AIConfig,
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    timeout_seconds: int | None = None,
    api_key: str | None = None,
) -> AIConfig:
    """Resolve runtime config from stored defaults + CLI overrides + env vars."""
    runtime_provider = provider or stored.provider
    if runtime_provider not in _VALID_PROVIDERS:
        runtime_provider = "local"

    if provider is not None and provider != stored.provider:
        defaults = _provider_defaults(runtime_provider)
        runtime_model = model or defaults["model"]
        runtime_base_url = base_url or defaults["base_url"]
        runtime_api_key_env = (
            defaults["api_key_env"] if api_key_env is None else api_key_env
        )
        runtime_timeout = (
            timeout_seconds if timeout_seconds is not None else _DEFAULT_TIMEOUT_SECONDS
        )
    else:
        runtime_model = model or stored.model
        runtime_base_url = base_url or stored.base_url
        runtime_api_key_env = stored.api_key_env if api_key_env is None else api_key_env
        runtime_timeout = (
            stored.timeout_seconds if timeout_seconds is None else timeout_seconds
        )

    if runtime_timeout <= 0:
        raise ValueError("timeout must be > 0")

    resolved_api_key = api_key
    if resolved_api_key is None:
        env_name = runtime_api_key_env or _DEFAULT_API_KEY_ENV
        resolved_api_key = os.environ.get(env_name)

    return AIConfig(
        provider=runtime_provider,
        model=runtime_model,
        base_url=runtime_base_url,
        api_key_env=runtime_api_key_env,
        timeout_seconds=runtime_timeout,
        api_key=resolved_api_key,
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


def resolve_prompt_path(prompts_dir: Path, prompt_ref: str) -> Path:
    """Resolve prompt name/path to a YAML file path."""
    raw = prompt_ref.strip()
    if not raw:
        raise ValueError("Prompt name/path cannot be empty.")

    direct = Path(raw)
    if direct.exists() and direct.is_file():
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
            return candidate

    tried = ", ".join(str(p) for p in candidates)
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

    model = raw.get("model")
    if model is not None and not isinstance(model, str):
        raise ValueError(f"Prompt key 'model' must be a string in '{path}'")
    model = model.strip() if isinstance(model, str) else None

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
        model=model or None,
        temperature=temperature,
        source_path=path,
    )


def select_decks_with_subdecks(
    decks: list[dict[str, Any]],
    include_decks: list[str] | None,
) -> list[dict[str, Any]]:
    """Select decks by recursive include semantics.

    If include_decks is empty, all decks are selected.
    """
    targets = [d.strip() for d in (include_decks or []) if d.strip()]
    if not targets:
        return decks

    selected: list[dict[str, Any]] = []
    for deck in decks:
        deck_name = str(deck.get("name", ""))
        if any(_matches_deck_or_subdeck(deck_name, target) for target in targets):
            selected.append(deck)
    return selected


def _matches_deck_or_subdeck(deck_name: str, target: str) -> bool:
    return deck_name == target or deck_name.startswith(f"{target}::")


def _parse_json_object(content: str) -> dict[str, Any]:
    content = content.strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        candidate = content[first_brace : last_brace + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("AI response was not a valid JSON object")


class OpenAICompatibleInlineEditor:
    """Minimal OpenAI-compatible chat-completions client for inline JSON edits."""

    def __init__(self, config: AIConfig):
        self._config = config

    def edit_note(
        self,
        prompt_config: PromptConfig,
        note_payload: dict[str, Any],
    ) -> dict[str, Any]:
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        model = prompt_config.model or self._config.model
        payload = {
            "model": model,
            "temperature": prompt_config.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": prompt_config.prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": "inline_json_edit",
                            "requirements": [
                                "Return JSON only.",
                                "Return the same structure as note.",
                                "Keep note_key unchanged.",
                            ],
                            "note": note_payload,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }

        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self._config.timeout_seconds,
        )
        if resp.status_code >= 400:
            raise ValueError(
                f"AI request failed ({resp.status_code}): {resp.text[:200]}"
            )

        raw = resp.json()
        content = raw["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("AI response content is not text")
        return _parse_json_object(content)


def _matching_field_names(
    fields: dict[str, str],
    patterns: list[str],
) -> list[str]:
    return [
        name
        for name in fields
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
    ]


def run_inline_prompt_on_serialized_collection(
    serialized_data: dict[str, Any],
    include_decks: list[str] | None,
    prompt_config: PromptConfig,
    editor: InlineJSONEditor,
) -> PromptRunResult:
    """Run prompt-driven inline JSON edits over selected decks."""
    decks = serialized_data.get("decks")
    if not isinstance(decks, list):
        raise ValueError("serialized_data must contain 'decks' list")

    selected = select_decks_with_subdecks(decks, include_decks)
    selected_copy = copy.deepcopy(selected)
    result = PromptRunResult()

    for deck in selected_copy:
        deck_name = str(deck.get("name", ""))
        notes = deck.get("notes", [])
        if not isinstance(notes, list):
            continue

        result.processed_decks += 1
        deck_changed = False

        for note in notes:
            if not isinstance(note, dict):
                continue
            result.processed_notes += 1

            note_key = note.get("note_key")
            if not isinstance(note_key, str) or not note_key.strip():
                result.warnings.append(
                    f"{deck_name}/<missing-note_key>: skipped note without note_key"
                )
                continue

            note_type = str(note.get("note_type", ""))
            if not prompt_config.matches_note_type(note_type):
                continue

            fields = note.get("fields", {})
            if not isinstance(fields, dict):
                continue
            string_fields = {
                field_name: value
                for field_name, value in fields.items()
                if isinstance(field_name, str) and isinstance(value, str)
            }
            if not string_fields:
                continue

            target_field_names = _matching_field_names(
                string_fields,
                prompt_config.target_fields,
            )
            if not target_field_names:
                continue

            send_field_names = _matching_field_names(
                string_fields,
                prompt_config.send_fields,
            )
            send_seen = set(send_field_names)
            for field_name in target_field_names:
                if field_name not in send_seen:
                    send_field_names.append(field_name)
                    send_seen.add(field_name)

            note_payload = {
                "note_key": note_key,
                "note_type": note_type,
                "fields": {
                    field_name: string_fields[field_name]
                    for field_name in send_field_names
                },
            }

            result.prompted_notes += 1
            try:
                edited = editor.edit_note(prompt_config, note_payload)
            except Exception as e:
                result.warnings.append(f"{deck_name}/{note_key}: {e}")
                continue

            if not isinstance(edited, dict):
                result.warnings.append(
                    f"{deck_name}/{note_key}: response is not a JSON object"
                )
                continue
            if edited.get("note_key") != note_key:
                result.warnings.append(
                    f"{deck_name}/{note_key}: response note_key mismatch"
                )
                continue

            edited_fields = edited.get("fields")
            if not isinstance(edited_fields, dict):
                result.warnings.append(
                    f"{deck_name}/{note_key}: response missing fields object"
                )
                continue
            missing_or_bad = [
                field_name
                for field_name in send_field_names
                if not isinstance(edited_fields.get(field_name), str)
            ]
            if missing_or_bad:
                result.warnings.append(
                    f"{deck_name}/{note_key}: response fields invalid: "
                    f"{', '.join(missing_or_bad)}"
                )
                continue

            for field_name in target_field_names:
                edited_text = edited_fields[field_name]
                original_text = string_fields[field_name]
                if edited_text == original_text:
                    continue

                fields[field_name] = edited_text
                deck_changed = True
                result.changed_fields += 1
                result.changes.append(
                    PromptChange(
                        prompt_name=prompt_config.name,
                        deck_name=deck_name,
                        note_key=note_key,
                        note_type=note_type,
                        field_name=field_name,
                        original_text=original_text,
                        edited_text=edited_text,
                    )
                )

        if deck_changed:
            result.changed_decks.append(deck)

    return result
