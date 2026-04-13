"""Collection-local model registry with provider routing and optional pricing."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

from ankiops.config import LLM_DIR

_TOKENS_PER_MTOK = Decimal("1000000")
_USD_CENTS_QUANTUM = Decimal("0.01")
_MODELS_FILE_NAME = "models.yaml"
_SUPPORTED_MODEL_KEYS = {
    "name",
    "api_id",
    "provider",
    "base_url",
    "api_key_env",
    "input_usd_per_mtok",
    "output_usd_per_mtok",
}


class ModelRegistryError(ValueError):
    """Raised when a collection model registry file is invalid."""


@dataclass(frozen=True)
class CostEstimate:
    input_usd: Decimal
    output_usd: Decimal

    @property
    def total_usd(self) -> Decimal:
        return self.input_usd + self.output_usd

    def format(self) -> str:
        return format_usd_cents(self.total_usd)


@dataclass(frozen=True)
class ProviderModel:
    name: str
    api_id: str
    provider: str
    base_url: str
    api_key_env: str
    input_usd_per_mtok: Decimal | None = None
    output_usd_per_mtok: Decimal | None = None

    def estimate_cost(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> CostEstimate | None:
        if self.input_usd_per_mtok is None or self.output_usd_per_mtok is None:
            return None
        return CostEstimate(
            input_usd=(
                Decimal(input_tokens) * self.input_usd_per_mtok / _TOKENS_PER_MTOK
            ),
            output_usd=(
                Decimal(output_tokens) * self.output_usd_per_mtok / _TOKENS_PER_MTOK
            ),
        )

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class ModelRegistry:
    models: tuple[ProviderModel, ...]
    _models_by_name: dict[str, ProviderModel] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        by_name: dict[str, ProviderModel] = {}
        for model in self.models:
            if model.name in by_name:
                raise ModelRegistryError(
                    f"duplicate model name '{model.name}' in registry"
                )
            by_name[model.name] = model

        if not by_name:
            raise ModelRegistryError("model registry must contain at least one model")

        object.__setattr__(self, "_models_by_name", by_name)

    def parse(self, value: str) -> ProviderModel | None:
        return self._models_by_name.get(value.strip())

    def format_names(self) -> str:
        return ", ".join(self._models_by_name)


def _models_path(collection_dir: Path) -> Path:
    return collection_dir / LLM_DIR / _MODELS_FILE_NAME


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ModelRegistryError("model registry must be a YAML mapping")
    return raw


def _require_string(
    value: Any,
    *,
    key: str,
    item_label: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelRegistryError(f"{item_label}: '{key}' must be a non-empty string")
    return value.strip()


def _parse_decimal(
    value: Any,
    *,
    key: str,
    item_label: str,
) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ModelRegistryError(f"{item_label}: '{key}' must be numeric")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ModelRegistryError(f"{item_label}: '{key}' must be numeric") from error


def _parse_model_entry(entry: Any, *, index: int) -> ProviderModel:
    item_label = f"models[{index}]"
    if not isinstance(entry, dict):
        raise ModelRegistryError(f"{item_label}: model entry must be a mapping")

    unknown = sorted(set(entry.keys()) - _SUPPORTED_MODEL_KEYS)
    if unknown:
        raise ModelRegistryError(f"{item_label}: unknown key(s): {', '.join(unknown)}")

    return ProviderModel(
        name=_require_string(entry.get("name"), key="name", item_label=item_label),
        api_id=_require_string(
            entry.get("api_id"),
            key="api_id",
            item_label=item_label,
        ),
        provider=_require_string(
            entry.get("provider"),
            key="provider",
            item_label=item_label,
        ),
        base_url=_require_string(
            entry.get("base_url"),
            key="base_url",
            item_label=item_label,
        ),
        api_key_env=_require_string(
            entry.get("api_key_env"),
            key="api_key_env",
            item_label=item_label,
        ),
        input_usd_per_mtok=_parse_decimal(
            entry.get("input_usd_per_mtok"),
            key="input_usd_per_mtok",
            item_label=item_label,
        ),
        output_usd_per_mtok=_parse_decimal(
            entry.get("output_usd_per_mtok"),
            key="output_usd_per_mtok",
            item_label=item_label,
        ),
    )


def _parse_registry(path: Path) -> ModelRegistry:
    mapping = _read_yaml_mapping(path)
    unknown_top_level = sorted(set(mapping.keys()) - {"models"})
    if unknown_top_level:
        raise ModelRegistryError(
            "unknown top-level key(s): " + ", ".join(unknown_top_level)
        )

    models = mapping.get("models")
    if not isinstance(models, list) or not models:
        raise ModelRegistryError("'models' must be a non-empty list")

    parsed = tuple(
        _parse_model_entry(entry, index=index) for index, entry in enumerate(models)
    )
    return ModelRegistry(models=parsed)


def load_model_registry(*, collection_dir: Path) -> ModelRegistry:
    path = _models_path(collection_dir)
    if not path.exists():
        raise ModelRegistryError(
            f"{path}: model registry file not found. "
            "Run 'ankiops init' to eject llm/models.yaml."
        )
    if not path.is_file():
        raise ModelRegistryError(f"{path}: model registry must be a file")
    try:
        return _parse_registry(path)
    except ModelRegistryError as error:
        raise ModelRegistryError(f"{path}: {error}") from error


def parse_model(
    value: str,
    *,
    collection_dir: Path,
) -> ProviderModel | None:
    return load_model_registry(collection_dir=collection_dir).parse(value)


def format_supported_model_names(*, collection_dir: Path) -> str:
    return load_model_registry(collection_dir=collection_dir).format_names()


def format_usd_cents(amount: Decimal) -> str:
    quantized = amount.quantize(_USD_CENTS_QUANTUM, rounding=ROUND_HALF_UP)
    return f"${quantized:.2f}"
