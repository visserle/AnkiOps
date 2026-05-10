"""Catalog helpers for runtime v2."""

from __future__ import annotations

from ankiops.llm.model_registry import ModelSpec

from .domain.capabilities import ModelCapabilities, resolve_model_capabilities
from .domain.errors import CapabilityError


def capabilities_from_model_spec(model: ModelSpec) -> ModelCapabilities:
    """Resolve runtime v2 capabilities from a collection model spec."""

    return resolve_model_capabilities(
        provider=model.provider,
        model_id=model.model_id,
    )


def ensure_model_supported(model: ModelSpec) -> None:
    """Validate that a model spec supports strict structured JSON execution."""

    try:
        capabilities_from_model_spec(model)
    except CapabilityError as error:
        raise ValueError(
            f"Model '{model.model}' ({model.provider}/{model.model_id}) "
            f"is not supported: {error}"
        ) from error
