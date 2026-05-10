"""Model transport capability definitions for runtime v2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import CapabilityError


class TransportMode(Enum):
    OPENAI_RESPONSES_STRUCTURED = "openai_responses"
    OPENAI_COMPAT_JSON_SCHEMA_STRICT = "openai_compat"
    ANTHROPIC_TOOL_USE_STRICT = "anthropic_native"
    GEMINI_NATIVE_STRUCTURED = "gemini_native"


class SchemaLimitsProfile(Enum):
    OPENAI_SUBSET = "openai_subset"
    ANTHROPIC_SUBSET = "anthropic_subset"
    GEMINI_SUBSET = "gemini_subset"


@dataclass(frozen=True)
class ModelCapabilities:
    provider: str
    model_id: str
    transport_mode: TransportMode
    supports_strict_schema: bool
    supports_streaming_structured: bool = False
    schema_limits_profile: SchemaLimitsProfile = SchemaLimitsProfile.OPENAI_SUBSET


def resolve_model_capabilities(
    *,
    provider: str,
    model_id: str,
    transport_mode: TransportMode | None = None,
    supports_strict_schema: bool | None = None,
    supports_streaming_structured: bool | None = None,
    schema_limits_profile: SchemaLimitsProfile | None = None,
) -> ModelCapabilities:
    normalized_provider = provider.strip().lower()
    resolved_transport = transport_mode or _default_transport(normalized_provider)
    resolved_strict_schema = (
        supports_strict_schema
        if supports_strict_schema is not None
        else _default_strict_schema_support(normalized_provider)
    )

    capabilities = ModelCapabilities(
        provider=normalized_provider,
        model_id=model_id,
        transport_mode=resolved_transport,
        supports_strict_schema=resolved_strict_schema,
        supports_streaming_structured=(
            supports_streaming_structured
            if supports_streaming_structured is not None
            else False
        ),
        schema_limits_profile=(
            schema_limits_profile
            if schema_limits_profile is not None
            else _default_schema_limits_profile(resolved_transport)
        ),
    )
    require_strict_schema_support(capabilities)
    return capabilities


def require_strict_schema_support(capabilities: ModelCapabilities) -> None:
    if not capabilities.supports_strict_schema:
        raise CapabilityError("Model does not support strict structured JSON output")


def _default_transport(provider: str) -> TransportMode:
    if provider == "openai":
        return TransportMode.OPENAI_RESPONSES_STRUCTURED
    if provider == "anthropic":
        return TransportMode.ANTHROPIC_TOOL_USE_STRICT
    if provider in {"gemini", "google", "google_ai_studio"}:
        return TransportMode.GEMINI_NATIVE_STRUCTURED
    return TransportMode.OPENAI_COMPAT_JSON_SCHEMA_STRICT


def _default_strict_schema_support(provider: str) -> bool:
    return provider in {
        "openai",
        "openai-compatible",
        "openai_compatible",
        "anthropic",
        "gemini",
        "google",
        "google_ai_studio",
        "openrouter",
        "groq",
        "xai",
        "ollama",
    }


def _default_schema_limits_profile(
    transport_mode: TransportMode,
) -> SchemaLimitsProfile:
    if transport_mode is TransportMode.ANTHROPIC_TOOL_USE_STRICT:
        return SchemaLimitsProfile.ANTHROPIC_SUBSET
    if transport_mode is TransportMode.GEMINI_NATIVE_STRUCTURED:
        return SchemaLimitsProfile.GEMINI_SUBSET
    return SchemaLimitsProfile.OPENAI_SUBSET
