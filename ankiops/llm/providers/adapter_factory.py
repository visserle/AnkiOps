"""Provider adapter factory for runtime."""

from __future__ import annotations

from ankiops.llm.model_registry import ModelSpec

from ..domain.capabilities import ModelCapabilities, TransportMode
from .adapter_base import StructuredProviderAdapter
from .anthropic_tool_strict_adapter import AnthropicToolStrictAdapter
from .gemini_structured_adapter import GeminiStructuredAdapter
from .openai_compat_adapter import OpenAICompatStructuredAdapter
from .openai_responses_adapter import OpenAIResponsesStructuredAdapter


def create_structured_adapter(
    *,
    model: ModelSpec,
    capabilities: ModelCapabilities,
    api_key: str,
    timeout_seconds: int,
) -> StructuredProviderAdapter:
    if capabilities.transport_mode is TransportMode.OPENAI_RESPONSES_STRUCTURED:
        return OpenAIResponsesStructuredAdapter(
            api_key=api_key,
            base_url=model.base_url,
            timeout_seconds=timeout_seconds,
        )

    if capabilities.transport_mode is TransportMode.OPENAI_COMPAT_JSON_SCHEMA_STRICT:
        return OpenAICompatStructuredAdapter(
            api_key=api_key,
            base_url=model.base_url,
            provider=model.provider,
            timeout_seconds=timeout_seconds,
            retries=model.retries,
            retry_backoff_seconds=model.retry_backoff_seconds,
            retry_backoff_jitter=model.retry_backoff_jitter,
        )

    if capabilities.transport_mode is TransportMode.ANTHROPIC_TOOL_USE_STRICT:
        return AnthropicToolStrictAdapter(
            api_key=api_key,
            base_url=model.base_url,
            timeout_seconds=timeout_seconds,
            retries=model.retries,
            retry_backoff_seconds=model.retry_backoff_seconds,
            retry_backoff_jitter=model.retry_backoff_jitter,
        )

    if capabilities.transport_mode is TransportMode.GEMINI_NATIVE_STRUCTURED:
        return GeminiStructuredAdapter(
            api_key=api_key,
            base_url=model.base_url,
            timeout_seconds=timeout_seconds,
            retries=model.retries,
            retry_backoff_seconds=model.retry_backoff_seconds,
            retry_backoff_jitter=model.retry_backoff_jitter,
        )

    raise ValueError(f"Unsupported transport mode: {capabilities.transport_mode}")
