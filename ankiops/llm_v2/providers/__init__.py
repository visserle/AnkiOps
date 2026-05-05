"""Provider adapters for runtime v2."""

from .adapter_base import AdapterRequest, StructuredProviderAdapter


def create_structured_adapter(*args, **kwargs):
    from .adapter_factory import create_structured_adapter as _create_structured_adapter

    return _create_structured_adapter(*args, **kwargs)

__all__ = [
    "AdapterRequest",
    "StructuredProviderAdapter",
    "create_structured_adapter",
]
