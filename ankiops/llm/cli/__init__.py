"""CLI wiring for LLM runtime."""

from .commands import configure_llm_parser, run_llm

__all__ = ["configure_llm_parser", "run_llm"]
