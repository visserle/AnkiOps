"""Shared assertion helpers for readable, uniform tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def assert_summary(
    summary: Any,
    *,
    created: int | None = None,
    updated: int | None = None,
    moved: int | None = None,
    deleted: int | None = None,
    errors: int | None = None,
    total: int | None = None,
) -> None:
    """Assert selected summary counters while leaving others unconstrained."""
    if created is not None:
        assert summary.created == created
    if updated is not None:
        assert summary.updated == updated
    if moved is not None:
        assert summary.moved == moved
    if deleted is not None:
        assert summary.deleted == deleted
    if errors is not None:
        assert summary.errors == errors
    if total is not None:
        assert summary.total == total


def assert_unique(values: Sequence[str]) -> None:
    """Assert all values are unique in order-agnostic contexts."""
    assert len(values) == len(set(values))
