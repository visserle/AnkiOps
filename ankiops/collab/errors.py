"""Collab command error messages."""

from __future__ import annotations


class RepositoryCollisionError(ValueError):
    """A requested GitHub repository belongs to an unrelated history."""
