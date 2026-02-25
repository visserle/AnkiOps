"""Small declarative case models used by scenario tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CollectionState = Literal["FRESH", "RUN", "CORR"]


@dataclass(frozen=True)
class StateCase:
    """A scenario state case with explicit pytest id."""

    id: str
    state: CollectionState


def mk_state_cases(*, fresh: str, run: str, corr: str) -> list[StateCase]:
    """Build the common FRESH/RUN/CORR case list."""
    return [
        StateCase(id=fresh, state="FRESH"),
        StateCase(id=run, state="RUN"),
        StateCase(id=corr, state="CORR"),
    ]
