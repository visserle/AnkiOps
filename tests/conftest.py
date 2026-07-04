"""Collab fixtures for AnkiOps tests."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from tests.support.deck_files import DeckFileHarness
from tests.support.fake_anki import MockAnki
from tests.support.sync_world import SyncWorld


def pytest_addoption(parser):
    parser.addoption(
        "--live-llm-model",
        action="store",
        default=os.getenv("ANKIOPS_LIVE_LLM_MODEL", ""),
        help=(
            "Model name for live LLM integration tests. "
            "Must exist in ankiops/llm/_models.yaml or collection/llm/_models.yaml."
        ),
    )


@pytest.fixture(scope="session")
def default_note_types_dir(tmp_path_factory):
    note_types_dir = tmp_path_factory.mktemp("default-note-types") / "note_types"
    DeckFileHarness().eject_default_note_types(note_types_dir)
    return note_types_dir


@pytest.fixture(scope="session")
def fs(default_note_types_dir):
    """DeckFileHarness pre-loaded with built-in note types (collab across tests)."""
    adapter = DeckFileHarness()
    adapter.set_note_types(adapter.load_note_types(default_note_types_dir))
    return adapter


@pytest.fixture(scope="session")
def choice_config(fs):
    """AnkiOpsChoice config for validation tests."""
    return next(config for config in fs._note_types if config.name == "AnkiOpsChoice")


@pytest.fixture
def mock_anki() -> MockAnki:
    return MockAnki()


@pytest.fixture
def world(tmp_path, mock_anki) -> SyncWorld:
    """High-level sync helper for scenario-style tests."""
    return SyncWorld(root=tmp_path, mock_anki=mock_anki)


@pytest.fixture(autouse=True)
def mock_input():
    """Always answer 'y' to confirmation prompts."""
    with patch("builtins.input", return_value="y"):
        yield
