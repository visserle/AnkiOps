"""Shared fixtures for AnkiOps tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ankiops.config import get_note_types_dir
from ankiops.fs import FileSystemAdapter
from tests.support.fake_anki import MockAnki
from tests.support.sync_world import SyncWorld


@pytest.fixture(scope="session")
def fs():
    """FileSystemAdapter pre-loaded with built-in note types (shared across tests)."""
    adapter = FileSystemAdapter()
    adapter.set_configs(adapter.load_note_type_configs(get_note_types_dir()))
    return adapter


@pytest.fixture(scope="session")
def choice_config(fs):
    """AnkiOpsChoice config for validation tests."""
    return next(
        config for config in fs._note_type_configs if config.name == "AnkiOpsChoice"
    )


@pytest.fixture
def mock_anki() -> MockAnki:
    return MockAnki()


@pytest.fixture
def run_ankiops(mock_anki):
    """Patch both Anki invoke entry points with the stateful fake backend."""
    with (
        patch("ankiops.anki_client.invoke", side_effect=mock_anki.invoke),
        patch("ankiops.anki.invoke", side_effect=mock_anki.invoke),
    ):
        yield


@pytest.fixture
def world(tmp_path, mock_anki, run_ankiops) -> SyncWorld:
    """High-level sync helper for scenario-style tests."""
    return SyncWorld(root=tmp_path, mock_anki=mock_anki)


@pytest.fixture(autouse=True)
def mock_input():
    """Always answer 'y' to confirmation prompts."""
    with patch("builtins.input", return_value="y"):
        yield
