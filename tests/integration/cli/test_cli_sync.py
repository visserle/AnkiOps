"""CLI sync behavior tests."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ankiops.cli import main, run_ma
from ankiops.models import CollectionImportResult, UntrackedDeck


def _run_ma_with_summary(tmp_path, caplog, summary: CollectionImportResult):
    fake_anki = MagicMock()
    fake_anki.get_active_profile.return_value = "TestProfile"
    args = SimpleNamespace(no_auto_commit=True)

    with (
        patch("ankiops.cli.connect_or_exit", return_value=fake_anki),
        patch("ankiops.cli.require_collection_dir", return_value=tmp_path),
        patch("ankiops.cli.sync_media_to_anki"),
        patch("ankiops.cli.sync_note_types", return_value=""),
        patch("ankiops.cli.import_collection", return_value=summary),
        caplog.at_level(logging.WARNING),
    ):
        run_ma(args)


def test_run_ma_warns_for_untracked_decks(tmp_path, caplog):
    summary = CollectionImportResult(
        results=[],
        untracked_decks=[UntrackedDeck("AnkiOnlyDeck", 99, [101, 102])],
    )
    _run_ma_with_summary(tmp_path, caplog, summary)

    assert "untracked Anki deck(s)" in caplog.text
    assert "AnkiOnlyDeck" in caplog.text


def test_run_ma_has_no_untracked_warning_when_none(tmp_path, caplog):
    summary = CollectionImportResult(results=[], untracked_decks=[])
    _run_ma_with_summary(tmp_path, caplog, summary)

    assert "untracked Anki deck(s)" not in caplog.text


@pytest.mark.parametrize(
    "argv",
    [
        ["ankiops", "am", "--keep-orphans"],
        ["ankiops", "ma", "--only-add-new"],
    ],
)
def test_removed_full_sync_flags_are_rejected(argv):
    with patch("sys.argv", argv):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 2
