"""CLI sync behavior tests."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ankiops.cli import main, run_am, run_ma
from ankiops.models import (
    Change,
    ChangeType,
    CollectionResult,
    SyncResult,
    UntrackedDeck,
)


def _run_ma_with_summary(tmp_path, caplog, summary: CollectionResult):
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
    summary = CollectionResult.for_import(
        results=[],
        untracked_decks=[UntrackedDeck("AnkiOnlyDeck", 99, [101, 102])],
    )
    _run_ma_with_summary(tmp_path, caplog, summary)

    assert "untracked Anki deck(s)" in caplog.text
    assert "AnkiOnlyDeck" in caplog.text


def test_run_ma_has_no_untracked_warning_when_none(tmp_path, caplog):
    summary = CollectionResult.for_import(results=[], untracked_decks=[])
    _run_ma_with_summary(tmp_path, caplog, summary)

    assert "untracked Anki deck(s)" not in caplog.text


def test_run_ma_logs_import_errors_with_actionable_details(tmp_path, caplog):
    sync_result = SyncResult.for_notes(
        name="Rhetorik",
        file_path=tmp_path / "Rhetorik.md",
    )
    sync_result.errors.append(
        "Note type mismatch for note_key: nk-1: markdown uses 'AnkiOpsQA' "
        "but Anki has 'AnkiOpsCloze'. Anki cannot convert existing notes "
        "between note types. Remove this note's key comment "
        "(<!-- note_key: nk-1 -->) to force creating a new note with the "
        "new type on the next import."
    )
    summary = CollectionResult.for_import(results=[sync_result], untracked_decks=[])

    _run_ma_with_summary(tmp_path, caplog, summary)

    assert "Import errors:" in caplog.text
    assert "Rhetorik" in caplog.text
    assert "Remove this note's key comment" in caplog.text
    assert "Review and resolve errors above" in caplog.text


def test_run_ma_auto_commit_runs_before_media_sync(tmp_path):
    fake_anki = MagicMock()
    fake_anki.get_active_profile.return_value = "TestProfile"
    args = SimpleNamespace(no_auto_commit=False)
    order: list[str] = []

    def _record_git(*_args, **_kwargs):
        order.append("git")
        return True

    def _record_media(*_args, **_kwargs):
        order.append("media")
        result = SyncResult.for_media()
        result.checked = 1
        return result

    def _record_note_types(*_args, **_kwargs):
        order.append("note_types")
        return ""

    def _record_import(*_args, **_kwargs):
        order.append("import")
        return CollectionResult.for_import(results=[], untracked_decks=[])

    with (
        patch("ankiops.cli.connect_or_exit", return_value=fake_anki),
        patch("ankiops.cli.require_collection_dir", return_value=tmp_path),
        patch("ankiops.cli.SQLiteDbAdapter.load", return_value=MagicMock()),
        patch("ankiops.cli.git_snapshot", side_effect=_record_git),
        patch("ankiops.cli.sync_media_to_anki", side_effect=_record_media),
        patch("ankiops.cli.sync_note_types", side_effect=_record_note_types),
        patch("ankiops.cli.import_collection", side_effect=_record_import),
    ):
        run_ma(args)

    assert order == ["git", "media", "note_types", "import"]


def test_run_ma_logs_media_status_in_normal_mode(tmp_path, caplog):
    fake_anki = MagicMock()
    fake_anki.get_active_profile.return_value = "TestProfile"
    args = SimpleNamespace(no_auto_commit=True)
    media_result = SyncResult.for_media()
    media_result.checked = 12
    media_result.unchanged = 12

    with (
        patch("ankiops.cli.connect_or_exit", return_value=fake_anki),
        patch("ankiops.cli.require_collection_dir", return_value=tmp_path),
        patch("ankiops.cli.SQLiteDbAdapter.load", return_value=MagicMock()),
        patch("ankiops.cli.sync_media_to_anki", return_value=media_result),
        patch("ankiops.cli.sync_note_types", return_value=""),
        patch(
            "ankiops.cli.import_collection",
            return_value=CollectionResult.for_import(results=[], untracked_decks=[]),
        ),
        caplog.at_level(logging.INFO),
    ):
        run_ma(args)

    assert "Media: 12 files checked — no changes" in caplog.text


def test_run_am_logs_missing_media_summary(tmp_path, caplog):
    fake_anki = MagicMock()
    fake_anki.get_active_profile.return_value = "TestProfile"
    args = SimpleNamespace(no_auto_commit=True)
    media_result = SyncResult.for_media()
    media_result.changes = [Change(ChangeType.SYNC, "a.png", "a.png")]
    media_result.checked = 5
    media_result.missing = 2

    with (
        patch("ankiops.cli.connect_or_exit", return_value=fake_anki),
        patch("ankiops.cli.require_collection_dir", return_value=tmp_path),
        patch("ankiops.cli.SQLiteDbAdapter.load", return_value=MagicMock()),
        patch(
            "ankiops.cli.export_collection",
            return_value=CollectionResult.for_export(results=[], extra_changes=[]),
        ),
        patch("ankiops.cli.sync_media_from_anki", return_value=media_result),
        caplog.at_level(logging.INFO),
    ):
        run_am(args)

    assert "Media: 5 files checked" in caplog.text
    assert "1 pulled, 2 missing in Anki" in caplog.text


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
