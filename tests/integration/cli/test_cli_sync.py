"""CLI sync behavior tests."""

import logging
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ankiops.anki_client import AnkiConnectionError
from ankiops.cli import main, run_fix_image_widths, run_serialize


class _FailingProfileAnki:
    def get_active_profile(self):
        raise AnkiConnectionError("Connection reset by peer")


def test_run_ma_warns_for_untracked_anki_decks(world, caplog):
    world.add_anki_note(
        deck_name="AnkiOnlyDeck",
        fields={"Question": "remote", "Answer": "remote"},
        note_key="remote-key",
    )

    with caplog.at_level(logging.WARNING):
        world.run_ma()

    assert "untracked Anki deck(s)" in caplog.text
    assert "AnkiOnlyDeck" in caplog.text
    world.assert_anki_note(deck_name="AnkiOnlyDeck", note_key="remote-key")


def test_run_ma_has_no_untracked_warning_for_declared_collection(world, caplog):
    world.write_deck("DeclaredDeck", "Q: local\nA: answer")

    with caplog.at_level(logging.WARNING):
        world.run_ma()

    assert "untracked Anki deck(s)" not in caplog.text
    world.assert_anki_note(
        deck_name="DeclaredDeck",
        question="local",
        answer="answer",
    )
    assert world.extract_note_keys("DeclaredDeck")


def test_run_ma_warns_for_keyless_anki_notes_it_protects(world, caplog):
    world.write_qa_deck("ProtectDeck", [("managed", "local", "managed-key")])
    keyless_id = world.add_anki_note(
        deck_name="ProtectDeck",
        fields={"Question": "draft", "Answer": "keep me"},
    )

    with caplog.at_level(logging.WARNING):
        world.run_ma()

    assert "Protected 1 keyless Anki note(s) during import." in caplog.text
    assert "ProtectDeck" in caplog.text
    assert keyless_id in world.mock_anki.notes


def test_run_am_warns_for_keyless_markdown_notes_it_protects(world, caplog):
    world.write_deck("DraftDeck", "Q: draft\nA: keep me")

    with caplog.at_level(logging.WARNING):
        world.run_am()

    assert "Protected 1 local markdown note(s)" in caplog.text
    assert "DraftDeck" in caplog.text
    assert "Use 'ankiops ma'" in caplog.text
    world.assert_deck_contains("DraftDeck", "Q: draft")


def test_run_ma_logs_import_errors_with_actionable_details(world, caplog):
    world.write_deck(
        "Rhetorik",
        (
            "<!-- note_key: shared-key -->\n"
            "<!-- note_type: AnkiOpsQA -->\n"
            "Q: local question\n"
            "A: local answer"
        ),
    )
    world.add_anki_note(
        deck_name="Rhetorik",
        note_type="AnkiOpsCloze",
        fields={"Text": "{{c1::remote}}", "AnkiOps Key": "shared-key"},
    )

    with caplog.at_level(logging.ERROR):
        world.run_ma()

    assert "Import errors:" in caplog.text
    assert "Rhetorik" in caplog.text
    assert "field names differ" in caplog.text
    assert "Review and resolve errors above" in caplog.text


def test_run_ma_auto_commit_snapshots_declared_state_before_sync_mutates_media(
    world,
):
    world.write_deck("MediaDeck", "Q: prompt\nA: ![img](media/img.png)")
    world.write_media("img.png", b"image-content")
    world.init_git()

    world.run_ma(no_auto_commit=False)

    committed = subprocess.run(
        ["git", "show", "HEAD:MediaDeck.md"],
        cwd=world.root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    working = world.read_deck("MediaDeck")

    assert "media/img.png" in committed
    assert "<!-- note_key:" not in committed
    assert "media/img_" in working
    assert "<!-- note_key:" in working


def test_run_ma_logs_real_media_status(world, caplog):
    world.write_deck("MediaDeck", "Q: prompt\nA: ![img](media/img.png)")
    world.write_media("img.png", b"image-content")

    with caplog.at_level(logging.INFO):
        world.run_ma()

    assert "Media: 1 files checked" in caplog.text
    assert "1 hashed" in caplog.text
    assert "1 synced" in caplog.text


def test_run_am_logs_missing_anki_media_summary(world, caplog):
    world.write_deck(
        "MediaDeck",
        "<!-- note_key: key-1 -->\nQ: prompt\nA: ![missing](media/missing.png)",
    )
    note_id = world.add_anki_note(
        deck_name="MediaDeck",
        fields={"Question": "prompt", "Answer": "![missing](media/missing.png)"},
        note_key="key-1",
    )
    world.seed_db_link("key-1", note_id)

    with caplog.at_level(logging.INFO):
        world.run_am()

    assert "Media: 1 files checked" in caplog.text
    assert "0 pulled, 1 missing in Anki" in caplog.text


def test_cli_version_flag_prints_package_version(capsys):
    with (
        patch("ankiops.cli._get_cli_version", return_value="9.9.9"),
        patch("sys.argv", ["ankiops", "--version"]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "ankiops 9.9.9"


def test_cli_help_lists_version_flag(capsys):
    with patch("sys.argv", ["ankiops", "--help"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--version" in captured.out


def test_cli_collab_publish_accepts_public_visibility_flag():
    captured = []

    with (
        patch(
            "ankiops.cli.run_collab_impl",
            side_effect=lambda args: captured.append(args),
        ),
        patch(
            "sys.argv",
            ["ankiops", "collab", "publish", "Deck", "owner/repo", "--public"],
        ),
    ):
        main()

    assert captured[0].deck == "Deck"
    assert captured[0].repo == "owner/repo"
    assert captured[0].public is True


def test_cli_init_exits_cleanly_on_anki_connection_error(caplog):
    with (
        patch("ankiops.cli.connect_or_exit", return_value=_FailingProfileAnki()),
        patch("ankiops.cli.configure_logging"),
        patch("sys.argv", ["ankiops", "init"]),
        caplog.at_level(logging.ERROR),
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1
    assert "Error communicating with Anki" in caplog.text


def test_cli_welcome_mentions_version_and_version_flag(capsys):
    with (
        patch("ankiops.cli._get_cli_version", return_value="9.9.9"),
        patch("sys.argv", ["ankiops"]),
    ):
        main()

    captured = capsys.readouterr()
    assert "AnkiOps v9.9.9" in captured.out
    assert "ankiops --version" in captured.out


def test_run_serialize_rejects_no_subdecks_without_deck():
    args = SimpleNamespace(
        output=None,
        deck=None,
        no_subdecks=True,
    )

    with pytest.raises(SystemExit) as exc:
        run_serialize(args)

    assert exc.value.code == 2


def test_run_fix_image_widths_rejects_no_subdecks_without_deck():
    args = SimpleNamespace(
        deck=None,
        no_subdecks=True,
        tolerance=5,
        width=None,
        no_auto_commit=True,
    )

    with pytest.raises(SystemExit) as exc:
        run_fix_image_widths(args)

    assert exc.value.code == 2
