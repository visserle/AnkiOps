"""CLI AI argument parsing behavior tests."""

from unittest.mock import patch

import pytest

from ankiops.cli import main


def test_ai_requires_task_flag():
    with patch("sys.argv", ["ankiops", "ai"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 2


def test_ai_parses_task_runtime_and_repeatable_include_decks():
    captured = {}

    def _fake_handler(args):
        captured["args"] = args

    with (
        patch("ankiops.cli.run_ai_task", side_effect=_fake_handler),
        patch(
            "sys.argv",
            [
                "ankiops",
                "ai",
                "--task",
                "grammar",
                "-d",
                "Biology",
                "--include-deck",
                "Biology::Cells",
                "--profile",
                "openai-fast",
                "--max-in-flight",
                "6",
                "--temperature",
                "0.5",
                "--progress",
                "on",
            ],
        ),
    ):
        main()

    args = captured["args"]
    assert args.task == "grammar"
    assert args.include_deck == ["Biology", "Biology::Cells"]
    assert args.profile == "openai-fast"
    assert args.max_in_flight == 6
    assert args.temperature == pytest.approx(0.5)
    assert args.progress == "on"


def test_ai_config_does_not_require_task():
    captured = {}

    def _fake_handler(args):
        captured["args"] = args

    with (
        patch("ankiops.cli.run_ai_config", side_effect=_fake_handler),
        patch("sys.argv", ["ankiops", "ai", "config", "--provider", "groq"]),
    ):
        main()

    args = captured["args"]
    assert args.provider == "groq"
    assert args.profile is None


def test_ai_rejects_out_of_range_temperature_before_handler():
    with (
        patch("ankiops.cli.run_ai_task") as run_ai_task,
        patch(
            "sys.argv",
            ["ankiops", "ai", "--task", "grammar", "--temperature", "2.5"],
        ),
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 2
    run_ai_task.assert_not_called()


def test_ai_rejects_non_positive_runtime_timeout_for_config():
    with (
        patch("ankiops.cli.run_ai_config") as run_ai_config,
        patch("sys.argv", ["ankiops", "ai", "config", "--timeout", "0"]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 2
    run_ai_config.assert_not_called()
