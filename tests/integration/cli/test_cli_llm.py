from __future__ import annotations

from unittest.mock import patch

import pytest

from ankiops.cli import main


def test_cli_llm_run_dispatches():
    with (
        patch("ankiops.cli.require_initialized_collection_dir"),
        patch("ankiops.cli.run_task") as run_task,
        patch("sys.argv", ["ankiops", "llm", "run", "grammar", "--dry-run"]),
    ):
        main()

    run_task.assert_called_once()
    assert run_task.call_args.kwargs["task_name"] == "grammar"
    assert run_task.call_args.kwargs["dry_run"] is True


def test_cli_llm_tasks_exits_on_invalid_config(tmp_path):
    with (
        patch("ankiops.cli.require_initialized_collection_dir", return_value=tmp_path),
        patch(
            "ankiops.cli.list_tasks",
            return_value=([], {"/tmp/grammar.yaml": "bad config"}),
        ),
        patch("sys.argv", ["ankiops", "llm", "tasks"]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1
