from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ankiops.cli import main, run_llm
from ankiops.llm.errors import LlmFatalError
from ankiops.llm.models import TaskCatalog


def test_cli_llm_dispatches_task_run():
    with (
        patch("ankiops.cli.require_initialized_collection_dir"),
        patch("ankiops.cli.run_task") as run_task,
        patch("sys.argv", ["ankiops", "llm", "grammar"]),
    ):
        main()

    run_task.assert_called_once()
    assert run_task.call_args.kwargs["task_name"] == "grammar"


def test_cli_llm_list_exits_on_invalid_config(tmp_path):
    with (
        patch("ankiops.cli.require_initialized_collection_dir", return_value=tmp_path),
        patch("ankiops.cli.FileSystemAdapter.load_note_type_configs", return_value=[]),
        patch(
            "ankiops.cli.load_llm_task_catalog",
            return_value=TaskCatalog({}, {"/tmp/grammar.yaml": "bad config"}),
        ),
        patch("sys.argv", ["ankiops", "llm"]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1


def test_run_llm_exits_cleanly_on_fatal_provider_error(tmp_path, caplog):
    args = SimpleNamespace(
        task_name="grammar",
        model=None,
        no_auto_commit=True,
    )

    with (
        patch("ankiops.cli.require_initialized_collection_dir", return_value=tmp_path),
        patch(
            "ankiops.cli.run_task",
            side_effect=LlmFatalError("Provider authentication failed: bad key"),
        ),
        caplog.at_level(logging.ERROR),
    ):
        with pytest.raises(SystemExit) as exc:
            run_llm(args)

    assert exc.value.code == 1
    assert "Provider authentication failed: bad key" in caplog.text
