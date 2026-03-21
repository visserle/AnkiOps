from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ankiops.cli import main, run_llm
from ankiops.llm.anthropic_models import SONNET
from ankiops.llm.db import LlmJobListItem
from ankiops.llm.models import LlmJobResult, LlmJobStatus, TaskCatalog, TaskRunSummary


def test_cli_llm_dispatches_task_run(capsys):
    success_result = LlmJobResult(
        job_id=24,
        status="completed",
        summary=TaskRunSummary(task_name="grammar", model=SONNET),
        failed=False,
        persisted=False,
    )
    with (
        patch("ankiops.cli.require_initialized_collection_dir"),
        patch("ankiops.cli.run_task", return_value=success_result) as run_task,
        patch("sys.argv", ["ankiops", "llm", "run", "grammar"]),
    ):
        main()

    run_task.assert_called_once()
    assert run_task.call_args.kwargs["task_name"] == "grammar"
    captured = capsys.readouterr()
    assert "LLM job: 24" in captured.out


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
        llm_action="run",
        task_name="grammar",
        model=None,
        no_auto_commit=True,
    )

    with (
        patch("ankiops.cli.require_initialized_collection_dir", return_value=tmp_path),
        patch(
            "ankiops.cli.run_task",
            side_effect=ValueError("Provider authentication failed: bad key"),
        ),
        caplog.at_level(logging.ERROR),
    ):
        with pytest.raises(SystemExit) as exc:
            run_llm(args)

    assert exc.value.code == 1
    assert "Provider authentication failed: bad key" in caplog.text


def test_run_llm_exits_non_zero_when_task_has_note_errors(tmp_path, caplog):
    args = SimpleNamespace(
        llm_action="run",
        task_name="grammar",
        model=None,
        no_auto_commit=True,
    )
    failed_result = LlmJobResult(
        job_id=42,
        status="failed",
        summary=TaskRunSummary(
            task_name="grammar",
            model=SONNET,
            errors=2,
            updated=1,
        ),
        failed=True,
        persisted=False,
    )

    with (
        patch("ankiops.cli.require_initialized_collection_dir", return_value=tmp_path),
        patch("ankiops.cli.run_task", return_value=failed_result),
        caplog.at_level(logging.ERROR),
    ):
        with pytest.raises(SystemExit) as exc:
            run_llm(args)

    assert exc.value.code == 1
    assert "LLM task finished with 2 error(s)" in caplog.text


def test_cli_llm_jobs_lists_recent_runs(tmp_path, capsys):
    jobs = [
        LlmJobListItem(
            job_id=24,
            task_name="grammar",
            model_name="sonnet",
            status=LlmJobStatus.COMPLETED,
            persisted=True,
            created_at="2026-03-21T10:00:00Z",
            finished_at="2026-03-21T10:00:04Z",
        )
    ]
    with (
        patch("ankiops.cli.require_initialized_collection_dir", return_value=tmp_path),
        patch("ankiops.cli.list_llm_jobs", return_value=jobs),
        patch("sys.argv", ["ankiops", "llm", "jobs"]),
    ):
        main()

    captured = capsys.readouterr()
    assert "24" in captured.out
    assert "task=grammar" in captured.out


def test_run_llm_show_exits_when_job_not_found(tmp_path):
    args = SimpleNamespace(
        llm_action="show",
        job_id="999999",
    )
    with (
        patch("ankiops.cli.require_initialized_collection_dir", return_value=tmp_path),
        patch("ankiops.cli.show_job", return_value=None),
    ):
        with pytest.raises(SystemExit) as exc:
            run_llm(args)

    assert exc.value.code == 1


def test_run_llm_show_exits_on_disallowed_full_uuid(tmp_path, caplog):
    args = SimpleNamespace(
        llm_action="show",
        job_id="24c99f81-a7dd-48b5-a29c-cc14beff9ddc",
    )
    with (
        patch("ankiops.cli.require_initialized_collection_dir", return_value=tmp_path),
        patch(
            "ankiops.cli.show_job",
            side_effect=ValueError(
                "Job ID must be numeric, or use 'latest'."
            ),
        ),
        caplog.at_level(logging.ERROR),
    ):
        with pytest.raises(SystemExit) as exc:
            run_llm(args)

    assert exc.value.code == 1
    assert "Job ID must be numeric" in caplog.text
