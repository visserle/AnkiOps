from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ankiops.cli import main, run_llm
from ankiops.llm.llm_db import LlmJobListItem
from ankiops.llm.llm_models import (
    ExecutionMode,
    LlmJobResult,
    LlmJobStatus,
    PlanFieldSurface,
    TaskCatalog,
    TaskConfig,
    TaskPlanResult,
    TaskRunSummary,
)
from ankiops.llm.model_registry import ProviderModel

TEST_MODEL = ProviderModel(
    name="claude-sonnet-4-6",
    api_id="claude-sonnet-4-6",
    provider="anthropic",
    base_url="https://api.anthropic.com/v1/",
    api_key_env="ANTHROPIC_API_KEY",
    input_usd_per_mtok=3,
    output_usd_per_mtok=15,
)


def _plan_result() -> TaskPlanResult:
    return TaskPlanResult(
        task_name="grammar",
        model=TEST_MODEL,
        deck_scope="*",
        serializer_scope="collection",
        system_prompt_path="/tmp/llm/system_prompt.md",
        prompt_path="/tmp/llm/prompts/grammar.md",
        system_prompt="System prompt",
        task_prompt="Task prompt",
        request_defaults=(
            "timeout=60s max_tokens=2048 temperature=default retries=2 "
            "retry_backoff=0.5s retry_jitter=true mode=online "
            "concurrency=8"
        ),
        summary=TaskRunSummary(
            task_name="grammar",
            model=TEST_MODEL,
            decks_seen=1,
            decks_matched=1,
            notes_seen=2,
            eligible=2,
            skipped_deck_scope=0,
            skipped_no_editable_fields=0,
            errors=0,
            requests=2,
        ),
        field_surface=[
            PlanFieldSurface(
                note_type="AnkiOpsQA",
                candidate_notes=1,
                editable_fields=["Question", "Answer", "Source"],
                read_only_fields=[],
                hidden_fields=["AI Notes"],
            )
        ],
        requests_estimate=2,
        input_tokens_estimate=80,
        output_tokens_cap=4096,
    )


def test_cli_llm_dispatches_run():
    success_result = LlmJobResult(
        job_id=24,
        status="completed",
        summary=TaskRunSummary(task_name="grammar", model=TEST_MODEL),
        failed=False,
        persisted=False,
    )
    with (
        patch("ankiops.cli.require_initialized_collection_dir"),
        patch("ankiops.cli.run_task", return_value=success_result) as run_task,
        patch("sys.argv", ["ankiops", "llm", "grammar", "--run"]),
    ):
        main()

    run_task.assert_called_once()
    assert run_task.call_args.kwargs["task_name"] == "grammar"
    assert run_task.call_args.kwargs["deck_override"] is None


def test_cli_llm_dispatches_run_with_deck_override():
    success_result = LlmJobResult(
        job_id=24,
        status="completed",
        summary=TaskRunSummary(task_name="grammar", model=TEST_MODEL),
        failed=False,
        persisted=False,
    )
    with (
        patch("ankiops.cli.require_initialized_collection_dir"),
        patch("ankiops.cli.run_task", return_value=success_result) as run_task,
        patch("sys.argv", ["ankiops", "llm", "grammar", "--run", "--deck", "Target"]),
    ):
        main()

    run_task.assert_called_once()
    assert run_task.call_args.kwargs["deck_override"] == "Target"


def test_cli_llm_dispatches_plan():
    with (
        patch("ankiops.cli.require_initialized_collection_dir"),
        patch("ankiops.cli.plan_task", return_value=_plan_result()) as plan_task,
        patch("sys.argv", ["ankiops", "llm", "grammar"]),
    ):
        main()

    plan_task.assert_called_once()
    assert plan_task.call_args.kwargs["task_name"] == "grammar"
    assert plan_task.call_args.kwargs["deck_override"] is None


def test_cli_llm_dispatches_plan_with_online_override():
    with (
        patch("ankiops.cli.require_initialized_collection_dir"),
        patch("ankiops.cli.plan_task", return_value=_plan_result()) as plan_task,
        patch("sys.argv", ["ankiops", "llm", "grammar", "--online"]),
    ):
        main()

    plan_task.assert_called_once()
    assert plan_task.call_args.kwargs["mode_override"] == "online"


def test_cli_llm_dispatches_plan_with_deck_override():
    with (
        patch("ankiops.cli.require_initialized_collection_dir"),
        patch("ankiops.cli.plan_task", return_value=_plan_result()) as plan_task,
        patch("sys.argv", ["ankiops", "llm", "grammar", "--deck", "Target"]),
    ):
        main()

    plan_task.assert_called_once()
    assert plan_task.call_args.kwargs["deck_override"] == "Target"


def test_run_llm_plan_logs_system_prompt_path_and_full_prompt(tmp_path, caplog):
    args = SimpleNamespace(
        task_name="grammar",
        run=False,
        job_id=None,
        model=None,
        deck=None,
        no_auto_commit=False,
    )

    with (
        patch("ankiops.cli.require_initialized_collection_dir", return_value=tmp_path),
        patch("ankiops.cli.plan_task", return_value=_plan_result()),
        caplog.at_level(logging.INFO),
    ):
        run_llm(args)

    assert "System prompt file: /tmp/llm/system_prompt.md" in caplog.text
    assert "Task prompt file: /tmp/llm/prompts/grammar.md" in caplog.text
    assert "<system>\nSystem prompt\n</system>" in caplog.text
    assert "<task>\nTask prompt\n</task>" in caplog.text


def test_cli_llm_status_exits_on_invalid_config(tmp_path):
    with (
        patch("ankiops.cli.require_initialized_collection_dir", return_value=tmp_path),
        patch("ankiops.cli.FileSystemAdapter.load_note_type_configs", return_value=[]),
        patch(
            "ankiops.cli.load_llm_task_catalog",
            return_value=TaskCatalog({}, {"/tmp/grammar.yaml": "bad config"}),
        ),
        patch("ankiops.cli.list_llm_jobs", return_value=[]),
        patch("sys.argv", ["ankiops", "llm"]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1


def test_cli_llm_status_lists_tasks_and_recent_jobs(tmp_path):
    jobs = [
        LlmJobListItem(
            job_id=24,
            task_name="grammar",
            model_name="claude-sonnet-4-6",
            execution_mode=ExecutionMode.ONLINE,
            status=LlmJobStatus.COMPLETED,
            persisted=True,
            created_at="2026-03-21T10:00:00Z",
            finished_at="2026-03-21T10:00:04Z",
            resume_from_job_id=None,
        )
    ]
    catalog = TaskCatalog(
        {
            "grammar": TaskConfig(
                name="grammar",
                model=TEST_MODEL,
                system_prompt="system",
                prompt="prompt",
            )
        },
        {},
    )
    with (
        patch("ankiops.cli.require_initialized_collection_dir", return_value=tmp_path),
        patch("ankiops.cli.FileSystemAdapter.load_note_type_configs", return_value=[]),
        patch("ankiops.cli.load_llm_task_catalog", return_value=catalog),
        patch("ankiops.cli.list_llm_jobs", return_value=jobs) as list_jobs,
        patch("sys.argv", ["ankiops", "llm"]),
    ):
        main()

    list_jobs.assert_called_once_with(collection_dir=tmp_path)


def test_cli_llm_show_parses_latest_alias(tmp_path):
    with (
        patch("ankiops.cli.require_initialized_collection_dir", return_value=tmp_path),
        patch("ankiops.cli.show_job", return_value=None) as show_job,
        patch("sys.argv", ["ankiops", "llm", "--job", "latest"]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1
    show_job.assert_called_once()
    assert show_job.call_args.kwargs["job_id"] == "latest"


def test_cli_llm_dispatches_resume_with_job_selector(tmp_path):
    success_result = LlmJobResult(
        job_id=25,
        status="completed",
        summary=TaskRunSummary(task_name="grammar", model=TEST_MODEL),
        failed=False,
        persisted=True,
    )
    with (
        patch("ankiops.cli.require_initialized_collection_dir", return_value=tmp_path),
        patch("ankiops.cli.resume_task", return_value=success_result) as resume_task,
        patch("sys.argv", ["ankiops", "llm", "--job", "latest", "--resume"]),
    ):
        main()

    resume_task.assert_called_once()
    assert resume_task.call_args.kwargs["resume_job_id"] == "latest"


def test_run_llm_run_exits_cleanly_on_fatal_provider_error(tmp_path, caplog):
    args = SimpleNamespace(
        task_name="grammar",
        run=True,
        job_id=None,
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


def test_run_llm_show_exits_when_job_not_found(tmp_path):
    args = SimpleNamespace(
        task_name=None,
        run=False,
        job_id="999999",
        model=None,
        no_auto_commit=False,
    )
    with (
        patch("ankiops.cli.require_initialized_collection_dir", return_value=tmp_path),
        patch("ankiops.cli.show_job", return_value=None),
    ):
        with pytest.raises(SystemExit) as exc:
            run_llm(args)

    assert exc.value.code == 1


def test_run_llm_show_accepts_latest_alias(tmp_path):
    args = SimpleNamespace(
        task_name=None,
        run=False,
        job_id="latest",
        model=None,
        no_auto_commit=False,
    )
    with (
        patch("ankiops.cli.require_initialized_collection_dir", return_value=tmp_path),
        patch("ankiops.cli.show_job", return_value=None) as show_job,
    ):
        with pytest.raises(SystemExit) as exc:
            run_llm(args)

    assert exc.value.code == 1
    show_job.assert_called_once()
    assert show_job.call_args.kwargs["job_id"] == "latest"


def test_run_llm_run_logs_compact_job_summary(tmp_path, caplog):
    args = SimpleNamespace(
        task_name="grammar",
        run=True,
        job_id=None,
        model=None,
        deck=None,
        no_auto_commit=True,
    )
    success_result = LlmJobResult(
        job_id=24,
        status="completed",
        summary=TaskRunSummary(task_name="grammar", model=TEST_MODEL),
        failed=False,
        persisted=False,
    )

    with (
        patch("ankiops.cli.require_initialized_collection_dir", return_value=tmp_path),
        patch("ankiops.cli.run_task", return_value=success_result),
        caplog.at_level(logging.INFO),
    ):
        run_llm(args)

    assert (
        "LLM job #24 completed (no markdown changes persisted). "
        "Cost: $0.00. Inspect: ankiops llm --job 24"
    ) in caplog.text


@pytest.mark.parametrize(
    "args",
    [
        SimpleNamespace(
            task_name=None,
            run=True,
            job_id=None,
            model=None,
            deck=None,
            no_auto_commit=False,
        ),
        SimpleNamespace(
            task_name=None,
            run=False,
            job_id=None,
            model="claude-haiku-4-5",
            deck=None,
            no_auto_commit=False,
        ),
        SimpleNamespace(
            task_name=None,
            run=False,
            job_id=None,
            model=None,
            deck=None,
            no_auto_commit=True,
        ),
        SimpleNamespace(
            task_name=None,
            run=False,
            job_id=None,
            model=None,
            deck=None,
            no_auto_commit=False,
            resume=True,
        ),
        SimpleNamespace(
            task_name="grammar",
            run=False,
            job_id="latest",
            model=None,
            deck=None,
            no_auto_commit=False,
        ),
        SimpleNamespace(
            task_name=None,
            run=True,
            job_id="latest",
            model=None,
            deck=None,
            no_auto_commit=False,
        ),
        SimpleNamespace(
            task_name=None,
            run=False,
            job_id="latest",
            model=None,
            deck=None,
            no_auto_commit=True,
            resume=False,
        ),
        SimpleNamespace(
            task_name="grammar",
            run=False,
            job_id=None,
            model=None,
            deck=None,
            no_auto_commit=True,
        ),
        SimpleNamespace(
            task_name=None,
            run=False,
            job_id=None,
            model=None,
            deck="Target",
            no_auto_commit=False,
        ),
        SimpleNamespace(
            task_name=None,
            run=False,
            job_id="latest",
            model=None,
            deck="Target",
            no_auto_commit=False,
        ),
        SimpleNamespace(
            task_name="grammar",
            run=False,
            job_id=None,
            model=None,
            deck="Target*",
            no_auto_commit=False,
        ),
    ],
)
def test_run_llm_usage_errors_exit_code_two(tmp_path, args):
    with patch("ankiops.cli.require_initialized_collection_dir", return_value=tmp_path):
        with pytest.raises(SystemExit) as exc:
            run_llm(args)

    assert exc.value.code == 2
