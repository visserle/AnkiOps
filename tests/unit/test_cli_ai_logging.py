"""Unit tests for AI CLI logging output and throttling behavior."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ankiops.ai.types import TaskProgressUpdate
from ankiops.cli_ai import (
    _log_performance_summary,
    _TaskProgressLogger,
    run_ai_task,
)


def _task_args(*, include_deck: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        task="grammar",
        temperature=None,
        no_auto_commit=True,
        progress="off",
        include_deck=include_deck or [],
        batch_size=None,
        profile=None,
        provider=None,
        model=None,
        base_url=None,
        api_key_env=None,
        timeout=None,
        max_in_flight=None,
        api_key=None,
    )


def _task_config() -> SimpleNamespace:
    return SimpleNamespace(
        id="grammar",
        batch="batch",
        batch_size=8,
        temperature=0.0,
    )


def _runtime_config() -> SimpleNamespace:
    return SimpleNamespace(
        requires_api_key=False,
        api_key=None,
        profile="ollama-fast",
        provider="ollama",
        model="gemma3:4b",
        max_in_flight=4,
    )


def _task_result(
    *,
    processed_decks: int = 1,
    processed_notes: int = 5,
    matched_notes: int = 5,
    changed_fields: int = 0,
    warnings: list[str] | None = None,
    dropped_warnings: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        processed_decks=processed_decks,
        processed_notes=processed_notes,
        matched_notes=matched_notes,
        changed_fields=changed_fields,
        changes=[],
        changed_decks=[],
        warnings=warnings or [],
        dropped_warnings=dropped_warnings,
    )


def _mock_client() -> SimpleNamespace:
    metrics = SimpleNamespace(
        logical_requests=1,
        http_attempts=1,
        retries=0,
        total_attempt_seconds=0.5,
    )
    return SimpleNamespace(metrics=lambda: metrics)


def _run_ai_task_with_result(
    *,
    tmp_path,
    result: SimpleNamespace,
    include_deck: list[str] | None = None,
) -> None:
    args = _task_args(include_deck=include_deck)
    runner = MagicMock()
    runner.run.return_value = result

    with (
        patch(
            "ankiops.cli_ai.require_initialized_collection_dir",
            return_value=tmp_path,
        ),
        patch(
            "ankiops.cli_ai.prepare_ai_run",
            return_value=(_task_config(), _runtime_config()),
        ),
        patch(
            "ankiops.cli_ai.serialize_collection",
            return_value={"collection": {}, "decks": []},
        ),
        patch("ankiops.cli_ai.build_async_editor", return_value=_mock_client()),
        patch("ankiops.cli_ai.TaskRunner", return_value=runner),
        patch("ankiops.cli_ai.deserialize_collection_data"),
        patch("ankiops.cli_ai.time.perf_counter", side_effect=[10.0, 12.0]),
    ):
        run_ai_task(args)


def test_run_ai_task_logs_start_summary(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        _run_ai_task_with_result(
            tmp_path=tmp_path,
            result=_task_result(),
            include_deck=["Biology", "Biology::Cells"],
        )

    assert (
        "AI: running task 'grammar' with profile 'ollama-fast' "
        "(ollama/gemma3:4b)"
    ) in caplog.text
    assert (
        "AI: batch 'batch' (size 8), max in flight 4, temperature 0.0"
    ) in caplog.text
    assert "AI: scope Biology, Biology::Cells" in caplog.text


def test_run_ai_task_logs_no_change_message(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        _run_ai_task_with_result(
            tmp_path=tmp_path,
            result=_task_result(changed_fields=0),
        )

    assert "AI changes (0 field(s)):" in caplog.text
    assert "No changes to write." in caplog.text


def test_run_ai_task_logs_warning_overflow_summary(tmp_path, caplog):
    warnings = [f"warning {index}" for index in range(21)]
    with caplog.at_level(logging.INFO):
        _run_ai_task_with_result(
            tmp_path=tmp_path,
            result=_task_result(
                changed_fields=0,
                warnings=warnings,
                dropped_warnings=3,
            ),
        )

    assert "AI warnings (24):" in caplog.text
    assert "warning 0" in caplog.text
    assert "warning 19" in caplog.text
    assert "warning 20" not in caplog.text
    assert "... and 4 more warning(s)" in caplog.text


def test_progress_logger_throttles_and_emits_final_update(caplog):
    progress_logger = _TaskProgressLogger(interval_seconds=5.0)

    def _update(
        *,
        elapsed_seconds: float,
        completed_chunks: int,
        warning_count: int,
    ) -> TaskProgressUpdate:
        return TaskProgressUpdate(
            elapsed_seconds=elapsed_seconds,
            queued_chunks=4,
            completed_chunks=completed_chunks,
            in_flight=max(0, 4 - completed_chunks),
            processed_decks=1,
            processed_notes=completed_chunks,
            matched_notes=completed_chunks,
            changed_fields=completed_chunks,
            warning_count=warning_count,
        )

    with caplog.at_level(logging.INFO):
        progress_logger(
            _update(elapsed_seconds=1.0, completed_chunks=1, warning_count=0)
        )
        progress_logger(
            _update(elapsed_seconds=2.0, completed_chunks=2, warning_count=1)
        )
        progress_logger(
            _update(elapsed_seconds=3.0, completed_chunks=3, warning_count=1)
        )
        progress_logger(
            _update(elapsed_seconds=4.0, completed_chunks=4, warning_count=1)
        )

    progress_lines = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("AI progress:")
    ]
    assert len(progress_lines) == 2
    assert "2/4 chunk(s)" in progress_lines[0]
    assert "in flight 2, elapsed 2.0s" in progress_lines[0]
    assert "4/4 chunk(s)" in progress_lines[1]
    assert "in flight 0, elapsed 4.0s" in progress_lines[1]


def test_log_performance_summary_uses_human_readable_output(caplog):
    result = _task_result(
        processed_decks=2,
        processed_notes=10,
        matched_notes=8,
        changed_fields=3,
        warnings=["first warning"],
        dropped_warnings=2,
    )
    metrics = SimpleNamespace(
        logical_requests=4,
        http_attempts=5,
        retries=1,
        total_attempt_seconds=1.25,
    )
    client = SimpleNamespace(metrics=lambda: metrics)

    with (
        caplog.at_level(logging.INFO),
        patch("ankiops.cli_ai.time.perf_counter", return_value=13.0),
    ):
        _log_performance_summary(10.0, result, client)

    assert (
        "AI: completed in 3.00s — 10 scanned, 8 matched, 3 field(s) changed, "
        "3 warning(s), 2 deck(s)"
    ) in caplog.text
    assert (
        "AI transport: 4 logical request(s), 5 HTTP attempt(s), 1 retry, "
        "250.0ms avg attempt latency"
    ) in caplog.text


def test_run_ai_task_normalizes_multiline_warning_and_adds_short_hint(
    tmp_path,
    caplog,
):
    warning = (
        "Deck A/n1: response returned 1 unexpected field key(s): 'Bad Key';\n"
        "expected write_fields: 'Question'"
    )
    with caplog.at_level(logging.INFO):
        _run_ai_task_with_result(
            tmp_path=tmp_path,
            result=_task_result(changed_fields=0, warnings=[warning]),
        )

    warning_lines = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    rendered_warning = next(
        line for line in warning_lines if line.startswith("Deck A/n1:")
    )
    assert "\n" not in rendered_warning
    assert "  " not in rendered_warning
    assert "hint: check task write_fields and returned patch keys." in rendered_warning


def test_run_ai_task_truncates_long_warning_line_to_cli_budget(tmp_path, caplog):
    warning = f"Deck A/n1: {'x' * 240}"
    with caplog.at_level(logging.INFO):
        _run_ai_task_with_result(
            tmp_path=tmp_path,
            result=_task_result(changed_fields=0, warnings=[warning]),
        )

    warning_lines = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    rendered_warning = next(
        line for line in warning_lines if line.startswith("Deck A/n1:")
    )
    assert len(rendered_warning) <= 180
    assert rendered_warning.endswith("...")
