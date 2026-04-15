from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import replace
from importlib import resources
from pathlib import Path
from textwrap import dedent

import pytest

from ankiops.collection_serializer import serialize_collection
from ankiops.db import SQLiteDbAdapter
from ankiops.fs import FileSystemAdapter
from ankiops.llm.llm_db import LlmDb
from ankiops.llm.runner import LlmTaskExecutor, _materialize_task_context
from ankiops.llm.task_types import LlmItemStatus

LIVE_DECK_NAME = "LiveGrammarDeck"
STANDARD_TASK_NAME = "grammar"
GROQ_MODEL_NAME = "groq-oss-120b"
GROQ_MODEL_ID = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, 1)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(value, 0.0)


def _max_allowed_errors(
    *, total_notes: int, env_name: str, default_ratio: float
) -> int:
    ratio = min(_env_float(env_name, default_ratio), 1.0)
    return int(total_notes * ratio)


def _note_markdown(
    *,
    note_key: str,
    question: str,
    answer: str,
    source: str,
    ai_notes: str,
) -> str:
    return dedent(
        f"""
        <!-- note_key: {note_key} -->
        Q: {question}
        A: {answer}
        S: {source}
        AI: {ai_notes}
        """
    ).strip()


def _join_notes(notes: list[str]) -> str:
    return "\n\n---\n\n".join(notes)


def _build_mixed_deck_markdown(total_notes: int) -> str:
    notes: list[str] = []
    for index in range(total_notes):
        note_key = f"live-mix-{index:03d}"
        if index % 3 == 0:
            question = f"teh workflow dont recieve updates for item {index}."
        elif index % 3 == 1:
            question = f"This sentence is already correct for item {index}."
        else:
            question = f"teh process dont recieve support for item {index}."
        notes.append(
            _note_markdown(
                note_key=note_key,
                question=question,
                answer=f"Keep unchanged answer {index}.",
                source=f"source-{index}",
                ai_notes=f"internal-notes-{index}",
            )
        )
    return _join_notes(notes)


def _build_pressure_deck_markdown(total_notes: int) -> str:
    notes: list[str] = []
    for index in range(total_notes):
        notes.append(
            _note_markdown(
                note_key=f"live-pressure-{index:03d}",
                question=(
                    f"teh pressure test sentence dont recieve fixes for card {index}."
                ),
                answer=f"Keep unchanged pressure answer {index}.",
                source=f"pressure-source-{index}",
                ai_notes=f"pressure-internal-notes-{index}",
            )
        )
    return _join_notes(notes)


def _eject_groq_llm_files(collection_dir: Path) -> None:
    llm_dir = collection_dir / "llm"
    _write(
        llm_dir / "_models.yaml",
        f"""
        - model: {GROQ_MODEL_NAME}
          model_id: {GROQ_MODEL_ID}
          provider: groq
          base_url: {GROQ_BASE_URL}
          api_key: $GROQ_API_KEY
        """,
    )
    _write(
        llm_dir / "_system_prompt.md",
        resources.files("ankiops.llm")
        .joinpath("_system_prompt.md")
        .read_text(encoding="utf-8"),
    )

    grammar_task = (
        resources.files("ankiops.llm")
        .joinpath("grammar.yaml")
        .read_text(encoding="utf-8")
    )
    grammar_lines = grammar_task.splitlines()
    if grammar_lines and grammar_lines[0].startswith("model:"):
        grammar_lines[0] = f"model: {GROQ_MODEL_NAME}"
    _write(
        llm_dir / "grammar.yaml",
        "\n".join(grammar_lines),
    )


def _bootstrap_collection(collection_dir: Path, *, deck_markdown: str) -> None:
    db = SQLiteDbAdapter.open(collection_dir)
    db.close()

    note_types_dir = collection_dir / "note_types"
    FileSystemAdapter().eject_builtin_note_types(note_types_dir)
    _eject_groq_llm_files(collection_dir)
    _write(collection_dir / f"{LIVE_DECK_NAME}.md", deck_markdown)


def _open_job_detail(collection_dir: Path, job_id: int):
    db = LlmDb.open(collection_dir)
    try:
        detail = db.get_job_detail(job_id)
        return detail, job_id, db
    except Exception:
        db.close()
        raise


def _run_live_grammar_task(collection_dir: Path):
    live_concurrency = _env_int("ANKIOPS_LIVE_LLM_CONCURRENCY", 1)
    live_max_output_tokens = _env_int("ANKIOPS_LIVE_LLM_MAX_OUTPUT_TOKENS", 512)

    context = _materialize_task_context(
        collection_dir=collection_dir,
        task_name=STANDARD_TASK_NAME,
        model_override=None,
        deck_override=None,
        serialize_collection_fn=serialize_collection,
    )
    tuned_request = replace(
        context.task.request,
        max_output_tokens=live_max_output_tokens,
    )
    tuned_model = replace(
        context.task.model,
        concurrency=live_concurrency,
    )
    tuned_task = replace(
        context.task,
        model=tuned_model,
        request=tuned_request,
    )
    executor = LlmTaskExecutor(
        collection_dir=collection_dir,
        materialized_context=replace(context, task=tuned_task),
        no_auto_commit=True,
    )
    return asyncio.run(executor.execute())


def _collect_attempt_metrics(db: LlmDb, *, job_id: int) -> dict[str, int]:
    row = db._conn.execute(
        """
        SELECT
            COUNT(a.id) AS attempts_count,
            COALESCE(SUM(CASE WHEN a.retry_count > 0 THEN 1 ELSE 0 END), 0)
                AS retried_attempts,
            COALESCE(SUM(a.retry_count), 0) AS retry_events,
            COALESCE(MAX(a.retry_count), 0) AS max_retry_count,
            COALESCE(
                SUM(
                    CASE
                        WHEN a.rate_limit_headers_json IS NOT NULL
                            AND a.rate_limit_headers_json != '{}'
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS attempts_with_non_empty_rate_headers,
            COALESCE(
                SUM(
                    CASE
                        WHEN p.response_raw_text IS NOT NULL
                            AND TRIM(p.response_raw_text) != ''
                        THEN 1
                        ELSE 0
                    END
                ),
                0
            ) AS attempts_with_raw_text
        FROM llm_job_item i
        LEFT JOIN llm_item_attempt a ON a.job_item_id = i.id
        LEFT JOIN llm_attempt_payload p ON p.attempt_id = a.id
        WHERE i.job_id = ?
        """,
        (job_id,),
    ).fetchone()
    assert row is not None
    return {
        "attempts_count": int(row["attempts_count"]),
        "retried_attempts": int(row["retried_attempts"]),
        "retry_events": int(row["retry_events"]),
        "max_retry_count": int(row["max_retry_count"]),
        "attempts_with_non_empty_rate_headers": int(
            row["attempts_with_non_empty_rate_headers"]
        ),
        "attempts_with_raw_text": int(row["attempts_with_raw_text"]),
    }


def _emit_metrics(
    *,
    collection_dir: Path,
    scenario: str,
    elapsed_seconds: float,
    result,
    attempt_metrics: dict[str, int],
) -> Path:
    requests = result.summary.requests
    requests_per_second = requests / elapsed_seconds if elapsed_seconds > 0 else 0.0
    provider_share = (
        result.summary.provider_latency_ms_total / (elapsed_seconds * 1000)
        if elapsed_seconds > 0
        else 0.0
    )
    metrics = {
        "scenario": scenario,
        "job_id": result.job_id,
        "status": result.status,
        "failed": result.failed,
        "persisted": result.persisted,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "requests_per_second": round(requests_per_second, 3),
        "provider_latency_share": round(provider_share, 3),
        "summary": {
            "eligible": result.summary.eligible,
            "updated": result.summary.updated,
            "unchanged": result.summary.unchanged,
            "errors": result.summary.errors,
            "canceled": result.summary.canceled,
            "requests": result.summary.requests,
            "input_tokens": result.summary.input_tokens,
            "output_tokens": result.summary.output_tokens,
            "provider_latency_ms_total": result.summary.provider_latency_ms_total,
            "provider_retries": result.summary.provider_retries,
        },
        "attempt_metrics": attempt_metrics,
    }
    metrics_path = collection_dir / f"llm-live-metrics-{scenario}.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "[live_llm] scenario="
        f"{scenario} requests={result.summary.requests} "
        f"elapsed={elapsed_seconds:.2f}s rps={requests_per_second:.2f} "
        f"retries={result.summary.provider_retries}"
    )
    return metrics_path


@pytest.fixture(scope="module")
def require_live_llm() -> None:
    if os.getenv("ANKIOPS_RUN_LIVE_LLM_TESTS") != "1":
        pytest.skip(
            "Set ANKIOPS_RUN_LIVE_LLM_TESTS=1 to run live LLM integration tests."
        )
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY is required for live LLM integration tests.")


@pytest.fixture(scope="module")
def require_live_llm_stress() -> None:
    if os.getenv("ANKIOPS_RUN_LIVE_LLM_STRESS") != "1":
        pytest.skip("Set ANKIOPS_RUN_LIVE_LLM_STRESS=1 to run stress live LLM tests.")


@pytest.mark.live_llm
@pytest.mark.usefixtures("require_live_llm")
def test_live_grammar_single_note_smoke(tmp_path: Path) -> None:
    source_markdown = """
    <!-- note_key: live-nk-1 -->
    Q: teh cat dont recieve treats.
    A: Keep this answer unchanged.
    S: style guide
    AI: internal notes
    """
    _bootstrap_collection(tmp_path, deck_markdown=source_markdown)

    started = time.monotonic()
    result = _run_live_grammar_task(tmp_path)
    elapsed = time.monotonic() - started

    assert result.status == "completed"
    assert not result.failed
    assert result.persisted
    assert result.summary.eligible == 1
    assert result.summary.updated == 1
    assert result.summary.unchanged == 0
    assert result.summary.errors == 0
    assert result.summary.requests == 1

    content = (tmp_path / f"{LIVE_DECK_NAME}.md").read_text(encoding="utf-8")
    lower_content = content.lower()
    assert "<!-- note_key: live-nk-1 -->" in content
    assert "teh" not in lower_content
    assert "dont" not in lower_content
    assert "recieve" not in lower_content
    assert "A: Keep this answer unchanged." in content
    assert "AI: internal notes" in content

    detail, job_id, db = _open_job_detail(tmp_path, result.job_id)
    try:
        assert detail is not None
        assert detail.summary.updated == 1
        assert detail.summary.errors == 0
        assert detail.summary.requests == 1
        assert len(detail.items) == 1
        assert detail.items[0].item_status is LlmItemStatus.SUCCEEDED_UPDATED

        attempts_count = db._conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM llm_item_attempt a
            JOIN llm_job_item i ON i.id = a.job_item_id
            WHERE i.job_id = ?
            """,
            (job_id,),
        ).fetchone()
        payload_rows = db._conn.execute(
            """
            SELECT p.response_raw_text
            FROM llm_attempt_payload p
            JOIN llm_item_attempt a ON a.id = p.attempt_id
            JOIN llm_job_item i ON i.id = a.job_item_id
            WHERE i.job_id = ?
            ORDER BY a.id ASC
            """,
            (job_id,),
        ).fetchall()
        attempt_metrics = _collect_attempt_metrics(db, job_id=job_id)
    finally:
        db.close()

    assert attempts_count is not None
    assert int(attempts_count["total"]) == 1
    assert len(payload_rows) == 1
    assert isinstance(payload_rows[0]["response_raw_text"], str)
    assert payload_rows[0]["response_raw_text"].strip()

    metrics_path = _emit_metrics(
        collection_dir=tmp_path,
        scenario="smoke",
        elapsed_seconds=elapsed,
        result=result,
        attempt_metrics=attempt_metrics,
    )
    assert metrics_path.exists()


@pytest.mark.live_llm
@pytest.mark.usefixtures("require_live_llm")
def test_live_grammar_mixed_correctness_robustness_and_telemetry(
    tmp_path: Path,
) -> None:
    mixed_notes = _env_int("ANKIOPS_LIVE_LLM_MIXED_NOTES", 12)
    source_markdown = _build_mixed_deck_markdown(mixed_notes)
    _bootstrap_collection(tmp_path, deck_markdown=source_markdown)

    started = time.monotonic()
    result = _run_live_grammar_task(tmp_path)
    elapsed = time.monotonic() - started

    assert result.status == "completed"
    assert not result.failed
    assert result.persisted
    assert result.summary.eligible == mixed_notes
    assert result.summary.errors == 0
    assert result.summary.requests == mixed_notes
    assert result.summary.updated >= mixed_notes // 3
    assert result.summary.unchanged >= 1

    content = (tmp_path / f"{LIVE_DECK_NAME}.md").read_text(encoding="utf-8")
    for index in range(min(6, mixed_notes)):
        assert f"A: Keep unchanged answer {index}." in content
        assert f"AI: internal-notes-{index}" in content

    detail, job_id, db = _open_job_detail(tmp_path, result.job_id)
    try:
        assert detail is not None
        assert len(detail.items) == mixed_notes
        assert all(
            item.item_status
            in {
                LlmItemStatus.SUCCEEDED_UPDATED,
                LlmItemStatus.SUCCEEDED_UNCHANGED,
            }
            for item in detail.items
        )

        totals_row = db._conn.execute(
            """
            SELECT
                COUNT(a.id) AS attempts_count,
                COUNT(p.attempt_id) AS payload_count
            FROM llm_job_item i
            LEFT JOIN llm_item_attempt a ON a.job_item_id = i.id
            LEFT JOIN llm_attempt_payload p ON p.attempt_id = a.id
            WHERE i.job_id = ?
            """,
            (job_id,),
        ).fetchone()
        attempt_metrics = _collect_attempt_metrics(db, job_id=job_id)
    finally:
        db.close()

    assert totals_row is not None
    assert int(totals_row["attempts_count"]) == mixed_notes
    assert int(totals_row["payload_count"]) == mixed_notes
    assert attempt_metrics["attempts_with_raw_text"] == mixed_notes
    assert attempt_metrics["max_retry_count"] <= 2

    metrics_path = _emit_metrics(
        collection_dir=tmp_path,
        scenario="mixed",
        elapsed_seconds=elapsed,
        result=result,
        attempt_metrics=attempt_metrics,
    )
    assert metrics_path.exists()

    max_seconds = _env_float("ANKIOPS_LIVE_LLM_MIXED_MAX_SECONDS", 300.0)
    assert elapsed <= max_seconds


@pytest.mark.live_llm
@pytest.mark.usefixtures("require_live_llm")
def test_live_grammar_pressure_robustness_and_speed(tmp_path: Path) -> None:
    pressure_notes = _env_int("ANKIOPS_LIVE_LLM_PRESSURE_NOTES", 18)
    source_markdown = _build_pressure_deck_markdown(pressure_notes)
    _bootstrap_collection(tmp_path, deck_markdown=source_markdown)

    started = time.monotonic()
    result = _run_live_grammar_task(tmp_path)
    elapsed = time.monotonic() - started

    assert result.persisted
    assert result.summary.eligible == pressure_notes
    assert result.summary.requests == pressure_notes
    max_errors = _max_allowed_errors(
        total_notes=pressure_notes,
        env_name="ANKIOPS_LIVE_LLM_PRESSURE_MAX_ERROR_RATIO",
        default_ratio=0.25,
    )
    assert result.summary.errors <= max_errors
    assert result.summary.canceled == 0
    assert result.summary.updated >= max(pressure_notes // 3, 1)

    detail, job_id, db = _open_job_detail(tmp_path, result.job_id)
    try:
        assert detail is not None
        assert len(detail.items) == pressure_notes
        assert all(
            item.item_status
            in {
                LlmItemStatus.SUCCEEDED_UPDATED,
                LlmItemStatus.SUCCEEDED_UNCHANGED,
                LlmItemStatus.NOTE_ERROR,
                LlmItemStatus.PROVIDER_ERROR,
            }
            for item in detail.items
        )
        assert all(
            item.item_status not in {LlmItemStatus.FATAL_ERROR, LlmItemStatus.CANCELED}
            for item in detail.items
        )
        attempt_metrics = _collect_attempt_metrics(db, job_id=job_id)
    finally:
        db.close()

    assert attempt_metrics["attempts_count"] == pressure_notes
    assert attempt_metrics["attempts_with_raw_text"] >= (
        pressure_notes - result.summary.errors
    )
    assert attempt_metrics["max_retry_count"] <= 2

    metrics_path = _emit_metrics(
        collection_dir=tmp_path,
        scenario="pressure",
        elapsed_seconds=elapsed,
        result=result,
        attempt_metrics=attempt_metrics,
    )
    assert metrics_path.exists()

    max_seconds = _env_float("ANKIOPS_LIVE_LLM_PRESSURE_MAX_SECONDS", 900.0)
    assert elapsed <= max_seconds


@pytest.mark.live_llm
@pytest.mark.usefixtures("require_live_llm", "require_live_llm_stress")
def test_live_grammar_stress_large_batch_optional(tmp_path: Path) -> None:
    stress_notes = _env_int("ANKIOPS_LIVE_LLM_STRESS_NOTES", 48)
    source_markdown = _build_pressure_deck_markdown(stress_notes)
    _bootstrap_collection(tmp_path, deck_markdown=source_markdown)

    started = time.monotonic()
    result = _run_live_grammar_task(tmp_path)
    elapsed = time.monotonic() - started

    assert result.persisted
    assert result.summary.eligible == stress_notes
    assert result.summary.requests == stress_notes
    max_errors = _max_allowed_errors(
        total_notes=stress_notes,
        env_name="ANKIOPS_LIVE_LLM_STRESS_MAX_ERROR_RATIO",
        default_ratio=0.35,
    )
    assert result.summary.errors <= max_errors
    assert result.summary.canceled == 0

    detail, job_id, db = _open_job_detail(tmp_path, result.job_id)
    try:
        assert detail is not None
        assert len(detail.items) == stress_notes
        assert all(
            item.item_status
            in {
                LlmItemStatus.SUCCEEDED_UPDATED,
                LlmItemStatus.SUCCEEDED_UNCHANGED,
                LlmItemStatus.NOTE_ERROR,
                LlmItemStatus.PROVIDER_ERROR,
            }
            for item in detail.items
        )
        assert all(
            item.item_status not in {LlmItemStatus.FATAL_ERROR, LlmItemStatus.CANCELED}
            for item in detail.items
        )
        attempt_metrics = _collect_attempt_metrics(db, job_id=job_id)
    finally:
        db.close()

    assert attempt_metrics["attempts_count"] == stress_notes
    assert attempt_metrics["attempts_with_raw_text"] >= (
        stress_notes - result.summary.errors
    )

    metrics_path = _emit_metrics(
        collection_dir=tmp_path,
        scenario="stress",
        elapsed_seconds=elapsed,
        result=result,
        attempt_metrics=attempt_metrics,
    )
    assert metrics_path.exists()

    max_seconds = _env_float("ANKIOPS_LIVE_LLM_STRESS_MAX_SECONDS", 1800.0)
    assert elapsed <= max_seconds
