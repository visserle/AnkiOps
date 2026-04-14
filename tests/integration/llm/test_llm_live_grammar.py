from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from textwrap import dedent

import pytest

from ankiops.db import SQLiteDbAdapter
from ankiops.fs import FileSystemAdapter
from ankiops.llm.llm_db import LlmDb
from ankiops.llm.runner import run_task
from ankiops.llm.task_types import LlmFinalStatus

LIVE_DECK_NAME = "LiveGrammarDeck"
STANDARD_TASK_NAME = "grammar"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def _eject_standard_llm_files(collection_dir: Path) -> None:
    llm_dir = collection_dir / "llm"
    _write(
        llm_dir / "_models.yaml",
        resources.files("ankiops.llm")
        .joinpath("_models.yaml")
        .read_text(encoding="utf-8"),
    )
    _write(
        llm_dir / "_system_prompt.md",
        resources.files("ankiops.llm")
        .joinpath("_system_prompt.md")
        .read_text(encoding="utf-8"),
    )
    _write(
        llm_dir / "grammar.yaml",
        resources.files("ankiops.llm")
        .joinpath("grammar.yaml")
        .read_text(encoding="utf-8"),
    )


def _bootstrap_collection(collection_dir: Path, *, deck_markdown: str) -> None:
    db = SQLiteDbAdapter.open(collection_dir)
    db.close()

    note_types_dir = collection_dir / "note_types"
    FileSystemAdapter().eject_builtin_note_types(note_types_dir)
    _eject_standard_llm_files(collection_dir)
    _write(collection_dir / f"{LIVE_DECK_NAME}.md", deck_markdown)


def _open_job_detail(collection_dir: Path, job_id: int):
    db = LlmDb.open(collection_dir)
    try:
        detail = db.get_job_detail(job_id)
        return detail, job_id, db
    except Exception:
        db.close()
        raise


@pytest.fixture(scope="module")
def require_live_llm() -> None:
    if os.getenv("ANKIOPS_RUN_LIVE_LLM_TESTS") != "1":
        pytest.skip(
            "Set ANKIOPS_RUN_LIVE_LLM_TESTS=1 to run live LLM integration tests."
        )
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is required for live LLM integration tests.")


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

    result = run_task(
        collection_dir=tmp_path,
        task_name=STANDARD_TASK_NAME,
        no_auto_commit=True,
    )

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

    detail, job_id, db = _open_job_detail(tmp_path, result.job_id)
    try:
        assert detail is not None
        assert detail.summary.updated == 1
        assert detail.summary.errors == 0
        assert detail.summary.requests == 1
        assert len(detail.items) == 1
        assert detail.items[0].final_status is LlmFinalStatus.SUCCEEDED_UPDATED

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
    finally:
        db.close()

    assert attempts_count is not None
    assert int(attempts_count["total"]) == 1
    assert len(payload_rows) == 1
    assert isinstance(payload_rows[0]["response_raw_text"], str)
    assert payload_rows[0]["response_raw_text"].strip()


@pytest.mark.live_llm
@pytest.mark.usefixtures("require_live_llm")
def test_live_grammar_three_note_end_to_end(tmp_path: Path) -> None:
    source_markdown = """
    <!-- note_key: live-nk-a -->
    Q: teh route dont recieve updates.
    A: Keep unchanged A.
    S: source A
    AI: notes A

    ---

    <!-- note_key: live-nk-b -->
    Q: This sentence is already correct.
    A: Keep unchanged B.
    S: source B
    AI: notes B

    ---

    <!-- note_key: live-nk-c -->
    Q: teh plan dont recieve support.
    A: Keep unchanged C.
    S: source C
    AI: notes C
    """
    _bootstrap_collection(tmp_path, deck_markdown=source_markdown)

    result = run_task(
        collection_dir=tmp_path,
        task_name=STANDARD_TASK_NAME,
        no_auto_commit=True,
    )

    assert result.status == "completed"
    assert not result.failed
    assert result.persisted
    assert result.summary.eligible == 3
    assert result.summary.errors == 0
    assert result.summary.requests == 3
    assert result.summary.updated in {2, 3}
    assert result.summary.unchanged == 3 - result.summary.updated

    content = (tmp_path / f"{LIVE_DECK_NAME}.md").read_text(encoding="utf-8")
    lower_content = content.lower()
    assert "<!-- note_key: live-nk-a -->" in content
    assert "<!-- note_key: live-nk-b -->" in content
    assert "<!-- note_key: live-nk-c -->" in content
    assert "teh route dont recieve updates." not in lower_content
    assert "teh plan dont recieve support." not in lower_content

    detail, job_id, db = _open_job_detail(tmp_path, result.job_id)
    try:
        assert detail is not None
        status_by_note_key = {item.note_key: item.final_status for item in detail.items}
        assert status_by_note_key["live-nk-a"] is LlmFinalStatus.SUCCEEDED_UPDATED
        assert status_by_note_key["live-nk-c"] is LlmFinalStatus.SUCCEEDED_UPDATED
        assert status_by_note_key["live-nk-b"] in {
            LlmFinalStatus.SUCCEEDED_UNCHANGED,
            LlmFinalStatus.SUCCEEDED_UPDATED,
        }

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
        linked_rows = db._conn.execute(
            """
            SELECT
                i.note_key,
                COUNT(a.id) AS attempts_count,
                COUNT(p.attempt_id) AS payload_count
            FROM llm_job_item i
            LEFT JOIN llm_item_attempt a ON a.job_item_id = i.id
            LEFT JOIN llm_attempt_payload p ON p.attempt_id = a.id
            WHERE i.job_id = ?
            GROUP BY i.id, i.note_key
            ORDER BY i.ordinal ASC
            """,
            (job_id,),
        ).fetchall()
    finally:
        db.close()

    assert totals_row is not None
    assert int(totals_row["attempts_count"]) == 3
    assert int(totals_row["payload_count"]) == 3
    assert len(linked_rows) == 3
    assert [row["note_key"] for row in linked_rows] == [
        "live-nk-a",
        "live-nk-b",
        "live-nk-c",
    ]
    assert all(int(row["attempts_count"]) == 1 for row in linked_rows)
    assert all(int(row["payload_count"]) == 1 for row in linked_rows)
