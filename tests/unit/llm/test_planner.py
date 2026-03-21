from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from ankiops.db import SQLiteDbAdapter
from ankiops.fs import FileSystemAdapter
from ankiops.llm.db import LlmDbAdapter
from ankiops.llm.runner import plan_task

TASK_FILE = Path("llm/tasks/grammar.yaml")
PROMPT_FILE = Path("llm/prompts/grammar.md")
SYSTEM_PROMPT_FILE = Path("llm/system_prompt.md")
TEST_DECK = "TestDeck"
TEST_DECK_MARKDOWN = """
<!-- note_key: nk-1 -->
Q: this is a broken question.
A: this is a broken answer.
S: grammar book
AI: hidden content

---

<!-- note_key: nk-2 -->
Q: pick one
C1: yes
C2: no
A: 1
"""


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def _prepare_collection(tmp_path: Path) -> Path:
    db = SQLiteDbAdapter.open(tmp_path)
    db.close()

    fs = FileSystemAdapter()
    fs.eject_builtin_note_types(tmp_path / "note_types")
    _write(tmp_path / f"{TEST_DECK}.md", TEST_DECK_MARKDOWN)
    _write(
        tmp_path / SYSTEM_PROMPT_FILE,
        "You are a strict editor.",
    )
    _write(
        tmp_path / PROMPT_FILE,
        "fix grammar",
    )
    _write(
        tmp_path / TASK_FILE,
        """
        model: sonnet
        prompt_file: ../prompts/grammar.md
        fields:
          exceptions:
            - read_only: ["Source"]
            - note_types: ["AnkiOpsChoice"]
              read_only: ["Answer"]
            - hidden: ["AI Notes"]
        request:
          max_output_tokens: 2048
        """,
    )
    return tmp_path


def test_plan_task_summarizes_scope_surface_and_cost_cap(tmp_path: Path):
    collection = _prepare_collection(tmp_path)
    original_content = (collection / f"{TEST_DECK}.md").read_text(encoding="utf-8")

    plan = plan_task(
        collection_dir=collection,
        task_name="grammar",
    )

    assert plan.task_name == "grammar"
    assert str(plan.model) == "sonnet"
    assert plan.summary.decks_seen == 1
    assert plan.summary.decks_matched == 1
    assert plan.summary.notes_seen == 2
    assert plan.summary.eligible == 2
    assert plan.summary.errors == 0
    assert plan.requests_estimate == 2
    assert plan.output_tokens_cap == 4096
    assert plan.input_tokens_estimate > 0
    assert plan.format_cost_cap().startswith("LLM estimated cost: $")

    surface_by_type = {surface.note_type: surface for surface in plan.field_surface}
    assert "AnkiOpsQA" in surface_by_type
    assert "AnkiOpsChoice" in surface_by_type
    assert "AI Notes" in surface_by_type["AnkiOpsQA"].hidden_fields
    assert "Answer" in surface_by_type["AnkiOpsChoice"].read_only_fields

    db = LlmDbAdapter.open(collection)
    try:
        row = db._conn.execute("SELECT COUNT(*) AS total FROM llm_job").fetchone()
    finally:
        db.close()

    assert row is not None
    assert int(row["total"]) == 0
    assert (
        (collection / f"{TEST_DECK}.md").read_text(encoding="utf-8")
        == original_content
    )


def test_plan_task_ignores_unrelated_invalid_task_files(tmp_path: Path):
    collection = _prepare_collection(tmp_path)
    _write(
        collection / "llm/tasks/translate.yaml",
        """
        model: sonnet
        prompt_file: ../prompts/grammar.md
        sdk: anthropic
        """,
    )

    plan = plan_task(
        collection_dir=collection,
        task_name="grammar",
    )

    assert plan.summary.eligible == 2
