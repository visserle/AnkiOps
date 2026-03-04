from __future__ import annotations

from importlib import resources
from pathlib import Path
from textwrap import dedent

import pytest

from ankiops.db import SQLiteDbAdapter
from ankiops.fs import FileSystemAdapter
from ankiops.init import initialize_collection
from ankiops.llm.config_loader import load_llm_task_catalog
from ankiops.llm.models import NotePatch
from ankiops.llm.runner import run_task

TASK_FILE = Path("llm/tasks/grammar.yaml")
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
DEFAULT_TASK_EXTRA = """
fields:
  exceptions:
    - read_only: ["Source"]
    - note_types: ["AnkiOpsChoice"]
      read_only: ["Answer"]
    - hidden: ["AI Notes"]
"""


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def _task_config(
    *,
    model: str = "claude-sonnet-4-20250514",
    prompt: str = "fix grammar",
    extra: str = "",
) -> str:
    suffix = f"\n{dedent(extra).strip()}" if extra.strip() else ""
    return f"name: grammar\nmodel: {model}\nprompt: {prompt}{suffix}\n"


def _write_task(collection_dir: Path, *, content: str) -> None:
    _write(collection_dir / TASK_FILE, content)


def _load_note_type_configs(collection_dir: Path):
    fs = FileSystemAdapter()
    note_types_dir = collection_dir / "note_types"
    fs.eject_builtin_note_types(note_types_dir)
    return fs.load_note_type_configs(note_types_dir)


def _patch_collection_paths(monkeypatch, collection_dir: Path) -> None:
    monkeypatch.setattr("ankiops.config.get_collection_dir", lambda: collection_dir)
    monkeypatch.setattr(
        "ankiops.collection_serializer.get_collection_dir",
        lambda: collection_dir,
    )
    monkeypatch.setattr(
        "ankiops.config.get_note_types_dir",
        lambda: collection_dir / "note_types",
    )
    monkeypatch.setattr(
        "ankiops.collection_serializer.get_note_types_dir",
        lambda: collection_dir / "note_types",
    )


def _prepare_runner_collection(
    tmp_path: Path,
    monkeypatch,
    *,
    task_content: str | None = None,
) -> Path:
    _patch_collection_paths(monkeypatch, tmp_path)
    db = SQLiteDbAdapter.load(tmp_path)
    db.close()

    _load_note_type_configs(tmp_path)
    _write(tmp_path / f"{TEST_DECK}.md", TEST_DECK_MARKDOWN)
    _write_task(
        tmp_path,
        content=task_content or _task_config(extra=DEFAULT_TASK_EXTRA),
    )
    return tmp_path


class _StubClient:
    def __init__(self, patches: list[NotePatch]) -> None:
        self._patches = patches

    def generate_patch(self, **_kwargs) -> NotePatch:
        return self._patches.pop(0)


@pytest.fixture
def note_type_configs(tmp_path: Path):
    return _load_note_type_configs(tmp_path)


def test_initialize_collection_ejects_packaged_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr("ankiops.init.get_collection_dir", lambda: tmp_path)
    monkeypatch.setattr("ankiops.init._setup_git", lambda _collection_dir: None)

    collection_dir = initialize_collection("TestProfile")

    packaged_tasks = sorted(
        resource.name
        for resource in resources.files("ankiops.llm.tasks").iterdir()
        if resource.is_file() and resource.suffix == ".yaml"
    )
    ejected_tasks = sorted(
        path.name for path in (tmp_path / "llm/tasks").glob("*.yaml")
    )

    assert collection_dir == tmp_path
    assert ejected_tasks == packaged_tasks
    assert "model: claude-sonnet-4-20250514" in (
        tmp_path / "llm/tasks/grammar.yaml"
    ).read_text(encoding="utf-8")


def test_initialize_collection_preserves_existing_task(tmp_path, monkeypatch):
    monkeypatch.setattr("ankiops.init.get_collection_dir", lambda: tmp_path)
    monkeypatch.setattr("ankiops.init._setup_git", lambda _collection_dir: None)

    existing_task = tmp_path / TASK_FILE
    existing_task.parent.mkdir(parents=True, exist_ok=True)
    existing_task.write_text("name: grammar\nprompt: keep mine\n", encoding="utf-8")

    initialize_collection("TestProfile")

    assert existing_task.read_text(encoding="utf-8") == (
        "name: grammar\nprompt: keep mine\n"
    )


def test_load_llm_task_catalog_loads_valid_task(note_type_configs, tmp_path: Path):
    _write_task(
        tmp_path,
        content=_task_config(
            extra="""
            decks:
              include: ["Parent"]
              include_subdecks: false
            fields:
              exceptions:
                - read_only: ["Source"]
                - note_types: ["AnkiOpsChoice"]
                  read_only: ["Answer"]
                - hidden: ["AI Notes"]
            """
        ),
    )

    catalog = load_llm_task_catalog(tmp_path, note_type_configs=note_type_configs)

    assert not catalog.errors
    task = catalog.tasks_by_name["grammar"]
    assert task.model == "claude-sonnet-4-20250514"
    assert task.api_key_env == "ANTHROPIC_API_KEY"
    assert task.decks.include == ["Parent"]
    assert task.decks.include_subdecks is False


@pytest.mark.parametrize(
    ("task_content", "expected_error"),
    [
        (
            _task_config(
                extra="""
                fields:
                  exceptions:
                    - note_types: ["AnkiOpsQA"]
                      read_only: ["DoesNotExist"]
                """
            ),
            "DoesNotExist",
        ),
        (
            _task_config(
                extra="""
                decks:
                  include: ["Parent"]
                  include_subdecks: "yes"
                """
            ),
            "decks.include_subdecks",
        ),
        (
            _task_config(model="gpt-5"),
            "Claude model id",
        ),
        (
            _task_config(extra="sdk: anthropic"),
            "unknown task key(s): sdk",
        ),
    ],
    ids=[
        "invalid-exception-field",
        "invalid-include-subdecks",
        "non-claude-model",
        "legacy-provider-key",
    ],
)
def test_load_llm_task_catalog_rejects_invalid_tasks(
    note_type_configs,
    tmp_path: Path,
    task_content: str,
    expected_error: str,
):
    _write_task(tmp_path, content=task_content)

    catalog = load_llm_task_catalog(tmp_path, note_type_configs=note_type_configs)

    assert not catalog.tasks_by_name
    assert expected_error in next(iter(catalog.errors.values()))


def test_load_llm_task_catalog_ignores_non_task_dirs(
    note_type_configs,
    tmp_path: Path,
):
    _write_task(tmp_path, content=_task_config())
    _write(
        tmp_path / "llm/actions/old.yaml",
        """
        name: old
        prompt: should be ignored
        """,
    )
    _write(
        tmp_path / "llm/providers/anthropic.yaml",
        """
        name: anthropic
        model: claude-sonnet-4-20250514
        """,
    )

    catalog = load_llm_task_catalog(tmp_path, note_type_configs=note_type_configs)

    assert not catalog.errors
    assert set(catalog.tasks_by_name) == {"grammar"}


def test_run_task_updates_only_editable_fields(tmp_path, monkeypatch):
    collection = _prepare_runner_collection(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "ankiops.llm.runner.git_snapshot", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        "ankiops.llm.runner.ClaudeClient",
        lambda _task: _StubClient(
            [
                NotePatch(
                    note_key="nk-1",
                    edits={"Question": "This is a fixed question."},
                ),
                NotePatch(note_key="nk-2", edits={}),
            ]
        ),
    )

    summary = run_task(
        collection_dir=collection,
        task_name="grammar",
        dry_run=False,
        no_auto_commit=True,
    )

    content = (collection / f"{TEST_DECK}.md").read_text(encoding="utf-8")

    assert summary.model == "claude-sonnet-4-20250514"
    assert summary.updated == 1
    assert summary.unchanged == 1
    assert "Q: This is a fixed question." in content
    assert "S: grammar book" in content
    assert "AI: hidden content" in content
    assert "A: 1" in content


def test_run_task_rejects_read_only_updates(tmp_path, monkeypatch):
    collection = _prepare_runner_collection(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "ankiops.llm.runner.git_snapshot", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        "ankiops.llm.runner.ClaudeClient",
        lambda _task: _StubClient(
            [
                NotePatch(note_key="nk-1", edits={}),
                NotePatch(note_key="nk-2", edits={"Answer": "2"}),
            ]
        ),
    )

    with pytest.raises(SystemExit) as exc:
        run_task(
            collection_dir=collection,
            task_name="grammar",
            dry_run=True,
            no_auto_commit=True,
        )

    assert exc.value.code == 1
    assert "A: 1" in (collection / f"{TEST_DECK}.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("task_content", "expected_scope"),
    [
        (
            _task_config(
                extra=f"""
                decks:
                  include: ["{TEST_DECK}"]
                fields:
                  exceptions:
                    - read_only: ["Source"]
                """
            ),
            {"strict": True, "deck": TEST_DECK, "no_subdecks": False},
        ),
        (
            _task_config(
                extra=f"""
                decks:
                  include: ["{TEST_DECK}"]
                  include_subdecks: false
                fields:
                  exceptions:
                    - read_only: ["Source"]
                """
            ),
            {"strict": True, "deck": TEST_DECK, "no_subdecks": True},
        ),
        (
            _task_config(
                extra="""
                decks:
                  include: ["Test*"]
                fields:
                  exceptions:
                    - read_only: ["Source"]
                """
            ),
            {"strict": True, "deck": None, "no_subdecks": False},
        ),
    ],
    ids=["exact-deck", "exact-deck-without-subdecks", "wildcard-deck"],
)
def test_run_task_uses_expected_serialize_scope(
    tmp_path: Path,
    monkeypatch,
    task_content: str,
    expected_scope: dict[str, object],
):
    collection = _prepare_runner_collection(
        tmp_path,
        monkeypatch,
        task_content=task_content,
    )
    captured: dict[str, object] = {}

    def _fake_serialize(
        _collection_dir,
        *,
        strict: bool = False,
        deck: str | None = None,
        no_subdecks: bool = False,
    ):
        captured["strict"] = strict
        captured["deck"] = deck
        captured["no_subdecks"] = no_subdecks
        return {
            "collection": {"serialized_at": "2026-03-02T00:00:00Z"},
            "decks": [],
        }

    monkeypatch.setattr("ankiops.llm.runner.serialize_collection", _fake_serialize)
    monkeypatch.setattr(
        "ankiops.llm.runner.ClaudeClient",
        lambda _task: _StubClient([]),
    )

    run_task(
        collection_dir=collection,
        task_name="grammar",
        dry_run=True,
        no_auto_commit=True,
    )

    assert captured == expected_scope
