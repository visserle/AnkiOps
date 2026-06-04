from __future__ import annotations

from ankiops.fs import FileSystemAdapter
from ankiops.sources import (
    SyncSource,
    discover_sync_sources,
    load_configs_for_source,
    markdown_files_for_source,
)


def test_discover_sync_sources_includes_local_and_collab(tmp_path):
    (tmp_path / "note_types").mkdir()
    (tmp_path / "collab" / "owner" / "repo" / "note_types").mkdir(parents=True)
    (tmp_path / "collab" / "other" / "deck" / "note_types").mkdir(parents=True)

    sources = discover_sync_sources(tmp_path)

    assert [source.display_name for source in sources] == [
        "local",
        "collab/other/deck",
        "collab/owner/repo",
    ]
    assert sources[1].github_url == "https://github.com/other/deck.git"
    assert sources[2].github_url == "https://github.com/owner/repo.git"


def test_collab_markdown_files_ignore_reserved_docs(tmp_path):
    source = SyncSource.collab(tmp_path, "owner", "repo")
    source.root.mkdir(parents=True)
    (source.root / "Deck.md").write_text("Q: a\nA: b", encoding="utf-8")
    (source.root / "README.md").write_text("# docs", encoding="utf-8")
    (source.root / "LICENSE.md").write_text("# license", encoding="utf-8")
    (source.root / "CHANGELOG.md").write_text("# changes", encoding="utf-8")
    (source.root / "_draft.md").write_text("Q: x\nA: y", encoding="utf-8")

    assert [path.name for path in markdown_files_for_source(source)] == ["Deck.md"]


def test_reserved_markdown_names_are_not_ignored_in_local_root(tmp_path):
    source = SyncSource.local(tmp_path)
    for name in ["README.md", "LICENSE.md", "CHANGELOG.md", "_draft.md"]:
        (tmp_path / name).write_text("Q: local\nA: deck", encoding="utf-8")

    assert [path.name for path in markdown_files_for_source(source)] == [
        "CHANGELOG.md",
        "LICENSE.md",
        "README.md",
        "_draft.md",
    ]


def test_collab_configs_are_scoped_by_source(tmp_path):
    source = SyncSource.collab(tmp_path, "owner", "repo")
    FileSystemAdapter().eject_builtin_note_types(source.note_types_dir)

    names = {config.name for config in load_configs_for_source(source)}

    assert "collab/owner/repo/AnkiOpsQA" in names
    assert "AnkiOpsQA" not in names


def test_collab_note_type_names_are_path_like(tmp_path):
    source = SyncSource.collab(tmp_path, "owner", "repo")

    assert source.scope_note_type_name("AnkiOpsQA") == (
        "collab/owner/repo/AnkiOpsQA"
    )
    assert source.scope_note_type_name("collab/owner/repo/AnkiOpsQA") == (
        "collab/owner/repo/AnkiOpsQA"
    )
    assert source.unscoped_note_type_name("collab/owner/repo/AnkiOpsQA") == (
        "AnkiOpsQA"
    )
