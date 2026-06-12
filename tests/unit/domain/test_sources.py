from __future__ import annotations

from ankiops.deck_sources import (
    DeckSource,
    deck_files_for_source,
    discover_deck_sources,
    load_note_types_for_source,
)
from tests.support.deck_files import DeckFileHarness


def test_discover_deck_sources_includes_local_and_shared(tmp_path):
    (tmp_path / "note_types").mkdir()
    (tmp_path / "shared" / "owner" / "repo" / "note_types").mkdir(parents=True)
    (tmp_path / "shared" / "other" / "deck" / "note_types").mkdir(parents=True)

    sources = discover_deck_sources(tmp_path)

    assert [source.display_name for source in sources] == [
        "local",
        "shared/other/deck",
        "shared/owner/repo",
    ]
    assert sources[1].github_url == "https://github.com/other/deck.git"
    assert sources[2].github_url == "https://github.com/owner/repo.git"


def test_shared_markdown_files_ignore_reserved_docs(tmp_path):
    source = DeckSource.shared(tmp_path, "owner", "repo")
    source.root.mkdir(parents=True)
    (source.root / "Deck.md").write_text("Q: a\nA: b", encoding="utf-8")
    (source.root / "README.md").write_text("# docs", encoding="utf-8")
    (source.root / "LICENSE.md").write_text("# license", encoding="utf-8")
    (source.root / "CHANGELOG.md").write_text("# changes", encoding="utf-8")
    (source.root / "_draft.md").write_text("Q: x\nA: y", encoding="utf-8")

    assert [path.name for path in deck_files_for_source(source)] == ["Deck.md"]


def test_reserved_markdown_names_are_not_ignored_in_local_root(tmp_path):
    source = DeckSource.local(tmp_path)
    for name in ["README.md", "LICENSE.md", "CHANGELOG.md", "_draft.md"]:
        (tmp_path / name).write_text("Q: local\nA: deck", encoding="utf-8")

    assert [path.name for path in deck_files_for_source(source)] == [
        "CHANGELOG.md",
        "LICENSE.md",
        "README.md",
        "_draft.md",
    ]


def test_shared_configs_are_scoped_by_source(tmp_path):
    source = DeckSource.shared(tmp_path, "owner", "repo")
    DeckFileHarness().eject_default_note_types(source.note_types_dir)

    names = {config.name for config in load_note_types_for_source(source)}

    assert "shared/owner/repo/AnkiOpsQA" in names
    assert "AnkiOpsQA" not in names


def test_shared_note_type_names_are_path_like(tmp_path):
    source = DeckSource.shared(tmp_path, "owner", "repo")

    assert source.scope_note_type_name("AnkiOpsQA") == ("shared/owner/repo/AnkiOpsQA")
    assert source.scope_note_type_name("shared/owner/repo/AnkiOpsQA") == (
        "shared/owner/repo/AnkiOpsQA"
    )
    assert source.unscoped_note_type_name("shared/owner/repo/AnkiOpsQA") == (
        "AnkiOpsQA"
    )
