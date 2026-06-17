from __future__ import annotations

import pytest

from ankiops.deck_sources import (
    DeckSource,
    discover_deck_sources,
    load_note_types_for_collection,
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


def test_markdown_files_ignore_reserved_docs_in_all_sources(tmp_path):
    local_source = DeckSource.local(tmp_path)
    shared_source = DeckSource.shared(tmp_path, "owner", "repo")
    shared_source.root.mkdir(parents=True)

    for root in [local_source.root, shared_source.root]:
        (root / "Deck.md").write_text("Q: a\nA: b", encoding="utf-8")
        (root / "README.md").write_text("# docs", encoding="utf-8")
        (root / "LICENSE.md").write_text("# license", encoding="utf-8")
        (root / "CHANGELOG.md").write_text("# changes", encoding="utf-8")
        (root / "CONTRIBUTING.md").write_text("# contributing", encoding="utf-8")
        (root / "_draft.md").write_text("Q: x\nA: y", encoding="utf-8")

    assert [path.name for path in local_source.deck_files()] == [
        "Deck.md",
        "_draft.md",
    ]
    assert [path.name for path in shared_source.deck_files()] == [
        "Deck.md",
        "_draft.md",
    ]


def test_ambiguous_deck_file_names_raise(tmp_path):
    source = DeckSource.local(tmp_path)
    (tmp_path / "a___b.md").write_text("Q: local\nA: deck", encoding="utf-8")

    with pytest.raises(ValueError, match="Ambiguous deck filename 'a___b.md'"):
        source.deck_files()


def test_shared_configs_are_scoped_by_source(tmp_path):
    source = DeckSource.shared(tmp_path, "owner", "repo")
    DeckFileHarness().eject_default_note_types(source.note_types_dir)

    names = {config.name for config in load_note_types_for_source(source)}

    assert "shared/owner/repo/AnkiOpsQA" in names
    assert "AnkiOpsQA" not in names


def test_collection_configs_include_local_and_scoped_shared_types(tmp_path):
    harness = DeckFileHarness()
    harness.eject_default_note_types(tmp_path / "note_types")
    shared_source = DeckSource.shared(tmp_path, "owner", "repo")
    harness.eject_default_note_types(shared_source.note_types_dir)

    names = {config.name for config in load_note_types_for_collection(tmp_path)}

    assert "AnkiOpsQA" in names
    assert "shared/owner/repo/AnkiOpsQA" in names


def test_shared_note_type_names_are_path_like(tmp_path):
    source = DeckSource.shared(tmp_path, "owner", "repo")

    assert source.scope_note_type_name("AnkiOpsQA") == ("shared/owner/repo/AnkiOpsQA")
    assert source.scope_note_type_name("shared/owner/repo/AnkiOpsQA") == (
        "shared/owner/repo/AnkiOpsQA"
    )
    assert source.unscoped_note_type_name("shared/owner/repo/AnkiOpsQA") == (
        "AnkiOpsQA"
    )
