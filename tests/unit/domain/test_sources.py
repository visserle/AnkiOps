from __future__ import annotations

from pathlib import Path

import pytest

from ankiops.anki_manifest import anki_applicable_paths
from ankiops.deck_sources import (
    DeckSource,
    discover_deck_sources,
    load_note_types_for_collection,
    load_note_types_for_source,
)
from ankiops.git import GitRepository
from tests.support.deck_files import DeckFileHarness


def test_discover_deck_sources_includes_local_and_collab(tmp_path):
    (tmp_path / "note_types").mkdir()
    (tmp_path / "collab" / "owner" / "repo" / "note_types").mkdir(parents=True)
    (tmp_path / "collab" / "other" / "deck" / "note_types").mkdir(parents=True)
    GitRepository(tmp_path / "collab" / "owner" / "repo").init_repo()
    GitRepository(tmp_path / "collab" / "other" / "deck").init_repo()

    sources = discover_deck_sources(tmp_path)

    assert [source.display_name for source in sources] == [
        "local",
        "other/deck",
        "owner/repo",
    ]
    assert sources[1].github_url == "https://github.com/other/deck.git"
    assert sources[2].github_url == "https://github.com/owner/repo.git"


def test_discover_deck_sources_rejects_non_repository_collab_directory(tmp_path):
    invalid_root = tmp_path / "collab" / "owner" / "repo"
    invalid_root.mkdir(parents=True)

    with pytest.raises(ValueError, match="not an independent Git repository"):
        discover_deck_sources(tmp_path)


@pytest.mark.parametrize(
    "relative_path",
    [Path("owner/repo"), Path("collab/owner"), Path("collab/../repo")],
)
def test_deck_source_rejects_noncanonical_relative_paths(tmp_path, relative_path):
    with pytest.raises(ValueError, match="Expected .*source"):
        DeckSource(tmp_path, relative_path)


def test_markdown_files_ignore_reserved_docs_in_all_sources(tmp_path):
    local_source = DeckSource.local(tmp_path)
    collab_source = DeckSource.collab(tmp_path, "owner/repo")
    collab_source.root.mkdir(parents=True)

    for root in [local_source.root, collab_source.root]:
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
    assert [path.name for path in collab_source.deck_files()] == [
        "Deck.md",
        "_draft.md",
    ]


def test_ambiguous_deck_file_names_raise(tmp_path):
    source = DeckSource.local(tmp_path)
    (tmp_path / "a___b.md").write_text("Q: local\nA: deck", encoding="utf-8")

    with pytest.raises(ValueError, match="Ambiguous deck filename 'a___b.md'"):
        source.deck_files()


def test_collab_configs_are_scoped_by_source(tmp_path):
    source = DeckSource.collab(tmp_path, "owner/repo")
    DeckFileHarness().eject_default_note_types(source.note_types_dir)

    names = {config.name for config in load_note_types_for_source(source)}

    assert "collab/owner/repo/AnkiOpsQA" in names
    assert "AnkiOpsQA" not in names


def test_collection_note_types_include_local_and_scoped_collab_configs(tmp_path):
    harness = DeckFileHarness()
    harness.eject_default_note_types(tmp_path / "note_types")
    collab_source = DeckSource.collab(tmp_path, "owner/repo")
    harness.eject_default_note_types(collab_source.note_types_dir)
    GitRepository(collab_source.root).init_repo()

    names = {config.name for config in load_note_types_for_collection(tmp_path)}

    assert "AnkiOpsQA" in names
    assert "collab/owner/repo/AnkiOpsQA" in names


def test_collab_note_type_names_are_path_like(tmp_path):
    source = DeckSource.collab(tmp_path, "owner/repo")

    assert source.scope_note_type_name("AnkiOpsQA") == ("collab/owner/repo/AnkiOpsQA")
    assert source.scope_note_type_name("collab/owner/repo/AnkiOpsQA") == (
        "collab/owner/repo/AnkiOpsQA"
    )
    assert source.unscoped_note_type_name("collab/owner/repo/AnkiOpsQA") == (
        "AnkiOpsQA"
    )


def test_source_location_and_kind_are_derived_from_identity(tmp_path):
    local = DeckSource.local(tmp_path)
    collab = DeckSource.collab(tmp_path, "owner/repo")

    assert vars(local) == {
        "collection_root": tmp_path,
        "relative_path": Path("."),
    }
    assert vars(collab) == {
        "collection_root": tmp_path,
        "relative_path": Path("collab/owner/repo"),
    }
    assert local.source_path == "."
    assert collab.source_path == "collab/owner/repo"
    assert local.root == tmp_path
    assert not local.is_collab
    assert collab.root == tmp_path / "collab" / "owner" / "repo"
    assert collab.is_collab


def test_anki_applicable_paths_contains_only_files_used_by_loaded_decks(tmp_path):
    source = DeckSource.collab(tmp_path, "owner/repo")
    DeckFileHarness().eject_default_note_types(source.note_types_dir)
    (source.root / "Deck.md").write_text(
        "<!-- note_key: key -->\nQ: question\nA: ![image](media/shared.png)\n",
        encoding="utf-8",
    )
    (source.root / "README.md").write_text("docs\n", encoding="utf-8")
    media = source.root / "media"
    media.mkdir()
    (media / "shared.png").write_bytes(b"shared")
    (media / "private.png").write_bytes(b"private")

    paths = anki_applicable_paths(source)

    assert paths == frozenset(
        {
            "Deck.md",
            "media/shared.png",
            "note_types/AnkiOpsQA/Back.template.anki",
            "note_types/AnkiOpsQA/Front.template.anki",
            "note_types/AnkiOpsQA/note_type.yaml",
            "note_types/AnkiOpsStyling.css",
            "note_types/SyntaxHighlighting.css",
        }
    )


def test_anki_applicable_paths_keeps_removed_references_relevant_to_update(tmp_path):
    source = DeckSource.local(tmp_path)
    DeckFileHarness().eject_default_note_types(source.note_types_dir)
    deck = source.root / "Deck.md"
    deck.write_text(
        "<!-- note_key: key -->\nQ: question\nA: ![image](media/shared.png)\n",
        encoding="utf-8",
    )
    media = source.root / "media"
    media.mkdir()
    shared = media / "shared.png"
    shared.write_bytes(b"shared")
    before = anki_applicable_paths(source)

    deck.write_text(
        "<!-- note_key: key -->\nQ: question\nA: no image\n",
        encoding="utf-8",
    )
    shared.unlink()
    after = anki_applicable_paths(source)

    applicable = before | after
    assert "media/shared.png" not in after
    assert "media/shared.png" in applicable
    assert "media/private.png" not in applicable
