"""Source files that can change Anki."""

from __future__ import annotations

from pathlib import Path

from ankiops.deck_sources import DeckSource, load_note_types_for_source
from ankiops.markdown import read_deck_file
from ankiops.media import extract_media_references


def anki_applicable_paths(source: DeckSource) -> frozenset[str]:
    """Return repository-relative files read by files-to-Anki for one source."""
    note_types = load_note_types_for_source(source)
    note_types_by_name = {note_type.name: note_type for note_type in note_types}
    paths: set[str] = set()
    note_types_used: set[str] = set()

    for deck_path in source.deck_files():
        paths.add(deck_path.relative_to(source.root).as_posix())
        deck = read_deck_file(
            deck_path,
            note_types=note_types,
            context_root=source.root,
        )
        paths.update(
            f"media/{filename}"
            for filename in extract_media_references(deck.raw_content)
        )
        note_types_used.update(note.note_type for note in deck.notes)

    for name in note_types_used:
        paths.update(
            (Path("note_types") / path).as_posix()
            for path in note_types_by_name[name].source_files
        )

    return frozenset(paths)
