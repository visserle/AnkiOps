"""Read-only manifest of source files that can change Anki."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from blake3 import blake3

from ankiops.deck_sources import DeckSource, load_note_types_for_source
from ankiops.markdown import read_deck_file
from ankiops.media import extract_media_references


@dataclass(frozen=True)
class AnkiApplicableManifest:
    root: Path
    paths: frozenset[str]

    def content_hash(self) -> str:
        """Hash only the current contents represented by this manifest."""
        digest = blake3()
        for relative_path in sorted(self.paths):
            encoded_path = relative_path.encode()
            digest.update(len(encoded_path).to_bytes(4, "big"))
            digest.update(encoded_path)
            path = self.root / relative_path
            if not path.is_file():
                digest.update(b"missing")
                continue
            digest.update(b"file")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    digest.update(chunk)
        return digest.hexdigest()


def source_anki_manifest(source: DeckSource) -> AnkiApplicableManifest:
    """Return repository-relative files read by files-to-Anki for one source."""
    note_types = load_note_types_for_source(source)
    note_types_by_name = {note_type.name: note_type for note_type in note_types}
    paths: set[str] = set()
    note_types_used: set[str] = set()

    for deck_path in source.deck_files():
        paths.add(deck_path.relative_to(source.root).as_posix())
        raw_content = deck_path.read_text(encoding="utf-8")
        paths.update(
            f"media/{filename}" for filename in extract_media_references(raw_content)
        )
        deck = read_deck_file(
            deck_path,
            note_types=note_types,
            context_root=source.root,
        )
        note_types_used.update(note.note_type for note in deck.notes)

    for name in note_types_used:
        paths.update(
            (Path("note_types") / path).as_posix()
            for path in note_types_by_name[name].source_files
        )

    return AnkiApplicableManifest(source.root, frozenset(paths))


def anki_applicable_paths_changed(
    changed_paths: set[str],
    before: AnkiApplicableManifest,
    after: AnkiApplicableManifest,
) -> bool:
    """Return whether changes touch paths applicable before or after an update."""
    return bool(changed_paths & (before.paths | after.paths))
