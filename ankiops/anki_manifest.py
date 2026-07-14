"""Read-only manifest of source files that can change Anki."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
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

    for scoped_name in note_types_used:
        name = source.unscoped_note_type_name(scoped_name)
        paths.update(_note_type_paths(source, name))

    return AnkiApplicableManifest(source.root, frozenset(paths))


def anki_applicable_paths_changed(
    changed_paths: set[str],
    before: AnkiApplicableManifest,
    after: AnkiApplicableManifest,
) -> bool:
    """Return whether changes touch paths applicable before or after an update."""
    return bool(changed_paths & (before.paths | after.paths))


def _note_type_paths(source: DeckSource, name: str) -> set[str]:
    note_type_dir = source.note_types_dir / name
    manifest_path = note_type_dir / "note_type.yaml"
    info = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    references = _note_type_references(note_type_dir, info)
    paths = {manifest_path.relative_to(source.root).as_posix()}
    paths.update(
        _contained_note_type_path(source, note_type_dir / reference)
        for reference in references
    )
    return paths


def _note_type_references(note_type_dir: Path, info: Any) -> list[str]:
    if not isinstance(info, dict):
        return []
    styling = info.get("styling")
    styling_refs = [styling] if isinstance(styling, str) else list(styling or [])

    templates = info.get("templates")
    if templates is not None:
        template_refs = [
            str(template[side]).strip()
            for template in templates
            for side in ("front", "back")
        ]
        return [*styling_refs, *template_refs]

    template_refs = ["Front.template.anki", "Back.template.anki"]
    index = 2
    while (note_type_dir / f"Front{index}.template.anki").is_file() and (
        note_type_dir / f"Back{index}.template.anki"
    ).is_file():
        template_refs.extend(
            [f"Front{index}.template.anki", f"Back{index}.template.anki"]
        )
        index += 1
    return [*styling_refs, *template_refs]


def _contained_note_type_path(source: DeckSource, path: Path) -> str:
    note_types_root = Path(os.path.abspath(source.note_types_dir))
    candidate = Path(os.path.abspath(path))
    try:
        relative = candidate.relative_to(note_types_root)
    except ValueError as error:
        raise ValueError(
            f"Note-type asset '{path}' is outside {source.note_types_dir}."
        ) from error
    return (Path("note_types") / relative).as_posix()
