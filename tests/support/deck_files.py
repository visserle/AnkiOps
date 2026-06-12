from pathlib import Path

from ankiops.html_to_markdown import HTMLToMarkdown
from ankiops.markdown import (
    DeckFile,
    find_deck_files,
    infer_note_type,
    read_deck_file,
    write_deck_file,
)
from ankiops.markdown_to_html import MarkdownToHTML
from ankiops.media import calculate_blake3, update_references
from ankiops.note_types import NoteType, eject_default_note_types, load_note_types


class DeckFileHarness:
    def __init__(self) -> None:
        self._note_types: list[NoteType] = []
        self._md_to_html = MarkdownToHTML()
        self._html_to_md = HTMLToMarkdown()

    def set_note_types(self, note_types: list[NoteType]) -> None:
        self._note_types = note_types

    def read_deck_file(
        self,
        file_path: Path,
        *,
        context_root: Path | None = None,
    ) -> DeckFile:
        return read_deck_file(
            file_path,
            note_types=self._note_types,
            context_root=context_root,
        )

    def write_deck_file(self, file_path: Path, content: str) -> None:
        write_deck_file(file_path, content)

    def find_deck_files(self, directory: Path) -> list[Path]:
        return find_deck_files(directory)

    def load_note_types(self, note_types_dir: Path) -> list[NoteType]:
        note_types = load_note_types(note_types_dir)
        self.set_note_types(note_types)
        return note_types

    def _infer_note_type(
        self,
        fields: dict[str, str],
        *,
        labels: set[str] | None = None,
    ) -> str:
        return infer_note_type(self._note_types, fields, labels=labels)

    def eject_default_note_types(self, dst_dir: Path) -> None:
        eject_default_note_types(dst_dir)

    def calculate_blake3(self, file_path: Path) -> str:
        return calculate_blake3(file_path)

    def update_media_references(
        self,
        directory: Path,
        rename_map: dict[str, str],
    ) -> int:
        return update_references(directory, rename_map)

    def convert_to_html(self, markdown_text: str) -> str:
        return self._md_to_html.convert(markdown_text)

    def convert_to_markdown(self, html_text: str) -> str:
        return self._html_to_md.convert(html_text)
