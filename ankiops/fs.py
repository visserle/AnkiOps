"""FileSystem Adapter for Reading/Writing Markdown, Configs, and Media."""

import logging
import re
import shutil
from importlib import resources
from pathlib import Path
from urllib.parse import unquote

import yaml
from blake3 import blake3

from ankiops.config import NOTE_SEPARATOR
from ankiops.html_converter import HTMLToMarkdown
from ankiops.markdown_converter import MarkdownToHTML
from ankiops.models import (
    ANKIOPS_KEY_FIELD,
    Field,
    MarkdownFile,
    Note,
    NoteTypeConfig,
)

logger = logging.getLogger(__name__)

_NOTE_KEY_PATTERN = re.compile(r"<!--\s*note_key:\s*([a-zA-Z0-9-]+)\s*-->")
_CODE_FENCE_PATTERN = re.compile(r"^(```|~~~)")


class FileSystemAdapter:
    def __init__(self):
        self._md_to_html = MarkdownToHTML()
        self._html_to_md = HTMLToMarkdown()
        self._note_type_configs: list[NoteTypeConfig] = []
        self._note_type_cache_signature: tuple | None = None

    def set_configs(self, configs: list[NoteTypeConfig]):
        self._note_type_configs = configs
        # External callers can override configs in-memory; invalidate disk cache.
        self._note_type_cache_signature = None

    def read_markdown_file(self, file_path: Path) -> MarkdownFile:
        raw_content = file_path.read_text(encoding="utf-8")
        remaining = raw_content

        blocks = remaining.split(NOTE_SEPARATOR)
        parsed_notes = []

        # Build prefix map
        prefix_to_type_field = {}
        for config in self._note_type_configs:
            for f in config.fields:
                if f.prefix:
                    prefix_to_type_field[f.prefix] = f.name

        for block in blocks:
            if not block.strip() or set(block.strip()) <= {"-"}:
                continue

            lines = block.strip().split("\n")
            note_key: str | None = None
            fields: dict[str, str] = {}
            current_field: str | None = None
            current_content: list[str] = []
            in_code_block = False
            seen: set[str] = set()

            for line in lines:
                stripped = line.lstrip()
                if _CODE_FENCE_PATTERN.match(stripped):
                    in_code_block = not in_code_block
                    if current_field:
                        current_content.append(line)
                    continue

                key_match = _NOTE_KEY_PATTERN.match(line)
                if key_match:
                    note_key = key_match.group(1)
                    continue

                if in_code_block:
                    if current_field:
                        current_content.append(line)
                    continue

                matched_field = None
                for prefix, field_name in prefix_to_type_field.items():
                    if line.startswith(prefix + " ") or line == prefix:
                        if field_name in seen:
                            ctx = (
                                f"in note_key: {note_key}"
                                if note_key
                                else "in this note"
                            )
                            raise ValueError(f"Duplicate field '{prefix}' {ctx}.")

                        seen.add(field_name)
                        if current_field:
                            fields[current_field] = "\n".join(current_content).strip()

                        matched_field = field_name
                        current_content = (
                            [line[len(prefix) + 1 :]]
                            if line.startswith(prefix + " ")
                            else []
                        )
                        current_field = field_name
                        break

                if matched_field is None and current_field:
                    current_content.append(line)

            if current_field:
                fields[current_field] = "\n".join(current_content).strip()

            if not fields:
                raise ValueError(
                    "Found content but no valid field prefixes in block starting "
                    f"with: '{block.strip()[:50]}...'"
                )

            # Infer Note Type
            note_type = self._infer_note_type(fields)
            note = Note(note_key=note_key, note_type=note_type, fields=fields)

            # Validate
            cfg = next(c for c in self._note_type_configs if c.name == note_type)
            errors = note.validate(cfg)
            if errors:
                raise ValueError("Invalid note in block:\n  " + "\n  ".join(errors))

            parsed_notes.append(note)

        return MarkdownFile(
            file_path=file_path,
            raw_content=raw_content,
            notes=parsed_notes,
        )

    def _infer_note_type(self, fields: dict[str, str]) -> str:
        reserved_names = {ANKIOPS_KEY_FIELD.name}
        note_fields = {key for key in fields.keys() if key not in reserved_names}

        candidates = []
        for config in self._note_type_configs:
            type_all_fields = {f.name for f in config.fields}
            if not note_fields.issubset(type_all_fields):
                continue

            type_ident_fields = {f.name for f in config.fields if f.identifying}

            if config.is_choice:
                base_ident = {f for f in type_ident_fields if "Choice" not in f}
                choice_fields = {f for f in type_all_fields if "Choice" in f}
                if base_ident.issubset(note_fields) and (note_fields & choice_fields):
                    candidates.append(config.name)
            else:
                if type_ident_fields.issubset(note_fields):
                    candidates.append(config.name)

        if not candidates:
            raise ValueError(
                "Cannot determine note type from fields: " + ", ".join(fields.keys())
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous note type: matches multiple types: {', '.join(candidates)}"
            )

        return candidates[0]

    def write_markdown_file(self, file_path: Path, content: str) -> None:
        file_path.write_text(content, encoding="utf-8")

    def find_markdown_files(self, directory: Path) -> list[Path]:
        return sorted(directory.glob("*.md"))

    def _resolve_note_type_dirs(self, note_types_dir: Path) -> list[Path]:
        directories_to_scan = []
        builtin_dir = Path(__file__).parent / "note_types"
        if builtin_dir.exists():
            directories_to_scan.append(builtin_dir)

        if note_types_dir.exists():
            try:
                if not builtin_dir.exists() or not note_types_dir.samefile(builtin_dir):
                    directories_to_scan.append(note_types_dir)
            except Exception:
                directories_to_scan.append(note_types_dir)

        return directories_to_scan

    def _note_type_signature(self, note_types_dir: Path) -> tuple:
        """Build a filesystem signature to invalidate note type cache on changes."""
        signature_parts: list[tuple[str, str, int, int]] = []
        for target_dir in self._resolve_note_type_dirs(note_types_dir):
            signature_parts.append(("dir", str(target_dir.resolve()), 0, 0))
            for subdir in sorted(target_dir.iterdir(), key=lambda p: p.name):
                if not subdir.is_dir():
                    continue
                signature_parts.append(
                    ("subdir", str(subdir.relative_to(target_dir)), 0, 0)
                )
                for file_path in sorted(subdir.rglob("*"), key=lambda p: str(p)):
                    if not file_path.is_file():
                        continue
                    stat = file_path.stat()
                    signature_parts.append(
                        (
                            "file",
                            str(file_path.relative_to(target_dir)),
                            stat.st_mtime_ns,
                            stat.st_size,
                        )
                    )
        return tuple(signature_parts)

    def load_note_type_configs(self, note_types_dir: Path) -> list[NoteTypeConfig]:
        configs_dict: dict[str, NoteTypeConfig] = {}
        signature = self._note_type_signature(note_types_dir)
        if self._note_type_configs and self._note_type_cache_signature == signature:
            return self._note_type_configs

        directories_to_scan = self._resolve_note_type_dirs(note_types_dir)

        # Builtin & folder-based
        for target_dir in directories_to_scan:
            for subdir in target_dir.iterdir():
                if not subdir.is_dir():
                    continue

                config_path = subdir / "note_type.yaml"
                if not config_path.exists():
                    continue

                with open(config_path, "r", encoding="utf-8") as file:
                    info = yaml.safe_load(file) or {}

                name = info.get("name", subdir.name)

                is_builtin = name.startswith("AnkiOps")
                if not is_builtin and "styling" not in info:
                    raise ValueError(
                        f"Note type '{name}' is missing mandatory 'styling' key "
                        "in note_type.yaml"
                    )

                fields = []
                for f in info.get("fields", []):
                    fields.append(
                        Field(f["name"], f["prefix"], identifying=f["identifying"])
                    )

                if ANKIOPS_KEY_FIELD.name not in [f.name for f in fields]:
                    fields.append(ANKIOPS_KEY_FIELD)

                # CSS
                styling_input = info.get("styling", [])
                if isinstance(styling_input, str):
                    styling_input = [styling_input]

                css_parts = []
                for p in styling_input:
                    p_path = subdir / p
                    if p_path.exists():
                        css_parts.append(p_path.read_text(encoding="utf-8"))
                css = "\n\n/* --- Added by Local Override --- */\n\n".join(css_parts)

                # Templates
                raw_templates = info.get("templates", [])
                templates = []

                if not raw_templates:
                    # Implicit loading
                    front = subdir / "Front.template.anki"
                    back = subdir / "Back.template.anki"
                    if front.exists() and back.exists():
                        name_card1 = "Cloze" if info.get("is_cloze") else "Card 1"
                        templates.append(
                            {
                                "Name": name_card1,
                                "Front": front.read_text(encoding="utf-8"),
                                "Back": back.read_text(encoding="utf-8"),
                            }
                        )

                        i = 2
                        while True:
                            front_n = subdir / f"Front{i}.template.anki"
                            back_n = subdir / f"Back{i}.template.anki"
                            if front_n.exists() and back_n.exists():
                                templates.append(
                                    {
                                        "Name": f"Card {i}",
                                        "Front": front_n.read_text(encoding="utf-8"),
                                        "Back": back_n.read_text(encoding="utf-8"),
                                    }
                                )
                                i += 1
                            else:
                                break
                else:
                    for t in raw_templates:
                        front = subdir / t["front"]
                        back = subdir / t["back"]
                        templates.append(
                            {
                                "Name": str(t.get("name", "Card")),
                                "Front": front.read_text(encoding="utf-8")
                                if front.exists()
                                else "",
                                "Back": back.read_text(encoding="utf-8")
                                if back.exists()
                                else "",
                            }
                        )

                configs_dict[name] = NoteTypeConfig(
                    name=name,
                    fields=fields,
                    css=css,
                    is_cloze=bool(info.get("is_cloze", False)),
                    is_choice=bool(info.get("is_choice", False)),
                    templates=templates,
                )

        configs = list(configs_dict.values())
        NoteTypeConfig.validate_configs(configs)
        self._note_type_configs = configs
        self._note_type_cache_signature = signature
        return configs

    def calculate_blake3(self, file_path: Path) -> str:
        h = blake3()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest(length=4)

    def update_media_references(
        self, directory: Path, rename_map: dict[str, str]
    ) -> int:

        if not rename_map:
            return 0

        updated_files = 0
        # Markdown image links, HTML src attributes, and Anki sound refs.
        pattern = re.compile(
            r'(!\[.*?\]\()(?:<(.+?)>|([^()]+(?:\([^()]*\)[^()]*)*))(\)(?:\{[^}]*\})?)'
            r'|(src=["\'])(.+?)(["\'])'
            r'|(\[sound:)(.+?)(\])'
        )

        def replace_callback(match: re.Match) -> str:
            # Determine which pattern matched
            if match.group(1) is not None:
                prefix = match.group(1)
                # Path can be in group 2 (angle brackets) or group 3 (plain).
                path = match.group(2) or match.group(3)
                suffix = match.group(4)
                is_markdown = True
            elif match.group(5) is not None:
                prefix, path, suffix = match.group(5), match.group(6), match.group(7)
                is_markdown = False
            else:
                prefix, path, suffix = match.group(8), match.group(9), match.group(10)
                is_markdown = False

            decoded_path = unquote(path)
            lookup_path = decoded_path.strip("<>").replace("\\", "/")

            if lookup_path in rename_map:
                new_path = rename_map[lookup_path]
                if is_markdown:
                    # Wrap markdown links in angle brackets for safer parsing.
                    if not new_path.startswith("<"):
                        new_path = f"<{new_path}>"
                return f"{prefix}{new_path}{suffix}"
            return match.group(0)

        for md_file in directory.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            new_content = pattern.sub(replace_callback, content)
            if new_content != content:
                md_file.write_text(new_content, encoding="utf-8")
                updated_files += 1

        return updated_files

    def convert_to_html(self, markdown_text: str) -> str:
        return self._md_to_html.convert(markdown_text)

    def convert_to_markdown(self, html_text: str) -> str:
        return self._html_to_md.convert(html_text)

    def eject_builtin_note_types(self, dst_dir: Path) -> None:
        """Copy built-in note type definitions to the filesystem."""
        src_root = resources.files("ankiops.note_types")
        with resources.as_file(src_root) as src_path:
            shutil.copytree(
                src_path,
                dst_dir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__init__.py", "__pycache__", "*.pyc"),
            )
        logger.debug(f"Ejected built-in note types to {dst_dir}")
