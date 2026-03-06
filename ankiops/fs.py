"""FileSystem Adapter for Reading/Writing Markdown, Configs, and Media."""

import logging
import re
import shutil
from importlib import resources
from pathlib import Path
from urllib.parse import unquote

import yaml
from blake3 import blake3

from ankiops import anki
from ankiops.config import LOCAL_MEDIA_DIR, NOTE_SEPARATOR
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
_LABEL_CANDIDATE_PATTERN = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*:)(?:\s|$)")
_RESERVED_NOTE_FIELD_NAMES = frozenset({ANKIOPS_KEY_FIELD.name})


class FileSystemAdapter:
    def __init__(self):
        self._md_to_html = MarkdownToHTML()
        self._html_to_md = HTMLToMarkdown()
        self._note_type_configs: list[NoteTypeConfig] = []
        self._label_to_field: dict[str, str] = {}
        self._note_type_cache_signature: tuple | None = None
        self._inferred_note_type_by_signature: dict[
            tuple[frozenset[str], frozenset[str] | None], str
        ] = {}

    def set_configs(self, configs: list[NoteTypeConfig]):
        self._note_type_configs = configs
        self._label_to_field = self._build_label_to_field_map(configs)
        # External callers can override configs in-memory; invalidate disk cache.
        self._note_type_cache_signature = None
        self._inferred_note_type_by_signature = {}

    def _display_parse_error_path(
        self,
        file_path: Path,
        *,
        context_root: Path | None,
    ) -> str:
        if context_root is None:
            return file_path.name

        try:
            return str(file_path.resolve().relative_to(context_root.resolve()))
        except Exception:
            return file_path.name

    def _with_parse_error_context(
        self,
        message: str,
        *,
        display_path: str,
        line_number: int,
    ) -> str:
        return f"{message} (file: {display_path}, line: {line_number})"

    def read_markdown_file(
        self,
        file_path: Path,
        *,
        context_root: Path | None = None,
    ) -> MarkdownFile:
        raw_content = file_path.read_text(encoding="utf-8")
        parsed_notes = []
        config_by_name = {config.name: config for config in self._note_type_configs}
        display_path = self._display_parse_error_path(
            file_path, context_root=context_root
        )
        separator_newlines = NOTE_SEPARATOR.count("\n")
        block_start = 0
        block_start_line = 1

        while True:
            separator_index = raw_content.find(NOTE_SEPARATOR, block_start)
            if separator_index == -1:
                block = raw_content[block_start:]
                next_block_start = -1
            else:
                block = raw_content[block_start:separator_index]
                next_block_start = separator_index + len(NOTE_SEPARATOR)

            stripped_block = block.strip()
            if not stripped_block or not stripped_block.replace("-", ""):
                if next_block_start == -1:
                    break
                block_start_line += block.count("\n") + separator_newlines
                block_start = next_block_start
                continue

            leading_trimmed_len = len(block) - len(block.lstrip())
            leading_line_offset = block[:leading_trimmed_len].count("\n")
            note_start_line = block_start_line + leading_line_offset

            lines = stripped_block.split("\n")
            note_key: str | None = None
            fields: dict[str, str] = {}
            current_field: str | None = None
            current_content: list[str] = []
            in_code_block = False
            seen: set[str] = set()
            seen_labels: set[str] = set()
            first_field_line: int | None = None

            for offset, line in enumerate(lines):
                current_line_number = note_start_line + offset
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
                for label, field_name in self._label_to_field.items():
                    if line.startswith(label + " ") or line == label:
                        if field_name in seen:
                            raise ValueError(
                                self._with_parse_error_context(
                                    f"Duplicate field '{label}'.",
                                    display_path=display_path,
                                    line_number=current_line_number,
                                )
                            )

                        seen.add(field_name)
                        seen_labels.add(label)
                        if current_field:
                            fields[current_field] = "\n".join(current_content).strip()

                        matched_field = field_name
                        current_content = (
                            [line[len(label) + 1 :]]
                            if line.startswith(label + " ")
                            else []
                        )
                        current_field = field_name
                        if first_field_line is None:
                            first_field_line = current_line_number
                        break

                if matched_field is None and current_field:
                    current_content.append(line)
                elif matched_field is None:
                    label_match = _LABEL_CANDIDATE_PATTERN.match(stripped)
                    if label_match:
                        unknown_label = label_match.group(1)
                        if unknown_label not in self._label_to_field:
                            raise ValueError(
                                self._with_parse_error_context(
                                    (
                                        f"Unknown field label '{unknown_label}'. "
                                        "Please check your note type labels. "
                                        "Use `ankiops note-type --info` to list defined"
                                        " labels."
                                    ),
                                    display_path=display_path,
                                    line_number=current_line_number,
                                )
                            )

            if current_field:
                fields[current_field] = "\n".join(current_content).strip()

            if not fields:
                raise ValueError(
                    self._with_parse_error_context(
                        (
                            "Found content but no valid field labels in block "
                            f"starting with: '{block.strip()[:50]}...'"
                        ),
                        display_path=display_path,
                        line_number=note_start_line,
                    )
                )

            # Infer Note Type
            try:
                note_type = self._infer_note_type(fields, labels=seen_labels)
            except ValueError as error:
                raise ValueError(
                    self._with_parse_error_context(
                        str(error),
                        display_path=display_path,
                        line_number=first_field_line or note_start_line,
                    )
                ) from error
            note = Note(note_key=note_key, note_type=note_type, fields=fields)

            # Validate
            note_type_config = config_by_name[note_type]
            errors = note.validate(note_type_config)
            if errors:
                raise ValueError(
                    self._with_parse_error_context(
                        "Invalid note in block:\n  " + "\n  ".join(errors),
                        display_path=display_path,
                        line_number=first_field_line or note_start_line,
                    )
                )

            parsed_notes.append(note)

            if next_block_start == -1:
                break
            block_start_line += block.count("\n") + separator_newlines
            block_start = next_block_start

        return MarkdownFile(
            file_path=file_path,
            raw_content=raw_content,
            notes=parsed_notes,
        )

    def _infer_note_type(
        self,
        fields: dict[str, str],
        *,
        labels: set[str] | None = None,
    ) -> str:
        note_fields_frozen = frozenset(
            key for key in fields.keys() if key not in _RESERVED_NOTE_FIELD_NAMES
        )
        note_labels_frozen = frozenset(labels) if labels is not None else None
        signature = (note_fields_frozen, note_labels_frozen)
        cached = self._inferred_note_type_by_signature.get(signature)
        if cached is not None:
            return cached

        note_fields = set(note_fields_frozen)
        note_labels = set(note_labels_frozen) if note_labels_frozen else None
        candidates: list[str] = []

        for config in self._note_type_configs:
            type_all_fields = {field.name for field in config.fields}
            if not note_fields.issubset(type_all_fields):
                continue

            if note_labels is not None:
                type_all_labels = {
                    str(field.label)
                    for field in config.fields
                    if field.label is not None
                }
                if not note_labels.issubset(type_all_labels):
                    continue

            type_ident_fields = {
                field.name for field in config.fields if field.identifying
            }

            if config.is_choice:
                base_ident_fields = {
                    field.name
                    for field in config.fields
                    if field.identifying and "Choice" not in field.name
                }
                choice_fields = {
                    field.name for field in config.fields if "Choice" in field.name
                }
                if not base_ident_fields.issubset(note_fields):
                    continue
                if not (note_fields & choice_fields):
                    continue

                if note_labels is not None:
                    base_ident_labels = {
                        str(field.label)
                        for field in config.fields
                        if (
                            field.identifying
                            and field.label is not None
                            and "Choice" not in field.name
                        )
                    }
                    choice_labels = {
                        str(field.label)
                        for field in config.fields
                        if field.label is not None and "Choice" in field.name
                    }
                    if not base_ident_labels.issubset(note_labels):
                        continue
                    if not (note_labels & choice_labels):
                        continue
            else:
                if not type_ident_fields.issubset(note_fields):
                    continue
                if note_labels is not None:
                    type_ident_labels = {
                        str(field.label)
                        for field in config.fields
                        if field.identifying and field.label is not None
                    }
                    if not type_ident_labels.issubset(note_labels):
                        continue

            candidates.append(config.name)

        if not candidates:
            raise ValueError(
                "Cannot determine note type from fields: " + ", ".join(fields.keys())
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous note type: matches multiple types: {', '.join(candidates)}"
            )

        resolved = candidates[0]
        self._inferred_note_type_by_signature[signature] = resolved
        return resolved

    def write_markdown_file(self, file_path: Path, content: str) -> None:
        file_path.write_text(content, encoding="utf-8")

    def find_markdown_files(self, directory: Path) -> list[Path]:
        return sorted(directory.glob("*.md"))

    def _note_type_signature(self, note_types_dir: Path) -> tuple:
        """Build a filesystem signature to invalidate note type cache on changes."""
        if not note_types_dir.exists() or not note_types_dir.is_dir():
            return (("missing", str(note_types_dir.resolve()), 0, 0),)

        signature_parts: list[tuple[str, str, int, int]] = []
        target_dir = note_types_dir
        signature_parts.append(("dir", str(target_dir.resolve()), 0, 0))
        for subdir in sorted(
            target_dir.iterdir(), key=lambda path_entry: path_entry.name
        ):
            if not subdir.is_dir():
                continue
            signature_parts.append(
                ("subdir", str(subdir.relative_to(target_dir)), 0, 0)
            )
            for file_path in sorted(
                subdir.rglob("*"), key=lambda path_entry: str(path_entry)
            ):
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
        if not note_types_dir.exists() or not note_types_dir.is_dir():
            raise ValueError(
                f"Note types directory not found: {note_types_dir}. "
                "Initialize or create local note_types definitions first."
            )

        configs_dict: dict[str, NoteTypeConfig] = {}
        signature = self._note_type_signature(note_types_dir)
        if self._note_type_configs and self._note_type_cache_signature == signature:
            return self._note_type_configs

        for subdir in sorted(
            note_types_dir.iterdir(), key=lambda path_entry: path_entry.name
        ):
            if not subdir.is_dir():
                continue

            config_path = subdir / "note_type.yaml"
            if not config_path.exists():
                raise ValueError(
                    f"Note type directory '{subdir.name}' is missing note_type.yaml."
                )

            with open(config_path, "r", encoding="utf-8") as file:
                info = yaml.safe_load(file) or {}
            if not isinstance(info, dict):
                raise ValueError(
                    f"Note type '{subdir.name}' config must be a YAML mapping."
                )

            name = subdir.name
            if "name" in info:
                raise ValueError(
                    f"Note type '{name}' must not define 'name' in note_type.yaml. "
                    "Use the directory name as the note type name."
                )

            if "styling" not in info:
                raise ValueError(
                    f"Note type '{name}' is missing mandatory 'styling' key "
                    "in note_type.yaml"
                )

            fields = []
            for field_data in info.get("fields", []):
                fields.append(
                    Field(
                        field_data["name"],
                        field_data["label"],
                        identifying=field_data["identifying"],
                    )
                )

            if ANKIOPS_KEY_FIELD.name not in [
                field_config.name for field_config in fields
            ]:
                fields.append(ANKIOPS_KEY_FIELD)

            # CSS
            styling_input = info.get("styling")
            if isinstance(styling_input, str):
                styling_files = [styling_input]
            elif isinstance(styling_input, list):
                styling_files = []
                for css_file in styling_input:
                    if not isinstance(css_file, str) or not css_file.strip():
                        raise ValueError(
                            f"Note type '{name}' has invalid styling entry "
                            f"'{css_file}'. Expected non-empty file names."
                        )
                    styling_files.append(css_file.strip())
            else:
                raise ValueError(
                    f"Note type '{name}' has invalid 'styling' value. "
                    "Expected a string or list of strings."
                )
            if not styling_files:
                raise ValueError(
                    f"Note type '{name}' must reference at least one styling file."
                )

            css_parts = []
            for css_file in styling_files:
                css_path = subdir / css_file
                if not css_path.exists() or not css_path.is_file():
                    raise ValueError(
                        f"Note type '{name}' references missing styling file "
                        f"'{css_file}'."
                    )
                css_parts.append(css_path.read_text(encoding="utf-8"))
            css = "\n\n/* --- Added by Local Override --- */\n\n".join(css_parts)

            # Templates
            raw_templates = info.get("templates")
            templates = []

            if raw_templates is None:
                # Implicit loading
                front = subdir / "Front.template.anki"
                back = subdir / "Back.template.anki"
                if not front.exists() or not back.exists():
                    missing = []
                    if not front.exists():
                        missing.append("Front.template.anki")
                    if not back.exists():
                        missing.append("Back.template.anki")
                    missing_text = ", ".join(missing)
                    raise ValueError(
                        f"Note type '{name}' is missing template file(s): "
                        f"{missing_text}."
                    )

                name_card1 = "Cloze" if info.get("is_cloze") else "Card 1"
                templates.append(
                    {
                        "Name": name_card1,
                        "Front": front.read_text(encoding="utf-8"),
                        "Back": back.read_text(encoding="utf-8"),
                    }
                )

                template_index = 2
                while True:
                    front_n = subdir / f"Front{template_index}.template.anki"
                    back_n = subdir / f"Back{template_index}.template.anki"
                    has_front = front_n.exists()
                    has_back = back_n.exists()
                    if has_front and has_back:
                        templates.append(
                            {
                                "Name": f"Card {template_index}",
                                "Front": front_n.read_text(encoding="utf-8"),
                                "Back": back_n.read_text(encoding="utf-8"),
                            }
                        )
                        template_index += 1
                    elif has_front != has_back:
                        missing_file = (
                            f"Back{template_index}.template.anki"
                            if has_front
                            else f"Front{template_index}.template.anki"
                        )
                        raise ValueError(
                            f"Note type '{name}' has incomplete template pair for "
                            f"Card {template_index}; missing '{missing_file}'."
                        )
                    else:
                        break
            else:
                if not isinstance(raw_templates, list) or not raw_templates:
                    raise ValueError(
                        f"Note type '{name}' must define a non-empty 'templates' list."
                    )
                for template_data in raw_templates:
                    if not isinstance(template_data, dict):
                        raise ValueError(
                            f"Note type '{name}' has invalid template entry "
                            f"'{template_data}'. Expected a mapping."
                        )

                    front_ref = template_data.get("front")
                    back_ref = template_data.get("back")
                    if not isinstance(front_ref, str) or not front_ref.strip():
                        raise ValueError(
                            f"Note type '{name}' has template with invalid 'front'."
                        )
                    if not isinstance(back_ref, str) or not back_ref.strip():
                        raise ValueError(
                            f"Note type '{name}' has template with invalid 'back'."
                        )

                    front = subdir / front_ref.strip()
                    back = subdir / back_ref.strip()
                    if not front.exists() or not front.is_file():
                        raise ValueError(
                            f"Note type '{name}' references missing template file "
                            f"'{front_ref}'."
                        )
                    if not back.exists() or not back.is_file():
                        raise ValueError(
                            f"Note type '{name}' references missing template file "
                            f"'{back_ref}'."
                        )

                    templates.append(
                        {
                            "Name": str(template_data.get("name", "Card")),
                            "Front": front.read_text(encoding="utf-8"),
                            "Back": back.read_text(encoding="utf-8"),
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
        if not configs:
            raise ValueError(
                f"No note type definitions found in {note_types_dir}. "
                "Add at least one note type directory with note_type.yaml."
            )
        NoteTypeConfig.validate_configs(configs)
        self._note_type_configs = configs
        self._label_to_field = self._build_label_to_field_map(configs)
        self._note_type_cache_signature = signature
        self._inferred_note_type_by_signature = {}
        return configs

    def _build_label_to_field_map(
        self, configs: list[NoteTypeConfig]
    ) -> dict[str, str]:
        label_to_field: dict[str, str] = {}
        for config in configs:
            for field_config in config.fields:
                if field_config.label is None:
                    continue
                existing_field = label_to_field.get(field_config.label)
                if existing_field is not None and existing_field != field_config.name:
                    raise ValueError(
                        f"Label '{field_config.label}' maps to both "
                        f"'{existing_field}' and '{field_config.name}'."
                    )
                label_to_field[field_config.label] = field_config.name
        return label_to_field

    def calculate_blake3(self, file_path: Path) -> str:
        hash_state = blake3()
        with open(file_path, "rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(65536), b""):
                hash_state.update(chunk)
        return hash_state.hexdigest(length=4)

    def update_media_references(
        self, directory: Path, rename_map: dict[str, str]
    ) -> int:
        if not rename_map:
            return 0

        updated_files = 0
        # Markdown image links, HTML src attributes, and Anki sound refs.
        pattern = re.compile(
            r"(!\[.*?\]\()(?:<(.+?)>|([^()]+(?:\([^()]*\)[^()]*)*))(\)(?:\{[^}]*\})?)"
            r'|(src=["\'])(.+?)(["\'])'
            r"|(\[sound:)(.+?)(\])"
        )

        def replace_callback(match: re.Match) -> str:
            # Determine which pattern matched
            media_context = "other"
            if match.group(1) is not None:
                opener = match.group(1)
                # Path can be in group 2 (angle brackets) or group 3 (plain).
                path = match.group(2) or match.group(3)
                suffix = match.group(4)
                is_markdown = True
                media_context = "markdown"
            elif match.group(5) is not None:
                opener, path, suffix = match.group(5), match.group(6), match.group(7)
                is_markdown = False
                media_context = "html"
            else:
                opener, path, suffix = match.group(8), match.group(9), match.group(10)
                is_markdown = False
                media_context = "sound"

            decoded_path = unquote(path)
            lookup_path = decoded_path.strip("<>").replace("\\", "/")

            if lookup_path in rename_map:
                new_path = rename_map[lookup_path]
                # Anki [sound:...] references always use bare file names.
                if media_context == "sound" and new_path.startswith(
                    f"{LOCAL_MEDIA_DIR}/"
                ):
                    new_path = new_path[len(LOCAL_MEDIA_DIR) + 1 :]
                if is_markdown:
                    # Wrap markdown links in angle brackets for safer parsing.
                    if not new_path.startswith("<"):
                        new_path = f"<{new_path}>"
                return f"{opener}{new_path}{suffix}"
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
