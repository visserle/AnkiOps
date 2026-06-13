# AnkiOps

[![Tests](https://github.com/visserle/AnkiOps/actions/workflows/test.yml/badge.svg)](https://github.com/visserle/AnkiOps/actions/workflows/test.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://img.shields.io/pypi/v/ankiops.svg)](https://pypi.org/project/ankiops/) 

AnkiOps is a bidirectional Anki ↔ Markdown bridge where each deck becomes a Markdown file. Edit in plain text, version with Git, enhance with LLMs, and sync changes both ways. 

## Features

- **Full Anki support**: Safe, performant bidirectional syncing of notes with custom note types, (sub-)decks, and media files
- **Markdown-first**: Edit in your favourite editor and render Markdown features in Anki (including syntax highlighting)
- **Simple CLI interface** for importing and exporting between Anki and the filesystem
- **LLM-integration**: Run programmable tasks such as content review, grammar fixes, or translations on your collection
- **GitHub shared**: Create shared deck sources on GitHub, add shared decks, update them, and submit changes through PRs (experimental)

> [!NOTE]
> AnkiOps only acts on note types defined within the `note_types/` folder. You can add note types from Anki using various methods described below.

<!-- ## Example

The following Markdown file is a valid, new Anki deck:

```markdown
Q: How fast do actions potentials propagate in the human body?


![](img_hash.png){width=700}
```

With the first import, AnkiOps adds a stable `note_key` comment above each note for tracking.

same for note type:Inference is only for fresh notes that do not yet have metadata. -->

## Installation


1. **Install AnkiOps via [pipx](https://github.com/pypa/pipx)**: Pipx makes AnkiOps available in your terminal.

```bash
pipx install ankiops
```

2. **Initialize AnkiOps**: Make sure that Anki is running, with AnkiOpsConnect enabled. (For shared operations, the AnkiOps addon must be manually installed which is currently experimental.) Run `ankiops init` in an empty collection directory. AnkiOps creates `.ankiops.db`, `llm/`, and `note_types/`. The tutorial flag also creates a sample Markdown deck.

```bash
ankiops init --tutorial
```



3. **Import Markdown into Anki**: Import the current collection of Markdown decks into Anki.

```bash
ankiops ma # alias for markdown-to-anki (import)
```

4. **Sync Anki back to Markdown**: After you review or edit cards in Anki, export those changes. Each sync makes one side match the other.

```bash
ankiops am # alias for anki-to-markdown (export)
```

## FAQ

### How is this different from other Markdown tools?

Most Markdown-to-Anki tools import one way: you write Markdown and push it to Anki. AnkiOps lets you edit in either place and sync back. It stores each deck as one Markdown file, so you browse decks instead of hundreds of per-card files. It also keeps custom note type definitions beside your decks, which lets you edit both card content and card structure from your editor.

### Is it safe to use?

Yes, AnkiOps will never modify notes that are not defined within the `note_types/` folder. Your existing collection won't be affected and you can safely mix managed and unmanaged notes within one deck. Further, AnkiOps only syncs if the activated profiles matches the one it was initialized with. Concerning the Markdown files, AnkiOps automatically creates a Git commit of your collection folder before every sync, so you can always roll your files back if needed.

### How do I create new notes?

Create a new Markdown file in your initialized AnkiOps folder. On the first import, AnkiOps uses the file name as the deck name. Subdecks use `__` (for example, `Anatomy::Heart` maps to `Anatomy__Heart.md`). Separate notes with a blank line, three dashes `---`, and another blank line. You can add new notes anywhere in an existing file.

```markdown
<!-- note_key: 123487556abc -->
<!-- note_type: AnkiOpsQA -->
<!-- tags: AnkiOps -->
Q: Question text here
A: Answer text here
E: Extra information (optional)
M: Content behind a "more" button (optional)

---

<!-- note_key: 123474567def -->
<!-- note_type: AnkiOpsCloze -->
T: Text with the {{c1::first}} cloze.
Text with the {{c1::second}} cloze.
E: Some *formatted* extra info.

![image with set width](im.png){width=700}

---

Q: What is this?
C1: A multiple choice note
C2: with
C3: randomized answers.
A: 1, 3
```

On the next import, AnkiOps assigns a `note_key` and `note_type` comment to the last note.

### How are different note types handled?

AnkiOps reads note types exclusively from your local `note_types/` directory. `ankiops init` ejects default note types as bootstrap files; those local files are then the only source of truth and can be modified as needed. Each note type is identified by a unique set of field labels. These labels are defined in`note_types/name/note_type.yaml` and can be customized as needed. The set operations of each unique note type are defined by the `identifying` fields in the yamls. For an overview of the current configuration, use `ankiops note-types`.

### How can I migrate my existing notes into AnkiOps?

You can migrate existing notes in three ways: write matching note type files in `note_types/`, copy note types from Anki with `ankiops note-types --add <name>`, or convert notes to the default AnkiOps note types with `Change Note Type…` in the Anki browser.

For Anki browser conversion, use this workflow:

1. Convert your existing notes to the matching AnkiOps note types via `Change Note Type…` in the Anki browser.
2. Export your notes from Anki to Markdown using `ankiops am`.
3. Review the Git diff after the first re-import. Original Anki HTML may not match CommonMark, so the first Markdown-to-Anki sync can change formatting. Formatting issues can be fixed by hand or via LLM tasks.

### How does it work?

AnkiOps assigns a stable `note_key` to each managed note. In Markdown, it writes the key as a single-line HTML comment above the note, such as `<!-- note_key: a1b2c3d4e5f6 -->`. AnkiOps also writes a derived `note_type` comment, such as `<!-- note_type: AnkiOpsQA -->`, so you can see the resolved type in the file. Note types are inferred by sets of identifying fields (e.g. `Q:`, `A:`), defined in the note type folder. AnkiOps note keys do not depend on Anki's note IDs. The `.ankiops.db` database maps Anki note IDs to AnkiOps note keys and stores sync metadata. During sync, AnkiOps uses note keys to decide which notes to create, update, or delete. It stores media in `media/` with hashed file names to avoid name conflicts.

### What is the recommended workflow?

We recommend VS Code because it previews Markdown and can paste clipboard images into the `media/` folder that AnkiOps creates during init.

### How can I share my AnkiOps collection? (experimental)

Use `ankiops shared` from a Git-backed collection. AnkiOps stores each shared source at `shared/<owner>/<repo>` and pulls from `https://github.com/<owner>/<repo>.git` on the `main` branch. AnkiOps scopes note types from that source as `shared/<owner>/<repo>/<note_type>`, so two shared decks can use the same local note type names without colliding.

To create a shared source from one of your decks:

```bash
ankiops shared create "Deck Name" owner/repo
```

`create` requires a clean Git index and `note_key` metadata on every selected note. It includes the selected deck and its subdecks, copies referenced media and used note types into the shared source, commits the move, and pushes the subtree to GitHub. If the GitHub repository does not exist and the `gh` CLI is available, AnkiOps creates a private repo by default. Pass `--public` to create a public repo.

To use a shared deck:

```bash
ankiops shared add owner/repo
ankiops shared update owner/repo --to-anki
```

`add` adds the GitHub repository as a subtree. `update` refreshes one source, or all sources when you omit `owner/repo`. `--to-anki` runs Markdown-to-Anki after the update.

To submit local edits back:

```bash
ankiops shared submit owner/repo --from-anki
```

`--from-anki` exports Anki edits to Markdown first. `submit` then commits the shared source, creates a subtree branch, and opens a pull request with `gh` when possible. Without `gh`, AnkiOps leaves you with the branch name and GitHub remote so you can push and open the PR yourself.

### How do I upgrade AnkiOps to the latest version?

AnkiOps is in early development, so breaking changes are expected. Use `pipx upgrade ankiops` to upgrade AnkiOps. Delete local AnkiOps support files except your Markdown decks, re-initialize the same folder with `ankiops init`, and export from Anki with `ankiops am`. If your collection is in Git, inspect the diff before you continue syncing.

### What is the Add-on for?

The Add-on has two purposes. It adds `am` and `ma` buttons to the Anki toolbar for quick sync andit implements AnkiOpsConnect, a rewrite of AnkiConnect that enables operations for the shared features (mainly the conversion of note types without losing schedule information). The Add-on is still experimental and not available on AnkiWeb yet. To install it, download the folder and put it in your Anki add-ons directory.

### How can I develop AnkiOps locally?

Fork this repository and initialize the tutorial from the repository root while Anki is running. The commands below create `collection/` with the sample Markdown deck and run the CLI from source.

```bash
git clone https://github.com/visserle/ankiops.git
cd ankiops
uv sync
uv run python -m main init --tutorial
uv run python -m main ma
```

### Are Pull Requests welcome?

Yes. Bug fixes, feature work, and documentation PRs are welcome. Open an issue first for behavior changes.

### What commands and flags are available in the CLI?

**Global:**
- `--debug` - Enable debug logging
- `--version` - Show installed AnkiOps version
- `--help` - Show help message

**`init`:**
- `--tutorial` - Create tutorial markdown file

**`anki-to-markdown` / `am`:**
- `--no-auto-commit`, `-n` - Skip automatic git commit

**`markdown-to-anki` / `ma`:**
- `--no-auto-commit`, `-n` - Skip automatic git commit

**`note-types`:**
- `ankiops note-types` - Show note types, identifying labels, and the label registry
- `ankiops note-types --add <name>` - Copy a note type from Anki into local `note_types/` with interactive label/identifying prompts

**`serialize`:**
- `--output`, `-o` - Output file path (default: `AnkiCollection.json`, `<deck-stem>.json` when `--deck` is set)
- `--deck` - Serialize only one deck (includes subdecks by default)
- `--no-subdecks` - With `--deck`, exclude subdecks (exact deck only)

**`deserialize`:**
- `--input`, `-i` - Input file path (default: `AnkiCollection.json`)
- `--overwrite` - Overwrite existing markdown files

**`fix-image-widths`:**
- `--deck` - Fix only one deck (includes subdecks by default)
- `--no-subdecks` - With `--deck`, exclude subdecks (exact deck only)
- `--tolerance` - Pixel tolerance for near-equal width fixes (default: `5`)
- `--width` - Force all Markdown images in scope to this width
- `--no-auto-commit`, `-n` - Skip automatic git commit

**`llm`:**
- `ankiops llm` - Show LLM status dashboard (tasks + recent jobs)
- `ankiops llm <task_name> [--model <model>] [--deck <deck_name>]` - Plan one configured task
- `ankiops llm <task_name> --run [--model <model>] [--deck <deck_name>] [--no-auto-commit]` - Run one configured task job
- `ankiops llm --job <job_id|latest>` - Show one LLM job in detail

**`shared`:**
- `ankiops shared create <deck> <owner>/<repo> [--public|--private]` - Create a GitHub shared source from a local deck tree
- `ankiops shared add <owner>/<repo>` - Add a GitHub shared source to the collection
- `ankiops shared update [owner/repo] [--to-anki]` - Update one shared source, or all sources when omitted
- `ankiops shared submit <owner>/<repo> [--from-anki]` - Prepare a submission branch and PR for local shared edits
- `ankiops shared list` - Show known shared sources
