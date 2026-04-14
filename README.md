# AnkiOps

[![Tests](https://github.com/visserle/AnkiOps/actions/workflows/test.yml/badge.svg)](https://github.com/visserle/AnkiOps/actions/workflows/test.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://img.shields.io/pypi/v/ankiops.svg)](https://pypi.org/project/ankiops/) 

AnkiOps is a bi-directional Anki ↔ Markdown bridge where each deck becomes a Markdown file. Edit in plain text, version with Git, enhance with LLMs, sync changes both ways. 

## Features

- **Full Anki support**: Safe, performant bidirectional syncing of notes with custom note types, (sub-)decks, and media files
- **Markdown-first**: Edit in your favourite editor; render Markdown features on Anki's desktop and mobile apps (including syntax highlighting)
- **Simple CLI interface**: After initialization, just two commands are needed for importing and exporting between Anki and your filesystem 
- **LLM-integration**: Serialize your collection to JSON for batch processing tasks such as content review, grammar fixes, or translations (wip)
- **Git-based collaboration**: Stable note keys allow for sharing decks via GitHub repositories with built-in sync commands (nyi)

> [!NOTE]
> AnkiOps only acts on note types defined within the `note_types/` folder. You can add note types from Anki using `ankiops note-types --add <name>`.

<!-- ## Example

The following Markdown file is a valid, new Anki deck:

```markdown
Q: How fast do actions potentials propagate in the human body?


![](img_hash.png){width=700}
```

With the first import, AnkiOps adds a stable `note_key` comment above each note for tracking. -->

## Installation


1. **Install AnkiOps via [pipx](https://github.com/pypa/pipx)**: Pipx will make AnkiOps globally available in your terminal.

```bash
pipx install ankiops
```

2. **Initialize AnkiOps**: Make sure that Anki is running, with the [AnkiConnect add-on](https://ankiweb.net/shared/info/2055492159) enabled. Initialize AnkiOps in any empty directory of your choosing. The command creates a database file for synchronization, an `llm/` directory for custom LLM tasks, and a `note_types/` directory for the note types AnkiOps will act on (following Infrastructure as Code principles). The additional tutorial flag creates a sample Markdown deck you can experiment with.

```bash
ankiops init --tutorial
```

3. **Execute AnkiOps**: Import the tutorial deck into Anki using:

```bash
ankiops ma # alias for markdown-to-anki (import)
```

4. **Keep everything in sync**: When editing your Markdown files, sync Markdown → Anki (and vice versa), as each sync makes one side match the other. After reviewing and editing your cards in Anki, you can sync Anki → Markdown using the following command:

```bash
ankiops am # alias for anki-to-markdown (export)
```

## FAQ

### How is this different from other Markdown or Obsidian tools?

Most available tools are one-way importers: you write in Markdown or Obsidian and push to Anki, but edits in Anki don't sync back. AnkiOps is bi-directional: you can edit in either Anki or Markdown and sync in both directions. It uses a one-file-per-deck structure, making your collection easier to navigate than approaches that use one file per card. Further, custom note types are supported while maintaining a clear working environment. This essentially lets you manage your entire Anki collection from your favorite text editor.

### Is it safe to use?

Yes, AnkiOps will never modify notes that are not defined within the `note_types/` folder. Your existing collection won't be affected and you can safely mix managed and unmanaged notes within one deck. Further, AnkiOps only syncs if the activated profiles matches the one it was initialized with. Concerning your Markdown files, AnkiOps automatically creates a Git commit of your collection folder before every sync, so you can always roll your files back if needed.

### How do I create new notes?

Create a new Markdown file in your initialized AnkiOps folder. For the first import, the file name acts as the deck name. Subdecks use `__` (for example, `Anatomy::Heart` --> `Anatomy__Heart.md`). Notes must be separated by a new line, three dashes `---`, and another new line. You can add new notes anywhere in an existing file.

```markdown
<!-- note_key: 123487556abc -->
Q: Question text here
A: Answer text here
E: Extra information (optional)
M: Content behind a "more" button (optional)

---

<!-- note_key: 123474567def -->
T: Text with {{c1::multiple}} {{c2::cloze deletions}}.
E: Some *formatted* extra info.

![image with set width](im.png){width=700}

---

Q: What is this?
C1: A multiple choice note
C2: with
C3: automatically randomized answers.
A: 1, 3
```

In this example, the last note is a new note which will get a `note_key` comment assigned on the next import.

### How are different note types handled?

AnkiOps reads note types exclusively from your local `note_types/` directory. `ankiops init` ejects default note types as bootstrap files; those local files are then the only source of truth and can be modified as needed. Each note type is identified by a unique set of field labels. These labels are defined in`note_types/name/note_type.yaml` and can be customized as needed. The set operations of each unique note type are defined by the `identifying` fields in the yamls. For an overview of the current configuration, use `ankiops note-types`.

### How can I migrate my existing notes into AnkiOps?

There are three ways to migrate your existing collection.
You can create new note types configuration files in the `note_types/` folder that match your existing note types in Anki by hand, use the `ankiops note-types --add <name>` command to copy note types from Anki, or convert your existing notes to the default AnkiOps note types using `Change Note Type…` in the Anki browser.

For the last option specifically, the recommended workflow is:

1. Convert your existing notes to the matching AnkiOps note types via `Change Note Type…` in the Anki browser.
2. Export your notes from Anki to Markdown using `ankiops am`.
3. In the first re-import, some formatting may change because the original HTML from Anki may not follow the CommonMark standard. Formatting of your cards can be done automatically at a low cost using the included JSON serializer and AI tooling.

### How does it work?

AnkiOps assigns a stable `note_key` to each managed note. It is represented by a single-line HTML tag (e.g., `<!-- note_key: a1b2c3d4e5f6 -->`) above a note in the Markdown. AnkiOps note keys are profile-independent, in contrast to Anki's note IDs. One AnkiOps folder represents one Anki profile. The `.ankiops.db`database stores the mapping between Anki's note IDs and AnkiOps note keys, along with other metadata. When syncing, AnkiOps uses these note keys to determine which notes to create, update, or delete in either Anki or Markdown. Media files are stored in a `media/` folder with hashed file names to avoid conflicts.

### What is the recommended workflow?

We recommend using VS Code. It has excellent AI integration, a native Markdown previewer, and supports image pasting from the clipboard directly into the `/media` folder (automatically set up).

### How can I share my AnkiOps collection?

TODO

### How do I upgrade AnkiOps to the latest version?

Use `pipx upgrade ankiops` to upgrade AnkiOps to the latest version. Delete all local AnkiOps files (except for your Markdown decks), re-initialize AnkiOps in the same folder using `ankiops init`, and sync from Anki via `ankiops am`. Since all files are git-tracked, you can easily spot any changes and roll back if needed.

### How can I develop AnkiOps locally?

Fork this repository and initialize the tutorial in your root folder (make sure Anki is running). This will create a folder called `collection` with the sample Markdown in it. Paths will adapt automatically to the development environment. You can run AnkiOps locally using the main script.

```bash
git clone https://github.com/visserle/ankiops.git
cd ankiops
uv sync
uv run python -m main init --tutorial
uv run python -m main ma
```

### Are Pull Requests welcome?

Yes! We welcome contributions of all kinds, including bug fixes, new features, documentation improvements, and more. Please open an issue or submit a pull request if you'd like to contribute.

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

**`serialize`:**
- `--output`, `-o` - Output file path (default: `<collection-name>.json`)
- `--deck` - Serialize only one deck (includes subdecks by default)
- `--no-subdecks` - With `--deck`, exclude subdecks (exact deck only)

**`deserialize`:**
- `--input`, `-i` - Input file path (default: `<collection-name>.json`)
- `--overwrite` - Overwrite existing markdown files

**`llm`:**
- `ankiops llm` - Show LLM status dashboard (tasks + recent jobs)
- `ankiops llm <task_name> [--model <model>] [--deck <deck_name>]` - Plan one configured task
- `ankiops llm <task_name> --run [--model <model>] [--deck <deck_name>] [--no-auto-commit]` - Run one configured task job
- `ankiops llm --job <job_id|latest>` - Show one LLM job in detail
- `ankiops llm --job <job_id|latest> --resume [--no-auto-commit]` - Resume unfinished/error items from a prior job

**`note-types`:**
- `ankiops note-types` - Show note types, identifying labels, and the label registry
- `ankiops note-types --add <name>` - Copy a note type from Anki into local `note_types/` with interactive label/identifying prompts

## LLM Integration (wip, experimental)

AnkiOps includes a LLM pipeline for repeatable task execution. 

> [!NOTE]
> LLM integration is still experimental and subject to change.

After `ankiops init`, AnkiOps bootstraps:

- `llm/models.yaml`
- `llm/system_prompt.md`
- `llm/grammar.yaml`
- `llm/translate.yaml`
- `llm/.llm.db` (job history, auto-added to `.gitignore`)

Set the key required by the model entry you use (from `llm/models.yaml`):

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

Plan, run, and inspect jobs:

```bash
ankiops llm                         # status dashboard (tasks + recent jobs)
ankiops llm grammar                 # dry-run plan
ankiops llm grammar --run           # run task job
ankiops llm grammar --deck Biology  # one exact deck (subdecks excluded)
ankiops llm grammar --run --model haiku
ankiops llm --job latest
ankiops llm --job latest --resume
```
### Task File Format (`llm/<task-name>.yaml`)

```yaml
model: sonnet
system_prompt: !file system_prompt.md
task_prompt: |
  Correct grammar, spelling, and punctuation in editable fields.
  Preserve meaning, Markdown structure, cloze syntax, code fences, math, and URLs.
  Do not add facts or change correctness.

fields:
  exceptions:
    - hidden: ["AI Notes"]
    - note_types: ["AnkiOpsChoice"]
      read_only: ["Answer"]
```

- Required keys: `model`, `system_prompt`, `task_prompt`
- `model` must reference a model name from `llm/models.yaml`
- `system_prompt` and `task_prompt` each accept either inline text or a YAML file tag (`!file <relative-path>`) resolved relative to the task file
- Default templates use `system_prompt: !file system_prompt.md`
- `fields.exceptions` is optional
- `fields.exceptions` controls per-note-type field access: `read_only` fields are sent for context but cannot be edited, while `hidden` fields are omitted from LLM input/output
- Without `--deck`, tasks run against the full collection; `--deck <name>` scopes to one exact deck
- Request/execution tuning uses internal defaults; only model can be overridden from CLI (`--model`)

Optional file-linked prompt example:

```yaml
model: sonnet
system_prompt: !file system_prompt.md
task_prompt: !file grammar.md
```

### Model Registry (`llm/models.yaml`)

`llm/models.yaml` is ejected during `ankiops init` and is the source of truth for available models. You can add any OpenAI-compatible provider/model by defining an entry with a `base_url`, `api_key`, and `model_id`.

```yaml
- model: qwen3-32b
  model_id: qwen3-32b
  provider: my-openai-compatible
  base_url: https://api.example.com/v1
  api_key: $EXAMPLE_API_KEY
```

`api_key` accepts either an env-var reference (`$EXAMPLE_API_KEY`) or a literal API key string.

Pricing fields are optional (`input_usd_per_mtok`, `output_usd_per_mtok`) and only used for cost estimates.

### Runtime Behavior

- `ankiops llm` validates all task configs and exits non-zero on errors
- AnkiOps creates a pre-LLM git snapshot unless `--no-auto-commit` is passed
- `ankiops llm <task>` prints the full prompt (`<system> ... </system>` + `<task> ... </task>`) used for planning; file paths are shown only when `system_prompt` or `task_prompt` use `!file`
- Only notes in scope with at least one editable, non-empty field are sent to the model
- If any note errors, staged note edits are not persisted
- Every job is recorded in `llm/.llm.db` with per-note status, token usage, latency, and errors
- Use `ankiops llm --job <job_id|latest>` for one job's history and diagnostics
