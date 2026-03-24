# AnkiOps

[![Tests](https://github.com/visserle/AnkiOps/actions/workflows/test.yml/badge.svg)](https://github.com/visserle/AnkiOps/actions/workflows/test.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://img.shields.io/pypi/v/ankiops.svg)](https://pypi.org/project/ankiops/) 
[![Hypercommit](https://img.shields.io/badge/Hypercommit-DB2475)](https://hypercommit.com/ankiops)

**Anki ↔ Markdown, with bidirectional sync, custom note types, and LLM integration**

**AnkiOps** is a bidirectional bridge between Anki and your filesystem. Each deck becomes a Markdown file you can edit directly, track with Git, and enhance with automated LLM tasks. With two-way sync, media support, and custom note types, AnkiOps lets you manage your entire collection from your favorite editor.

## Features

- Simple CLI interface: `ankiops init`, `ankiops markdown-to-anki`, `ankiops anki-to-markdown`, `ankiops note-types`, `ankiops llm`
- Fully round-trip sync that handles notes (creation, deletion, movements across decks, conflicts), note types, and media
- Markdown rendering with nearly all features (including syntax-highlighted code blocks for desktop and mobile)
- Support for custom note types following Infrastructure as Code (IaC) principles
- Built-in Git integration with auto-commit for tracking all changes
- High-performance processing using hashing: handles thousands of cards across hundreds of decks in mere seconds
- Serialize entire collections to JSON format for automated AI processing

## How to Get Started


1. **Install AnkiOps via [pipx](https://github.com/pypa/pipx)**: Pipx will make AnkiOps globally available in your terminal.

```bash
pipx install ankiops
```

2. **Initialize AnkiOps**: Make sure that Anki is running, with the [AnkiConnect add-on](https://ankiweb.net/shared/info/2055492159) enabled. Initialize AnkiOps in any empty directory of your choosing. This is where your text-based decks will live. The additional tutorial flag creates a sample Markdown deck.

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

AnkiOps reads note types exclusively from your local `note_types/` directory. `ankiops init` ejects default note types as bootstrap files; those local files are then the only source of truth and can be modified as needed. Each note type is identified by a unique set of field labels. These labels are defined in`note_types/name/note_type.yaml` and can be customized as needed. For an overview of the current configuration, use `ankiops note-types`.

### How does it work?

AnkiOps assigns a stable `note_key` to each managed note. It is represented by a single-line HTML tag (e.g., `<!-- note_key: a1b2c3d4e5f6 -->`) above a note in the Markdown. AnkiOps note keys are profile-independent, in contrast to Anki's note IDs. One AnkiOps folder represents one Anki profile. The `.ankiops.db`database stores the mapping between Anki's note IDs and AnkiOps note keys, along with other metadata. When syncing, AnkiOps uses these note keys to determine which notes to create, update, or delete in either Anki or Markdown. Media files are stored in a `media/` folder with hashed file names to avoid conflicts.
### What is the recommended workflow?

We recommend using VS Code. It has excellent AI integration, a native Markdown previewer, and supports image pasting from the clipboard directly into the `/media` folder (automatically set up).

### How can I share my AnkiOps collection?

TODO

### How can I migrate my existing notes into AnkiOps?

For standard note types, migration is straightforward:

1. Convert your existing notes to the matching AnkiOps note types via `Change Note Type…` in the Anki browser.
2. Export your notes from Anki to Markdown using `ankiops am`.
3. In the first re-import, some formatting may change because the original HTML from Anki may not follow the CommonMark standard. Formatting of your cards can be done automatically at a low cost using the included JSON serializer and AI tooling.

If your existing note format doesn't map cleanly to the AnkiOps format (e.g., notes with additional or custom fields), you'll need to adapt the code accordingly. This should be fairly simple for most cases: define your note type with unique field labels (and unique field names within that note type) for automatic note type detection, and add your card's templates to `ankiops/note_types`.

### How can I develop AnkiOps locally?

Fork this repository and initialize the tutorial in your root folder (make sure Anki is running). This will create a folder called `collection` with the sample Markdown in it. Paths will adapt automatically to the development environment. You can run AnkiOps locally using the main script.

```bash
git clone https://github.com/visserle/ankiops.git
cd ankiops
uv sync
uv run python -m main init --tutorial
uv run python -m main ma
```

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
- `ankiops llm <task_name> [--model <opus|sonnet|haiku>] [--deck <deck_name>]` - Plan one configured task
- `ankiops llm <task_name> --run [--model <opus|sonnet|haiku>] [--online|--batch] [--deck <deck_name>] [--no-auto-commit]` - Run one configured task job
- `ankiops llm --job <job_id|latest>` - Show one LLM job in detail
- `ankiops llm --job <job_id|latest> --resume [--online|--batch] [--no-auto-commit]` - Resume unfinished/error items from a prior job

**`note-types`:**
- `ankiops note-types` - Show note types, identifying labels, and the label registry
- `ankiops note-types --import <name>` - Copy a note type from Anki into local `note_types/` with interactive label/identifying prompts

## LLM Integration

AnkiOps includes a LLM pipeline for repeatable task execution. 

> [!NOTE]
> LLM integration is still experimental and subject to change.

After `ankiops init`, AnkiOps bootstraps:

- `llm/system_prompt.md`
- `llm/tasks/grammar.yaml`
- `llm/prompts/grammar.md`
- `llm/llm.db` (job history, auto-added to `.gitignore`)

Set your Anthropic key before running tasks:

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

Plan, run, and inspect jobs:

```bash
ankiops llm                         # status dashboard (tasks + recent jobs)
ankiops llm grammar                 # dry-run plan
ankiops llm grammar --run           # run task job
ankiops llm grammar --run --batch
ankiops llm grammar --run --online
ankiops llm grammar --deck Biology  # one exact deck (subdecks excluded)
ankiops llm grammar --run --model haiku
ankiops llm --job latest
ankiops llm --job latest --resume
```
### Task File Format (`llm/tasks/<task-name>.yaml`)

```yaml
model: sonnet
prompt_file: ../prompts/grammar.md
system_prompt_file: ../system_prompt.md
api_key_env: ANTHROPIC_API_KEY
timeout_seconds: 60

decks:
  include: ["*"]
  exclude: []
  include_subdecks: true

fields:
  exceptions:
    - hidden: ["AI Notes"]
    - note_types: ["AnkiOpsChoice"]
      read_only: ["Answer"]

request:
  temperature: 0
  max_output_tokens: 2048
  retries: 2
  retry_backoff_seconds: 0.5
  retry_backoff_jitter: true
```

- Required keys: `model`, `prompt_file`
- Supported models: `opus`, `sonnet`, `haiku`
- `prompt_file` is resolved relative to the task file and must stay within `llm/`
- `system_prompt_file` is optional (defaults to `llm/system_prompt.md`), resolved relative to the task file, and must stay within `llm/`
- `api_key_env` defaults to `ANTHROPIC_API_KEY` if omitted
- `decks.include` defaults to `["*"]`, `decks.exclude` defaults to `[]`, and `decks.include_subdecks` defaults to `true`
- CLI override: `ankiops llm <task> --deck <name>` forces one exact deck with `include_subdecks=false`
- `decks` patterns use shell-style matching (`*`, `?`, character classes); non-wildcard names match exact deck names (and optionally subdecks)
- `fields.exceptions` controls per-note-type field access: `read_only` fields are sent for context but cannot be edited, while `hidden` fields are omitted from LLM input/output
- `request` tuning defaults: `retries=2`, `retry_backoff_seconds=0.5`, `retry_backoff_jitter=true`, `max_output_tokens=2048`
- `execution` is optional; defaults are `mode=online`, `concurrency=8`, `fail_fast=true`, `batch_poll_seconds=15`
- `execution.mode` chooses `online` (concurrent Messages API) or `batch` (Message Batches API)
- `execution.concurrency` applies to `online` mode; `execution.batch_poll_seconds` applies to `batch` mode
- `execution.fail_fast=true` cancels pending online work on fatal failures
- Batch mode example:
  ```yaml
  execution:
    mode: batch
    batch_poll_seconds: 15
    fail_fast: true
  ```

### Runtime Behavior

- `ankiops llm` validates all task configs and exits non-zero on errors
- AnkiOps creates a pre-LLM git snapshot unless `--no-auto-commit` is passed
- `ankiops llm <task>` prints the resolved system/task prompt file paths and the full prompt (`<system> ... </system>` + `<task> ... </task>`) used for planning
- Only notes in scope with at least one editable, non-empty field are sent to the model
- Jobs use an atomic failure policy by default: if any note errors, staged note edits are not persisted
- Every job is recorded in `llm/llm.db` with per-note status, token usage, latency, and errors
- Use `ankiops llm --job <job_id|latest>` for one job's history and diagnostics
