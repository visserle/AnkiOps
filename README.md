# AnkiOps

[![Tests](https://github.com/visserle/AnkiOps/actions/workflows/test.yml/badge.svg)](https://github.com/visserle/AnkiOps/actions/workflows/test.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://img.shields.io/pypi/v/ankiops.svg)](https://pypi.org/project/ankiops/)

AnkiOps is a bidirectional bridge between Anki and the filesystem. Each deck becomes a Markdown file so you can manage your collection from your favorite text editor. Edit in plain text, version with Git, enhance with LLMs, and sync changes both ways.

## Advantages

- ✏️ Edit Anki decks as highly readable Markdown files
- 🔄 Two-way sync of notes, note types, decks and media files
- ⚡ Sync thousands of notes in under a second
- ⚙️ Bring your own note types
- ✨ Improve content with programmable LLM tasks
- 👥 Share decks on GitHub and collaborate with others

## How It Works

### 1 Markdown Files

AnkiOps is Markdown-first: Each file represents an Anki deck and notes are separated by a blank line, three dashes, and another blank line. Already existing decks can be exported from Anki, or created from scratch as in the following example: 

```markdown
Q: Question text here
A: Answer text here:
Multiple lines supported
E: Extra information (optional)
M: Content behind a "more" button (optional)

---

T: Text with
- {{c1::multiple}}
- {{c2::cloze deletions}}.
E: Some *extra* info:

![image with set width](media/im.png){width=700}

---

<!-- tags: exam -->
Q: What is this?
C1: A multiple choice note
C2: with
C3: automatically randomized answers
A: 1, 3
```

The three notes represent a question-answer note, a cloze deletion note, and a multiple-choice note. Notes types are inferred automatically by the used labels (e.g. `Q:`, `A:`) for the card fields (Question, Answer). Labels and note types are fully customizable.

You can use any Markdown syntax (except the horizontal rule) in the note content, including italics, bold text, lists, tables, images, math equations, syntax-highlighted code blocks, and more. AnkiOps automatically converts Markdown to HTML for Anki, and back to Markdown when syncing from Anki.

After the first import into Anki (`ankiops files-to-anki`), AnkiOps adds metadata comments for each note:

```markdown
<!-- note_key: 2fd62bcaa861 -->
<!-- note_type: AnkiOpsQA -->
Q: Question text here
A: Answer text here:
Multiple lines supported
E: Extra information (optional)
M: Content behind a "more" button (optional)

---

<!-- note_key: ef0108255d7d -->
<!-- note_type: AnkiOpsCloze -->
T: Text with
- {{c1::multiple}}
- {{c2::cloze deletions}}.
E: Some *extra* info:

![image with set width](media/im_hash.png){width=700}

---

<!-- note_key: 332e64bba6fe -->
<!-- note_type: AnkiOpsChoice -->
<!-- tags: exam -->
Q: What is this?
C1: A multiple choice note
C2: with
C3: automatically randomized answers
A: 1, 3
```

- The `note_key` is a stable identifier independent of Anki's note IDs and it is used to track notes across syncs. 
- The `note_type` comment is added just for the user's reference. Neither comment should be edited by hand.
- The `tags` comment  is user-editable and synced with Anki's tags.

Markdown files can also be created by exporting existing notes from Anki with `ankiops anki-to-files`, as explained in the following sections.

### 2 Collection Structure

This is the basic structure of an AnkiOps collection:

````
├── note_types/
│   ├── AnkiOpsQA/
│   │   ├── Front.template.anki
│   │   ├── Back.template.anki
│   │   └── note_type.yaml
│   ├── AnkiOpsCloze/
│   ├── AnkiOpsChoice/
│   ├── AnkiOpsStyling.css
│   └── SyntaxHighlighting.css
├── media/
│   └── image1_hash.png
├── llm/
├── .ankiops.db
├── Deck1.md
└── Deck1__Subdeck1.md
````

The `.ankiops.db` file is the heart of AnkiOps. It connects the `note_key` values in the Markdown files to Anki's internal note IDs.

### 3 Note Types

> [!NOTE]
> AnkiOps only acts on note types defined within the `note_types/` folder.

AnkiOps automatically infers the note type for each note by a set of identifying field labels (e.g. a note with the labels `Q:` for the field Question and `A:` for Answer dafults to `AnkiOpsQA`). For the inference, each note type must have unique sets of identifying labels. The labels are defined in the `note_type.yaml` file within each note type folder. In the yaml file, you define field names, field labels (identifying and non-identifying), card templates, and the styling. Note types are fully customizable. 

If you want AnkiOps to manage your own note types, create a folder in `note_types/` with the note type name and a `note_type.yaml` file and the required templates. The following table shows the default AnkiOps note types and their identifying labels, which can be changes as needed.

| Default note type | Identifying labels | 
| --- | --- | 
| `AnkiOpsQA` | `Q:`, `A:` | 
| `AnkiOpsCloze` | `T:` | 
| `AnkiOpsClozeHideAll` | `THA:`  |
| `AnkiOpsReversed` | `F:`, `B:`  |
| `AnkiOpsInput` | `Q:`, `I:` |
| `AnkiOpsChoice` | `Q:`, choice labels such as `C1:`, plus `A:`  |
| `AnkiOpsImageOcclusion` | `IO_*:` labels for image occlusion fields |

Generic, non-identifying labels such as `E:` for Extra can be added to any note type. To see the assigned labels in your collection, run `ankiops note-types`, or look up the note type definitions in `note_types/` manually. All notes managed by AnkiOps have an additional field called `AnkiOps Key` that stores the `note_key` in Anki. 

### 4 Synchronization

AnkiOps has two sync commands: 
- `ankiops anki-to-files` or short **`ankiops af`** for syncing Anki to files, and 
- `ankiops files-to-anki` or short **`ankiops fa`** for syncing files to Anki.

The commands apply changes from one side to the other. Both operations create, update, move, and delete managed notes, and handle note types and media files. Before syncing, an automatic Git snapshot is created.

### 5 LLM Integration

LLM tasks apply programmable edits to existing notes, from content review and grammar fixes to translation. In contrast to prompting an AI agent to edit the Markdown files, LLM tasks send batches of serialized notes (JSON) to the model. This ensures that the LLM only sees the content it needs to interact with and does not skip any notes. LLM tasks are fully customizable and can be run on the entire collection or a single deck. 

Please refer to the [LLM task guide](https://github.com/visserle/AnkiOps/blob/main/ankiops/llm/resources/README.md) for more details on creating and customizing LLM tasks.

### 6 Add-On

For basic usage, you can use AnkiOps without the add-on. The add-on enables AnkiOpsConnect, which AnkiOps needs for operations related to collaboration. If you do not want to install the add-on, you can use AnkiOps with AnkiConnect (AnkiOpsConnect is twice as fast though). 

Another feature of the add-on are the toolbar buttons for `af` and `fa`:

<img src="toolbar.png" alt="alt text" width="450" />

To install the add-on, download the folder and put it in your Anki add-ons directory.

### 7 Collaboration

AnkiOps supports collaborative decks on GitHub. You can publish a deck as a public GitHub repository and collaborate through pull requests. Each shared deck lives at `collab/<owner>/<repo>/` as its own Git repository, alongside your private decks. `ankiops collab publish` moves the selected deck and its referenced media and note types into that folder, then publishes the repository to GitHub.

Others can add the deck to their collection with `ankiops collab subscribe <owner>/<repo>`. Subscribers edit its Markdown files and send changes through pull requests. You review and merge those pull requests on GitHub. See the [public example collab deck](https://github.com/visserle/Collaborate-With-AnkiOps) for setup and the complete workflow. Available collab commands are:

- `ankiops collab publish <deck> <owner>/<repo>`: Publish a local deck tree as a public GitHub repository.
- `ankiops collab subscribe <owner>/<repo>`: Add a public collab deck to your collection.
- `ankiops collab status [owner/repo]`: Show local changes, upstream updates, and pull-request state for one or all collab decks.
- `ankiops collab update <owner>/<repo>`: Bring GitHub changes into one collab deck.
- `ankiops collab submit <owner>/<repo> [-t|--title text]`: Open or update a pull request with your changes.

## How To Get Started

1. Install AnkiOps via [uv](https://docs.astral.sh/uv/getting-started/installation/) from PyPI. This will make the `ankiops` command available in your shell in an isolated virtual environment. No need to get Python separately.

```bash
uv tool install ankiops
```

2. **Initialize AnkiOps**: Make sure that Anki is running, with Anki(Ops)Connect enabled. Run `ankiops init` in an empty directory of your choice. AnkiOps creates a Git repository, `.ankiops.db`, the `note_types/` folder, and recommended configurations for VS Code. The tutorial flag further creates a sample Markdown deck.

```bash
cd my-collection
ankiops init --tutorial
```

3. **Sync files to Anki**: Apply the current files, including Markdown decks, media, and note types, to Anki. `fa` is short for `files-to-anki`.

```bash
ankiops fa
```

4. **Sync Anki back to files**: After you review or edit cards in Anki, apply those changes to your local files. Each sync makes one side match the other. Inspect the Git diff to easily track all changes.

```bash
ankiops af # anki-to-files
```

## FAQ

### Why should I use AnkiOps?

There is a joke among software engineers that eventually every filesystem grows into a database, and every database grows into a filesystem. Both approaches have their merits. AnkiOps enables the filesystem solution for the Anki database. It mirrors a full collection in a folder of your choice, where each deck becomes a Markdown file, and media and note-type definitions are stored along with it. Having your collection represented in the filesystem enables straightforward version control, automation, and collaboration.

### How is this different from other Markdown tools?

Most Markdown-to-Anki tools import one way: you write Markdown and push it to Anki. AnkiOps lets you edit in either place and sync back. It stores each deck as one Markdown file, so you browse decks instead of hundreds of per-card files. It also keeps custom note type definitions beside your decks, which lets you edit both card content and card structure from your editor.

### Is it safe to use?

Yes, AnkiOps will never modify notes that are not defined within the `note_types/` folder. Your existing collection won't be affected, and you can safely mix managed and unmanaged notes within one deck. Additionally, AnkiOps only syncs if the activated profile matches the one it was initialized with. A unique hash is appended to media file names to prevent conflicts. AnkiOps automatically creates a Git commit of your collection folder before every sync, so you can always revert changes if needed.

### What is the recommended workflow?

We recommend VS Code because it has a stable Markdown previewer and handles images with the dedicated `media/` folder well. AnkiOps ejects a configuration for VS Code that works out of the box with all features provided by AnkiOps.

### How can I migrate my existing notes into AnkiOps?

You can migrate existing notes in three ways: write matching note type files in `note_types/`, copy note types from Anki with `ankiops note-types --add <name>`, or convert notes to the default AnkiOps note types with `Change Note Type…` in the Anki browser.

For Anki browser conversion, use this workflow:

1. Convert your existing notes to the matching AnkiOps note types via `Change Note Type…` in the Anki browser.
2. Export your notes from Anki to Markdown using `ankiops am`.
3. Review the Git diff after the first re-import. Original Anki HTML may not match CommonMark, so the first Markdown-to-Anki sync can change formatting slightly. Formatting issues can be fixed by hand or via the LLM task fix-html.

### How can I make use of LLMs with AnkiOps?

If you want it simple, just prompt an LLM (Codex, Claude Code, etc.) to edit your Markdown files. If you want it integrated, AnkiOps has a dedicated LLM module that runs through your serialized (JSON) collection and applies edits to the Markdown files. Only works with OpenAI's response API.

### How do I upgrade AnkiOps to the latest version?

AnkiOps is in early development, so breaking changes are expected. Use `uv upgrade ankiops` to upgrade AnkiOps. Delete local AnkiOps support files except your Markdown decks, re-initialize the same folder with `ankiops init`, and export from Anki with `ankiops am`. Since all files are git-tracked, you can easily spot any changes and roll back if needed.

### What other features are there?


**Image Width Normalization**

Handling multiple images in a single Anki note often leads to inconsistent widths. AnkiOps can normalize similar widths or force a width across a deck:

```bash
ankiops fix-image-widths  # normalize widths within 10px tolerance
ankiops fix-image-widths --deck "Deck1" --width 500  # fix width
```

**JSON serialization**

Serialize a collection (or one deck tree) to portable JSON and deserialize it back using the `ankiops serialize` and `ankiops deserialize` commands. 


## Command Reference

Global options:

- `--debug`: Enable debug logging.
- `--version`: Show the installed version.
- `--help`: Show help.

Core commands:

- `ankiops init [--tutorial]`: Initialize the current directory as an AnkiOps collection.
- `ankiops anki-to-files` or `ankiops af`: Sync Anki changes into files.
- `ankiops files-to-anki` or `ankiops fa`: Sync file changes into Anki.
- `ankiops note-types`: Show note types and the label registry.
- `ankiops note-types --add <name>`: Copy a note type from Anki into `note_types/`.

Sync flags:

- `--no-auto-commit`, `-n`: Skip the pre-sync Git snapshot for `af`, `fa`, `deserialize`, `fix-image-widths`, and `llm <task> --run`.

Serialization:

- `ankiops serialize [-o path] [--deck name] [--no-subdecks]`
- `ankiops deserialize [-i path] [--overwrite]`

Image tools:

- `ankiops fix-image-widths [--deck name] [--no-subdecks] [--tolerance px] [--width px]`

LLM tools:

- `ankiops llm`: Show configured tasks and recent jobs.
- `ankiops llm <task> [--model model] [--deck deck]`: Show a dry-run plan.
- `ankiops llm <task> --run [--model model] [--deck deck]`: Run a task.
- `ankiops llm --job <id|latest>`: Inspect one job.

Collab deck tools:

- `ankiops collab publish <deck> <owner>/<repo>`
- `ankiops collab subscribe <owner>/<repo>`
- `ankiops collab status [owner/repo]`
- `ankiops collab update <owner>/<repo>`
- `ankiops collab submit <owner>/<repo> [-t|--title text]`

## Contributing

Bug fixes, documentation improvements, tests, and PRs are welcome!

## Inspired by

- [hashcards](https://github.com/eudoxia0/hashcards): Markdown-first flashcards
- [AnkiCollab](https://github.com/CravingCrates/AnkiCollab-Plugin): Collaborative Anki decks

## License

MIT
