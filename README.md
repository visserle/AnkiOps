# AnkiOps

[![Tests](https://github.com/visserle/AnkiOps/actions/workflows/test.yml/badge.svg)](https://github.com/visserle/AnkiOps/actions/workflows/test.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://img.shields.io/pypi/v/ankiops.svg)](https://pypi.org/project/ankiops/)

AnkiOps is a bidirectional bridge between Anki and your filesystem. Each deck becomes a Markdown file so you can manage your collection from your favorite text editor. Edit in plain text, version with Git, enhance with LLMs, and sync changes both ways.

## Advantages

- ✏️ Edit Anki decks as highly readable Markdown files
- 🔄 Two-way sync of notes, note types, decks and media files
- ⚡ Sync thousands of notes in under a second
- ⚙️ Bring your own note types
- ✨ Improve your flashcards with programmable LLM tasks
- 👥 Share decks on GitHub and collaborate with others

## How It Works

### Markdown Files

One Markdown file is one deck. In a deck, each note is separated by a blank line, three dashes, and another blank line:

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

You can use any Markdown syntax (except the horizontal rule) in the note content, including italics, bold text, lists, tables, images, math equations, syntax-highlighted code blocks, and more. AnkiOps automatically converts Markdown to HTML for Anki and back again.

After the first sync with Anki, AnkiOps adds metadata comments for each note:

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

![image with set width](media/im.png){width=700}

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

The `note_key` is a stable identifier independent of Anki's note IDs and it is used to track notes across syncs. The `note_type` comment is added just for the user's reference. Neither comment should be edited by hand. The `tags` comment  is user-editable and synced with Anki's tags.

### Collection Structure

This is the basic structure of an AnkiOps collection:

````
├── note_types/
│   ├── AnkiOpsQA/
│   │   ├── Front.template.anki
│   │   ├── Back.template.anki
│   │   └── note_type.yaml
│   ├── AnkiOpsCloze/
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

### Note Types

> [!NOTE]
> AnkiOps only acts on note types defined within the `note_types/` folder.

AnkiOps automatically infers the note type for each note by a unique set of identifying field labels (e.g. `Q:` for Question and `A:` for Answer, which by default is `AnkiOpsQA`). Labels are defined in the `note_type.yaml` file within each note type folder. They define field names, field labels (identifying and non-identifying), card templates, and the styling. Note types are fully customizable. 

| Default note type | Identifying labels | 
| --- | --- | 
| `AnkiOpsQA` | `Q:`, `A:` | 
| `AnkiOpsCloze` | `T:` | 
| `AnkiOpsClozeHideAll` | `THA:`  |
| `AnkiOpsReversed` | `F:`, `B:`  |
| `AnkiOpsInput` | `Q:`, `I:` |
| `AnkiOpsChoice` | `Q:`, choice labels such as `C1:`, plus `A:`  |
| `AnkiOpsImageOcclusion` | `IO_*:` labels for image occlusion fields |

Generic, non-identifying labels such as `E:` for Extra can be added to any note type. To see the assigned labels in your collection, run `ankiops note-types`, or look up the note type definitions in `note_types/` manually. Note type inference depends on unique sets of identifying labels. All notes managed by AnkiOps have an additional field called `AnkiOps Key` that stores the `note_key` in Anki. 

### Synchronization

AnkiOps has two sync commands, which make one side match the other:

- `ankiops af` (anki to files): After editing notes, tags, or decks in Anki, and
- `ankiops fa` (files to anki): After editing Markdown decks, media, or note types.

Both sync operations create, update, move, and delete managed notes, and handle media files and note types. Before syncing, an automatic Git snapshot is created.

### Add-On

For basic usage, you can use AnkiOps without the add-on. The add-on enables AnkiOpsConnect, which AnkiOps needs for operations related to sharing. If you do not want to install the add-on, you can use AnkiOps with AnkiConnect (AnkiOpsConnect is twice as fast though). 

Another feature of the add-on are the toolbar buttons for `af` and `fa`:

<img src="toolbar.png" alt="alt text" width="450" />

To install the add-on, download the folder and put it in your Anki add-ons directory.

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

3. **Sync files to Anki**: Apply the current files, including Markdown decks, media, and note types, to Anki.

```bash
ankiops fa
```

4. **Sync Anki back to files**: After you review or edit cards in Anki, apply those changes to your local files. Each sync makes one side match the other. Inspect the Git diff to easily track all changes.

```bash
ankiops af
```

## FAQ

### Why should I use AnkiOps?

There is a joke among software engineers that eventually every filesystem grows into a database, and every database grows into a filesystem. Both approaches have their merits. AnkiOps enables the filesystem solution for the Anki database. It mirrors a full collection in a folder of your choice, where each deck becomes a Markdown file, and media and note-type definitions are stored along with it. Having your collection represented in the filesystem enables straightforward version control, automation, and collaboration.

### How is this different from other Markdown tools?

Most Markdown-to-Anki tools import one way: you write Markdown and push it to Anki. AnkiOps lets you edit in either place and sync back. It stores each deck as one Markdown file, so you browse decks instead of hundreds of per-card files. It also keeps custom note type definitions beside your decks, which lets you edit both card content and card structure from your editor.

### Is it safe to use?

Yes, AnkiOps will never modify notes that are not defined within the `note_types/` folder. Your existing collection won't be affected, and you can safely mix managed and unmanaged notes within one deck. Additionally, AnkiOps only syncs if the activated profile matches the one it was initialized with. For Markdown files, AnkiOps automatically creates a Git commit of your collection folder before every sync, so you can always revert changes if needed.

### What is the recommended workflow?

We recommend VS Code because it has a stable Markdown previewer and handles images with the dedicated `media/` folder well.

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

## How does sharing work? (experimental)

Shared decks are ordinary GitHub repositories cloned inside the collection. The
collection root remains one VS Code workspace, while VS Code shows each shared
source as its own repository.

Publish a local deck tree:

```bash
ankiops shared publish "Psychology" owner/psychology-deck
```

`shared publish` requires stable `note_key` metadata and authenticated GitHub CLI.
It moves the selected deck tree into `shared/<owner>/<repo>/`, copies referenced
media and note types, creates an independent repository, and pushes it to GitHub.
Published repositories are always public.

Subscribe to and update a shared deck:

```bash
ankiops shared subscribe owner/psychology-deck
ankiops shared update owner/psychology-deck
ankiops fa
```

Submit local shared edits:

```bash
ankiops shared status owner/psychology-deck
ankiops shared submit owner/psychology-deck \
  --message "Clarify attention terminology"
```

`shared update` changes files only; run `ankiops fa` after reviewing them.
`shared submit` commits changes only inside the selected source and opens a pull
request. Contributors without write permission use an authenticated fork
automatically. If local and upstream edits overlap, the subscribed deck remains
unchanged. AnkiOps preserves editable base, local, and upstream copies; edit the
marked Markdown it reports and rerun `shared update`.

If GitHub authentication, upload, or pull-request creation fails, AnkiOps keeps
the commit on the recovery branch shown in the output. Rerun the reported
`shared submit` command to reuse it. Your local Git configuration supplies the
commit author; the authenticated GitHub account uploads it and opens the pull
request. Submit output shows both identities.

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

Shared deck tools:

- `ankiops shared publish <deck> <owner>/<repo>`
- `ankiops shared subscribe <owner>/<repo>`
- `ankiops shared status [owner/repo]`
- `ankiops shared update [owner/repo]`
- `ankiops shared submit <owner>/<repo> [-m|--message text]`

## Contributing

Bug fixes, documentation improvements, tests, and PRs are welcome!

## License

MIT
